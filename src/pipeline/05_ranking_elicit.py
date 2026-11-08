"""
Stage 05: LLM ranking elicitation.

For each of the 60 SKUs (treated as out-of-stock), present its top-K
embedding neighbours to multiple LLMs and collect substitutability
rankings.  Each (model, SKU) pair is ranked under multiple random
permutations of the candidate order to mitigate positional bias.

Inputs
------
- data/processed/ModelInput60_formatted.jsonl  (pre-rendered product blocks)
  Produced by stage 04b_format_llm_inputs.py - run that first.
- data/processed/ModelInput60.jsonl            (raw product attributes,
  used only for title lookup in smoke-test output)
- outputs/embeddings/all_pairs_scores.parquet  (embedding similarities)

Outputs
-------
- outputs/rankings/raw_rankings.parquet
    One row per (model, sku, permutation) with the returned ranking.
- outputs/rankings/debug/
    One .txt file per failed attempt, containing model, SKU, perm,
    prompt, raw response, and exception - for post-mortem debugging.

Resume-safe: existing rows in raw_rankings.parquet are skipped on
restart.  Safe to Ctrl-C and re-run.

Pipeline position
-----------------
04b_format_llm_inputs.py  ->  ModelInput60_formatted.jsonl  (run first)
05_ranking_elicit.py      ->  raw_rankings.parquet           (this script)

Usage
-----
    # 0. Pre-render product blocks (run once before this script):
    python 04b_format_llm_inputs.py --preview

    # 1. Validate prompts and pipeline -- no API calls, no tokens:
    python 05_ranking_elicit.py --dry-run

    # 2. Smoke-test ONE real call per model (1 SKU x 1 perm):
    python 05_ranking_elicit.py --test

    # 3. Run a specific model only (start with local models):
    python 05_ranking_elicit.py --model deepseek-r1:32b
    python 05_ranking_elicit.py --model qwen3.6:27b

    # 4. Run all local models:
    python 05_ranking_elicit.py --local-only

    # 5. Full run (local first, then frontier):
    python 05_ranking_elicit.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd

# Load .env from project root
try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

# Project imports
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (
    MODEL_INPUT_FILE,
    FORMATTED_MODEL_INPUT_FILE,
    ALL_PAIRS_SCORES,
    RANKINGS_DIR,
    RANKINGS_FILE,
    EMBEDDING_TOP_K,
    RANKING_REPEATS,
    RANKING_MODELS_FRONTIER,
    RANKING_MODELS_LOCAL,
    RANKING_MODELS_ALL,
    OLLAMA_BASE_URL,
    RANKING_TEMPERATURE,
    RANKING_MAX_TOKENS,
    RANDOM_SEED,
)

DEBUG_DIR = os.path.join(RANKINGS_DIR, "debug")


# Data loading

def load_products(path: str) -> dict[str, dict]:
    """Load ModelInput60.jsonl -> {parent_asin: record}."""
    products: dict[str, dict] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            products[rec["parent_asin"]] = rec
    return products


def load_formatted_products(path: str) -> dict[str, str]:
    """Load ModelInput60_formatted.jsonl -> {parent_asin: formatted_block}."""
    blocks: dict[str, str] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            blocks[rec["parent_asin"]] = rec["formatted_block"]
    return blocks


def load_top_k_candidates(scores_path: str, k: int) -> dict[str, list[str]]:
    """
    Load all-pairs embedding scores and return, for each SKU, the
    top-k most similar other SKUs (by cosine similarity).
    """
    df = pd.read_parquet(scores_path)
    candidates: dict[str, list[str]] = {}
    for asin in df["asin_a"].unique():
        subset = (
            df[df["asin_a"] == asin]
            .sort_values("cosine_sim", ascending=False)
            .head(k)
        )
        candidates[asin] = subset["asin_b"].tolist()
    return candidates


# Prompt building

def build_prompt(
    oos_block: str,
    candidate_blocks: list[str],
    candidate_asins: list[str],
) -> str:
    """
    Build the ranking prompt for one (OOS SKU, candidate set) pair.

    /no_think is prepended to disable chain-of-thought on Qwen3 models,
    which would otherwise consume the entire token budget on hidden reasoning.
    """
    n = len(candidate_asins)
    asin_list = ", ".join(f'"{a}"' for a in candidate_asins)

    labelled_blocks = "\n\n".join(
        f"[{i}] {block}" for i, block in enumerate(candidate_blocks, 1)
    )

    prompt = f"""\
You are a customer who came to this grocery store to buy the product below, \
but it is out of stock. You must choose one of the listed alternatives or \
leave empty-handed.

OUT-OF-STOCK PRODUCT:
{oos_block}

AVAILABLE ALTERNATIVES:
{labelled_blocks}

Rank the alternatives from the one you would MOST likely put in your basket \
to the one you would LEAST likely accept as a substitute.

You are deciding based on which alternative best satisfies the same need as \
your original choice -- not which is most similar in the abstract. Use ONLY \
the product information shown above. Do not draw on any outside knowledge \
about brands, products, or retailers beyond what is explicitly stated.

Apply this reasoning in order:

1. FORMAT IS A HARD CONSTRAINT. If your original product is ground coffee, \
you will not substitute to pods or whole bean -- they require different \
equipment. A format mismatch means you would rather leave empty-handed. Rank \
format-incompatible alternatives at the very bottom.

2. CAFFEINE TYPE IS A HARD CONSTRAINT. If your original product is decaf, \
you will not substitute to regular, and vice versa -- where this is stated. \
Rank caffeine-incompatible alternatives at the bottom alongside format mismatches.

3. ROAST AND FLAVOUR MATCH. The closer the roast level, flavour profile, and \
grind type stated in the attributes, the more willing you are to accept the \
substitute.

4. PRICE PROXIMITY -- and direction matters. You will reluctantly accept \
paying a little more, but you will not pay significantly more than your \
original budget. A customer buying a $9 bag will not substitute to a $25 bag \
-- the gap signals a completely different purchase. A customer buying a $25 bag \
may reluctantly accept a $15 alternative, but not a $9 one. A slightly less \
flavour-similar product at a closer price is often more acceptable than a \
flavour-similar product at a very different price.

5. ALL OTHER STATED ATTRIBUTES -- package size, weight, quantity, origin, \
certifications, specialty claims -- use these to break ties where formats, \
caffeine, roast, and price are otherwise close.

Return ONLY a JSON object. Do not include <think> tags, markdown fences, \
or any text before or after the JSON. Your entire response must be valid JSON.

{{
  "ranking": ["{candidate_asins[0]}", ...],
  "reasoning": "One sentence explaining your top choice."
}}

Rules:
- "ranking" must list exactly {n} ASINs from [{asin_list}], best substitute first.
- Use the ASIN identifiers, not bracket numbers.
- Every ASIN must appear exactly once.
- "reasoning" must be a single string sentence.
"""
    return prompt


# JSON extraction (balanced-brace parser)

def _extract_json(text: str | None) -> dict[str, Any]:
    """
    Extract a JSON object from raw LLM output.

    Handles:
    - None / empty input
    - <think>...</think> blocks (including unclosed ones)
    - Markdown ```json ... ``` fences
    - Leading/trailing commentary
    - Multiple JSON objects (takes first valid one)
    - Nested JSON (balanced-brace search)
    - Rejects JSON arrays when an object is expected
    """
    if text is None:
        raise ValueError("Model returned None")
    if not text.strip():
        raise ValueError("Model returned empty string")

    # 1. Strip complete <think>...</think> blocks
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)

    # 2. Strip unclosed <think> block (model hit token limit mid-think)
    text = re.sub(r"<think>.*$", "", text, flags=re.DOTALL)

    text = text.strip()

    # 3. Strip markdown fences
    text = re.sub(r"^```(?:json)?\s*\n?", "", text)
    text = re.sub(r"\n?```\s*$", "", text)
    text = text.strip()

    # 4. Find the first '{' and extract a balanced JSON object
    start = text.find("{")
    if start == -1:
        raise ValueError(f"No '{{' found in response. Got: {repr(text[:200])}")

    # Walk forward counting braces to find the matching '}'
    depth = 0
    end = -1
    in_string = False
    escape_next = False
    for i, ch in enumerate(text[start:], start=start):
        if escape_next:
            escape_next = False
            continue
        if ch == "\\" and in_string:
            escape_next = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i
                break

    if end == -1:
        raise ValueError(
            f"Unmatched braces in response. Got: {repr(text[start:start+200])}"
        )

    candidate = text[start:end + 1]

    try:
        return json.loads(candidate)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Balanced-brace extract produced invalid JSON: {e}. "
            f"Candidate: {repr(candidate[:300])}"
        ) from e


# Model clients

class ModelClient:
    """Abstract base for LLM API clients."""

    model_name: str
    last_raw_response: str | None = None  # set before _extract_json; aids debugging

    def rank(self, prompt: str) -> dict[str, Any]:
        raise NotImplementedError


class GeminiClient(ModelClient):
    """Google Gemini via google-genai SDK."""

    def __init__(self, model_name: str):
        from google import genai
        self.model_name = model_name
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            raise ValueError("GEMINI_API_KEY not set")
        self.client = genai.Client(api_key=api_key)

    def rank(self, prompt: str) -> dict[str, Any]:
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config={
                "temperature": RANKING_TEMPERATURE,
                "max_output_tokens": RANKING_MAX_TOKENS,
                "response_mime_type": "application/json",
            },
        )
        self.last_raw_response = response.text
        return _extract_json(response.text)


class OpenAIClient(ModelClient):
    """OpenAI GPT models via openai SDK."""

    def __init__(self, model_name: str):
        from openai import OpenAI
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OPENAI_API_KEY not set")
        self.model_name = model_name
        self.client = OpenAI(api_key=api_key)

    def rank(self, prompt: str) -> dict[str, Any]:
        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=RANKING_MAX_TOKENS,
            response_format={"type": "json_object"},
        )
        raw = response.choices[0].message.content
        self.last_raw_response = raw
        finish = response.choices[0].finish_reason
        if finish == "length":
            print(f"\n  [WARNING] finish_reason=length — response truncated; "
                  f"consider raising RANKING_MAX_TOKENS")
        return _extract_json(raw)


class ClaudeClient(ModelClient):
    """Anthropic Claude via anthropic SDK."""

    def __init__(self, model_name: str):
        import anthropic
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")
        self.model_name = model_name
        self.client = anthropic.Anthropic(api_key=api_key)

    def rank(self, prompt: str) -> dict[str, Any]:
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=RANKING_MAX_TOKENS,
            messages=[{"role": "user", "content": prompt}],
        )
        raw = response.content[0].text
        self.last_raw_response = raw
        if response.stop_reason == "max_tokens":
            print(f"\n  [WARNING] stop_reason=max_tokens — response truncated; "
                  f"consider raising RANKING_MAX_TOKENS")
        return _extract_json(raw)


class OllamaClient(ModelClient):
    """
    Local models via Ollama's native /api/chat endpoint.

    We use the native endpoint (not the OpenAI-compat /v1/chat/completions)
    because the compat layer silently drops non-OpenAI fields like 'think'
    and 'format' from extra_body, making think:false ineffective for Qwen3.
    """

    def __init__(self, model_name: str):
        import urllib.request as _urllib
        self.model_name = model_name
        self._urllib = _urllib
        self._url = f"{OLLAMA_BASE_URL}/api/chat"

    def rank(self, prompt: str) -> dict[str, Any]:
        import json as _json
        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "think": False,       # disable Qwen3 chain-of-thought at the Ollama layer
            "stream": False,
            "format": "json",     # sampler-level JSON enforcement
            "options": {
                "num_ctx": 8192,              # override Ollama's 2048-token default
                "temperature": RANKING_TEMPERATURE,
                "num_predict": RANKING_MAX_TOKENS,
            },
        }
        req = self._urllib.Request(
            self._url,
            data=_json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        # timeout=1200: DeepSeek-R1:32b routinely takes 500-900s due to its
        # large thinking budget; 300s was too short and caused spurious timeouts.
        try:
            with self._urllib.urlopen(req, timeout=1200) as resp:
                body = _json.loads(resp.read())
        except Exception as e:
            raise RuntimeError(f"Ollama /api/chat request failed: {e}") from e

        raw = body.get("message", {}).get("content", "")
        self.last_raw_response = raw

        # Check for truncation
        done_reason = body.get("done_reason", "")
        if done_reason == "length":
            print(f"\n  [WARNING] done_reason=length — response truncated; "
                  f"consider raising RANKING_MAX_TOKENS or num_ctx")

        return _extract_json(raw)


def get_client(model_name: str) -> ModelClient:
    if model_name.startswith("gemini"):
        return GeminiClient(model_name)
    elif model_name.startswith("claude"):
        return ClaudeClient(model_name)
    elif model_name.startswith("gpt"):
        return OpenAIClient(model_name)
    else:
        return OllamaClient(model_name)


# Call + parse + validate

MAX_RETRIES = 3
RETRY_BACKOFF = [5, 15, 30]

RATE_LIMIT_DELAY = {
    "frontier": 1.0,
    "local": 0.0,
}


def _get_delay(model_name: str) -> float:
    if model_name in RANKING_MODELS_LOCAL:
        return RATE_LIMIT_DELAY["local"]
    return RATE_LIMIT_DELAY["frontier"]


def _save_debug(
    model: str,
    oos_asin: str,
    perm_idx: int,
    attempt: int,
    prompt: str,
    raw: str | None,
    exc: Exception,
) -> None:
    """Save a failed attempt to outputs/rankings/debug/ for post-mortem."""
    os.makedirs(DEBUG_DIR, exist_ok=True)
    safe_model = model.replace(":", "_").replace("/", "_")
    fname = f"{safe_model}__{oos_asin}__p{perm_idx}__a{attempt}.txt"
    fpath = os.path.join(DEBUG_DIR, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        f.write(f"model:    {model}\n")
        f.write(f"oos_asin: {oos_asin}\n")
        f.write(f"perm:     {perm_idx}\n")
        f.write(f"attempt:  {attempt}\n")
        f.write(f"error:    {exc}\n\n")
        f.write("--- RAW RESPONSE ---\n")
        f.write(repr(raw) if raw is not None else "None")
        f.write("\n\n--- PROMPT ---\n")
        f.write(prompt)


def call_and_parse(
    client: ModelClient,
    prompt: str,
    expected_asins: set[str],
    oos_asin: str = "",
    perm_idx: int = 0,
) -> dict[str, Any] | None:
    """
    Call the model, parse JSON, validate the ranking.
    Returns the parsed dict or None on failure after retries.
    Saves debug files on every failed attempt.
    """
    for attempt in range(MAX_RETRIES):
        try:
            result = client.rank(prompt)

            # Structural validation
            if "ranking" not in result:
                raise ValueError("Response missing 'ranking' key")

            ranking = result["ranking"]
            if not isinstance(ranking, list):
                raise ValueError(
                    f"'ranking' is {type(ranking).__name__}, expected list"
                )
            if not all(isinstance(x, str) for x in ranking):
                raise ValueError("'ranking' must contain only strings")

            returned = set(ranking)
            if returned != expected_asins:
                missing = expected_asins - returned
                extra = returned - expected_asins
                raise ValueError(
                    f"ASIN mismatch -- missing: {missing}, extra: {extra}"
                )
            if len(ranking) != len(expected_asins):
                raise ValueError(
                    f"Duplicate ASINs: got {len(ranking)} items, "
                    f"expected {len(expected_asins)}"
                )

            # Optional: validate reasoning field
            reasoning = result.get("reasoning", "")
            if not isinstance(reasoning, str):
                result["reasoning"] = str(reasoning)  # coerce, don't fail

            return result

        except (json.JSONDecodeError, ValueError) as e:
            raw = getattr(client, "last_raw_response", None)
            print(f"\n  [attempt {attempt + 1}] Parse/validation error: {e}")
            print(f"  Raw response length: {len(raw) if raw else 0} chars")
            if raw:
                print(f"  Raw response (first 500): {repr(raw[:500])}")
            _save_debug(
                client.model_name, oos_asin, perm_idx, attempt + 1,
                prompt, raw, e
            )

        except Exception as e:
            raw = getattr(client, "last_raw_response", None)
            print(f"\n  [attempt {attempt + 1}] API error: {type(e).__name__}: {e}")
            _save_debug(
                client.model_name, oos_asin, perm_idx, attempt + 1,
                prompt, raw, e
            )

        if attempt < MAX_RETRIES - 1:
            wait = RETRY_BACKOFF[attempt]
            print(f"  Retrying in {wait}s...")
            time.sleep(wait)

    return None


# Permutation logic

def get_permuted_candidates(
    candidates: list[str],
    oos_asin: str,
    perm_index: int,
    seed: int = RANDOM_SEED,
) -> list[str]:
    """
    Return a deterministic permutation of the candidate list.

    perm_index=0  original embedding order (highest cosine sim first).
    perm_index=1  fully reversed (lowest cosine sim first).
    perm_index=2,3  seeded random shuffles.

    Uses SHA-256 instead of Python's hash() for cross-process stability.
    (Python's built-in hash() is randomised per-process by default.)
    """
    if perm_index == 0:
        return list(candidates)
    if perm_index == 1:
        return list(reversed(candidates))

    digest = hashlib.sha256(
        f"{seed}|{oos_asin}|{perm_index}".encode("utf-8")
    ).digest()
    perm_seed = int.from_bytes(digest[:4], byteorder="big")
    rng = random.Random(perm_seed)
    shuffled = list(candidates)
    rng.shuffle(shuffled)
    return shuffled


# Checkpoint / resume

def load_completed(output_path: str) -> set[tuple[str, str, int]]:
    if not os.path.exists(output_path):
        return set()
    df = pd.read_parquet(output_path)
    return set(zip(df["model"], df["oos_asin"], df["perm_index"]))


def append_result(
    output_path: str,
    model: str,
    oos_asin: str,
    perm_index: int,
    candidate_order: list[str],
    ranking: list[str],
    reasoning: str,
    elapsed_s: float,
):
    row = pd.DataFrame([{
        "model": model,
        "oos_asin": oos_asin,
        "perm_index": perm_index,
        "candidate_order": json.dumps(candidate_order),
        "ranking": json.dumps(ranking),
        "reasoning": reasoning,
        "elapsed_s": round(elapsed_s, 2),
    }])

    if os.path.exists(output_path):
        existing = pd.read_parquet(output_path)
        combined = pd.concat([existing, row], ignore_index=True)
    else:
        combined = row

    combined.to_parquet(output_path, index=False)


# TESTING  (--dry-run and --test)

def _test_extract_json():
    """Unit-test _extract_json with a comprehensive set of cases."""
    cases = [
        ("clean JSON",
         '{"ranking": ["A", "B"], "reasoning": "A is best."}', True),
        ("markdown fences",
         '```json\n{"ranking": ["A"], "reasoning": "ok"}\n```', True),
        ("DeepSeek <think> block",
         '<think>reasoning...</think>\n{"ranking": ["A"], "reasoning": "ok"}', True),
        ("unclosed <think>",
         '<think>model hit token limit mid-think', False),
        ("leading text",
         'Here is my answer:\n{"ranking": ["A"], "reasoning": "ok"}', True),
        ("trailing text",
         '{"ranking": ["A"], "reasoning": "ok"}\nDone!', True),
        ("nested JSON in reasoning",
         '{"ranking": ["A", "B"], "reasoning": "A beats B because {score: 1}."}', True),
        ("empty string", "", False),
        ("None", None, False),
        ("JSON array instead of object", '["A", "B"]', False),
        ("invalid JSON", '{"ranking": ["A", "B"], "reasoning": }', False),
    ]

    passed = 0
    for name, raw, should_succeed in cases:
        try:
            result = _extract_json(raw)
            if should_succeed:
                print(f"   OK : {name}")
                passed += 1
            else:
                print(f"   FAIL (expected error, got result): {name} -> {result}")
        except (ValueError, json.JSONDecodeError) as e:
            if not should_succeed:
                print(f"   OK : {name} (correctly rejected: {e})")
                passed += 1
            else:
                print(f"   FAIL: {name}: {e}")

    return passed, len(cases)


def run_dry_run(products: dict, formatted_blocks: dict, candidates_map: dict):
    """Validate the full pipeline without any API calls."""
    print("\n" + "=" * 70)
    print("DRY RUN -- validating pipeline (zero API calls)")
    print("=" * 70)

    errors = []
    all_asins = sorted(products.keys())
    prompt_lengths: list[int] = []
    est_tokens: list[int] = []

    # 0. Formatted blocks coverage
    print(f"\n0. Formatted blocks coverage:")
    missing_blocks = [a for a in all_asins if a not in formatted_blocks]
    if missing_blocks:
        for a in missing_blocks:
            errors.append(f"  {a} missing from formatted blocks (run 04b first)")
        print(f"   X {len(missing_blocks)} ASINs missing")
    else:
        print(f"   All {len(all_asins)} ASINs have formatted blocks")

    # 1. Candidate coverage
    print(f"\n1. Candidate coverage:")
    for asin in all_asins:
        if asin not in candidates_map:
            errors.append(f"  {asin} has no candidates")
            continue
        cands = candidates_map[asin]
        if len(cands) != EMBEDDING_TOP_K:
            errors.append(f"  {asin}: expected {EMBEDDING_TOP_K}, got {len(cands)}")
        for c in cands:
            if c not in products:
                errors.append(f"  {asin}: candidate {c} not in products")
            if c not in formatted_blocks:
                errors.append(f"  {asin}: candidate {c} missing formatted block")
    if not errors:
        print(f"   All {len(all_asins)} SKUs have {EMBEDDING_TOP_K} valid candidates")

    # 2. Permutation logic
    print(f"\n2. Permutation logic:")
    perm_errors = 0
    for asin in all_asins[:5]:
        cands = candidates_map[asin]
        perm0 = get_permuted_candidates(cands, asin, 0)
        perm1 = get_permuted_candidates(cands, asin, 1)
        if perm0 != cands:
            perm_errors += 1; errors.append(f"  {asin} perm0 != original")
        if perm1 != list(reversed(cands)):
            perm_errors += 1; errors.append(f"  {asin} perm1 != reversed")
        for p in range(RANKING_REPEATS):
            perm = get_permuted_candidates(cands, asin, p)
            if set(perm) != set(cands) or len(perm) != len(cands):
                perm_errors += 1; errors.append(f"  {asin} perm={p} ASIN mismatch")
    if perm_errors == 0:
        print(f"   Permutation logic correct (spot-checked 5 SKUs x {RANKING_REPEATS} perms)")

    # 3. Cross-process permutation stability (SHA-256 seed)
    print(f"\n3. Cross-process permutation stability:")
    ta = all_asins[0]; tc = candidates_map[ta]
    p2a = get_permuted_candidates(tc, ta, 2, seed=RANDOM_SEED)
    p2b = get_permuted_candidates(tc, ta, 2, seed=RANDOM_SEED)
    if p2a == p2b:
        print(f"   SHA-256 seed is stable (same result on two calls)")
    else:
        errors.append("  FAIL: SHA-256 seed unstable!")

    # 4. Prompt generation
    print(f"\n4. Prompt generation ({len(all_asins)} SKUs x {RANKING_REPEATS} perms):")
    for asin in all_asins:
        if asin not in formatted_blocks or asin not in candidates_map:
            continue
        cands = candidates_map[asin]
        for p in range(RANKING_REPEATS):
            perm = get_permuted_candidates(cands, asin, p)
            try:
                oos_block = formatted_blocks[asin]
                cand_blocks = [formatted_blocks[a] for a in perm]
                prompt = build_prompt(oos_block, cand_blocks, perm)
                prompt_lengths.append(len(prompt))
                est_tokens.append(len(prompt) // 4)
            except Exception as e:
                errors.append(f"  {asin} perm={p}: prompt build failed -- {e}")

    if not any("prompt build failed" in e for e in errors) and prompt_lengths:
        print(f"   All {len(all_asins) * RANKING_REPEATS} prompts built successfully")
        print(f"   Chars:  min={min(prompt_lengths)}, max={max(prompt_lengths)}, "
              f"mean={sum(prompt_lengths)//len(prompt_lengths)}")
        print(f"   Tokens: min~{min(est_tokens)}, max~{max(est_tokens)}, "
              f"mean~{sum(est_tokens)//len(est_tokens)}")

    # 5. JSON extraction tests
    print(f"\n5. JSON extraction robustness:")
    passed, total = _test_extract_json()
    print(f"   {passed}/{total} cases passed")
    if passed < total:
        errors.append(f"  JSON extraction: {total - passed} case(s) failed")

    # 6. Call budget estimate
    print(f"\n6. Call budget:")
    n_local = len(RANKING_MODELS_LOCAL)
    n_frontier = len(RANKING_MODELS_FRONTIER)
    n_models = n_local + n_frontier
    total_calls = n_models * len(all_asins) * RANKING_REPEATS
    mean_tok = sum(est_tokens) // len(est_tokens) if est_tokens else 0
    print(f"   Models: {n_local} local + {n_frontier} frontier = {n_models}")
    print(f"   Calls:  {n_local * len(all_asins) * RANKING_REPEATS} local + "
          f"{n_frontier * len(all_asins) * RANKING_REPEATS} frontier = {total_calls} total")
    print(f"   Est. input tokens/call:  ~{mean_tok}")
    print(f"   Est. total input tokens: ~{total_calls * mean_tok:,}")

    print(f"\n{'=' * 70}")
    if errors:
        print(f"DRY RUN FAILED -- {len(errors)} error(s):")
        for e in errors:
            print(e)
    else:
        print("DRY RUN PASSED -- pipeline is ready.")
        print("Next step: python 05_ranking_elicit.py --test")
    print("=" * 70)
    return len(errors) == 0


def run_smoke_test(
    products: dict,
    formatted_blocks: dict,
    candidates_map: dict,
    model_names: list[str],
):
    """Make ONE real API call per model (1 SKU, perm=0). Does not write to main output."""
    print("\n" + "=" * 70)
    print("SMOKE TEST -- 1 call per model (real API calls)")
    print("=" * 70)

    all_asins = sorted(products.keys())
    test_asin = all_asins[len(all_asins) // 2]
    cands = candidates_map[test_asin]

    oos_block = formatted_blocks[test_asin]
    cand_blocks = [formatted_blocks[a] for a in cands]
    prompt = build_prompt(oos_block, cand_blocks, cands)
    expected = set(cands)

    test_output = os.path.join(RANKINGS_DIR, "_smoke_test.jsonl")
    results: list[dict] = []

    print(f"\nTest SKU : {test_asin} ({products[test_asin]['title'][:60]}...)")
    print(f"Candidates: {len(cands)}")
    print(f"Prompt   : {len(prompt)} chars (~{len(prompt)//4} tokens)")
    print(f"           (first 200): {repr(prompt[:200])}\n")

    for model_name in model_names:
        print(f"  {model_name}: ", end="", flush=True)

        try:
            client = get_client(model_name)
        except Exception as e:
            print(f"SKIP (client init failed: {e})")
            results.append({"model": model_name, "status": "client_error", "error": str(e)})
            continue

        t0 = time.time()
        result = call_and_parse(
            client, prompt, expected,
            oos_asin=test_asin, perm_idx=0,
        )
        elapsed = time.time() - t0

        if result is None:
            print(f"FAILED ({elapsed:.1f}s)")
            print(f"    Debug files saved to: {DEBUG_DIR}")
            results.append({"model": model_name, "status": "failed", "elapsed_s": elapsed})
        else:
            ranking = result["ranking"]
            reasoning = result.get("reasoning", "")
            print(f"OK ({elapsed:.1f}s)")
            print(f"    Top-3: {ranking[:3]}")
            print(f"    Reason: {reasoning[:120]}")
            results.append({
                "model": model_name,
                "status": "ok",
                "elapsed_s": round(elapsed, 2),
                "ranking": ranking,
                "reasoning": reasoning,
            })

    os.makedirs(RANKINGS_DIR, exist_ok=True)
    with open(test_output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r) + "\n")

    passed = sum(1 for r in results if r["status"] == "ok")
    print(f"\n{'=' * 70}")
    print(f"SMOKE TEST: {passed}/{len(model_names)} models passed")
    if passed == len(model_names):
        print("Next step: python 05_ranking_elicit.py --local-only")
    else:
        print(f"Failed: {[r['model'] for r in results if r['status'] != 'ok']}")
        print(f"Check debug files in: {DEBUG_DIR}")
    print(f"Test output: {test_output}")
    print("=" * 70)
    return passed == len(model_names)


# Main orchestration

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stage 05 -- LLM ranking elicitation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Workflow:
  1. --dry-run       Validate prompts & pipeline (zero API calls)
  2. --test          Smoke-test 1 call per model (real calls, separate output)
  3. --local-only    Run all local/Ollama models (no API tokens spent)
  4. (no flags)      Full run -- local first, then frontier
        """,
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--test", action="store_true")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--local-only", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 70)
    print("Stage 05 -- LLM Ranking Elicitation")
    print("=" * 70)

    print("\nLoading raw product data...")
    products = load_products(MODEL_INPUT_FILE)
    print(f"  {len(products)} products loaded.")

    print("Loading pre-formatted product blocks...")
    if not os.path.exists(FORMATTED_MODEL_INPUT_FILE):
        print(f"\nERROR: {FORMATTED_MODEL_INPUT_FILE} not found.")
        print("Run 04b_format_llm_inputs.py first.")
        sys.exit(1)
    formatted_blocks = load_formatted_products(FORMATTED_MODEL_INPUT_FILE)
    print(f"  {len(formatted_blocks)} formatted blocks loaded.")

    print("Loading embedding candidates...")
    candidates_map = load_top_k_candidates(ALL_PAIRS_SCORES, EMBEDDING_TOP_K)
    print(f"  Top-{EMBEDDING_TOP_K} candidates for {len(candidates_map)} SKUs.")

    if args.model:
        if args.model not in RANKING_MODELS_ALL:
            print(f"\nERROR: '{args.model}' not in RANKING_MODELS_ALL.")
            print(f"Available: {RANKING_MODELS_ALL}")
            sys.exit(1)
        model_list = [args.model]
    elif args.local_only:
        model_list = list(RANKING_MODELS_LOCAL)
    else:
        model_list = list(RANKING_MODELS_LOCAL) + list(RANKING_MODELS_FRONTIER)

    if args.dry_run:
        ok = run_dry_run(products, formatted_blocks, candidates_map)
        sys.exit(0 if ok else 1)

    if args.test:
        os.makedirs(RANKINGS_DIR, exist_ok=True)
        ok = run_smoke_test(products, formatted_blocks, candidates_map, model_list)
        sys.exit(0 if ok else 1)

    # Full run
    os.makedirs(RANKINGS_DIR, exist_ok=True)
    completed = load_completed(RANKINGS_FILE)
    print(f"  {len(completed)} (model, sku, perm) triples already completed.")

    all_asins = sorted(products.keys())
    work_queue: list[tuple[str, str, int]] = []
    for model_name in model_list:
        for oos_asin in all_asins:
            for perm_idx in range(RANKING_REPEATS):
                key = (model_name, oos_asin, perm_idx)
                if key not in completed:
                    work_queue.append(key)

    print(f"\n  Models: {model_list}")
    print(f"  Total: {len(model_list) * len(all_asins) * RANKING_REPEATS} calls")
    print(f"  Remaining: {len(work_queue)}")

    if not work_queue:
        print("\nAll calls completed. Nothing to do.")
        return

    from itertools import groupby
    model_order = {n: i for i, n in enumerate(model_list)}
    work_queue.sort(key=lambda x: model_order.get(x[0], 999))
    failures = 0

    for model_name, group in groupby(work_queue, key=lambda x: x[0]):
        tasks = list(group)
        print(f"\n{'─' * 60}")
        print(f"Model: {model_name}  ({len(tasks)} calls remaining)")
        print(f"{'─' * 60}")

        try:
            client = get_client(model_name)
        except Exception as e:
            print(f"  ERROR: Could not create client -- {e}")
            failures += len(tasks)
            continue

        delay = _get_delay(model_name)

        for i, (_, oos_asin, perm_idx) in enumerate(tasks, 1):
            candidate_asins = candidates_map[oos_asin]
            perm_order = get_permuted_candidates(candidate_asins, oos_asin, perm_idx)
            oos_block = formatted_blocks[oos_asin]
            cand_blocks = [formatted_blocks[a] for a in perm_order]
            prompt = build_prompt(oos_block, cand_blocks, perm_order)
            expected = set(candidate_asins)

            print(f"  [{i}/{len(tasks)}] {oos_asin} perm={perm_idx} ... ",
                  end="", flush=True)

            t0 = time.time()
            result = call_and_parse(
                client, prompt, expected,
                oos_asin=oos_asin, perm_idx=perm_idx,
            )
            elapsed = time.time() - t0

            if result is None:
                print(f"FAILED ({elapsed:.1f}s)")
                failures += 1
                continue

            append_result(
                output_path=RANKINGS_FILE,
                model=model_name,
                oos_asin=oos_asin,
                perm_index=perm_idx,
                candidate_order=perm_order,
                ranking=result["ranking"],
                reasoning=result.get("reasoning", ""),
                elapsed_s=elapsed,
            )
            print(f"OK ({elapsed:.1f}s)")

            if delay > 0 and i < len(tasks):
                time.sleep(delay)

    print(f"\n{'=' * 70}")
    print(f"Done.  Failures: {failures}/{len(work_queue)}")
    if os.path.exists(RANKINGS_FILE):
        final = pd.read_parquet(RANKINGS_FILE)
        print(f"Total rows in output: {len(final)}")
        for m in model_list:
            n = len(final[final["model"] == m])
            target = len(all_asins) * RANKING_REPEATS
            print(f"  {m}: {n}/{target}")
    print("=" * 70)


if __name__ == "__main__":
    main()
