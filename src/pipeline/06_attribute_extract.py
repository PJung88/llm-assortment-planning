"""
Stage 06: Attribute extraction + verification sheet.

Extracts the coffee attributes the Stage 05 ranking rubric relies on
(product type, format, pod system, caffeine, roast, flavoured, net quantity)
for the 60 SKUs, using Gemini on the SAME pre-rendered product text the
ranking models saw (ModelInput60_formatted.jsonl). Then builds a
verification sheet for manual checking, and - after you have filled it in -
produces the final verified attribute table plus a verification report.

Why the same text: the attributes are used to judge whether the ranking
models followed the rubric (e.g. format / caffeine hard constraints), so they
must reflect what those models were shown, not outside knowledge.

Anti-hallucination design
-------------------------
1. Every non-default value must come with an evidence quote copied verbatim
   from the product text; the script checks programmatically that the quote
   actually occurs in the text.
2. An independent keyword pass (no LLM) re-derives format, pod system,
   caffeine, roast and flavoured, and the Amazon category stratum from
   Stage 01 is carried along as a third reference for format.
3. Manual verification (see VERIFICATION below).

Note: Gemini is also one of the four ranking models, so an unverified
Gemini-extracted attribute table could favour Gemini in the rubric-compliance
check. The full manual check of the ground-truth attributes removes that
concern - do not skip it.

VERIFICATION (what the sheet asks you to do)
--------------------------------------------
Sheet "review" has one row per (SKU, attribute). Rows with must_review = Y:
  - product_type, format, pod_system, caffeine  -> ALL 60 SKUs (review_reason = full)
  - roast, flavored, net_quantity               -> random sample of 20 SKUs
                                                   (review_reason = random_sample)
                                                   + any flagged row (review_reason = flagged)
For each must_review row, read the product text (sheet "product_text" or the
title column) and fill `verified_value`:
  - ok              -> the LLM value is correct
  - <allowed value> -> the correct value (see sheet "allowed_values");
                       for net_quantity write e.g.  12 oz x 3  or  100 count
  - unresolvable    -> the text does not settle it
Add a short `note` for every correction.

Outputs (outputs/attributes/)
-----------------------------
  attributes_raw.jsonl            raw Gemini responses (resume-safe, append-only)
  attributes_verification.xlsx    the sheet you fill in  (CSV fallback if no openpyxl)
  attributes_final.parquet/.csv   one row per SKU, verified values   (--finalize)
  attributes_verification_report.txt  corrections + error rates     (--finalize)

Usage
-----
    python src/pipeline/06_attribute_extract.py --test        # 1 SKU, prints result, writes nothing
    python src/pipeline/06_attribute_extract.py               # extract all 60 (resume-safe) + build sheet
    python src/pipeline/06_attribute_extract.py --sheet-only  # rebuild sheet from existing raw file
    python src/pipeline/06_attribute_extract.py --finalize    # after manual review
    python src/pipeline/06_attribute_extract.py --overwrite-sheet  # replace an existing sheet (loses your edits!)

Requires: google-genai, pandas, pyarrow, python-dotenv, openpyxl; GEMINI_API_KEY in .env
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import sys
import time
import unicodedata
from pathlib import Path
from typing import Any

import pandas as pd

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parents[2] / ".env")
except ImportError:
    pass

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    FORMATTED_MODEL_INPUT_FILE,
    SAMPLE_FILE,
    OUTPUTS_DIR,
    RANDOM_SEED,
)

# Settings

EXTRACT_MODEL = "gemini-3.6-flash"
EXTRACT_TEMPERATURE = 0.0          # extraction is a reading task: no sampling noise wanted
EXTRACT_MAX_TOKENS = 8192
MAX_RETRIES = 3
CALL_DELAY_S = 1.0

ATTR_DIR = os.path.join(OUTPUTS_DIR, "attributes")
RAW_FILE = os.path.join(ATTR_DIR, "attributes_raw.jsonl")
SHEET_XLSX = os.path.join(ATTR_DIR, "attributes_verification.xlsx")
SHEET_CSV = os.path.join(ATTR_DIR, "attributes_verification.csv")
FINAL_PARQUET = os.path.join(ATTR_DIR, "attributes_final.parquet")
FINAL_CSV = os.path.join(ATTR_DIR, "attributes_final.csv")
REPORT_FILE = os.path.join(ATTR_DIR, "attributes_verification_report.txt")

N_RANDOM_SAMPLE = 20                # SKUs sampled for the descriptive attributes

ALLOWED: dict[str, list[str]] = {
    "product_type": ["pure_coffee", "coffee_with_additives", "coffee_alternative",
                     "non_coffee", "mixed_pack"],
    "format":       ["whole_bean", "ground", "pod_capsule", "instant",
                     "ready_to_drink", "concentrate", "other", "unknown"],
    "pod_system":   ["keurig_kcup", "nespresso_original", "nespresso_vertuo",
                     "tassimo", "other_or_unspecified", "not_applicable"],
    "caffeine":     ["regular", "decaf", "half_caf", "caffeine_free", "mixed", "not_stated"],
    "roast":        ["light", "medium", "medium_dark", "dark", "mixed", "not_stated"],
    "flavored":     ["yes", "no"],
}
QTY_UNITS = ["oz", "lb", "g", "kg", "fl_oz", "ml", "l", "count"]

ALL_ATTRS = list(ALLOWED) + ["net_quantity"]
FULL_REVIEW_ATTRS = {"product_type", "format", "pod_system", "caffeine"}   # ground truth
SAMPLED_ATTRS = {"roast", "flavored", "net_quantity"}                      # descriptive

# values that may legitimately have no evidence quote
NO_EVIDENCE_OK = {"not_stated", "not_applicable", "unknown", "no", "other_or_unspecified"}

# Prompt

PROMPT_TEMPLATE = """\
You are extracting product attributes for a research dataset. Read the product \
listing below and report ONLY what the text states. Do not use outside knowledge \
about brands or products.

PRODUCT LISTING:
<<<
{block}
>>>

The "Details" lines come from Amazon's structured fields and are sometimes wrong. \
If they conflict with the title, features or description, trust the title first, \
then features/description, and mention the conflict in "notes".

Attributes and allowed values:

1. product_type
   - pure_coffee: coffee (beans, ground, pods, instant, ready-to-drink) with nothing \
added except flavouring
   - coffee_with_additives: coffee with added creamer, sugar, milk, or functional \
ingredients (e.g. 3-in-1 mixes, cappuccino/latte mixes, mushroom/ganoderma coffee)
   - coffee_alternative: marketed as a coffee substitute (chicory, herbal "coffee")
   - non_coffee: no coffee at all (e.g. hot cocoa)
   - mixed_pack: a variety pack combining coffee with non-coffee items (tea, cocoa, ...)
2. format: whole_bean | ground | pod_capsule | instant | ready_to_drink | concentrate \
| other | unknown   (pod_capsule = any single-serve pod/cup/capsule; instant includes \
powder mixes and sticks)
3. pod_system (only if format is pod_capsule): keurig_kcup | nespresso_original | \
nespresso_vertuo | tassimo | other_or_unspecified ; otherwise not_applicable
4. caffeine: regular | decaf | half_caf | caffeine_free | mixed | not_stated \
("regular" only if the text states caffeinated/regular or a caffeine amount; \
if caffeine is not mentioned at all, use not_stated)
5. roast: light | medium | medium_dark | dark | mixed | not_stated \
(French/Italian roast = dark; "full city" = medium_dark; blonde = light)
6. flavored: yes | no   (yes only for ADDED flavouring such as vanilla or hazelnut; \
tasting notes like "notes of chocolate" are NOT flavouring)
7. net_quantity: the size of ONE unit and the number of units in the pack \
(e.g. "12 Ounce Bag (Pack of 3)" -> value 12, unit "oz", pack_count 3; \
"100 single-serve cups" -> value 100, unit "count", pack_count 1). \
Allowed units: oz, lb, g, kg, fl_oz, ml, l, count. Use null if not stated.

EVIDENCE RULE: for every attribute give "evidence": a short phrase (max ~15 words) \
copied EXACTLY, character for character, from the listing that supports the value. \
Use "" only when the value is not_stated, not_applicable, unknown, or flavored = no.

Return ONLY this JSON object, nothing else:
{{
  "product_type": {{"value": "...", "evidence": "..."}},
  "format":       {{"value": "...", "evidence": "..."}},
  "pod_system":   {{"value": "...", "evidence": "..."}},
  "caffeine":     {{"value": "...", "evidence": "..."}},
  "roast":        {{"value": "...", "evidence": "..."}},
  "flavored":     {{"value": "...", "evidence": "..."}},
  "net_quantity": {{"value": <number or null>, "unit": "<unit or null>", "pack_count": <integer or null>, "evidence": "..."}},
  "notes": "conflicts or ambiguities, or empty string"
}}
"""

PROMPT_HASH = hashlib.sha256(PROMPT_TEMPLATE.encode()).hexdigest()[:12]


def build_prompt(block: str) -> str:
    return PROMPT_TEMPLATE.format(block=block)


# Data loading

def load_blocks() -> dict[str, str]:
    blocks = {}
    with open(FORMATTED_MODEL_INPUT_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            blocks[r["parent_asin"]] = r["formatted_block"]
    return blocks


def load_strata() -> dict[str, str]:
    """Amazon-category stratum from Stage 01 (never shown to models) -> format reference."""
    strata = {}
    if not os.path.exists(SAMPLE_FILE):
        return strata
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            strata[r["parent_asin"]] = r.get("stratum", "")
    return strata


STRATUM_TO_FORMAT = {
    "Ground Coffee": "ground",
    "Whole Bean": "whole_bean",
    "Pods / Capsules": "pod_capsule",
    "Instant Coffee": "instant",
    "RTD & Cold Brew": "ready_to_drink|concentrate",   # Amazon lumps these together
}


def title_of(block: str) -> str:
    return block.split("\n", 1)[0].strip()


# Gemini call + validation

def _extract_json(text: str | None) -> dict[str, Any]:
    if not text or not text.strip():
        raise ValueError("empty response")
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"no JSON object in response: {text[:200]!r}")
    return json.loads(text[start:end + 1])


def validate(obj: dict[str, Any]) -> list[str]:
    """Return a list of schema problems (empty list = valid)."""
    problems = []
    for attr, allowed in ALLOWED.items():
        item = obj.get(attr)
        if not isinstance(item, dict):
            problems.append(f"{attr}: missing or not an object")
            continue
        if item.get("value") not in allowed:
            problems.append(f"{attr}: value {item.get('value')!r} not in {allowed}")
        if not isinstance(item.get("evidence", ""), str):
            problems.append(f"{attr}: evidence must be a string")
    q = obj.get("net_quantity")
    if not isinstance(q, dict):
        problems.append("net_quantity: missing or not an object")
    else:
        if q.get("value") is not None and not isinstance(q.get("value"), (int, float)):
            problems.append("net_quantity.value must be a number or null")
        if q.get("unit") is not None and q.get("unit") not in QTY_UNITS:
            problems.append(f"net_quantity.unit {q.get('unit')!r} not in {QTY_UNITS}")
        if q.get("pack_count") is not None and not isinstance(q.get("pack_count"), int):
            problems.append("net_quantity.pack_count must be an integer or null")
    return problems


class GeminiExtractor:
    def __init__(self, model_name: str = EXTRACT_MODEL):
        from google import genai
        key = os.environ.get("GEMINI_API_KEY")
        if not key:
            raise ValueError("GEMINI_API_KEY not set (put it in .env)")
        self.client = genai.Client(api_key=key)
        self.model_name = model_name

    def extract(self, block: str) -> tuple[dict[str, Any], str, int]:
        """Returns (parsed_json, raw_text, attempts). Retries on bad JSON / schema."""
        prompt = build_prompt(block)
        last_err = None
        for attempt in range(1, MAX_RETRIES + 1):
            raw = None
            try:
                resp = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config={
                        "temperature": EXTRACT_TEMPERATURE,
                        "max_output_tokens": EXTRACT_MAX_TOKENS,
                        "response_mime_type": "application/json",
                    },
                )
                raw = resp.text
                obj = _extract_json(raw)
                problems = validate(obj)
                if not problems:
                    return obj, raw, attempt
                last_err = "; ".join(problems)
            except Exception as e:  # network, quota, JSON
                last_err = f"{type(e).__name__}: {e}"
            print(f"    attempt {attempt} failed: {last_err}")
            time.sleep(2 ** attempt)
        raise RuntimeError(f"extraction failed after {MAX_RETRIES} attempts: {last_err}")


# RAW FILE (resume-safe)

def load_raw() -> dict[str, dict]:
    done = {}
    if os.path.exists(RAW_FILE):
        with open(RAW_FILE, encoding="utf-8") as f:
            for line in f:
                r = json.loads(line)
                done[r["parent_asin"]] = r
    return done


def append_raw(rec: dict) -> None:
    os.makedirs(ATTR_DIR, exist_ok=True)
    with open(RAW_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def run_extraction(blocks: dict[str, str]) -> None:
    done = load_raw()
    todo = [a for a in blocks if a not in done]
    print(f"{len(done)} already extracted, {len(todo)} to go (model {EXTRACT_MODEL}, prompt {PROMPT_HASH})")
    if not todo:
        return
    ex = GeminiExtractor()
    failures = []
    for i, asin in enumerate(todo, 1):
        print(f"[{i}/{len(todo)}] {asin}")
        t0 = time.time()
        try:
            obj, raw, attempts = ex.extract(blocks[asin])
        except RuntimeError as e:
            print(f"    FAILED: {e}")
            failures.append(asin)
            continue
        append_raw({
            "parent_asin": asin,
            "model": EXTRACT_MODEL,
            "temperature": EXTRACT_TEMPERATURE,
            "prompt_hash": PROMPT_HASH,
            "extracted_at": dt.datetime.now().isoformat(timespec="seconds"),
            "attempts": attempts,
            "elapsed_s": round(time.time() - t0, 2),
            "result": obj,
            "raw_response": raw,
        })
        time.sleep(CALL_DELAY_S)
    if failures:
        print(f"\n{len(failures)} SKUs failed: {failures}\nRe-run the script to retry them.")


# INDEPENDENT KEYWORD PASS (no LLM)

def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s or "")
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("–", "-").replace("—", "-").replace("‑", "-")
    return re.sub(r"\s+", " ", s).strip().lower()


FORMAT_KW = [
    ("pod_capsule",    r"k-?\s?cups?|\bpods?\b|capsules?|single[- ]serve|single cups?|nespresso|vertuo|tassimo|keurig"),
    ("instant",        r"\binstant\b|freeze[- ]dried|\bsticks\b|\bvia\b"),
    ("ready_to_drink", r"ready to drink|\bfl\.? ?oz\b|fluid ounce|\bcans?\b|\bbottles?\b"),
    ("concentrate",    r"concentrate"),
    ("whole_bean",     r"whole[- ]beans?|grano entero"),
    ("ground",         r"\bground\b|\bgrind\b"),
]
POD_KW = [
    ("nespresso_vertuo",   r"vertuo"),
    ("nespresso_original", r"nespresso"),
    ("tassimo",            r"tassimo"),
    ("keurig_kcup",        r"k-?\s?cups?|keurig"),
]
CAFF_KW = [
    ("half_caf",      r"half[- ]?caff?"),
    ("decaf",         r"\bdecaf|decaffeinated"),
    ("caffeine_free", r"(?<!% )caffeine[- ]free|chicory|herbal coffee"),
    ("regular",       r"(?<!de)caffeinated|\d+ ?mg (?:of )?caffeine|caffeine boost|regular caffeine|"
                      r"(?:high|medium|extra|full) caffeine"),
]
ROAST_KW = [
    ("medium_dark", r"medium[- ]dark|full city"),
    ("light",       r"light roast|blonde"),
    ("dark",        r"dark roast|french roast|italian roast|dark roasted"),
    ("medium",      r"medium roast|medium-roast"),
]
FLAVOR_KW = (r"vanilla|hazelnut|caramel|mocha|chocolate|cinnamon|maple|cherry|butterscotch|"
             r"pumpkin|coconut|flavou?red|irish cream|toffee|praline|pecan|walnut|peppermint|mint")


def _match_set(text: str, rules: list[tuple[str, str]]) -> set[str]:
    return {label for label, pat in rules if re.search(pat, text)}


def _resolve(title_hits: set[str], text_hits: set[str], default: str) -> str:
    """Title wins if it is unambiguous; else full text; multiple hits -> 'ambiguous:a|b'."""
    for hits in (title_hits, text_hits):
        if len(hits) == 1:
            return next(iter(hits))
        if len(hits) > 1:
            return "ambiguous:" + "|".join(sorted(hits))
    return default


def keyword_attrs(block: str) -> dict[str, str]:
    t, full = _norm(title_of(block)), _norm(block)
    fmt_t, fmt_f = _match_set(t, FORMAT_KW), _match_set(full, FORMAT_KW)
    pod_t, pod_f = _match_set(t, POD_KW), _match_set(full, POD_KW)
    # "vertuo" also matches the generic nespresso pattern -> keep the specific one
    for s in (pod_t, pod_f):
        if "nespresso_vertuo" in s:
            s.discard("nespresso_original")
    caff_t, caff_f = _match_set(t, CAFF_KW), _match_set(full, CAFF_KW)
    # a stated decaf/half-caf/caffeine-free overrides generic "caffeinated" wording
    for s in (caff_t, caff_f):
        if len(s) > 1:
            s.discard("regular")
    roast_t, roast_f = _match_set(t, ROAST_KW), _match_set(full, ROAST_KW)
    # "medium-dark roast" also matches the plain "dark roast" pattern -> keep the specific one
    for s in (roast_t, roast_f):
        if "medium_dark" in s:
            s.discard("dark")
    fmt = _resolve(fmt_t, fmt_f, "unknown")
    return {
        "format": fmt,
        "pod_system": _resolve(pod_t, pod_f, "other_or_unspecified") if fmt == "pod_capsule" else "not_applicable",
        "caffeine": _resolve(caff_t, caff_f, "not_stated"),
        "roast": _resolve(roast_t, roast_f, "not_stated"),
        # flavoured: title only - descriptions are full of tasting notes
        "flavored": "yes" if re.search(FLAVOR_KW, t) else "no",
    }


# Verification sheet

def evidence_found(evidence: str, block: str) -> bool | None:
    if not evidence:
        return None
    return _norm(evidence) in _norm(block)


def fmt_qty(q: dict) -> str:
    if not q or q.get("value") is None:
        return "not_stated"
    v = q["value"]
    v = int(v) if float(v).is_integer() else v
    s = f"{v} {q.get('unit') or '?'}"
    if q.get("pack_count") and q["pack_count"] != 1:
        s += f" x {q['pack_count']}"
    return s


def build_sheet(blocks: dict[str, str], overwrite: bool = False) -> str:
    raw = load_raw()
    missing = [a for a in blocks if a not in raw]
    if missing:
        print(f"WARNING: {len(missing)} SKUs not extracted yet; they are left out of the sheet: {missing}")
    target = SHEET_XLSX
    try:
        import openpyxl  # noqa: F401
        have_xlsx = True
    except ImportError:
        have_xlsx = False
        target = SHEET_CSV
        print("openpyxl not installed -> writing CSV instead (pip install openpyxl for the xlsx)")
    if os.path.exists(target) and not overwrite:
        raise SystemExit(f"{target} already exists — it may contain your review. "
                         f"Use --overwrite-sheet to replace it.")

    strata = load_strata()
    asins = sorted(a for a in blocks if a in raw)
    rng = random.Random(RANDOM_SEED)
    sample = set(rng.sample(asins, min(N_RANDOM_SAMPLE, len(asins))))

    rows = []
    for asin in asins:
        block, res = blocks[asin], raw[asin]["result"]
        kw = keyword_attrs(block)
        amazon_fmt = STRATUM_TO_FORMAT.get(strata.get(asin, ""), "")
        for attr in ALL_ATTRS:
            item = res.get(attr, {}) or {}
            llm_val = fmt_qty(item) if attr == "net_quantity" else item.get("value", "")
            ev = item.get("evidence", "") or ""
            found = evidence_found(ev, block)
            kw_val = kw.get(attr, "")
            az_val = amazon_fmt if attr == "format" else ""

            flags = []
            if found is False:
                flags.append("evidence_not_in_text")
            if not ev and llm_val not in NO_EVIDENCE_OK and attr != "net_quantity":
                flags.append("missing_evidence")
            if kw_val and kw_val != llm_val:
                flags.append("keyword_disagrees")
            if az_val and llm_val not in az_val.split("|"):
                flags.append("amazon_category_disagrees")
            if llm_val in ("unknown", "mixed", "other", "mixed_pack"):
                flags.append(f"llm_says_{llm_val}")

            if attr in FULL_REVIEW_ATTRS:
                reason = "full"
            elif asin in sample:
                reason = "random_sample"      # keeps the sample random even if also flagged
            elif flags:
                reason = "flagged"
            else:
                reason = ""
            rows.append({
                "parent_asin": asin,
                "title": title_of(block),
                "attribute": attr,
                "llm_value": llm_val,
                "keyword_value": kw_val,
                "amazon_category": az_val,
                "evidence": ev,
                "evidence_in_text": "" if found is None else ("yes" if found else "NO"),
                "flags": ", ".join(flags),
                "must_review": "Y" if reason else "N",
                "review_reason": reason,
                "verified_value": "",
                "note": "",
            })
        if res.get("notes"):
            rows[-1]["note"] = f"[LLM note] {res['notes']}"

    review = pd.DataFrame(rows)
    text = pd.DataFrame({"parent_asin": asins, "product_text": [blocks[a] for a in asins]})
    allowed = pd.DataFrame(
        [{"attribute": k, "allowed_values": ", ".join(v)} for k, v in ALLOWED.items()]
        + [{"attribute": "net_quantity", "allowed_values": "e.g. '12 oz x 3', '100 count', 'not_stated'; units: " + ", ".join(QTY_UNITS)}]
        + [{"attribute": "(any)", "allowed_values": "ok = LLM value correct; unresolvable = text does not settle it"}]
    )
    os.makedirs(ATTR_DIR, exist_ok=True)

    if have_xlsx:
        with pd.ExcelWriter(target, engine="openpyxl") as xw:
            review.to_excel(xw, sheet_name="review", index=False)
            text.to_excel(xw, sheet_name="product_text", index=False)
            allowed.to_excel(xw, sheet_name="allowed_values", index=False)
        _style_sheet(target)
    else:
        review.to_csv(target, index=False, encoding="utf-8-sig")
        text.to_csv(target.replace(".csv", "_product_text.csv"), index=False, encoding="utf-8-sig")

    n_must = (review.must_review == "Y").sum()
    print(f"\nSheet written: {target}")
    print(f"  {len(review)} rows, {n_must} to review "
          f"({(review.review_reason == 'full').sum()} full, "
          f"{(review.review_reason == 'random_sample').sum()} random sample, "
          f"{(review.review_reason == 'flagged').sum()} flagged)")
    print(f"  evidence quotes not found in text: {(review.evidence_in_text == 'NO').sum()}")
    return target


def _style_sheet(path: str) -> None:
    from openpyxl import load_workbook
    from openpyxl.styles import Alignment, Font, PatternFill
    wb = load_workbook(path)
    ws = wb["review"]
    widths = {"A": 13, "B": 45, "C": 13, "D": 18, "E": 18, "F": 16, "G": 40,
              "H": 9, "I": 30, "J": 7, "K": 13, "L": 18, "M": 35}
    for col, w in widths.items():
        ws.column_dimensions[col].width = w
    ws.freeze_panes = "D2"
    ws.auto_filter.ref = ws.dimensions
    yellow = PatternFill("solid", fgColor="FFF2CC")
    red = PatternFill("solid", fgColor="F8CBAD")
    green = PatternFill("solid", fgColor="E2EFDA")
    for c in ws[1]:
        c.font = Font(bold=True)
    for row in ws.iter_rows(min_row=2):
        must, flags, ev_ok = row[9].value, row[8].value, row[7].value
        if must == "Y":
            row[11].fill = yellow
        if flags:
            row[8].fill = red
        if ev_ok == "NO":
            row[7].fill = red
        for c in (row[1], row[6], row[12]):
            c.alignment = Alignment(wrap_text=True, vertical="top")
    ws.cell(row=1, column=12).fill = green
    wt = wb["product_text"]
    wt.column_dimensions["A"].width = 13
    wt.column_dimensions["B"].width = 120
    for row in wt.iter_rows(min_row=2):
        row[1].alignment = Alignment(wrap_text=True, vertical="top")
    wb["allowed_values"].column_dimensions["A"].width = 14
    wb["allowed_values"].column_dimensions["B"].width = 110
    wb.save(path)


# FINALIZE (after manual review)

QTY_RE = re.compile(r"^\s*([\d.]+)\s*([a-z_]+)\s*(?:x\s*(\d+))?\s*$")


def _parse_qty(s: str) -> str:
    s = str(s).strip().lower()
    if s == "not_stated":
        return s
    m = QTY_RE.match(s)
    if not m or m.group(2) not in QTY_UNITS:
        raise ValueError(f"cannot parse net_quantity {s!r} (use e.g. '12 oz x 3')")
    return s


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def finalize(allow_incomplete: bool = False) -> None:
    if os.path.exists(SHEET_XLSX):
        df = pd.read_excel(SHEET_XLSX, sheet_name="review", dtype=str).fillna("")
    elif os.path.exists(SHEET_CSV):
        df = pd.read_csv(SHEET_CSV, dtype=str).fillna("")
    else:
        raise SystemExit("No verification sheet found — run the extraction first.")

    df["verified_value"] = df["verified_value"].str.strip()
    todo = df[(df.must_review == "Y") & (df.verified_value == "")]
    if len(todo) and not allow_incomplete:
        print(todo[["parent_asin", "attribute", "review_reason"]].to_string(index=False))
        raise SystemExit(f"\n{len(todo)} must-review rows are still empty. "
                         f"Finish them, or pass --allow-incomplete for a provisional table.")

    errors, finals, status = [], [], []
    for _, r in df.iterrows():
        v = r.verified_value
        if v == "":
            finals.append(r.llm_value)
            status.append("unreviewed")
        elif v.lower() in ("ok", "✔", "✓", "y", "yes_ok"):
            finals.append(r.llm_value)
            status.append("confirmed")
        elif v.lower() == "unresolvable":
            finals.append("unresolvable")
            status.append("unresolvable")
        else:
            try:
                if r.attribute == "net_quantity":
                    v = _parse_qty(v)
                elif v not in ALLOWED[r.attribute]:
                    raise ValueError(f"{v!r} not in {ALLOWED[r.attribute]}")
            except ValueError as e:
                errors.append(f"{r.parent_asin} / {r.attribute}: {e}")
            finals.append(v)
            status.append("corrected" if v != r.llm_value else "confirmed")
    if errors:
        raise SystemExit("Invalid verified_value entries:\n  " + "\n  ".join(errors))
    df["final_value"], df["status"] = finals, status

    # wide table: one row per SKU
    wide = df.pivot(index="parent_asin", columns="attribute", values="final_value")[ALL_ATTRS]
    wide.insert(0, "title", df.drop_duplicates("parent_asin").set_index("parent_asin")["title"])
    wide["is_coffee"] = wide["product_type"].isin(["pure_coffee", "coffee_with_additives"])
    wide = wide.reset_index()
    wide.to_parquet(FINAL_PARQUET, index=False)
    wide.to_csv(FINAL_CSV, index=False, encoding="utf-8-sig")

    # report
    lines = [
        "Stage 06 attribute verification report",
        f"generated {dt.datetime.now().isoformat(timespec='seconds')}",
        f"extraction model {EXTRACT_MODEL}, temperature {EXTRACT_TEMPERATURE}, prompt {PROMPT_HASH}",
        f"SKUs: {df.parent_asin.nunique()}",
        "",
        f"{'attribute':<14}{'reviewed':>9}{'confirmed':>10}{'corrected':>10}{'unresolv.':>10}"
        f"{'  error rate on random sample (95% Wilson CI)':<}",
    ]
    for attr in ALL_ATTRS:
        a = df[df.attribute == attr]
        rev = a[a.status != "unreviewed"]
        rs = rev[rev.review_reason.isin(["full", "random_sample"])]   # unbiased rows only
        k, n = (rs.status == "corrected").sum(), len(rs)
        lo, hi = wilson(k, n)
        basis = "all SKUs" if attr in FULL_REVIEW_ATTRS else "random sample"
        lines.append(
            f"{attr:<14}{len(rev):>9}{(rev.status == 'confirmed').sum():>10}"
            f"{(rev.status == 'corrected').sum():>10}{(rev.status == 'unresolvable').sum():>10}"
            f"   {k}/{n} = {k / n:.1%} [{lo:.1%}, {hi:.1%}] ({basis})" if n else
            f"{attr:<14}{len(rev):>9}{'-':>10}{'-':>10}{'-':>10}   no unbiased rows reviewed"
        )
    lines += ["", "Note: 'flagged'-only rows are excluded from the error-rate estimate "
                  "(they were selected because they looked wrong).", "",
              "Keyword pass vs LLM (before review), agreement where keyword gave a definite value:"]
    for attr in ["format", "pod_system", "caffeine", "roast", "flavored"]:
        a = df[(df.attribute == attr) & (df.keyword_value != "") & (~df.keyword_value.str.startswith("ambiguous"))]
        lines.append(f"  {attr:<12} {(a.keyword_value == a.llm_value).sum()}/{len(a)}")
    lines += ["", "Corrections:"]
    for _, r in df[df.status == "corrected"].iterrows():
        lines.append(f"  {r.parent_asin} {r.attribute}: {r.llm_value} -> {r.final_value}"
                     + (f"   ({r.note})" if r.note else ""))
    lines += ["", f"Non-coffee / coffee-alternative / mixed SKUs (is_coffee = False): "
                  f"{', '.join(wide.loc[~wide.is_coffee, 'parent_asin'])}"]
    if (df.status == "unreviewed").any() and (df.must_review == "Y").any():
        n_left = ((df.status == "unreviewed") & (df.must_review == "Y")).sum()
        if n_left:
            lines.insert(4, f"*** PROVISIONAL: {n_left} must-review rows not yet reviewed ***")
    Path(REPORT_FILE).write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    print(f"\nWrote {FINAL_PARQUET}\n      {FINAL_CSV}\n      {REPORT_FILE}")


# Cli

def run_test(blocks: dict[str, str]) -> None:
    asin = sorted(blocks)[0]
    print(f"Test extraction on {asin}: {title_of(blocks[asin])}\n")
    obj, raw, attempts = GeminiExtractor().extract(blocks[asin])
    print(json.dumps(obj, indent=2, ensure_ascii=False))
    print(f"\nattempts: {attempts}")
    print("keyword pass:", keyword_attrs(blocks[asin]))
    for attr in ALL_ATTRS:
        ev = (obj.get(attr) or {}).get("evidence", "")
        print(f"  evidence found for {attr:<13}: {evidence_found(ev, blocks[asin])}")
    print("\n(nothing written)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--test", action="store_true", help="extract 1 SKU and print; writes nothing")
    ap.add_argument("--sheet-only", action="store_true", help="rebuild the sheet from the existing raw file")
    ap.add_argument("--overwrite-sheet", action="store_true", help="allow replacing an existing sheet")
    ap.add_argument("--finalize", action="store_true", help="read the reviewed sheet, write final table + report")
    ap.add_argument("--allow-incomplete", action="store_true", help="finalize even if must-review rows are empty")
    args = ap.parse_args()

    if args.finalize:
        finalize(args.allow_incomplete)
        return
    blocks = load_blocks()
    if args.test:
        run_test(blocks)
        return
    if not args.sheet_only:
        run_extraction(blocks)
    build_sheet(blocks, overwrite=args.overwrite_sheet)


if __name__ == "__main__":
    main()
