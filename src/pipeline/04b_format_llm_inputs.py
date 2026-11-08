"""
Stage 04b: Format product records for LLM prompt injection.

Reads  : data/processed/ModelInput60.jsonl
Writes : data/processed/ModelInput60_formatted.jsonl

Each output record contains:
    parent_asin     : str   - product identifier (unchanged)
    formatted_block : str   - fully rendered text block ready to be
                              inserted verbatim into the ranking prompt

This stage sits between stage 04 (embedding scores) and stage 05
(LLM elicitation). Separating it means:
  - You can inspect exactly what the LLM sees before spending API tokens
  - Formatting changes require only re-running this script, not stage 05
  - Stage 05 becomes a pure orchestration layer with no formatting logic

Pipeline position
-----------------
02_export_model_inputs.py   ->  ModelInput60.jsonl        (structured JSON)
04_embedding_score.py       ->  all_pairs_scores.parquet  (similarity scores)
04b_format_llm_inputs.py    ->  ModelInput60_formatted.jsonl  (this script)
05_ranking_elicit.py        ->  raw_rankings.parquet      (LLM rankings)

Formatting decisions
--------------------
- ALL feature bullets are included (no cap). Features often carry the
  most substitutability-relevant signal (roast, grind, certifications).
- Full description is included without truncation.
- Details: all fields are passed EXCEPT a small blocklist of noise keys
  (internal identifiers, logistics fields, duplicates). See _NOISE_KEYS.
- Each section is labelled and structured for clean LLM parsing.

Usage
-----
    python 04b_format_llm_inputs.py
    python 04b_format_llm_inputs.py --preview     # print first 3 blocks
    python 04b_format_llm_inputs.py --preview 10  # print first N blocks
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import MODEL_INPUT_FILE, PROCESSED_DIR

# Output path - same directory as ModelInput60.jsonl
FORMATTED_FILE = os.path.join(PROCESSED_DIR, "ModelInput60_formatted.jsonl")

# Noise filter
# Keys dropped from the details dict. These carry no substitutability signal:
# internal identifiers, logistics fields, and fields already in the title.
_NOISE_KEYS = {
    "UPC",
    "Manufacturer",                   # duplicates Brand / already in title
    "Is Discontinued By Manufacturer",
    "Item model number",
    "Domestic Shipping",
    "International Shipping",
    "Product Dimensions",             # Package Dimensions is the cleaner form
}


# Formatting helpers

def _fmt_price(price) -> str:
    return f"${price:.2f}" if price is not None else "not listed"


def _fmt_details(details: dict) -> str:
    """
    All detail fields except noise keys, one per line.
    Preserves the original key order from the source record.
    """
    lines = [
        f"  {key}: {val}"
        for key, val in details.items()
        if key not in _NOISE_KEYS
    ]
    return "\n".join(lines) if lines else "  (none listed)"


def _fmt_features(features: list[str]) -> str:
    """All feature bullets, one per line. No cap."""
    if not features:
        return "  (none listed)"
    return "\n".join(f"  - {feat.strip()}" for feat in features)


def _fmt_description(desc: list[str]) -> str:
    """Full description joined from paragraphs. No truncation."""
    text = " ".join(desc).strip() if desc else ""
    return f"  {text}" if text else "  (none provided)"


def format_product_block(product: dict, label: str | None = None) -> str:
    """
    Render one product record as a structured text block.

    The label (e.g. "1", "2", ...) is used when formatting candidates
    so the LLM can distinguish them visually. Omit for the OOS product.
    """
    header = f"[{label}] " if label else ""
    sections = [
        f"{header}{product['title']}",
        f"  ASIN : {product['parent_asin']}",
        f"  Price: {_fmt_price(product.get('price'))}",
        "  Details:",
        _fmt_details(product.get("details", {})),
        "  Features:",
        _fmt_features(product.get("features", [])),
        "  Description:",
        _fmt_description(product.get("description", [])),
    ]
    return "\n".join(sections)


# Main

def load_products(path: str) -> list[dict]:
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def run(preview_n: int = 0):
    print("=" * 60)
    print("Stage 04b — Format LLM Inputs")
    print("=" * 60)

    products = load_products(MODEL_INPUT_FILE)
    print(f"\nLoaded {len(products)} products from {MODEL_INPUT_FILE}")

    output_records = []
    for product in products:
        block = format_product_block(product)
        output_records.append({
            "parent_asin": product["parent_asin"],
            "formatted_block": block,
        })

    # Write output
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    with open(FORMATTED_FILE, "w", encoding="utf-8") as f:
        for rec in output_records:
            f.write(json.dumps(rec) + "\n")

    print(f"Wrote {len(output_records)} formatted records → {FORMATTED_FILE}")

    # Stats
    lengths = [len(r["formatted_block"]) for r in output_records]
    est_tokens = [l // 4 for l in lengths]
    print(f"\nBlock length  : min={min(lengths)} chars, "
          f"max={max(lengths)} chars, "
          f"mean={sum(lengths)//len(lengths)} chars")
    print(f"Est. tokens   : min≈{min(est_tokens)}, "
          f"max≈{max(est_tokens)}, "
          f"mean≈{sum(est_tokens)//len(est_tokens)}")
    print(f"\nWith 7 candidates per prompt + OOS block:")
    prompt_est = (sum(est_tokens) // len(est_tokens)) * 8  # 7 candidates + 1 OOS
    print(f"  Est. product content per prompt: ~{prompt_est} tokens")
    print(f"  (Plus ~250 tokens for prompt instructions)")

    if preview_n > 0:
        print(f"\n{'─' * 60}")
        print(f"PREVIEW — first {min(preview_n, len(output_records))} formatted blocks")
        print(f"{'─' * 60}")
        for rec in output_records[:preview_n]:
            print(f"\n{rec['formatted_block']}")
            print(f"{'─' * 60}")

    print("\n✓ Done. Run 05_ranking_elicit.py next.")


def main():
    parser = argparse.ArgumentParser(
        description="Stage 04b — Format product records for LLM prompt injection"
    )
    parser.add_argument(
        "--preview",
        nargs="?",
        const=3,
        type=int,
        metavar="N",
        help="Print first N formatted blocks to stdout (default 3 if flag given)",
    )
    args = parser.parse_args()
    run(preview_n=args.preview or 0)


if __name__ == "__main__":
    main()
