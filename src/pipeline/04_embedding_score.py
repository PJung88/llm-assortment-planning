"""
Stage 04: embed the 60 products and score every ordered pair.

Embeds all 60 coffee SKUs and scores every ordered pair:
  all_pairs_scores.parquet - cosine similarity for all 3,540 ordered pairs
                             (Stage 05 takes each SKU's top-7 neighbours
                             from this file as its ranking candidates)

Pipeline position: runs AFTER 02_export_model_inputs.py
Reads:  data/processed/EmbeddingInput60.jsonl   (parent_asin + text per product)
Writes: outputs/embeddings/all_pairs_scores.parquet   (3,540 rows)

Model
-----
BAAI/bge-large-en-v1.5  - open-source, ~1.3 GB, CPU-compatible.
First run downloads from Hugging Face and caches locally.
With normalize_embeddings=True, cosine similarity = dot product.

Steps
-----
1. Load 60 products from EmbeddingInput60.jsonl
2. Encode all 60 with bge-large-en-v1.5  ->  60 vectors in memory
3. Score ALL 3,540 ordered pairs via matrix multiplication  ->  all_pairs_scores.parquet
"""

import json
import os
import sys
import time
from itertools import permutations

import numpy as np
import pandas as pd

# Path setup
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import src.config as config

MODEL_NAME = "BAAI/bge-large-en-v1.5"


# Step 1 - Load products

def load_embedding_inputs():
    texts_by_asin = {}
    with open(config.EMBEDDING_INPUT_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                record = json.loads(line)
                texts_by_asin[record["parent_asin"]] = record["text"]

    print(f"[Step 1] Loaded {len(texts_by_asin)} products from EmbeddingInput60.jsonl")
    return texts_by_asin


# Step 2 - Encode with bge-large-en-v1.5

def encode_products(texts_by_asin):
    """
    Encode all products into L2-normalised vectors.
    Returns:
        asins  - ordered list of ASINs
        matrix - numpy array of shape (60, 1024)
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("\n  ERROR: sentence-transformers is not installed.")
        print("  Run:  pip install sentence-transformers --break-system-packages\n")
        sys.exit(1)

    print(f"[Step 2] Loading model: {MODEL_NAME}")

    t0 = time.time()
    model = SentenceTransformer(MODEL_NAME)
    print(f"         Model loaded in {time.time() - t0:.1f}s")

    asins = list(texts_by_asin.keys())
    texts = [texts_by_asin[a] for a in asins]

    print(f"         Encoding {len(texts)} products on CPU …")
    t0 = time.time()
    matrix = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=True,
        batch_size=16,
    )
    print(f"         Encoded in {time.time() - t0:.1f}s")
    print(f"         Embedding dimension: {matrix.shape[1]}")

    return asins, matrix


# Step 3 - Score ALL 3,540 ordered pairs

def score_all_pairs(asins, matrix):
    """
    Compute cosine similarity for every ordered (A, B) pair where A != B.

    Uses matrix multiplication: sim_matrix = matrix @ matrix.T
    This gives the full 60×60 similarity grid in one shot, then we
    extract the 3,540 off-diagonal entries.

    Output: all_pairs_scores.parquet
    Columns: asin_a, asin_b, cosine_sim, model_name
    """
    print(f"[Step 3] Scoring all {len(asins) * (len(asins) - 1)} ordered pairs …")

    # Full similarity matrix in one matrix multiply - very fast even on CPU
    sim_matrix = matrix @ matrix.T   # shape (60, 60)

    rows = []
    for i, asin_a in enumerate(asins):
        for j, asin_b in enumerate(asins):
            if i == j:
                continue
            rows.append({
                "asin_a":     asin_a,
                "asin_b":     asin_b,
                "cosine_sim": float(sim_matrix[i, j]),
                "model_name": MODEL_NAME,
            })

    df = pd.DataFrame(rows)

    os.makedirs(os.path.dirname(config.ALL_PAIRS_SCORES), exist_ok=True)
    df.to_parquet(config.ALL_PAIRS_SCORES, index=False)

    print(f"         Min: {df['cosine_sim'].min():.4f}  "
          f"Max: {df['cosine_sim'].max():.4f}  "
          f"Mean: {df['cosine_sim'].mean():.4f}")
    print(f"         Written: {config.ALL_PAIRS_SCORES}  ({len(df)} rows)")
    return df


# Main

def main():
    print("=" * 60)
    print("Stage 04 — Embedding Similarity Scores")
    print(f"           Model: {MODEL_NAME}")
    print("=" * 60)

    texts_by_asin = load_embedding_inputs()

    print()
    asins, matrix = encode_products(texts_by_asin)

    print()
    all_df = score_all_pairs(asins, matrix)

    print()
    print("=" * 60)
    print("Summary")
    print("=" * 60)
    print(f"  All pairs scored     : {len(all_df)}  →  all_pairs_scores.parquet")
    print("=" * 60)


if __name__ == "__main__":
    main()