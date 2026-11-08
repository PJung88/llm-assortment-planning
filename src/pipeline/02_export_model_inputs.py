"""
Stage 02: StratifiedSample60.jsonl -> the two model-facing exports.

ModelInput60.jsonl   : structured JSON for the LLM (title, price, details,
                        parent_asin, features, description). Deliberately
                        excludes categories/stratum/price_tier so the model
                        can't shortcut on sampling-methodology fields.
EmbeddingInput60.jsonl: same underlying content, flattened to plain text
                        for the embedding model.
"""
import json
import os

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

MODEL_VISIBLE_FIELDS = ["title", "price", "details", "parent_asin"]


def export_model_input():
    model_records = []
    with open(config.SAMPLE_FILE, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            model_record = {k: record[k] for k in MODEL_VISIBLE_FIELDS if k in record}
            model_record["features"] = record.get("features_clean", [])
            model_record["description"] = record.get("description_clean", [])
            model_records.append(model_record)

    with open(config.MODEL_INPUT_FILE, "w") as out:
        for r in model_records:
            out.write(json.dumps(r) + "\n")

    print(f"[export_model_input] wrote {len(model_records):,} records -> {config.MODEL_INPUT_FILE}")
    return model_records


def format_price(price):
    return "unspecified" if price is None else f"${price:.2f}"


def format_features(features):
    return "none listed" if not features else ", ".join(features)


def format_description(description):
    if not description:
        return "none provided"
    return " ".join(description) if isinstance(description, list) else str(description)


def format_details(details):
    if not details or not isinstance(details, dict) or len(details) == 0:
        return "none listed"
    return "; ".join(f"{k}: {v}" for k, v in details.items())


def flatten_record(record):
    parts = [
        f"Title: {record.get('title', 'unspecified')}.",
        f"Price: {format_price(record.get('price'))}.",
        f"Features: {format_features(record.get('features'))}.",
        f"Description: {format_description(record.get('description'))}.",
        f"Details: {format_details(record.get('details'))}.",
    ]
    return " ".join(parts)


def export_embedding_input(model_records=None):
    if model_records is None:
        model_records = []
        with open(config.MODEL_INPUT_FILE, "r") as f:
            for line in f:
                line = line.strip()
                if line:
                    model_records.append(json.loads(line))

    embedding_records = [
        {"parent_asin": r.get("parent_asin"), "text": flatten_record(r)}
        for r in model_records
    ]

    with open(config.EMBEDDING_INPUT_FILE, "w") as out:
        for r in embedding_records:
            out.write(json.dumps(r) + "\n")

    print(f"[export_embedding_input] wrote {len(embedding_records):,} records -> {config.EMBEDDING_INPUT_FILE}")
    return embedding_records


def main():
    model_records = export_model_input()
    export_embedding_input(model_records)


if __name__ == "__main__":
    main()
