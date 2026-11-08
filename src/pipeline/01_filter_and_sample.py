"""
Stage 01: raw Amazon metadata -> stratified 60-SKU coffee sample.

Runs the five filtering steps end to end via main(). Each step reads and
writes its own intermediate .jsonl file, so any single step can be re-run
or inspected on its own.
"""
import json
import os
import random
import statistics
from collections import Counter, defaultdict

from tqdm import tqdm

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config
from pipeline.text_utils import (
    clean_element, clean_field, clean_length_of, descriptive_length,
    get_category_path, meaningful_details_length, sanitized_list,
    split_field_elements,
)


def matches_coffee(record):
    cats = record.get("categories", [])
    prefix = config.TARGET_CATEGORY_PREFIX
    if not isinstance(cats, list) or len(cats) < len(prefix):
        return False
    if cats[:len(prefix)] != prefix:
        return False
    tail = cats[len(prefix):]
    for level in tail:
        if isinstance(level, str) and level.strip().lower() in config.EXCLUDE_CATEGORY_TOKENS:
            return False
    return True


def filter_by_category():
    """Step 1: category-path filter -> CoffeeOnly.jsonl"""
    coffee_records = []
    scanned = 0
    excluded_edge = 0
    prefix = config.TARGET_CATEGORY_PREFIX

    with open(config.RAW_FILE, "r") as f:
        for line in tqdm(f, desc="Filtering coffee"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scanned += 1
            cats = record.get("categories", [])
            if isinstance(cats, list) and cats[:len(prefix)] == prefix and not matches_coffee(record):
                excluded_edge += 1
            if matches_coffee(record):
                coffee_records.append(record)

    with open(config.COFFEE_ONLY_FILE, "w") as out:
        for record in coffee_records:
            out.write(json.dumps(record) + "\n")

    print(f"[filter_by_category] scanned={scanned:,} matched={len(coffee_records):,} excluded_edge={excluded_edge:,}")
    return len(coffee_records)


def trim_fields():
    """Step 2: keep only KEEP_FIELDS -> Field.jsonl"""
    kept_records = []
    scanned = 0
    missing_field_counts = Counter()

    with open(config.COFFEE_ONLY_FILE, "r") as f:
        for line in tqdm(f, desc="Trimming fields"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scanned += 1
            trimmed = {}
            for field in config.KEEP_FIELDS:
                if field in record:
                    trimmed[field] = record[field]
                else:
                    missing_field_counts[field] += 1
                    trimmed[field] = None
            kept_records.append(trimmed)

    with open(config.FIELD_FILE, "w") as out:
        for record in kept_records:
            out.write(json.dumps(record) + "\n")

    print(f"[trim_fields] scanned={scanned:,} written={len(kept_records):,} missing={dict(missing_field_counts)}")
    return len(kept_records)


def filter_by_price():
    """Step 3: keep only records with a clean numeric price -> PriceCoffee.jsonl"""
    kept_records = []
    scanned = 0
    dropped_no_price = 0

    with open(config.FIELD_FILE, "r") as f:
        for line in tqdm(f, desc="Filtering valid price"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scanned += 1
            raw_price = record.get("price", None)
            if isinstance(raw_price, (int, float)):
                kept_records.append(record)
            else:
                dropped_no_price += 1

    with open(config.PRICE_COFFEE_FILE, "w") as out:
        for record in kept_records:
            out.write(json.dumps(record) + "\n")

    pct_kept = (len(kept_records) / scanned) * 100 if scanned else 0
    print(f"[filter_by_price] scanned={scanned:,} kept={len(kept_records):,} ({pct_kept:.1f}%) dropped={dropped_no_price:,}")
    return len(kept_records)


def filter_by_text_sufficiency():
    """Step 4: minimum descriptive length + non-empty content -> TextFilteredCoffee.jsonl"""
    scanned = 0
    kept = []
    dropped_length = 0
    dropped_no_content = 0

    with open(config.PRICE_COFFEE_FILE, "r") as f:
        for line in tqdm(f, desc="Filtering by length + content"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            scanned += 1
            title = record.get("title", "")
            title_str = title.strip() if isinstance(title, str) else ""

            feat_len, feat_has_content = clean_field(record.get("features"))
            desc_len, desc_has_content = clean_field(record.get("description"))
            details_len = meaningful_details_length(record.get("details"))

            clean_total_length = len(title_str) + feat_len + desc_len
            passes_length = clean_total_length >= config.LENGTH_FLOOR
            passes_nonempty = feat_has_content or desc_has_content or (details_len > 0)

            if not passes_length:
                dropped_length += 1
                continue
            if not passes_nonempty:
                dropped_no_content += 1
                continue

            kept.append(record)

    with open(config.TEXT_FILTERED_FILE, "w") as out:
        for record in kept:
            out.write(json.dumps(record) + "\n")

    pct_kept = (len(kept) / scanned) * 100 if scanned else 0
    print(f"[filter_by_text_sufficiency] scanned={scanned:,} kept={len(kept):,} ({pct_kept:.1f}%) "
          f"dropped_length={dropped_length:,} dropped_no_content={dropped_no_content:,}")
    return len(kept)


def classify_uncategorized(title):
    t = (title or "").lower()
    for stratum, keywords in config.KEYWORD_RULES:
        if any(kw in t for kw in keywords):
            return stratum
    return None


# Title-based override for Amazon category mis-labels
# Amazon occasionally places pre-ground products under "Whole Coffee Beans".
# When the title clearly signals ground coffee and has no whole-bean signals,
# we correct the stratum rather than trust the wrong category path.

_GROUND_SIGNALS    = ["ground coffee", "ground"]
_WHOLE_BEAN_SIGNALS = ["whole bean", "whole beans", "coffee bean", "coffee beans",
                        "roasted bean", "unroasted bean", "green bean"]

def title_override_stratum(stratum, title):
    """
    Correct PATH_TO_STRATUM assignments that contradict the product title.

    Currently handles one known mis-label direction:
      Whole Bean -> Ground Coffee  when the title contains a ground signal
                                  but no whole-bean signal.
    """
    if stratum != "Whole Bean":
        return stratum, False   # (stratum, was_overridden)

    t = (title or "").lower()
    has_ground    = any(sig in t for sig in _GROUND_SIGNALS)
    has_whole_bean = any(sig in t for sig in _WHOLE_BEAN_SIGNALS)

    if has_ground and not has_whole_bean:
        return "Ground Coffee", True

    return stratum, False


def stratified_sample():
    """Step 5: stratified 60-SKU selection (format x price tier) -> StratifiedSample60.jsonl + report"""
    all_records = []
    unresolved_samples = []
    title_overrides = []   # track any corrected mis-labels for transparency

    with open(config.TEXT_FILTERED_FILE, "r") as f:
        for line in tqdm(f, desc="Loading + classifying"):
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue

            path = get_category_path(record)
            if path in config.PATH_TO_STRATUM:
                stratum = config.PATH_TO_STRATUM[path]
                # Correct Amazon category mis-labels using the title
                stratum, overridden = title_override_stratum(stratum, record.get("title"))
                if overridden:
                    title_overrides.append(record.get("title", "(no title)"))
            elif path == config.UNCATEGORIZED_PATH:
                stratum = classify_uncategorized(record.get("title"))
                if stratum is None:
                    if len(unresolved_samples) < 30:
                        unresolved_samples.append(record.get("title", "(no title)"))
                    continue
            else:
                continue

            record["_stratum"] = stratum
            record["_price"] = record.get("price")
            record["_length"] = descriptive_length(record)
            all_records.append(record)

    if title_overrides:
        print(f"  [title_override] Corrected {len(title_overrides)} mis-labeled product(s):")
        for t in title_overrides:
            print(f"    Whole Bean → Ground Coffee | {t[:90]}")

    by_stratum = defaultdict(list)
    for r in all_records:
        by_stratum[r["_stratum"]].append(r)

    tier_boundaries = {}
    for stratum, records in by_stratum.items():
        prices = sorted(r["_price"] for r in records)
        n = len(prices)
        t1 = prices[max(0, min(n - 1, round(n / 3) - 1))]
        t2 = prices[max(0, min(n - 1, round(2 * n / 3) - 1))]
        tier_boundaries[stratum] = (t1, t2)
        for r in records:
            if r["_price"] <= t1:
                r["_tier"] = "Budget"
            elif r["_price"] <= t2:
                r["_tier"] = "Mid-Range"
            else:
                r["_tier"] = "Premium"

    random.seed(config.RANDOM_SEED)
    all_lengths_sorted = sorted(r["_length"] for r in all_records)
    p99_idx = min(len(all_lengths_sorted) - 1, round(0.99 * (len(all_lengths_sorted) - 1)))
    global_p99_length = all_lengths_sorted[p99_idx]

    selected = []
    shortfalls = []
    outlier_excluded_total = 0

    for stratum, tiers in config.QUOTAS.items():
        for tier, quota in tiers.items():
            full_pool = [r for r in by_stratum.get(stratum, []) if r.get("_tier") == tier]
            pool = [r for r in full_pool if r["_length"] <= global_p99_length]
            outlier_excluded_total += (len(full_pool) - len(pool))

            chosen = pool if len(pool) <= quota else random.sample(pool, quota)
            selected.extend(chosen)
            if len(chosen) < quota:
                shortfalls.append((stratum, tier, len(chosen), quota, len(pool)))

    with open(config.SAMPLE_FILE, "w") as out:
        for r in selected:
            clean_record = {k: v for k, v in r.items() if not k.startswith("_")}
            clean_record["features_clean"] = sanitized_list(r.get("features"))
            clean_record["description_clean"] = sanitized_list(r.get("description"))
            clean_record["stratum"] = r["_stratum"]
            clean_record["price_tier"] = r["_tier"]
            out.write(json.dumps(clean_record) + "\n")

    _write_selection_report(all_records, unresolved_samples, by_stratum, tier_boundaries,
                             global_p99_length, outlier_excluded_total, selected, shortfalls)

    print(f"[stratified_sample] selected={len(selected)}/60  unresolved={len(unresolved_samples)}  "
          f"shortfalls={len(shortfalls)}")
    return len(selected)


def _write_selection_report(all_records, unresolved_samples, by_stratum, tier_boundaries,
                             global_p99_length, outlier_excluded_total, selected, shortfalls):
    with open(config.REPORT_FILE, "w") as out:
        out.write("=" * 90 + "\nSTRATIFIED SELECTION REPORT\n" + "=" * 90 + "\n\n")
        out.write(f"Candidate pool: {len(all_records) + len(unresolved_samples):,} scanned, "
                   f"{len(all_records):,} assigned to a stratum, "
                   f"{len(unresolved_samples):,} unresolved\n\n")
        if unresolved_samples:
            out.write("-" * 90 + "\nUNRESOLVED TITLES (needs manual assignment)\n" + "-" * 90 + "\n")
            for t in unresolved_samples:
                out.write(f"  {t}\n")
            out.write("\n")
        out.write("-" * 90 + "\nPER-STRATUM PRICE TERCILE BOUNDARIES\n" + "-" * 90 + "\n")
        for stratum, (t1, t2) in tier_boundaries.items():
            out.write(f"  {stratum:<20} Budget<=${t1:.2f}  Mid<=${t2:.2f}  Premium>${t2:.2f}  "
                       f"(pool: {len(by_stratum[stratum]):,})\n")
        out.write(f"\n  Global p99 length cutoff: {global_p99_length:,} chars, "
                   f"{outlier_excluded_total:,} excluded as outliers\n")
        out.write("\n" + "-" * 90 + "\nSELECTION RESULT\n" + "-" * 90 + "\n")
        for stratum, tiers in config.QUOTAS.items():
            for tier, quota in tiers.items():
                pool_size = len([r for r in by_stratum.get(stratum, []) if r.get("_tier") == tier])
                n_selected = len([r for r in selected if r["_stratum"] == stratum and r["_tier"] == tier])
                flag = "  <-- SHORTFALL" if n_selected < quota else ""
                out.write(f"{stratum:<20} {tier:<12} {n_selected:>9} {quota:>6} {pool_size:>15,}{flag}\n")
        out.write(f"\nTotal selected: {len(selected):,} / 60\n")


def main():
    os.makedirs(config.PROCESSED_DIR, exist_ok=True)
    filter_by_category()
    trim_fields()
    filter_by_price()
    filter_by_text_sufficiency()
    stratified_sample()


if __name__ == "__main__":
    main()
