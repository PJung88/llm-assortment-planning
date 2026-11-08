"""
Stage 07: Analysis of the 960 LLM rankings.

Thesis storyline for this stage:
  stability -> position bias -> cross-model agreement -> rubric compliance
  -> embedding comparison   (then: conversion to substitution shares -> OR model)

  A1  Stability      Does each model agree with itself across its 4 candidate orders?
                     Kendall's tau-b; top-1 and top-3 stability.
  A2  Position bias  Does the same candidate get a worse rank when shown later?
                     Within-(SKU, candidate) slope + permutation test.
  A3  Agreement      Do the 4 models agree? Pairwise tau-b, Kendall's W per SKU,
                     pairwise top-3 Jaccard, all-4 same top-1.
  A4  Rubric         Do models respect the prompt's format / caffeine hard constraints?
                     (needs verified attributes from Stage 06; skipped otherwise)
  A5  Exploratory    Where is disagreement concentrated? Descriptive subgroup
                     summaries only - no tests (60 SKUs, small cells).
  +   Embedding baseline: how far do the LLM rankings move from the embedding order?
      Lower agreement is NOT "better": the embedding is a retrieval mechanism, not
      behavioural ground truth; disagreement only shows the LLM stage adds structure.

Uncertainty: every headline number gets a 95% percentile CI from a SKU-level
bootstrap (SKUs resampled as whole units, with replacement, within the five
Stage 01 format groups; 2,000 replicates by default). The CIs quantify SKU-level
sensitivity within this 60-product pilot design; they are not population-level
confidence intervals for the full Amazon coffee market and do not capture LLM
run-to-run noise (one run per candidate order).

Chance references: simulated with the SAME pipeline as the observed statistics
(4 random rankings per model -> mean rank / Borda with embedding-order tie-break
-> W, top-1, top-3, Jaccard). No simulated baseline for tau: tau = 0 is the
no-association reference.

Caffeine rule (A4): the ranking prompt applies the caffeine constraint "where this
is stated", so the primary rule counts a caffeine conflict ONLY when both products
state their caffeine and the classes differ; any pair involving "not_stated" is not
a caffeine conflict. Sensitivity: "not_stated" treated as caffeinated (retail
convention), reported per model for the overall rubric and the caffeine-only dimension.

Inputs
------
  outputs/rankings/raw_rankings.parquet          (Stage 05; 960 rows)
  outputs/embeddings/all_pairs_scores.parquet    (Stage 04; defines the embedding order)
  data/processed/StratifiedSample60.jsonl        (Stage 01 format groups, for the bootstrap)
  outputs/attributes/attributes_final.parquet    (Stage 06 --finalize; optional)

Outputs (outputs/analysis/)
---------------------------
  headline_metrics.csv          headline/core estimates with 95% CI, chance reference, and reporting role
  rankings_aggregated.parquet   one row per (ranking_set, oos_asin, candidate), ranking_set =
                                each model's 4 orders combined + "consensus" (all 16).
                                For descriptive summaries and deterministic downstream
                                scenarios. NOT the input to a Plackett-Luce model: PL should
                                be fitted on the raw repeated rankings (raw_rankings.parquet),
                                which this stage leaves untouched.
  sku_agreement.csv             per-SKU agreement metrics + exploratory attributes/compatibility counts
  tables/*.csv                  detailed tables per analysis
  figures/*.png                 thesis figures
  analysis_report.txt           the headline numbers in one place

Aggregation rule
----------------
  mean rank over the rankings in the set (= Borda); ties broken by embedding order
  (higher cosine similarity wins). This deterministic tie-break is used for summaries and
  matched chance references; tie-break sensitivity diagnostics are retained in sku_agreement.csv.
  Raw repeated rankings remain the input for any Plackett-Luce estimation.

Usage
-----
    python src/pipeline/07_ranking_analysis.py
    python src/pipeline/07_ranking_analysis.py --n-perm 20000 --n-boot 2000 --n-sim 20000
    python src/pipeline/07_ranking_analysis.py --no-figures
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau, rankdata

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    ALL_PAIRS_SCORES,
    EMBEDDING_TOP_K,
    OUTPUTS_DIR,
    QUOTAS,
    RANDOM_SEED,
    RANKING_MODELS_ALL,
    RANKINGS_FILE,
    SAMPLE_FILE,
)

ANALYSIS_DIR = os.path.join(OUTPUTS_DIR, "analysis")
TABLE_DIR = os.path.join(ANALYSIS_DIR, "tables")
FIG_DIR = os.path.join(ANALYSIS_DIR, "figures")
ATTR_FILE = os.path.join(OUTPUTS_DIR, "attributes", "attributes_final.parquet")

K = EMBEDDING_TOP_K          # 7 candidates
N_PERMS = 4                   # candidate orders per (model, SKU)
TOP = 3                       # top-k used for set agreement
EXPECTED_N_SKUS = 60          # fixed by the Stage 05 design
EXPECTED_N_MODELS = 4         # fixed by the Stage 05 design
EXPECTED_N_ROWS = EXPECTED_N_SKUS * EXPECTED_N_MODELS * N_PERMS   # 960
EXPECTED_K = 7                 # Stage 05 retrieval depth
CONFIG_N_SKUS = sum(sum(tiers.values()) for tiers in QUOTAS.values())
EXPECTED_FORMAT_COUNTS = {fmt: sum(tiers.values()) for fmt, tiers in QUOTAS.items()}
EXPECTED_CELL_COUNTS = {(fmt, tier): n for fmt, tiers in QUOTAS.items() for tier, n in tiers.items()}

SHORT = {"gemini-3.6-flash": "Gemini", "claude-sonnet-4-5": "Claude",
         "qwen3.6:27b": "Qwen", "deepseek-r1:32b": "DeepSeek"}

# Single source of truth: the same format-group labels Stage 01 sampled on (config.QUOTAS).
EXPECTED_FORMAT_GROUPS = set(QUOTAS)

# Rubric compatibility (A4). Edit here if you change the definition.
# Formats in the same group count as compatible. Default: every format is its own group.
FORMAT_GROUP = {f: f for f in ["whole_bean", "ground", "pod_capsule", "instant",
                               "ready_to_drink", "concentrate"]}
UNDETERMINED_FORMAT = {"other", "unknown", "unresolvable", ""}
# Pods with two DIFFERENT named systems (e.g. K-Cup vs Nespresso) are incompatible;
# "other_or_unspecified" is treated as compatible with any system.
NAMED_POD_SYSTEMS = {"keurig_kcup", "nespresso_original", "nespresso_vertuo", "tassimo"}
# Caffeine classes per rule. A value missing from the map (not_stated under the
# primary rule, mixed, unresolvable) means "no stated caffeine" -> never a conflict.
CAFFEINE_RULES = {
    "stated_only": {"regular": "caf", "decaf": "nocaf", "caffeine_free": "nocaf",
                    "half_caf": "half"},
    "unstated_caffeinated": {"regular": "caf", "not_stated": "caf", "decaf": "nocaf",
                             "caffeine_free": "nocaf", "half_caf": "half"},
}
PRIMARY_CAFFEINE_RULE = "stated_only"


# Load + build arrays

def load_embedding_order() -> tuple[dict[str, list[str]], dict[tuple[str, str], float]]:
    df = pd.read_parquet(ALL_PAIRS_SCORES)
    order, cos = {}, {}
    for a, g in df.groupby("asin_a"):
        g = g.sort_values("cosine_sim", ascending=False).head(K)   # same rule as Stage 05
        order[a] = g["asin_b"].tolist()
        for b, s in zip(g["asin_b"], g["cosine_sim"]):
            cos[(a, b)] = float(s)
    return order, cos


def load_strata(skus: list[str]) -> np.ndarray:
    """Load and validate the Stage 01 sample; return format groups for the bootstrap.

    Validation is deliberately stricter than the bootstrap itself: the bootstrap
    resamples within format groups, while this function also verifies that the
    Stage 01 format x price-tier quotas in config.QUOTAS are exactly present.
    """
    if not os.path.exists(SAMPLE_FILE):
        raise SystemExit(
            f"Stage 01 sample file not found: {SAMPLE_FILE}. "
            "The format-stratified bootstrap cannot be run without it."
        )

    rows: list[dict] = []
    seen: set[str] = set()
    duplicate_asins: list[str] = []
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            asin = r.get("parent_asin")
            if asin in seen:
                duplicate_asins.append(asin)
            seen.add(asin)
            rows.append(r)

    if duplicate_asins:
        raise SystemExit(f"Duplicate ASINs in Stage 01 sample file: {sorted(set(duplicate_asins))}")
    if len(rows) != EXPECTED_N_SKUS:
        raise SystemExit(f"Stage 01 sample has {len(rows)} rows; locked design requires {EXPECTED_N_SKUS}.")

    by_asin = {r.get("parent_asin"): r for r in rows}
    missing = [a for a in skus if a not in by_asin]
    extra = sorted(set(by_asin) - set(skus))
    if missing or extra:
        raise SystemExit(
            "Stage 01 sample and analysis SKU sets differ. "
            f"Missing from sample={missing}; extra in sample={extra}"
        )

    missing_group = [a for a in skus if not by_asin[a].get("stratum")]
    missing_tier = [a for a in skus if not by_asin[a].get("price_tier")]
    if missing_group or missing_tier:
        raise SystemExit(
            f"Stage 01 labels missing: format group for {missing_group}; price tier for {missing_tier}"
        )

    fmt_counts = pd.Series([by_asin[a]["stratum"] for a in skus]).value_counts().to_dict()
    if fmt_counts != EXPECTED_FORMAT_COUNTS:
        raise SystemExit(
            "Stage 01 format counts do not match config.QUOTAS. "
            f"Observed={fmt_counts}; expected={EXPECTED_FORMAT_COUNTS}"
        )

    cell_counts = {}
    for a in skus:
        key = (by_asin[a]["stratum"], by_asin[a]["price_tier"])
        cell_counts[key] = cell_counts.get(key, 0) + 1
    if cell_counts != EXPECTED_CELL_COUNTS:
        raise SystemExit(
            "Stage 01 format x price-tier counts do not match config.QUOTAS. "
            f"Observed={cell_counts}; expected={EXPECTED_CELL_COUNTS}"
        )

    return np.array([by_asin[a]["stratum"] for a in skus])


def build_arrays(raw: pd.DataFrame, emb_order: dict[str, list[str]], models: list[str]):
    """
    R[m, s, p, c] = rank (1..K) that model m gave candidate c in candidate-order p for SKU s
    S[s, p, c]    = display slot (1..K) of candidate c in candidate-order p
    Candidate index c = position in the embedding order (0 = most similar).
    """
    skus = sorted(emb_order)
    R = np.zeros((len(models), len(skus), N_PERMS, K), dtype=float)
    S = np.zeros((len(skus), N_PERMS, K), dtype=float)
    idx = {(r.model, r.oos_asin, int(r.perm_index)): r for r in raw.itertuples()}
    for si, a in enumerate(skus):
        base = emb_order[a]
        for p in range(N_PERMS):
            order0 = None
            for mi, m in enumerate(models):
                r = idx[(m, a, p)]
                co, rk = json.loads(r.candidate_order), json.loads(r.ranking)
                if order0 is None:
                    order0 = co
                    S[si, p] = [co.index(c) + 1 for c in base]
                elif co != order0:
                    raise ValueError(f"candidate order differs across models for {a} perm {p}")
                R[mi, si, p] = [rk.index(c) + 1 for c in base]
    return skus, R, S


def integrity(raw: pd.DataFrame, emb_order: dict, models: list[str]) -> pd.DataFrame:
    """Fail fast unless the Stage 05 dataset exactly matches the design."""
    expected_models = set(models)
    expected_skus = set(emb_order)
    expected_perms = set(range(N_PERMS))
    expected_rows = EXPECTED_N_ROWS

    observed_models = set(raw["model"].unique())
    observed_skus = set(raw["oos_asin"].unique())
    duplicates = int(raw.duplicated(["model", "oos_asin", "perm_index"]).sum())

    parsed = raw.assign(
        co=raw.candidate_order.map(json.loads),
        rk=raw.ranking.map(json.loads),
    )
    parsed["co_key"] = parsed.co.map(lambda x: json.dumps(x, separators=(",", ":")))

    candidate_set_ok = parsed.apply(
        lambda r: (
            r.oos_asin in emb_order
            and len(r.co) == K
            and len(set(r.co)) == K
            and set(r.co) == set(emb_order[r.oos_asin])
        ),
        axis=1,
    )
    ranking_ok = parsed.apply(
        lambda r: len(r.rk) == K and len(set(r.rk)) == K and set(r.rk) == set(r.co),
        axis=1,
    )

    expected_keys = {
        (m, a, p) for m in models for a in expected_skus for p in range(N_PERMS)
    }
    observed_keys = set(
        zip(parsed["model"], parsed["oos_asin"], parsed["perm_index"].astype(int))
    )
    missing_keys = expected_keys - observed_keys
    unexpected_keys = observed_keys - expected_keys

    bad_perm_values = sorted(set(parsed["perm_index"].astype(int)) - expected_perms)
    order_consistency = parsed.groupby(["oos_asin", "perm_index"])["co_key"].nunique()
    inconsistent_orders = int((order_consistency != 1).sum())

    perm0 = parsed[parsed.perm_index == 0]
    matches_emb = perm0.apply(
        lambda r: r.oos_asin in emb_order and r.co == emb_order[r.oos_asin], axis=1
    )
    echo = parsed.apply(lambda r: r.rk == r.co, axis=1)

    rows = [
        ("rows", len(parsed)),
        ("expected rows", expected_rows),
        ("models", parsed.model.nunique()),
        ("SKUs", parsed.oos_asin.nunique()),
        ("duplicate (model, SKU, permutation) rows", duplicates),
        ("missing expected ranking cells", len(missing_keys)),
        ("unexpected ranking cells", len(unexpected_keys)),
        ("rows with invalid candidate set", int((~candidate_set_ok).sum())),
        ("invalid rankings (not an exact permutation of candidates)", int((~ranking_ok).sum())),
        ("SKU x permutation cells with candidate-order disagreement across models", inconsistent_orders),
        ("order 0 equals embedding top-7 order", f"{int(matches_emb.sum())}/{len(perm0)}"),
    ]
    for m in models:
        rows.append((
            f"rankings identical to the presented order ({SHORT.get(m, m)})",
            int(echo[parsed.model == m].sum()),
        ))

    failures = []
    if K != EXPECTED_K:
        failures.append(f"EMBEDDING_TOP_K={K} != locked retrieval depth {EXPECTED_K}")
    if CONFIG_N_SKUS != EXPECTED_N_SKUS:
        failures.append(
            f"config.QUOTAS implies {CONFIG_N_SKUS} SKUs != locked design count {EXPECTED_N_SKUS}"
        )
    if len(EXPECTED_FORMAT_GROUPS) != 5:
        failures.append(f"config.QUOTAS contains {len(EXPECTED_FORMAT_GROUPS)} format groups; locked design requires 5")
    if len(models) != EXPECTED_N_MODELS:
        failures.append(f"configured model count {len(models)} != locked design count {EXPECTED_N_MODELS}")
    if len(expected_skus) != EXPECTED_N_SKUS:
        failures.append(f"embedding focal-SKU count {len(expected_skus)} != locked design count {EXPECTED_N_SKUS}")
    if len(parsed) != expected_rows:
        failures.append(f"row count {len(parsed)} != expected {expected_rows}")
    if observed_models != expected_models:
        failures.append(
            f"model set mismatch: missing={sorted(expected_models - observed_models)}, "
            f"unexpected={sorted(observed_models - expected_models)}"
        )
    if observed_skus != expected_skus:
        failures.append(
            f"SKU set mismatch: missing={sorted(expected_skus - observed_skus)}, "
            f"unexpected={sorted(observed_skus - expected_skus)}"
        )
    if duplicates:
        failures.append(f"{duplicates} duplicate ranking keys")
    if missing_keys:
        failures.append(f"{len(missing_keys)} expected ranking cells are missing")
    if unexpected_keys:
        failures.append(f"{len(unexpected_keys)} unexpected ranking cells are present")
    if bad_perm_values:
        failures.append(f"unexpected permutation indices: {bad_perm_values}")
    if not candidate_set_ok.all():
        failures.append(f"{int((~candidate_set_ok).sum())} rows do not contain the expected top-{K} candidates")
    if not ranking_ok.all():
        failures.append(f"{int((~ranking_ok).sum())} rankings are not exact candidate permutations")
    if inconsistent_orders:
        failures.append(f"{inconsistent_orders} SKU/permutation cells differ in candidate order across models")
    if not matches_emb.all():
        failures.append("permutation 0 does not always equal the embedding top-7 order")

    if failures:
        detail = pd.DataFrame(rows, columns=["check", "value"]).to_string(index=False)
        raise SystemExit("Integrity check failed:\n  - " + "\n  - ".join(failures) + "\n\n" + detail)

    return pd.DataFrame(rows, columns=["check", "value"])


# Helpers

def tau(x, y) -> float:
    return float(kendalltau(x, y).statistic)      # tau-b (handles ties)


def kendall_w(ranks: np.ndarray) -> float:
    """ranks: raters x items, may contain ties (average ranks). Tie-corrected W."""
    m, n = ranks.shape
    Rj = ranks.sum(0)
    Ssq = ((Rj - Rj.mean()) ** 2).sum()
    T = 0.0
    for row in ranks:
        _, counts = np.unique(row, return_counts=True)
        T += (counts ** 3 - counts).sum()
    denom = m ** 2 * (n ** 3 - n) - m * T
    return float(12 * Ssq / denom) if denom > 0 else np.nan


def borda_order(mean_ranks: np.ndarray) -> np.ndarray:
    """Candidate indices best -> worst; ties broken by embedding position (lower index wins)."""
    return np.lexsort((np.arange(len(mean_ranks)), mean_ranks))


def topk_options(mean_ranks: np.ndarray, k: int) -> list[frozenset]:
    """All top-k sets consistent with some tie-break."""
    v = np.sort(mean_ranks)
    cut = v[k - 1]
    sure = [i for i in range(len(mean_ranks)) if mean_ranks[i] < cut]
    tied = [i for i in range(len(mean_ranks)) if mean_ranks[i] == cut]
    return [frozenset(sure) | frozenset(c) for c in itertools.combinations(tied, k - len(sure))]


def agree_range(option_lists: list[list[frozenset]]) -> tuple[bool, bool]:
    """(agree under every tie-break, agree under at least one tie-break)."""
    same = [len(set(c)) == 1 for c in itertools.product(*option_lists)]
    return all(same), any(same)


def jaccard(a: frozenset, b: frozenset) -> float:
    return len(a & b) / len(a | b)


def mean_pairwise_jaccard(sets: list[frozenset]) -> float:
    return float(np.mean([jaccard(a, b) for a, b in itertools.combinations(sets, 2)]))


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Holm-adjust p-values for a family of secondary tests."""
    p_values = np.asarray(p_values, dtype=float)
    m = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for j, idx in enumerate(order):
        running = max(running, (m - j) * p_values[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


# Bootstrap

def stratified_boot_index(strata: np.ndarray, n_boot: int, seed: int) -> np.ndarray:
    """(n_boot, n) SKU indices, resampled with replacement within each stratum."""
    rng = np.random.default_rng(seed)
    n = len(strata)
    out = np.empty((n_boot, n), dtype=int)
    for st in np.unique(strata):
        members = np.flatnonzero(strata == st)
        out[:, members] = rng.choice(members, size=(n_boot, len(members)), replace=True)
    return out


def ci_mean(v: np.ndarray, bidx: np.ndarray) -> tuple[float, float, float]:
    v = np.asarray(v, dtype=float)
    reps = v[bidx].mean(1)
    return float(v.mean()), float(np.percentile(reps, 2.5)), float(np.percentile(reps, 97.5))


def ci_ratio(num: np.ndarray, den: np.ndarray, bidx: np.ndarray) -> tuple[float, float, float]:
    num, den = np.asarray(num, float), np.asarray(den, float)
    if den.sum() == 0:
        return (np.nan, np.nan, np.nan)
    d = den[bidx].sum(1)
    reps = np.where(d > 0, num[bidx].sum(1) / np.where(d > 0, d, 1), np.nan)
    return (float(num.sum() / den.sum()), float(np.nanpercentile(reps, 2.5)),
            float(np.nanpercentile(reps, 97.5)))


# CHANCE REFERENCE (same pipeline as the observed statistics)

def simulate_null(n_models: int, n_sim: int, seed: int) -> dict:
    """
    Per simulated SKU: each model gives N_PERMS uniformly random rankings of K
    candidates; they are averaged (mean rank = Borda), ranked with average ties
    for W, and ordered with the embedding-index tie-break for top-1 / top-3.
    """
    rng = np.random.default_rng(seed)
    ranks = np.argsort(rng.random((n_sim, n_models, N_PERMS, K)), axis=3).argsort(axis=3) + 1
    mean = ranks.mean(axis=2)                                   # (n_sim, models, K)
    W = np.array([kendall_w(np.array([rankdata(r) for r in mean[i]])) for i in range(n_sim)])
    # Candidate index is embedding order; stable sort therefore implements the exact
    # same embedding-order tie-break as borda_order without an epsilon approximation.
    order = np.argsort(mean, axis=2, kind="stable")
    top1 = order[:, :, 0]
    tops = [[frozenset(order[i, m, :TOP]) for m in range(n_models)] for i in range(n_sim)]
    inter = np.array([len(frozenset.intersection(*t)) for t in tops])
    return {
        "W": W,
        "W_mean": float(np.nanmean(W)), "W_95pct": float(np.nanpercentile(W, 95)),
        "W_undefined_count": int(np.isnan(W).sum()),
        "all_same_top1": float((top1 == top1[:, [0]]).all(1).mean()),
        "all_same_top3": float(np.mean([len(set(t)) == 1 for t in tops])),
        "jaccard_mean": float(np.mean([mean_pairwise_jaccard(t) for t in tops])),
        "top3_intersection_dist": pd.Series(inter).value_counts(normalize=True)
                                    .reindex(range(TOP + 1), fill_value=0),
    }


# Analyses

def a1_stability(R, models, skus):
    """Returns summary table, pair-level taus, and per-SKU arrays for the bootstrap."""
    n_m, n_s = len(models), len(skus)
    tau_sku, same1, same3 = np.zeros((n_m, n_s)), np.zeros((n_m, n_s)), np.zeros((n_m, n_s))
    per = []
    for mi, m in enumerate(models):
        for si in range(n_s):
            rk = R[mi, si]
            ts = []
            for p1, p2 in itertools.combinations(range(N_PERMS), 2):
                t = tau(rk[p1], rk[p2])
                ts.append(t)
                per.append({"model": m, "oos_asin": skus[si], "perm_a": p1, "perm_b": p2, "tau": t})
            tau_sku[mi, si] = np.mean(ts)
            same1[mi, si] = len({int(np.argmin(r)) for r in rk}) == 1
            same3[mi, si] = len({frozenset(np.argsort(r, kind="stable")[:TOP]) for r in rk}) == 1
    per = pd.DataFrame(per)
    rows = []
    for mi, m in enumerate(models):
        # SKU is the inferential/descriptive unit: each SKU contributes one mean tau
        # (the six within-SKU pairwise taus remain available in t1_stability_pairs.csv).
        t = tau_sku[mi]
        rows.append({"model": m, "unit": "SKU", "n_SKUs": n_s,
                     "mean_tau": t.mean(), "median_tau": np.median(t),
                     "q25": np.quantile(t, .25), "q75": np.quantile(t, .75), "min": t.min(),
                     "share_tau_lt_0": (t < 0).mean(),
                     "share_SKUs_same_top1": same1[mi].mean(), "share_SKUs_same_top3": same3[mi].mean()})
    per_sku = pd.DataFrame([
        {"model": m, "oos_asin": skus[si], "mean_pairwise_tau": tau_sku[mi, si],
         "same_top1_all_4": same1[mi, si], "same_top3_all_4": same3[mi, si]}
        for mi, m in enumerate(models) for si in range(n_s)
    ])
    return pd.DataFrame(rows), per, per_sku, {"tau": tau_sku, "same1": same1, "same3": same3}


def a2_position_bias(R, S, models, n_perm: int, seed: int):
    """
    Within-(SKU, candidate) slope of rank on display slot, per model.
    Demeaning within each (SKU, candidate) removes candidate quality, so beta
    measures the pure effect of where a candidate was shown.
    beta > 0: shown later -> worse (higher) rank = primacy bias.
    Permutation test: under H0 the 4 rankings of a SKU are exchangeable w.r.t.
    the 4 presented orders, so the order labels are shuffled within each SKU.
    Also returns per-SKU sums (sum xy, sum xx) so beta can be bootstrapped.
    """
    rng = np.random.default_rng(seed)
    n_sku = S.shape[0]
    xs = S - S.mean(1, keepdims=True)                        # (sku, perm, cand)
    perms = np.argsort(rng.random((n_perm, n_sku, N_PERMS)), axis=2)
    sku_ix = np.arange(n_sku)[:, None]
    y_all = R - R.mean(2, keepdims=True)                     # (model, sku, perm, cand)
    sxy = (y_all * xs[None]).sum((2, 3))                     # (model, sku)
    sxx = (xs * xs).sum((1, 2))                              # (sku,)
    rows = []
    targets = [(m, [mi]) for mi, m in enumerate(models)] + [("all models", list(range(len(models))))]
    for name, mis in targets:
        y = y_all[mis]
        denom = len(mis) * sxx.sum()
        b_obs = float(sxy[mis].sum() / denom)
        null = np.empty(n_perm)
        for i in range(n_perm):
            xp = xs[sku_ix, perms[i]]
            null[i] = (y * xp[None]).sum() / denom
        p = (np.sum(np.abs(null) >= abs(b_obs)) + 1) / (n_perm + 1)
        rows.append({"model": name, "beta_rank_per_slot": b_obs,
                     "implied_rank_gap_slot1_vs_slot7": b_obs * (K - 1),
                     "p_value_two_sided": p, "n_permutations": n_perm})
    desc = []
    for mi, m in enumerate(models):
        for slot in range(1, K + 1):
            r = R[mi][S == slot]
            desc.append({"model": m, "slot": slot, "mean_rank": r.mean(), "top1_share": (r == 1).mean()})
    tests = pd.DataFrame(rows)
    tests["inference_role"] = np.where(tests.model == "all models", "primary pooled", "secondary model-specific")
    tests["p_value_holm"] = np.nan
    secondary = tests.model != "all models"
    tests.loc[secondary, "p_value_holm"] = holm_adjust(tests.loc[secondary, "p_value_two_sided"].to_numpy())
    return tests, pd.DataFrame(desc), {"sxy": sxy, "sxx": sxx}


def aggregate(R, models):
    """mean ranks per ranking set: dict name -> (n_sku, K)."""
    sets = {m: R[mi].mean(1) for mi, m in enumerate(models)}
    sets["consensus"] = R.mean((0, 2))
    return sets


def a3_agreement(R, sets, models, skus):
    n_sku = len(skus)
    top3 = {m: [frozenset(borda_order(sets[m][s])[:TOP]) for s in range(n_sku)] for m in models}
    pair_rows, tau_mat = [], pd.DataFrame(1.0, index=models, columns=models)
    for a, b in itertools.combinations(models, 2):
        t = np.array([tau(sets[a][s], sets[b][s]) for s in range(n_sku)])
        j = np.array([jaccard(top3[a][s], top3[b][s]) for s in range(n_sku)])
        pair_rows.append({"model_a": a, "model_b": b, "mean_tau": t.mean(), "median_tau": np.median(t),
                          "mean_top3_jaccard": j.mean(),
                          "share_SKUs_same_top3_set": np.mean([top3[a][s] == top3[b][s] for s in range(n_sku)])})
        tau_mat.loc[a, b] = tau_mat.loc[b, a] = t.mean()

    sku_rows = []
    for s in range(n_sku):
        M = np.array([sets[m][s] for m in models])
        ranks = np.array([rankdata(r) for r in M])
        tops = [borda_order(r) for r in M]
        t3 = [frozenset(t[:TOP]) for t in tops]
        a1_all, a1_any = agree_range([[frozenset([i]) for i in np.flatnonzero(r == r.min())] for r in M])
        a3_all, a3_any = agree_range([topk_options(r, TOP) for r in M])
        raw16 = {frozenset(np.argsort(R[mi, s, p], kind="stable")[:TOP])
                 for mi in range(len(models)) for p in range(N_PERMS)}
        sku_rows.append({
            "oos_asin": skus[s],
            # headline per-SKU metrics
            "kendall_w": kendall_w(ranks),
            "mean_pairwise_tau": np.mean([tau(M[i], M[j]) for i, j in itertools.combinations(range(len(models)), 2)]),
            f"mean_pairwise_top{TOP}_jaccard": mean_pairwise_jaccard(t3),
            "all_same_top1": len({t[0] for t in tops}) == 1,
            # diagnostics (CSV only)
            "all_same_top1_any_tiebreak": a1_any, "all_same_top1_every_tiebreak": a1_all,
            f"top{TOP}_shared_by_all": len(frozenset.intersection(*t3)),
            f"all_same_top{TOP}_set": len(set(t3)) == 1,
            f"all_same_top{TOP}_any_tiebreak": a3_any, f"all_same_top{TOP}_every_tiebreak": a3_all,
            f"all_same_top{TOP}_ordered": len({tuple(t[:TOP]) for t in tops}) == 1,
            f"all_16_rankings_same_top{TOP}_set": len(raw16) == 1,
            f"distinct_candidates_in_top{TOP}s": len(frozenset.union(*t3)),
            "boundary_tie_models": sum(len(topk_options(r, TOP)) > 1 for r in M),
        })
    return pd.DataFrame(pair_rows), tau_mat, pd.DataFrame(sku_rows)


def a_embedding_baseline(sets, models, skus):
    """Compare Borda mean ranks with embedding order, with explicit top-1 tie diagnostics."""
    emb = np.arange(1, K + 1)
    names = models + ["consensus"]
    per_sku = {name: np.array([tau(sets[name][s], emb) for s in range(len(skus))])
               for name in names}
    top1_fixed, top1_unique, top1_in_tie, boundary_ties = {}, {}, {}, {}
    for name in names:
        fixed, unique, in_tie, tied = [], [], [], []
        for s in range(len(skus)):
            r = sets[name][s]
            best = np.flatnonzero(r == r.min())
            fixed.append(borda_order(r)[0] == 0)
            unique.append(len(best) == 1 and best[0] == 0)
            in_tie.append(0 in best)
            tied.append(len(best) > 1)
        top1_fixed[name] = np.asarray(fixed, float)
        top1_unique[name] = np.asarray(unique, float)
        top1_in_tie[name] = np.asarray(in_tie, float)
        boundary_ties[name] = np.asarray(tied, float)

    rows = [{
        "ranking_set": n,
        "mean_tau_vs_embedding_order": per_sku[n].mean(),
        "median_tau": np.median(per_sku[n]),
        "share_top1_is_embedding_top1_fixed_tiebreak": top1_fixed[n].mean(),
        "share_unique_top1_is_embedding_top1": top1_unique[n].mean(),
        "share_embedding_top1_is_among_tied_best": top1_in_tie[n].mean(),
        "share_SKUs_with_top_boundary_tie": boundary_ties[n].mean(),
    } for n in names]
    return pd.DataFrame(rows), per_sku, top1_fixed


# A4 / A5

def load_attributes() -> pd.DataFrame | None:
    if not os.path.exists(ATTR_FILE):
        return None
    df = pd.read_parquet(ATTR_FILE)
    required = {"parent_asin", "product_type", "format", "pod_system", "caffeine", "is_coffee"}
    missing_cols = sorted(required - set(df.columns))
    if missing_cols:
        raise SystemExit(f"attributes_final.parquet is missing required columns: {missing_cols}")
    dup = df["parent_asin"].duplicated(keep=False)
    if dup.any():
        raise SystemExit(
            "attributes_final.parquet contains duplicate parent_asin values: "
            f"{sorted(df.loc[dup, 'parent_asin'].astype(str).unique().tolist())}"
        )
    attr_cols = ["product_type", "format", "pod_system", "caffeine", "is_coffee"]
    if df[attr_cols].isna().any().any():
        bad = df[attr_cols].columns[df[attr_cols].isna().any()].tolist()
        raise SystemExit(f"attributes_final.parquet contains missing finalized values in: {bad}")
    if not pd.api.types.is_bool_dtype(df["is_coffee"]):
        raise SystemExit(
            f"attributes_final.parquet is_coffee must be boolean, got dtype {df['is_coffee'].dtype}"
        )
    return df.set_index("parent_asin")


def compatible(src: pd.Series, cand: pd.Series, caffeine_rule: str) -> dict[str, bool | None]:
    """
    Format / caffeine / overall compatibility.
      format:   None if either format is undetermined.
      caffeine: None if either product has no stated caffeine class under the rule
                (no constraint applies); else True/False.
      overall:  False if format or caffeine conflicts; None if format undetermined;
                else True (no stated caffeine = no caffeine conflict, per the prompt).
    """
    fa, fb = src["format"], cand["format"]
    if fa in UNDETERMINED_FORMAT or fb in UNDETERMINED_FORMAT:
        fmt = None
    else:
        fmt = FORMAT_GROUP.get(fa, fa) == FORMAT_GROUP.get(fb, fb)
        if fmt and fa == "pod_capsule":
            pa, pb = src["pod_system"], cand["pod_system"]
            if pa in NAMED_POD_SYSTEMS and pb in NAMED_POD_SYSTEMS and pa != pb:
                fmt = False
    cmap = CAFFEINE_RULES[caffeine_rule]
    ca, cb = cmap.get(src["caffeine"]), cmap.get(cand["caffeine"])
    caf = None if ca is None or cb is None else ca == cb
    if fmt is False or caf is False:
        overall = False
    elif fmt is None:
        overall = None
    else:
        overall = True
    return {"format": fmt, "caffeine": caf, "overall": overall}


def a4_detail(R, models, skus, emb_order, attrs, caffeine_rule):
    comp = {(si, ci): compatible(attrs.loc[a], attrs.loc[b], caffeine_rule)
            for si, a in enumerate(skus) for ci, b in enumerate(emb_order[a])}
    rows = []
    for mi, m in enumerate(models):
        for si, a in enumerate(skus):
            for p in range(N_PERMS):
                rk = R[mi, si, p]
                top1 = int(np.argmin(rk))
                for dim in ["overall", "format", "caffeine"]:
                    ok = [comp[(si, c)][dim] for c in range(K)]
                    good = [c for c in range(K) if ok[c] is True]
                    bad = [c for c in range(K) if ok[c] is False]
                    und = [c for c in range(K) if ok[c] is None]
                    pairs = [(g, b) for g in good for b in bad]
                    rows.append({
                        "rule": caffeine_rule, "model": m, "oos_asin": a, "sku_i": si, "perm": p,
                        "dimension": dim, "source_is_coffee": bool(attrs.loc[a, "is_coffee"]),
                        "n_compatible": len(good), "n_incompatible": len(bad), "n_undetermined": len(und),
                        "violation_possible": bool(good and bad),
                        "unforced_top1_violation": bool(good) and ok[top1] is False,
                        "no_confirmed_compatible_candidate": not good,
                        "pairs_mixed": len(pairs),
                        "pairs_violated": sum(rk[b] < rk[g] for g, b in pairs),
                    })
    return pd.DataFrame(rows)


def a4_summarize(d: pd.DataFrame, models: list[str], n_sku: int, bidx: np.ndarray) -> pd.DataFrame:
    """Rates as ratio of sums; CIs by SKU bootstrap (per-SKU numerators/denominators)."""
    out = []
    for dim in ["overall", "format", "caffeine"]:
        for scope in ["all sources", "coffee sources"]:
            dd = d[d.dimension == dim]
            if scope == "coffee sources":
                dd = dd[dd.source_is_coffee]
            for m in models + ["all models"]:
                x = dd if m == "all models" else dd[dd.model == m]
                g = x.groupby("sku_i")
                per = lambda col: g[col].sum().reindex(range(n_sku), fill_value=0).values  # noqa: E731
                n_pos = per("violation_possible")
                n_unf = per("unforced_top1_violation")
                p_mix, p_vio = per("pairs_mixed"), per("pairs_violated")
                u, ul, uh = ci_ratio(n_unf, n_pos, bidx)
                pw, pl, ph = ci_ratio(p_vio, p_mix, bidx)
                out.append({
                    "dimension": dim, "scope": scope, "model": m,
                    "rankings_where_violation_possible": int(n_pos.sum()),
                    "unforced_top1_violations": int(n_unf.sum()),
                    "unforced_top1_violation_rate": u, "ci_low": ul, "ci_high": uh,
                    "pairwise_violation_rate": pw, "pw_ci_low": pl, "pw_ci_high": ph,
                    "rankings_with_no_confirmed_compatible_candidate": int(x.no_confirmed_compatible_candidate.sum()),
                })
    return pd.DataFrame(out)


def candidate_compatibility_by_sku(skus, emb_order, attrs, caffeine_rule: str) -> pd.DataFrame:
    """Candidate-set composition under the stated compatibility rule, independent of LLM rankings.

    These counts are diagnostics/inputs for downstream scenario design; they do NOT
    estimate substitution incidence or walk-away probability.
    """
    rows = []
    for a in skus:
        vals = [compatible(attrs.loc[a], attrs.loc[b], caffeine_rule)["overall"] for b in emb_order[a]]
        rows.append({
            "oos_asin": a,
            "compatibility_rule": caffeine_rule,
            "n_compatible": sum(v is True for v in vals),
            "n_incompatible": sum(v is False for v in vals),
            "n_undetermined": sum(v is None for v in vals),
        })
    return pd.DataFrame(rows)

def a5_exploratory(sku: pd.DataFrame, attrs: pd.DataFrame | None, per_sku_comp: pd.DataFrame | None):
    """Descriptive subgroup summaries only (no tests): 60 SKUs, small cells."""
    if attrs is None:
        return None, sku
    sku = sku.merge(attrs[["product_type", "format", "caffeine", "is_coffee"]],
                    left_on="oos_asin", right_index=True, how="left")
    if per_sku_comp is not None:
        sku = sku.merge(per_sku_comp, on="oos_asin", how="left")
        sku["compatible_group"] = pd.cut(sku.n_compatible, [-1, 1, 3, K], labels=["0-1", "2-3", "4-7"])
    groups = []
    for col in ["is_coffee", "compatible_group", "format"]:
        if col not in sku:
            continue
        for val, g in sku.groupby(col, observed=True):
            groups.append({"grouping": col, "group": val, "n_SKUs": len(g),
                           "mean_W": g.kendall_w.mean(), "mean_pairwise_tau": g.mean_pairwise_tau.mean(),
                           f"mean_top{TOP}_jaccard": g[f"mean_pairwise_top{TOP}_jaccard"].mean(),
                           "share_all_same_top1": g.all_same_top1.mean()})
    return pd.DataFrame(groups), sku


# Aggregated output for the matrix builder

def aggregated_table(R, sets, models, skus, emb_order, cos):
    rows = []
    for name in models + ["consensus"]:
        Rsub = R[[models.index(name)]] if name != "consensus" else R
        flat = Rsub.transpose(1, 3, 0, 2).reshape(len(skus), K, -1)   # sku, cand, rankings
        for si, a in enumerate(skus):
            pos = {c: i + 1 for i, c in enumerate(borda_order(sets[name][si]))}
            for ci, b in enumerate(emb_order[a]):
                r = flat[si, ci]
                rows.append({
                    "ranking_set": name, "oos_asin": a, "candidate_asin": b,
                    "embedding_rank": ci + 1, "cosine_sim": cos[(a, b)],
                    "mean_rank": r.mean(), "borda_rank": pos[ci],
                    "rank_sd": r.std(ddof=1), "top1_rate": (r == 1).mean(),
                    f"top{TOP}_rate": (r <= TOP).mean(), "n_rankings": len(r),
                })
    return pd.DataFrame(rows)


# Figures

# categorical slots 1-4 of the reference palette, bound to models (never to rank)
MODEL_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
INK, INK2, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)
    for s in ["left", "bottom"]:
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def make_figures(stab_sku, pos_desc, tau_mat, sku, inter_tbl, models, null):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    os.makedirs(FIG_DIR, exist_ok=True)
    col = {m: MODEL_COLORS[i] for i, m in enumerate(models)}
    names = [SHORT.get(m, m) for m in models]
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "axes.titleweight": "bold",
                         "figure.facecolor": SURFACE, "text.color": INK, "axes.labelcolor": INK2})
    save = lambda fig, n: (fig.tight_layout(), fig.savefig(os.path.join(FIG_DIR, n), dpi=200,  # noqa: E731
                                                          bbox_inches="tight"), plt.close(fig))

    # F1 stability
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    bp = ax.boxplot([stab_sku[stab_sku.model == m].mean_pairwise_tau.values for m in models], widths=0.45,
                    patch_artist=True, medianprops=dict(color=INK, linewidth=2),
                    flierprops=dict(marker="o", markersize=4, markerfacecolor=INK2, markeredgewidth=0, alpha=.5))
    for patch, m in zip(bp["boxes"], models):
        patch.set_facecolor(col[m]); patch.set_edgecolor(SURFACE); patch.set_linewidth(2)
    for w in bp["whiskers"] + bp["caps"]:
        w.set_color(INK2)
    ax.set_xticks(range(1, len(models) + 1), names)
    ax.set_ylabel("Mean within-SKU Kendall's τ across 6 order pairs")
    ax.set_ylim(-1.05, 1.05)
    ax.set_title("A1  Each model's agreement with itself across candidate orders", loc="left")
    _style(ax); save(fig, "f1_stability.png")

    # F2 position bias
    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for m in models:
        d = pos_desc[pos_desc.model == m]
        ax.plot(d.slot, d.mean_rank, color=col[m], linewidth=2, marker="o", markersize=8,
                markeredgecolor=SURFACE, markeredgewidth=2, label=SHORT.get(m, m))
        ax.annotate(SHORT.get(m, m), (d.slot.iloc[-1], d.mean_rank.iloc[-1]), xytext=(8, 0),
                    textcoords="offset points", va="center", fontsize=9, color=INK2)
    ax.axhline((K + 1) / 2, color=INK2, linewidth=1, linestyle="--")
    ax.text(1, (K + 1) / 2 + 0.03, "no position effect", fontsize=8, color=INK2, va="bottom")
    ax.set_xticks(range(1, K + 1)); ax.set_xlim(0.7, K + 1.1)
    ax.set_xlabel("Display slot (1 = shown first)"); ax.set_ylabel("Mean rank given (lower = better)")
    ax.legend(frameon=False, fontsize=9, loc="lower right", ncol=2)
    ax.set_title("A2  Rank by position in the prompt", loc="left")
    _style(ax); save(fig, "f2_position_bias.png")

    # F3 cross-model tau heatmap
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    mat = tau_mat.values.copy(); np.fill_diagonal(mat, np.nan)
    div = LinearSegmentedColormap.from_list("tau_div", ["#b84a45", "#f4f4f2", "#2a78d6"])
    im = ax.imshow(mat, cmap=div, vmin=-1, vmax=1)
    for i in range(len(models)):
        for j in range(len(models)):
            if i != j:
                ax.text(j, i, f"{mat[i, j]:.2f}", ha="center", va="center", fontsize=10,
                        color="white" if abs(mat[i, j]) > 0.6 else INK)
    ax.set_xticks(range(len(models)), names); ax.set_yticks(range(len(models)), names)
    ax.tick_params(colors=INK2, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("mean Kendall's τ", color=INK2); cb.outline.set_visible(False)
    ax.set_title("A3  Agreement between models", loc="left")
    save(fig, "f3_cross_model_tau.png")

    # F4 Kendall's W per SKU vs matched chance
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.hist(sku.kendall_w, bins=np.linspace(0, 1, 21), color=MODEL_COLORS[0], edgecolor=SURFACE, linewidth=2)
    ax.axvline(null["W_95pct"], color=INK2, linestyle="--", linewidth=1)
    ax.text(null["W_95pct"] - 0.01, ax.get_ylim()[1] * 0.92,
            "95th pct. of W for random rankings\n(same 4-run averaging)", ha="right", fontsize=8, color=INK2)
    ax.set_xlabel("Kendall's W across the 4 models (per SKU)"); ax.set_ylabel("Number of SKUs")
    ax.set_title("A3  Consensus per SKU", loc="left")
    _style(ax); save(fig, "f4_kendall_w.png")

    # F5 top-3 overlap vs matched chance
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    x = inter_tbl.shared_by_all_4.values
    ax.bar(x - 0.18, inter_tbl.share_SKUs, width=0.34, color=MODEL_COLORS[0], label="LLMs (4 models)")
    ax.bar(x + 0.18, inter_tbl.chance_share, width=0.34, color="#b9b8b2",
           label="Random rankings (same 4-run averaging)")
    for xi, v, n in zip(x, inter_tbl.share_SKUs, inter_tbl.n_SKUs):
        ax.text(xi - 0.18, v + 0.01, f"{n}", ha="center", fontsize=9, color=INK)
    ax.set_xticks(x, [str(i) for i in x])
    ax.set_xlabel(f"Candidates in the top {TOP} of all 4 models"); ax.set_ylabel("Share of SKUs")
    ax.legend(frameon=False, fontsize=9)
    ax.set_title(f"A3  Overlap of the models' top-{TOP} substitutes", loc="left")
    _style(ax); save(fig, "f5_top3_overlap.png")


# Main

def clean_previous_outputs():
    """Remove only files generated by this stage so stale results cannot survive a rerun."""
    os.makedirs(TABLE_DIR, exist_ok=True)
    os.makedirs(FIG_DIR, exist_ok=True)
    for path in [
        os.path.join(ANALYSIS_DIR, "headline_metrics.csv"),
        os.path.join(ANALYSIS_DIR, "rankings_aggregated.parquet"),
        os.path.join(ANALYSIS_DIR, "sku_agreement.csv"),
        os.path.join(ANALYSIS_DIR, "analysis_report.txt"),
    ]:
        if os.path.exists(path):
            os.remove(path)
    for folder, suffix in [(TABLE_DIR, ".csv"), (FIG_DIR, ".png")]:
        for path in Path(folder).glob(f"*{suffix}"):
            path.unlink()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-perm", type=int, default=10000, help="permutations for the position-bias test")
    ap.add_argument("--n-boot", type=int, default=2000, help="bootstrap replicates for the 95%% CIs")
    ap.add_argument("--n-sim", type=int, default=20000, help="simulated SKUs for the chance references")
    ap.add_argument("--no-figures", action="store_true")
    args = ap.parse_args()
    if args.n_perm < 1 or args.n_boot < 1 or args.n_sim < 1:
        raise SystemExit("--n-perm, --n-boot and --n-sim must all be positive integers")
    os.makedirs(ANALYSIS_DIR, exist_ok=True)
    clean_previous_outputs()

    raw = pd.read_parquet(RANKINGS_FILE)
    models = list(RANKING_MODELS_ALL)  # all four models are required
    emb_order, cos = load_embedding_order()

    t0 = integrity(raw, emb_order, models)
    skus, R, S = build_arrays(raw, emb_order, models)
    strata = load_strata(skus)
    bidx = stratified_boot_index(strata, args.n_boot, RANDOM_SEED)
    null = simulate_null(len(models), args.n_sim, RANDOM_SEED)

    t1, t1_per, t1_sku, a1 = a1_stability(R, models, skus)
    t2, t2_desc, a2 = a2_position_bias(R, S, models, args.n_perm, RANDOM_SEED)
    sets = aggregate(R, models)
    t3_pairs, tau_mat, sku = a3_agreement(R, sets, models, skus)
    t_emb, emb_tau, emb_top1 = a_embedding_baseline(sets, models, skus)

    # Headline metrics with CIs
    H = []
    def add(section, metric, who, est_lo_hi, chance=np.nan, role="core"):  # noqa: E306
        e, lo, hi = est_lo_hi
        H.append({"section": section, "reporting_role": role, "metric": metric, "model_or_set": who,
                  "estimate": e, "ci_low": lo, "ci_high": hi, "chance_reference": chance})

    for mi, m in enumerate(models):
        add("A1 stability", "mean Kendall tau-b across candidate orders", m, ci_mean(a1["tau"][mi], bidx), 0.0)
        add("A1 stability", "share SKUs same top-1 across 4 orders", m, ci_mean(a1["same1"][mi], bidx))
        add("A1 stability", f"share SKUs same top-{TOP} set across 4 orders", m, ci_mean(a1["same3"][mi], bidx))
    add("A2 position bias", "beta: rank change per later slot", "all models",
        ci_ratio(a2["sxy"].sum(0), len(models) * a2["sxx"], bidx), 0.0, role="primary")
    for mi, m in enumerate(models):
        add("A2 position bias", "beta: rank change per later slot", m,
            ci_ratio(a2["sxy"][mi], a2["sxx"], bidx), 0.0, role="secondary")
    add("A3 agreement", "mean pairwise Kendall tau-b between models", "4 models",
        ci_mean(sku.mean_pairwise_tau.values, bidx), 0.0)
    add("A3 agreement", "mean Kendall's W per SKU", "4 models",
        ci_mean(sku.kendall_w.values, bidx), null["W_mean"])
    add("A3 agreement", f"mean pairwise top-{TOP} Jaccard", "4 models",
        ci_mean(sku[f"mean_pairwise_top{TOP}_jaccard"].values, bidx), null["jaccard_mean"])
    add("A3 agreement", "share SKUs where all 4 models pick the same top-1", "4 models",
        ci_mean(sku.all_same_top1.values.astype(float), bidx), null["all_same_top1"])
    for name in models + ["consensus"]:
        add("Embedding baseline", "mean tau-b vs embedding order", name, ci_mean(emb_tau[name], bidx), 0.0, role="baseline")
        add("Embedding baseline", "share top-1 = embedding top-1 (fixed tie-break)", name, ci_mean(emb_top1[name], bidx), role="baseline")

    # A4 / A5
    attrs = load_attributes()
    t4 = t4_sens = per_sku_comp = None
    primary_a4_scope = a4_scope_note = None
    n_coffee = None
    if attrs is not None:
        missing = [a for a in skus if a not in attrs.index]
        if missing:
            raise SystemExit(f"attributes_final.parquet is missing SKUs: {missing}")
        coffee_flags = attrs.loc[skus, "is_coffee"].astype(bool)
        n_coffee = int(coffee_flags.sum())
        primary_a4_scope = "all sources" if n_coffee == len(skus) else "coffee sources"
        a4_scope_note = (
            "all 60 focal SKUs are verified coffee, so all-source and coffee-source scopes coincide"
            if primary_a4_scope == "all sources"
            else f"{len(skus) - n_coffee} focal SKU(s) are not verified coffee; coffee sources are primary and all sources are sensitivity"
        )

        d_main = a4_detail(R, models, skus, emb_order, attrs, PRIMARY_CAFFEINE_RULE)
        t4 = a4_summarize(d_main, models, len(skus), bidx)
        sens = [r for r in CAFFEINE_RULES if r != PRIMARY_CAFFEINE_RULE]
        t4_sens = pd.concat(
            [a4_summarize(a4_detail(R, models, skus, emb_order, attrs, rule), models, len(skus), bidx)
             .assign(caffeine_rule=rule) for rule in [PRIMARY_CAFFEINE_RULE] + sens],
            ignore_index=True,
        )
        # Caffeine-rule sensitivity: same scope as the headline A4 result; overall rubric and
        # caffeine-only (the caffeine-only line is where the coding choice matters most).
        t4_sens = t4_sens[(t4_sens.scope == primary_a4_scope)
                          & t4_sens.dimension.isin(["overall", "caffeine"])]
        for _, r in t4[t4.scope == primary_a4_scope].iterrows():
            role = ("primary" if (r.dimension == "overall" and r.model == "all models")
                    else "model-specific" if r.dimension == "overall"
                    else "component diagnostic")
            add("A4 rubric (" + r.dimension + ")", "unforced top-1 violation rate", r.model,
                (r.unforced_top1_violation_rate, r.ci_low, r.ci_high), role=role)
            add("A4 rubric (" + r.dimension + ")", "pairwise violation rate", r.model,
                (r.pairwise_violation_rate, r.pw_ci_low, r.pw_ci_high), role=role)
        per_sku_comp = candidate_compatibility_by_sku(
            skus, emb_order, attrs, PRIMARY_CAFFEINE_RULE
        )
    t5, sku = a5_exploratory(sku, attrs, per_sku_comp)
    headline = pd.DataFrame(H)

    # Chance table for the top-3 overlap figure
    inter = sku[f"top{TOP}_shared_by_all"].value_counts().reindex(range(TOP + 1), fill_value=0)
    t3_inter = pd.DataFrame({"shared_by_all_4": range(TOP + 1), "n_SKUs": inter.values,
                             "share_SKUs": inter.values / len(skus),
                             "chance_share": null["top3_intersection_dist"].values})

    # Write
    agg = aggregated_table(R, sets, models, skus, emb_order, cos)
    agg.to_parquet(os.path.join(ANALYSIS_DIR, "rankings_aggregated.parquet"), index=False)
    sku.to_csv(os.path.join(ANALYSIS_DIR, "sku_agreement.csv"), index=False)
    headline.to_csv(os.path.join(ANALYSIS_DIR, "headline_metrics.csv"), index=False)
    tables = {"t0_integrity": t0, "t1_stability": t1, "t1_stability_pairs": t1_per,
              "t1_stability_by_sku": t1_sku,
              "t2_position_bias_test": t2, "t2_mean_rank_by_slot": t2_desc,
              "t3_pairwise_model_agreement": t3_pairs, "t3_tau_matrix": tau_mat.reset_index(names="model"),
              "t3_top3_overlap_vs_chance": t3_inter, "t3_embedding_baseline": t_emb,
              "chance_references": pd.DataFrame([{k: v for k, v in null.items()
                                                  if k not in ("W", "top3_intersection_dist")}])}
    if t4 is not None:
        tables["t4_rubric_compliance"] = t4
        tables["t4_caffeine_rule_sensitivity"] = t4_sens
        tables["t4_candidate_compatibility_by_sku"] = per_sku_comp
        tables["t4_scope_decision"] = pd.DataFrame([{
            "n_focal_SKUs": len(skus), "n_verified_coffee_sources": n_coffee,
            "n_noncoffee_sources": len(skus) - n_coffee,
            "primary_scope": primary_a4_scope, "decision_note": a4_scope_note,
        }])
    if t5 is not None:
        tables["t5_exploratory_subgroups"] = t5
    for name, df in tables.items():
        df.to_csv(os.path.join(TABLE_DIR, f"{name}.csv"), index=False)
    if not args.no_figures:
        make_figures(t1_sku, t2_desc, tau_mat, sku, t3_inter, models, null)

    # Report
    def fmt_h(df):
        lines = []
        for sec, g in df.groupby("section", sort=False):
            lines.append(sec)
            for _, r in g.iterrows():
                who = SHORT.get(r.model_or_set, r.model_or_set)
                ch = "" if pd.isna(r.chance_reference) else f"   chance {r.chance_reference:.3f}"
                role = f"[{r.reporting_role}]"
                lines.append(f"  {role:<22} {r.metric:<52} {who:<11} {r.estimate:6.3f}  "
                             f"[{r.ci_low:6.3f}, {r.ci_high:6.3f}]{ch}")
        return "\n".join(lines)

    fmt = lambda df: df.to_string(index=False, float_format=lambda v: f"{v:.3f}")  # noqa: E731
    strata_counts = pd.Series(strata).value_counts().to_dict()
    L = ["Stage 07 — ranking analysis report", "=" * 40, "",
         "Integrity", fmt(t0), "",
         f"HEADLINE METRICS — estimate [95% CI], SKU bootstrap ({args.n_boot} replicates, "
         f"stratified by Stage 01 format group {strata_counts})",
         "Bootstrap CIs quantify SKU-level sensitivity within this 60-product pilot; they are not "
         "population-level CIs for the full Amazon coffee market and do not include LLM rerun noise.",
         "Chance references use the same 4 rankings -> Borda pipeline on random rankings "
         f"({args.n_sim} simulated SKUs); for tau and beta the reference is 0.",
         "Borda top-k summaries use the declared embedding-order tie-break; cross-model tie sensitivity is retained in sku_agreement.csv.",
         "Embedding top-1 retention is explicitly labelled fixed-tie-break; t3_embedding_baseline.csv also reports unique-top1 retention and top-boundary ties.", "",
         fmt_h(headline), "",
         f"A2 permutation test (p = (b+1)/(N+1); smallest possible p = {1 / (args.n_perm + 1):.1e})",
         "The pooled all-model test is the primary test. Four model-specific tests are secondary "
         "and use Holm-adjusted p-values.",
         fmt(t2[["model", "inference_role", "beta_rank_per_slot", "implied_rank_gap_slot1_vs_slot7",
                 "p_value_two_sided", "p_value_holm"]]), "",
         f"Kendall's W: 95th percentile under matched random rankings = {null['W_95pct']:.3f} "
         f"(descriptive reference; {int((sku.kendall_w > null['W_95pct']).sum())}/{len(skus)} SKUs above)", ""]
    if t4 is not None:
        s_top1 = t4_sens.pivot_table(index=["dimension", "model"], columns="caffeine_rule",
                                     values="unforced_top1_violation_rate").reset_index()
        s_pair = t4_sens.pivot_table(index=["dimension", "model"], columns="caffeine_rule",
                                     values="pairwise_violation_rate").reset_index()
        L += [f"A4 rubric compliance — primary scope: {primary_a4_scope}",
              f"Scope rule: {a4_scope_note}.",
              "Primary caffeine rule: conflict only when both products state caffeine.",
              fmt(t4[t4.scope == primary_a4_scope][["dimension", "model", "rankings_where_violation_possible",
                                                    "unforced_top1_violations", "unforced_top1_violation_rate",
                                                    "pairwise_violation_rate",
                                                    "rankings_with_no_confirmed_compatible_candidate"]]), "",
              f"A4 caffeine-rule sensitivity ({primary_a4_scope}; overall rubric and caffeine-only): unforced top-1 violation rate", fmt(s_top1), "",
              f"A4 caffeine-rule sensitivity ({primary_a4_scope}; overall rubric and caffeine-only): pairwise violation rate", fmt(s_pair), "",
              "Candidate-set compatibility counts are diagnostic only; n_compatible does NOT estimate substitution incidence or walk-away probability.",
              "A5 exploratory subgroup summaries (descriptive only; small cells)", fmt(t5), ""]
    else:
        L += ["A4/A5 skipped: outputs/attributes/attributes_final.parquet not found "
              "(run 06_attribute_extract.py --finalize first).", ""]
    L += ["Diagnostics (CSV only): sku_agreement.csv holds exact / ordered / all-16-run top-3 agreement,",
          "shared- and distinct-candidate counts, tie-break ranges, attributes, and candidate-set compatibility counts.",
          f"Wrote {ANALYSIS_DIR}: headline_metrics.csv, rankings_aggregated.parquet ({len(agg)} rows), "
          f"sku_agreement.csv, tables/ ({len(tables)} CSVs)" + ("" if args.no_figures else ", figures/ (5 PNGs)")]
    report = "\n".join(L)
    Path(os.path.join(ANALYSIS_DIR, "analysis_report.txt")).write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()