"""
Stage 09: Scenario inputs for the assortment model (Stage 10).

Builds and checks every input the optimiser needs, in one place, so each
assumption can be inspected and described on its own. No optimisation here.

What it produces (outputs/scenarios/)
-------------------------------------
  products.csv               one row per SKU: listing price (Stage 01), verified format
                             (Stage 06), coverage group, and whether the SKU counts toward
                             the "at least one per format group" rule (coffee products with a
                             determined format only; non-coffee and undetermined-format SKUs
                             can be stocked but do not count).
  demand.parquet             baseline demand per SKU for every demand instance, scaled so the
                             average SKU has demand 1 (total 60).
                               equal     all SKUs 1                         (1 instance)
                               moderate  Zipf s=0.5: top 20% of SKUs ~40%   (N_DRAWS instances)
                               strong    Zipf s=1.0: top 20% of SKUs ~66%   (N_DRAWS instances)
                             Popularity levels are assigned to SKUs at random (fixed seed), so
                             results do not depend on which products happen to be popular.
  substitution_inputs.parquet  the 7 substitution conditions taken from Stage 08, one row per
                             (condition, out-of-stock SKU, candidate). `weight` = strength
                             rescaled to sum to 1 over the allowed candidates (0 if not allowed;
                             all 0 for dead ends). With this scaling the outside option for a
                             switching rate alpha is simply (1 - alpha) / alpha for every SKU
                             that has an allowed candidate, so Stage 10 needs nothing else.
  conditions.csv             condition -> Stage 08 variant, with what each one isolates.
  scenario_grid.csv          market scenarios: demand shape x alpha x shelf size (27), plus the
                             alpha = 0 no-substitution references for the CONCENTRATED demand
                             shapes only. With equal demand and alpha = 0 every feasible
                             assortment retains the same demand (no unique optimum), so equal
                             demand is used only to compare substitution sources.
  scenario_inputs_report.txt  checks: coverage groups feasible, demand targets met, weights
                             valid, number of solves Stage 10 will run.

Usage
-----
    python src/pipeline/09_scenario_inputs.py
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR, RANDOM_SEED, SAMPLE_FILE  # noqa: E402

ATTR_FILE = os.path.join(OUTPUTS_DIR, "attributes", "attributes_final.parquet")
STRENGTHS_FILE = os.path.join(OUTPUTS_DIR, "substitution", "strengths.parquet")
OUT_DIR = os.path.join(OUTPUTS_DIR, "scenarios")

EXPECTED_N_SKUS = 60
K = 7

# Experiment parameters
DEMAND_SHAPES = {"equal": 0.0, "moderate": 0.5, "strong": 1.0}   # Zipf exponent s
DEMAND_TARGET_TOP20 = {"equal": 0.20, "moderate": 0.40, "strong": 0.66}
N_DRAWS = 20                                   # random popularity assignments per concentrated shape
ALPHAS = [0.45, 0.60, 0.75]                    # share of displaced shoppers who switch,
                                               # when all allowed candidates are stocked
CAPACITIES = [15, 30, 45]                      # max SKUs carried (of 60)
CENTRAL = {"demand_shape": "moderate", "alpha": 0.60, "capacity": 30}

# verified format (Stage 06) -> coverage group
COVERAGE_GROUP = {"ground": "ground", "whole_bean": "whole_bean", "pod_capsule": "pods",
                  "instant": "instant", "ready_to_drink": "rtd_cold_brew",
                  "concentrate": "rtd_cold_brew"}

# substitution condition -> Stage 08 variant
CONDITIONS = [
    ("pooled_pl",        "main",             "main: 4 LLMs pooled, Plackett-Luce (reference)"),
    ("gemini_pl",        "model_gemini",     "Gemini only, Plackett-Luce"),
    ("claude_pl",        "model_claude",     "Claude only, Plackett-Luce"),
    ("qwen_pl",          "model_qwen",       "Qwen only, Plackett-Luce"),
    ("deepseek_pl",      "model_deepseek",   "DeepSeek only, Plackett-Luce"),
    ("llm_points",       "points_llm",       "pooled LLM rankings, simple points rule"),
    ("embedding_points", "points_embedding", "embedding order, same points rule (no LLM)"),
]


def load_products() -> pd.DataFrame:
    for path in (SAMPLE_FILE, ATTR_FILE):
        if not os.path.exists(path):
            raise SystemExit(f"Missing input: {path}")
    with open(SAMPLE_FILE, encoding="utf-8") as f:
        sample = pd.DataFrame([json.loads(line) for line in f if line.strip()])
    attrs = pd.read_parquet(ATTR_FILE)

    problems = []
    if len(sample) != EXPECTED_N_SKUS or sample["parent_asin"].duplicated().any():
        problems.append(f"Stage 01 sample: {len(sample)} rows / duplicates present")
    missing = sorted(set(sample["parent_asin"]) - set(attrs["parent_asin"]))
    if missing:
        problems.append(f"{len(missing)} sample SKUs missing from attributes: {missing[:5]}")
    sub = attrs[attrs["parent_asin"].isin(sample["parent_asin"])]
    bad_coffee = sub.loc[~sub["is_coffee"].isin([True, False]) | sub["is_coffee"].isna(), "parent_asin"]
    if len(bad_coffee):
        problems.append(f"is_coffee must be True/False; bad values for {sorted(bad_coffee)[:5]}")
    price = pd.to_numeric(sample["price"], errors="coerce")
    if price.isna().any() or (price <= 0).any():
        problems.append("non-numeric or non-positive prices in the Stage 01 sample")
    if problems:
        raise SystemExit("Input validation failed:\n  " + "\n  ".join(problems))

    p = sample[["parent_asin", "title", "price", "stratum"]].rename(
        columns={"parent_asin": "asin", "stratum": "stage01_group"})
    p["price"] = price.values
    p = p.merge(attrs[["parent_asin", "format", "is_coffee"]].rename(columns={"parent_asin": "asin"}),
                on="asin", how="left")
    p["coverage_group"] = p["format"].map(COVERAGE_GROUP)
    p["counts_for_coverage"] = (p["is_coffee"] == True) & p["coverage_group"].notna()  # noqa: E712
    return p.sort_values("asin").reset_index(drop=True)


def build_demand(asins: list[str], rng: np.random.Generator) -> pd.DataFrame:
    n = len(asins)
    rows = []
    for shape, s in DEMAND_SHAPES.items():
        levels = np.arange(1, n + 1, dtype=float) ** -s
        levels = levels / levels.mean()                     # average SKU = 1
        draws = [0] if shape == "equal" else range(1, N_DRAWS + 1)
        for d in draws:
            assigned = levels if shape == "equal" else rng.permutation(levels)
            did = f"{shape}_{d:02d}"
            rows.extend(dict(demand_id=did, demand_shape=shape, draw=d, asin=a, demand=assigned[i])
                        for i, a in enumerate(asins))
    return pd.DataFrame(rows)


def build_substitution(products: pd.DataFrame) -> pd.DataFrame:
    if not os.path.exists(STRENGTHS_FILE):
        raise SystemExit(f"Missing Stage 08 output: {STRENGTHS_FILE}")
    st = pd.read_parquet(STRENGTHS_FILE)
    asins = set(products["asin"])
    parts, problems = [], []
    for cond, variant, _ in CONDITIONS:
        g = st[st["variant"] == variant].copy()
        if len(g) != EXPECTED_N_SKUS * K:
            problems.append(f"{variant}: {len(g)} rows, expected {EXPECTED_N_SKUS * K}")
            continue
        if set(g["oos_asin"]) != asins or not set(g["candidate_asin"]) <= asins:
            problems.append(f"{variant}: products/candidates outside the 60-SKU universe")
        allowed_strength = g["strength"].where(g["allowed"], 0.0)
        tot = allowed_strength.groupby(g["oos_asin"]).transform("sum")
        g["weight"] = np.where(tot > 0, allowed_strength / tot.where(tot > 0, 1.0), 0.0)
        g["condition"] = cond
        parts.append(g[["condition", "oos_asin", "candidate_asin", "embedding_rank", "strength",
                        "compat_status", "allowed", "weight", "dead_end", "structural_dead_end"]])
    if problems:
        raise SystemExit("Stage 08 input problems:\n  " + "\n  ".join(problems))
    return pd.concat(parts, ignore_index=True)


def build_grid() -> pd.DataFrame:
    rows = []
    for shape in DEMAND_SHAPES:
        for alpha in ALPHAS:
            for cap in CAPACITIES:
                rows.append(dict(demand_shape=shape, alpha=alpha, capacity=cap, role="main"))
        if shape != "equal":                               # alpha = 0 reference: concentrated only
            for cap in CAPACITIES:
                rows.append(dict(demand_shape=shape, alpha=0.0, capacity=cap,
                                 role="no_substitution_reference"))
    g = pd.DataFrame(rows)
    g["central"] = ((g.demand_shape == CENTRAL["demand_shape"]) & (g.alpha == CENTRAL["alpha"])
                    & (g.capacity == CENTRAL["capacity"]))
    g.insert(0, "scenario_id", [f"S{i:02d}" for i in range(1, len(g) + 1)])
    return g


def checks_and_report(products, demand, subst, grid) -> None:
    L = ["Stage 09 — scenario inputs", "=" * 60, ""]
    ok = True

    # coverage
    cov = products[products.counts_for_coverage].groupby("coverage_group").size()
    groups = sorted(set(COVERAGE_GROUP.values()))
    L.append("Coverage groups (coffee SKUs with a determined format):")
    for gname in groups:
        n = int(cov.get(gname, 0))
        L.append(f"  {gname:15s} {n:3d}")
        if n == 0:
            ok = False
            L.append("    !! no eligible SKU — coverage rule infeasible")
    excluded = products[~products.counts_for_coverage]
    L.append(f"  not counted    {len(excluded):3d}  (non-coffee or undetermined format; may still be stocked)")
    if min(CAPACITIES) < len(groups):
        ok = False
        L.append(f"  !! smallest shelf size {min(CAPACITIES)} < {len(groups)} coverage groups")

    # demand
    L += ["", f"Demand instances (average SKU = 1; total demand identical in every instance, only its "
              f"concentration differs). Popularity assignment seed: {RANDOM_SEED}"]
    for shape in DEMAND_SHAPES:
        d = demand[demand.demand_shape == shape]
        n_inst = d.demand_id.nunique()
        top = (d.groupby("demand_id")["demand"]
               .apply(lambda x: np.sort(x.to_numpy())[::-1][: len(x) // 5].sum() / x.sum()))
        means = d.groupby("demand_id")["demand"].mean()
        L.append(f"  {shape:9s} {n_inst:3d} instance(s); top-20% share {top.mean():.0%} "
                 f"(target {DEMAND_TARGET_TOP20[shape]:.0%}); max/min demand "
                 f"{d.demand.max() / d.demand.min():.0f}x")
        if not np.allclose(means, 1.0) or abs(top.mean() - DEMAND_TARGET_TOP20[shape]) > 0.015:
            ok = False
            L.append("    !! demand scaling or concentration off target")

    # substitution weights
    L += ["", "Substitution conditions (weights sum to 1 over allowed candidates; 0 for dead ends):"]
    for cond, variant, desc in CONDITIONS:
        g = subst[subst.condition == cond]
        sums = g.groupby("oos_asin")["weight"].sum()
        dead = int((sums == 0).sum())
        valid = np.allclose(sums[sums > 0], 1.0)
        ok &= bool(valid)
        L.append(f"  {cond:17s} <- {variant:17s} dead ends {dead:2d}  {'ok' if valid else '!! weights invalid'}")

    # grid
    n_main = int((grid.role == "main").sum())
    n_ref = int((grid.role != "main").sum())
    n_inst = demand.groupby("demand_shape")["demand_id"].nunique()
    solves_main = sum(n_inst[s] for s in grid.loc[grid.role == "main", "demand_shape"]) * len(CONDITIONS)
    solves_ref = sum(n_inst[s] for s in grid.loc[grid.role != "main", "demand_shape"])
    L += ["", f"Scenario grid: {n_main} main scenarios + {n_ref} alpha = 0 references "
          f"(concentrated demand only)",
          f"  alpha {ALPHAS}, shelf sizes {CAPACITIES}, central scenario {CENTRAL}",
          f"  Stage 10 solves: {solves_main} (main, x{len(CONDITIONS)} conditions) + {solves_ref} "
          f"(alpha = 0) = {solves_main + solves_ref}"]

    L += ["", f"Prices: ${products.price.min():.2f}–${products.price.max():.2f} "
          f"(median ${products.price.median():.2f}); per listing, pack sizes differ — "
          f"revenue is a secondary output only.",
          "", "ALL CHECKS PASSED" if ok else "SOME CHECKS FAILED — see !! lines above"]

    text = "\n".join(L)
    with open(os.path.join(OUT_DIR, "scenario_inputs_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)
    if not ok:
        raise SystemExit(1)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rng = np.random.default_rng(RANDOM_SEED)

    products = load_products()
    demand = build_demand(products["asin"].tolist(), rng)
    subst = build_substitution(products)
    grid = build_grid()

    products.to_csv(os.path.join(OUT_DIR, "products.csv"), index=False)
    demand.to_parquet(os.path.join(OUT_DIR, "demand.parquet"), index=False)
    demand.to_csv(os.path.join(OUT_DIR, "demand.csv"), index=False)
    subst.to_parquet(os.path.join(OUT_DIR, "substitution_inputs.parquet"), index=False)
    subst.to_csv(os.path.join(OUT_DIR, "substitution_inputs.csv"), index=False)
    pd.DataFrame(CONDITIONS, columns=["condition", "stage08_variant", "description"]).to_csv(
        os.path.join(OUT_DIR, "conditions.csv"), index=False)
    grid.to_csv(os.path.join(OUT_DIR, "scenario_grid.csv"), index=False)

    checks_and_report(products, demand, subst, grid)


if __name__ == "__main__":
    main()
