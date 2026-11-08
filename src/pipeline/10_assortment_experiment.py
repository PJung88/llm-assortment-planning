"""
Stage 10: Substitution-aware assortment experiment.

Question (RQ5): does the source of substitution information (none / embedding /
LLM, and which LLM) change which products a retailer should carry, and how many
customers are retained?

Model (one solve)
-----------------
Decide x_j in {0,1} for each of the 60 SKUs (1 = carried):
  - carry at most C SKUs;
  - carry at least one coffee SKU from each verified format group (coverage).
Customers of SKU i:
  - if x_i = 1 they buy i;
  - if x_i = 0 they choose among i's OWN stocked allowed candidates with weights w_ij,
    or walk away with weight v0 = (1 - alpha) / alpha:
        P(i -> j) = w_ij / (v0 + S_i),   S_i = sum_j w_ij x_j   (S_i is never rescaled);
  - one round of switching; top-7 candidates only; S_i = 0 -> all lost.
Objective: maximise expected retained demand
        sum_i d_i [ x_i + (1 - x_i) S_i / (v0 + S_i) ].
Listing-price revenue is computed afterwards on the same flows (secondary, never optimised).

Linear (MILP) form, solved exactly with CBC (alpha > 0):
        max  sum_i d_i (x_i + sum_j q_ij)
        s.t. q_i0 + sum_j q_ij = 1 - x_i         (displaced shares add up)
             q_ij <= x_j                         (only stocked candidates)
             q_ij <= (w_ij / v0) q_i0            (choice ratio)
             sum_j x_j <= C,  coverage,  0 <= q <= 1
q_ij exists only for allowed arcs (w_ij > 0). Three valid inequalities tighten the LP
relaxation for speed without changing any optimum: q_ij <= c_ij x_j and q_ij <= c_ij (1 - x_i)
with c_ij = w_ij / (v0 + w_ij) (the largest share j can ever receive), and
sum_j q_ij <= S_max / (v0 + S_max) (1 - x_i). Brute force and per-solve parity checks verify this. Because every q_ij earns the same reward,
the ratio constraint binds at the optimum and reproduces S/(v0+S) exactly (no big-M).
alpha = 0 uses a separate branch with no q variables: max sum_i d_i x_i.

Weights
-------
MAIN: Stage 09 `weight` (Stage 08 strengths rescaled to sum to 1 over allowed
candidates): every source has the same switching rate alpha at full availability, so
differences between sources reflect only WHERE customers go.
APPENDIX: no-rescaling (raw strength x allowed) and other Stage 08 variants.

Must-pass checks (the script exits non-zero if any fails; all are listed in the report)
-------------------------------------------------------------------------------------
  hand-calculated evaluator tests; 20 random brute-force instances = MILP;
  input integrity (A, B); per solve: proven optimal, binary x, q >= 0, exactly C SKUs,
  coverage, MILP objective = evaluator, MILP flows q_ij / q_i0 = evaluator choice shares;
  completeness (2,583 + 120 solves, no duplicates or gaps); invariants: regret >= 0,
  pooled regret = 0, optimum non-decreasing in C, fixed assortment non-decreasing in
  alpha, value of substitution >= 0, cross-grid diagonal = 0 and off-diagonal >= 0.

Pooled PL is the reference specification, not behavioural ground truth.

Inputs : outputs/scenarios/* (Stage 09); outputs/substitution/strengths.parquet and
         bootstrap_strengths.parquet (Stage 08).
Outputs: outputs/assortment/  (CSV tables, figures/, assortment_report.txt)

Usage
-----
    python src/pipeline/10_assortment_experiment.py --validate-only   # tests + input checks
    python src/pipeline/10_assortment_experiment.py --quick           # + central scenario (timing)
    python src/pipeline/10_assortment_experiment.py                   # full experiment (~1 h)
"""

from __future__ import annotations

import argparse
import itertools
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pulp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import OUTPUTS_DIR, RANDOM_SEED  # noqa: E402

# Validated with PuLP 3.x (CBC 2.10). PuLP 4 changed how variables are created
# (prob.add_variable), so stop early with a clear message instead of a TypeError.
if int(pulp.__version__.split(".")[0]) >= 4:
    raise SystemExit(f"PuLP {pulp.__version__} found; this script is validated with PuLP 3.x. "
                     'Install it with:  pip install "pulp>=2.7,<4"')

SCEN_DIR = os.path.join(OUTPUTS_DIR, "scenarios")
SUBST_DIR = os.path.join(OUTPUTS_DIR, "substitution")
OUT_DIR = os.path.join(OUTPUTS_DIR, "assortment")
FIG_DIR = os.path.join(OUT_DIR, "figures")

REFERENCE = "pooled_pl"
EXPECTED_CONDITIONS = ["pooled_pl", "gemini_pl", "claude_pl", "qwen_pl", "deepseek_pl",
                       "llm_points", "embedding_points"]
EXPECTED_N_SKUS = 60
EXPECTED_ALPHAS = {0.45, 0.60, 0.75}
EXPECTED_CAPACITIES = {15, 30, 45}
EXPECTED_SHAPES = {"equal": 1, "moderate": 20, "strong": 20}       # demand instances per shape
EXPECTED_STRUCTURAL_DEAD_ENDS = 6
CENTRAL = {"demand_shape": "moderate", "alpha": 0.60, "capacity": 30}
NO_COVERAGE_CAPACITIES = [15, 30, 45]
APPENDIX_VARIANTS = ["smooth_light", "smooth_heavy", "no_rules", "caffeine_unstated", "dead_end_keep"]
N_VALIDATION = 20
TOL = 1e-6          # relative tolerance for objective parity (total demand is 60)
FLOW_TOL = 1e-6     # absolute tolerance for per-arc flow shares (shares lie in [0, 1])


# Check registry: every must-pass check is recorded and fails loudly
class CheckFailed(RuntimeError):
    pass


CHECKS: list[tuple[str, bool, str]] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))
    if not ok:
        raise CheckFailed(f"CHECK FAILED — {name}: {detail}")


# Core: evaluator and MILP
def choice_shares(x: np.ndarray, W: np.ndarray, alpha: float):
    """Per focal SKU: walk-away share q0[i] and switching shares Q[i, j] of its DISPLACED
    customers (both 0 for carried SKUs). alpha = 0 -> nobody switches."""
    x = x.astype(float)
    n = len(x)
    dropped = 1.0 - x
    if alpha <= 0:
        return dropped.copy(), np.zeros((n, n))
    v0 = (1.0 - alpha) / alpha
    S = W @ x
    denom = v0 + S
    Q = dropped[:, None] * W * x[None, :] / denom[:, None]
    q0 = dropped * v0 / denom
    q0 = np.where((dropped > 0) & (S == 0), 1.0, q0)            # nothing stocked -> all walk away
    return q0, Q


def evaluate(x: np.ndarray, d: np.ndarray, W: np.ndarray, alpha: float,
             price: np.ndarray | None = None) -> dict:
    """Score a fixed assortment x. Independent of the MILP (used to check it)."""
    x = x.astype(float)
    q0, Q = choice_shares(x, W, alpha)
    direct = (d * x).sum()
    flows = d[:, None] * Q                                       # flows[i, j]: i's customers buying j
    switched = flows.sum()
    walked = (d * q0).sum()
    retained = direct + switched
    revenue = np.nan
    if price is not None:
        revenue = (d * x * price).sum() + (flows @ price).sum()  # switchers pay price_j, not price_i
    return {"retained": retained, "retention_rate": retained / d.sum(), "direct": direct,
            "switched": switched, "walked": walked, "revenue_listing": revenue}


def solve(d: np.ndarray, W: np.ndarray, alpha: float, C: int, groups: dict[str, list[int]],
          coverage: bool = True, forced_out: tuple[int, ...] = (), time_limit: int = 300) -> dict:
    n = len(d)
    prob = pulp.LpProblem("assortment", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x_{j}", cat="Binary") for j in range(n)]
    terms = [d[i] * x[i] for i in range(n)]
    q0v, qv = {}, {}
    if alpha > 0:                                                # alpha = 0: no-substitution branch
        v0 = (1.0 - alpha) / alpha
        for i in range(n):
            cands = np.nonzero(W[i] > 0)[0]                      # allowed arcs only
            if len(cands) == 0:
                continue                                         # dead end: dropping i loses all of d_i
            q0v[i] = pulp.LpVariable(f"q0_{i}", 0, 1)
            for j in cands:
                qv[(i, j)] = pulp.LpVariable(f"q_{i}_{j}", 0, 1)
            prob += q0v[i] + pulp.lpSum(qv[(i, j)] for j in cands) == 1 - x[i]
            # valid inequality (speed only): total switching <= S_max / (v0 + S_max) <= alpha
            s_max = W[i].sum()
            prob += pulp.lpSum(qv[(i, j)] for j in cands) <= (s_max / (v0 + s_max)) * (1 - x[i])
            for j in cands:
                c_ij = W[i, j] / (v0 + W[i, j])                  # largest share j can ever get
                prob += qv[(i, j)] <= c_ij * x[j]                # only stocked candidates (tightened)
                prob += qv[(i, j)] <= c_ij * (1 - x[i])          # valid inequality (speed only)
                prob += qv[(i, j)] <= (W[i, j] / v0) * q0v[i]    # choice ratio
                terms.append(d[i] * qv[(i, j)])
    prob += pulp.lpSum(terms)                                    # retained demand only; no revenue
    prob += pulp.lpSum(x) <= C
    if coverage:
        for g, idx in groups.items():
            prob += pulp.lpSum(x[j] for j in idx) >= 1, f"cover_{g}"
    for j in forced_out:
        prob += x[j] == 0
    t0 = time.perf_counter()
    prob.solve(pulp.PULP_CBC_CMD(msg=False, timeLimit=time_limit, gapRel=0))
    secs = time.perf_counter() - t0
    xr = np.array([v.value() if v.value() is not None else np.nan for v in x], dtype=float)
    return {"x_raw": xr, "x": np.rint(np.nan_to_num(xr)).astype(int),
            "q0": {i: v.value() for i, v in q0v.items()}, "q": {k: v.value() for k, v in qv.items()},
            "objective": pulp.value(prob.objective), "status": pulp.LpStatus[prob.status],
            "proven_optimal": (prob.status == pulp.LpStatusOptimal
                               and prob.sol_status == pulp.LpSolutionOptimal), "seconds": secs}


def verify_solution(r: dict, d, W, alpha, C, groups, coverage=True, forced_out=()) -> None:
    """Per-solve gates. Raises CheckFailed on any violation (no silent continuation)."""
    x = r["x"]
    if not r["proven_optimal"]:
        raise CheckFailed(f"solve not proven optimal after {r['seconds']:.0f} s (CBC stopped early, "
                          f"e.g. on the time limit); rerun with a larger --time-limit")
    if np.isnan(r["x_raw"]).any() or np.abs(r["x_raw"] - x).max() > 1e-6:
        raise CheckFailed("x not binary within tolerance")
    expected_size = min(C, len(d) - len(forced_out))
    if x.sum() != expected_size:                                 # demand > 0: adding a SKU always helps
        raise CheckFailed(f"selected {x.sum()} SKUs, expected exactly {expected_size}")
    if coverage and any(x[idx].sum() < 1 for idx in groups.values()):
        raise CheckFailed("coverage violated")
    if any(x[j] for j in forced_out):
        raise CheckFailed("forced-out SKU selected")
    ev = evaluate(x, d, W, alpha)
    if abs(ev["retained"] - r["objective"]) > TOL * max(1.0, abs(ev["retained"])):
        raise CheckFailed(f"MILP objective {r['objective']:.9f} != evaluator {ev['retained']:.9f}")
    if alpha > 0:                                                # flow-level parity, not just totals
        q0_e, Q_e = choice_shares(x, W, alpha)
        for (i, j), v in r["q"].items():
            if v < -FLOW_TOL or abs(v - Q_e[i, j]) > FLOW_TOL:
                raise CheckFailed(f"flow q[{i},{j}] = {v:.8f} != evaluator {Q_e[i, j]:.8f}")
        for i, v in r["q0"].items():
            if v < -FLOW_TOL or abs(v - q0_e[i]) > FLOW_TOL:
                raise CheckFailed(f"walk-away q0[{i}] = {v:.8f} != evaluator {q0_e[i]:.8f}")
        milp_walk = sum(d[i] * v for i, v in r["q0"].items()) + \
            sum(d[i] for i in range(len(d)) if i not in r["q0"] and not x[i])
        if abs(milp_walk - ev["walked"]) > TOL * max(1.0, ev["walked"]):
            raise CheckFailed("MILP walk-away total != evaluator walk-away")


def checked_solve(d, W, alpha, C, groups, coverage=True, forced_out=(), time_limit=300) -> dict:
    r = solve(d, W, alpha, C, groups, coverage=coverage, forced_out=forced_out, time_limit=time_limit)
    verify_solution(r, d, W, alpha, C, groups, coverage=coverage, forced_out=forced_out)
    return r


# Tests before any real data: hand calculations and random brute force
def hand_tests() -> None:
    """Evaluator vs hand calculations."""
    def close(a, b):
        return abs(a - b) < 1e-12

    d = np.array([1.0, 1.0, 1.0])
    for a in (0.45, 0.60, 0.75):
        # focal 0 with one candidate (1) of weight 1
        W = np.zeros((3, 3)); W[0, 1] = 1.0
        r = evaluate(np.array([0, 1, 1]), d, W, a)
        check(f"hand: one candidate stocked, alpha={a}: switch = alpha",
              close(r["switched"], a) and close(r["walked"], 1 - a), f"{r}")
        r = evaluate(np.array([0, 0, 1]), d, W, a)
        check(f"hand: one candidate dropped, alpha={a}: switch 0, walk 1",
              close(r["switched"], 0) and close(r["walked"], 2.0), f"{r}")   # focal 0 and SKU 1 both lost
        # focal 0 with candidates 1 (0.7) and 2 (0.3)
        W = np.zeros((3, 3)); W[0, 1], W[0, 2] = 0.7, 0.3
        full = evaluate(np.array([0, 1, 1]), d, W, a)
        check(f"hand: 0.7/0.3 all stocked, alpha={a}: switch = alpha", close(full["switched"], a))
        part = evaluate(np.array([0, 0, 1]), d, W, a)
        v0 = (1 - a) / a
        expect = 0.3 / (v0 + 0.3)
        check(f"hand: 0.7 dropped, alpha={a}: switch = 0.3/(v0+0.3), walk rises",
              close(part["switched"], expect) and part["switched"] < full["switched"]
              and (part["walked"] - 1.0) > (1 - a), f"{part}")
        # dead end: no candidates
        W = np.zeros((3, 3))
        r = evaluate(np.array([0, 1, 1]), d, W, a)
        check(f"hand: dead end dropped, alpha={a}: all lost", close(r["walked"], 1.0) and close(r["switched"], 0))
        # carried focal SKU unaffected by its candidates
        W = np.zeros((3, 3)); W[0, 1] = 1.0
        r1, r2 = evaluate(np.array([1, 1, 0]), d, W, a), evaluate(np.array([1, 0, 0]), d, W, a)
        check(f"hand: carried SKU keeps all its demand, alpha={a}",
              close(r1["direct"], 2.0) and close(r2["direct"], 1.0) and close(r2["switched"], 0.0))
    # accounting identity and full assortment on random data
    rng = np.random.default_rng(RANDOM_SEED)
    for _ in range(50):
        n = 8
        W = rng.uniform(0, 1, (n, n)) * (rng.uniform(0, 1, (n, n)) < 0.4)
        np.fill_diagonal(W, 0)
        d = rng.uniform(0.1, 3, n)
        x = rng.integers(0, 2, n)
        a = float(rng.choice([0.0, 0.45, 0.6, 0.75]))
        r = evaluate(x, d, W, a)
        ok = (close(r["direct"] + r["switched"] + r["walked"], d.sum()) and r["walked"] >= -1e-12
              and -1e-12 <= r["retained"] <= d.sum() + 1e-12)
        if not ok:
            check("hand: direct + switched + walk-away = total demand", False, f"{r}")
        full = evaluate(np.ones(n, dtype=int), d, W, a)
        if not close(full["retention_rate"], 1.0):
            check("hand: all SKUs carried -> 100% retention", False, f"{full}")
        # adding a stocked candidate never lowers retention; never raises walk-away
        j = int(np.argmin(x)) if (x == 0).any() else None
        if j is not None and a > 0:
            x2 = x.copy(); x2[j] = 1
            r2 = evaluate(x2, d, W, a)
            if r2["retained"] < r["retained"] - 1e-12:
                check("hand: adding a SKU never lowers retention", False, "")
    check("hand: direct + switched + walk-away = total; 100% when all carried; monotone in assortment", True)


def validate(n_instances: int, seed: int) -> pd.DataFrame:
    """Random tiny instances: brute force over every feasible assortment vs MILP."""
    rng = np.random.default_rng(seed)
    alphas = [0.0, 0.3, 0.45, 0.6, 0.75, 0.9]
    rows = []
    for k in range(n_instances):
        n = int(rng.integers(6, 11))
        n_groups = int(rng.integers(2, 4))
        member = rng.integers(-1, n_groups, n)                   # -1 = does not count for coverage
        anchors = rng.choice(n, size=n_groups, replace=False)    # distinct reserved SKU per group
        member[anchors] = np.arange(n_groups)
        groups = {f"g{g}": list(np.nonzero(member == g)[0]) for g in range(n_groups)}
        if any(len(v) == 0 for v in groups.values()):
            check("validation generator: every coverage group non-empty", False, f"instance {k}")
        C = int(rng.integers(n_groups, n))                       # n_groups <= C <= n - 1
        alpha = float(alphas[k % len(alphas)])
        d = rng.uniform(0.2, 3.0, n)
        W = np.zeros((n, n))
        rescale = bool(k % 2)                                    # test both weight conventions
        for i in range(n):
            m = int(rng.integers(0, 4))                          # 0 -> dead end
            js = rng.choice([j for j in range(n) if j != i], size=m, replace=False)
            W[i, js] = rng.uniform(0.01, 1.0, m)
        if rescale:
            s = W.sum(axis=1, keepdims=True)
            W = np.divide(W, s, out=np.zeros_like(W), where=s > 0)
        else:
            W = W / max(1.0, W.sum(axis=1).max())
        best, n_feasible = -np.inf, 0
        for size in range(0, C + 1):
            for combo in itertools.combinations(range(n), size):
                x = np.zeros(n, dtype=int)
                x[list(combo)] = 1
                if any(x[idx].sum() < 1 for idx in groups.values()):
                    continue
                n_feasible += 1
                best = max(best, evaluate(x, d, W, alpha)["retained"])
        r = checked_solve(d, W, alpha, C, groups)                # also runs all per-solve gates
        ok = abs(r["objective"] - best) <= TOL * max(1, best)
        rows.append(dict(instance=k, n=n, capacity=C, alpha=alpha, groups=n_groups,
                         dead_ends=int((W.sum(axis=1) == 0).sum()), rescaled=rescale,
                         feasible_assortments=n_feasible, brute_force_best=best,
                         milp_objective=r["objective"], match=ok))
    df = pd.DataFrame(rows)
    check(f"brute force = MILP on {n_instances} random instances", df.match.all(),
          f"{(~df.match).sum()} mismatches")
    return df


# Data + input integrity
def load_inputs():
    need = ["products.csv", "demand.parquet", "substitution_inputs.parquet",
            "scenario_grid.csv", "conditions.csv"]
    missing = [f for f in need if not os.path.exists(os.path.join(SCEN_DIR, f))]
    check("Stage 09 outputs present", not missing, f"missing {missing}")

    products = pd.read_csv(os.path.join(SCEN_DIR, "products.csv")).sort_values("asin").reset_index(drop=True)
    asins = products["asin"].tolist()
    n = len(asins)
    check("60 unique SKUs", n == EXPECTED_N_SKUS and products.asin.is_unique, f"{n}")
    check("prices positive", (products.price > 0).all())
    pos = {a: i for i, a in enumerate(asins)}
    groups = {g: products.index[(products.coverage_group == g) & products.counts_for_coverage].tolist()
              for g in sorted(products.loc[products.counts_for_coverage, "coverage_group"].unique())}
    check("5 coverage groups, each with an eligible SKU",
          len(groups) == 5 and all(len(v) > 0 for v in groups.values()), f"{ {g: len(v) for g, v in groups.items()} }")
    check("is_coffee is a real boolean column",
          products.is_coffee.map(lambda v: isinstance(v, (bool, np.bool_))).all(),
          f"dtype {products.is_coffee.dtype}, values {sorted(map(str, products.is_coffee.unique()))}")
    non_coffee = tuple(products.index[products.is_coffee == False].tolist())  # noqa: E712
    check("5 verified non-coffee SKUs", len(non_coffee) == 5, f"{len(non_coffee)}")
    price = products["price"].to_numpy(float)

    dem = pd.read_parquet(os.path.join(SCEN_DIR, "demand.parquet"))
    demand = {}
    for did, g in dem.groupby("demand_id"):
        check(f"demand {did}: same 60 SKUs", set(g.asin) == set(asins) and len(g) == n)
        d = np.zeros(n)
        d[[pos[a] for a in g.asin]] = g.demand.to_numpy()
        check(f"demand {did}: positive, sums to 60", (d > 0).all() and abs(d.sum() - n) < 1e-9, f"{d.sum()}")
        demand[did] = (g.demand_shape.iloc[0], int(g.draw.iloc[0]), d)
    counts = pd.Series([v[0] for v in demand.values()]).value_counts().to_dict()
    check("41 demand instances: 1 equal + 20 moderate + 20 strong", counts == EXPECTED_SHAPES, f"{counts}")

    sub = pd.read_parquet(os.path.join(SCEN_DIR, "substitution_inputs.parquet"))
    conditions = pd.read_csv(os.path.join(SCEN_DIR, "conditions.csv"))["condition"].tolist()
    check("exactly the 7 expected conditions", sorted(conditions) == sorted(EXPECTED_CONDITIONS), f"{conditions}")
    W_main, W_raw, masks, cand_sets = {}, {}, {}, {}
    for c in conditions:
        g = sub[sub.condition == c]
        check(f"{c}: every focal SKU has 7 candidates in the universe, none itself",
              len(g) == n * 7 and g.groupby("oos_asin").size().eq(7).all()
              and not g.duplicated(["oos_asin", "candidate_asin"]).any()
              and set(g.oos_asin) == set(asins) and set(g.candidate_asin) <= set(asins)
              and not (g.oos_asin == g.candidate_asin).any())
        cand_sets[c] = g.groupby("oos_asin").candidate_asin.apply(frozenset)
        W, R, M = np.zeros((n, n)), np.zeros((n, n)), np.zeros((n, n), dtype=bool)
        i, j = g.oos_asin.map(pos).to_numpy(), g.candidate_asin.map(pos).to_numpy()
        W[i, j] = g.weight.to_numpy()
        R[i, j] = np.where(g.allowed, g.strength, 0.0)
        M[i, j] = g.allowed.to_numpy()
        check(f"{c}: weights finite and non-negative",
              np.isfinite(g.weight).all() and (g.weight >= 0).all()
              and np.isfinite(g.strength).all() and (g.strength >= 0).all())
        rows = W.sum(axis=1)
        check(f"{c}: weights sum to 1 (or 0 for dead ends); none on disallowed arcs",
              np.all(np.isclose(rows, 1.0) | (rows == 0)) and not (W[~M] > 0).any())
        n_zero = int((rows == 0).sum())
        check(f"{c}: exactly {EXPECTED_STRUCTURAL_DEAD_ENDS} zero-weight rows (dead ends)",
              n_zero == EXPECTED_STRUCTURAL_DEAD_ENDS, f"{n_zero}")
        W_main[c], W_raw[c], masks[c] = W, R, M
    ref_mask = masks[REFERENCE]
    check("same allowed mask and candidate sets in all 7 conditions",
          all((masks[c] == ref_mask).all() and cand_sets[c].equals(cand_sets[REFERENCE]) for c in conditions))
    se = sub[sub.condition == REFERENCE].groupby("oos_asin").structural_dead_end.first().sum()
    check("exactly 6 structural candidate-set dead ends", se == EXPECTED_STRUCTURAL_DEAD_ENDS, f"{se}")

    grid = pd.read_csv(os.path.join(SCEN_DIR, "scenario_grid.csv"))
    m = grid[grid.role == "main"]
    r0 = grid[grid.role == "no_substitution_reference"]
    check("27 main scenarios = 3 shapes x alpha {0.45,0.60,0.75} x C {15,30,45}",
          len(m) == 27 and set(m.alpha.round(2)) == EXPECTED_ALPHAS and set(m.capacity) == EXPECTED_CAPACITIES
          and set(m.demand_shape) == set(EXPECTED_SHAPES) and not m.duplicated(["demand_shape", "alpha", "capacity"]).any())
    expected_r0 = {(s, c) for s in ("moderate", "strong") for c in EXPECTED_CAPACITIES}
    got_r0 = list(zip(r0.demand_shape, r0.capacity))
    check("alpha = 0 references = exactly {moderate, strong} x C {15,30,45}, no duplicates",
          len(got_r0) == 6 and set(got_r0) == expected_r0 and (r0.alpha == 0).all(), f"{sorted(got_r0)}")
    c = grid[grid.central]
    check("central scenario = moderate, alpha 0.60, C 30",
          len(c) == 1 and c.demand_shape.iloc[0] == CENTRAL["demand_shape"]
          and np.isclose(c.alpha.iloc[0], CENTRAL["alpha"]) and c.capacity.iloc[0] == CENTRAL["capacity"])
    return products, asins, pos, groups, non_coffee, price, demand, conditions, W_main, W_raw, ref_mask, grid


def variant_weights(strengths: pd.DataFrame, variant: str, pos: dict, n: int) -> np.ndarray:
    """Stage 08 variant -> weights rescaled over its own allowed candidates (main convention)."""
    g = strengths[strengths.variant == variant]
    W = np.zeros((n, n))
    W[g.oos_asin.map(pos).to_numpy(), g.candidate_asin.map(pos).to_numpy()] = \
        np.where(g.allowed, g.strength, 0.0)
    s = W.sum(axis=1, keepdims=True)
    return np.divide(W, s, out=np.zeros_like(W), where=s > 0)


def instances(demand: dict, shape: str) -> list[tuple[str, int, np.ndarray]]:
    return sorted([(k, dr, d) for k, (sh, dr, d) in demand.items() if sh == shape], key=lambda t: t[1])


def jaccard(a: np.ndarray, b: np.ndarray) -> float:
    inter, union = (a & b).sum(), (a | b).sum()
    return inter / union if union else 1.0


def names(x: np.ndarray, asins: list[str]) -> str:
    return "|".join(sorted(a for a, v in zip(asins, x) if v))


def to_x(assortment: str, asins: list[str]) -> np.ndarray:
    s = set(assortment.split("|")) if assortment else set()
    return np.array([a in s for a in asins], dtype=int)


# Main grid and alpha = 0 references
def run_main(grid, demand, conditions, W_main, groups, non_coffee, price, asins, time_limit,
             only_central=False):
    rows, central_x = [], {}
    nc_mask = np.zeros(len(asins), dtype=bool)
    nc_mask[list(non_coffee)] = True
    main = grid[grid.role == "main"]
    if only_central:
        main = main[main.central]
    Wref = W_main[REFERENCE]
    for sc in main.itertuples():
        for did, draw, d in instances(demand, sc.demand_shape):
            sols = {}
            for c in conditions:                                 # identical d, alpha, C, rules for all 7
                try:
                    sols[c] = checked_solve(d, W_main[c], sc.alpha, sc.capacity, groups, time_limit=time_limit)
                except CheckFailed as e:
                    raise CheckFailed(f"{e} [scenario {sc.scenario_id}: {sc.demand_shape}, alpha={sc.alpha}, "
                                      f"C={sc.capacity}, demand {did}, condition {c}]") from None
            x_ref = sols[REFERENCE]["x"]
            opt_ref = evaluate(x_ref, d, Wref, sc.alpha)["retained"]
            for c, r in sols.items():
                x = r["x"]
                ref = evaluate(x, d, Wref, sc.alpha, price)
                regret = (opt_ref - ref["retained"]) / opt_ref
                n_diff = int((x.astype(bool) & ~x_ref.astype(bool)).sum())
                rows.append(dict(
                    scenario_id=sc.scenario_id, demand_shape=sc.demand_shape, alpha=sc.alpha,
                    capacity=sc.capacity, central=bool(sc.central), demand_id=did, draw=draw,
                    condition=c, retained_own=r["objective"], retained_ref=ref["retained"],
                    retention_rate_ref=ref["retention_rate"], direct_ref=ref["direct"],
                    switched_ref=ref["switched"], walked_ref=ref["walked"],
                    revenue_listing_ref=ref["revenue_listing"],
                    retained_loss_abs=opt_ref - ref["retained"], regret=regret,
                    n_diff=n_diff,                                # raw SKU replacements vs x_ref
                    effective_n_diff=0 if abs(regret) <= 1e-9 else n_diff,   # 0 when only exact ties
                    jaccard=jaccard(x.astype(bool), x_ref.astype(bool)),
                    tied_optimum=bool(n_diff > 0 and abs(regret) <= 1e-9),
                    non_coffee_selected=int(x[nc_mask].sum()), n_selected=int(x.sum()),
                    solver_status=r["status"], seconds=r["seconds"], assortment=names(x, asins)))
                if sc.central:
                    central_x[(draw, c)] = x
        done = [r for r in rows if r["scenario_id"] == sc.scenario_id]
        print(f"  {sc.scenario_id} {sc.demand_shape:8s} alpha={sc.alpha:.2f} C={sc.capacity:2d}: "
              f"{len(done)} solves, slowest {max(r['seconds'] for r in done):.1f} s", flush=True)
    return pd.DataFrame(rows), central_x


def run_alpha0(grid, demand, W_main, groups, main_df, asins, time_limit):
    rows, n_solves = [], 0
    refs = grid[grid.role == "no_substitution_reference"]
    alphas = sorted(grid.loc[grid.role == "main", "alpha"].unique())
    Wref = W_main[REFERENCE]
    key = main_df[main_df.condition == REFERENCE].set_index(["demand_shape", "alpha", "capacity", "draw"])
    for sc in refs.itertuples():
        for did, draw, d in instances(demand, sc.demand_shape):
            r0 = checked_solve(d, np.zeros_like(Wref), 0.0, sc.capacity, groups, time_limit=time_limit)
            n_solves += 1
            x0 = r0["x"]
            if abs(r0["objective"] - d[x0.astype(bool)].sum()) > TOL * d.sum():
                check("alpha=0 objective = sum of selected demand", False, f"{did} C={sc.capacity}")
            for a in alphas:
                rk = key.loc[(sc.demand_shape, a, sc.capacity, draw)]
                opt = rk["retained_ref"]
                v0 = evaluate(x0, d, Wref, a)["retained"]
                x_ref = to_x(rk["assortment"], asins)
                rows.append(dict(demand_shape=sc.demand_shape, capacity=sc.capacity, alpha=a,
                                 draw=draw, demand_id=did, retained_no_sub_plan=v0,
                                 retained_sub_plan=opt, gain=opt - v0, gain_pct=(opt - v0) / v0,
                                 n_diff=int((x_ref.astype(bool) & ~x0.astype(bool)).sum())))
    return pd.DataFrame(rows), n_solves


# Completeness and invariants
def check_completeness(main_df, alpha0_n, grid, demand, conditions, quick):
    m = grid[grid.role == "main"]
    if quick:
        m = m[m.central]
    n_inst = {s: len(instances(demand, s)) for s in EXPECTED_SHAPES}
    expected = int(sum(n_inst[s] for s in m.demand_shape) * len(conditions))
    check(f"main solves complete ({expected} expected)", len(main_df) == expected, f"{len(main_df)}")
    check("no duplicate (scenario, demand, condition) rows",
          not main_df.duplicated(["scenario_id", "demand_id", "condition"]).any())
    per = main_df.groupby(["scenario_id", "demand_id"]).condition.nunique()
    check("every scenario x demand instance has all 7 conditions", per.eq(len(conditions)).all())
    check("all main solves 'Optimal'", (main_df.solver_status == "Optimal").all())
    if alpha0_n is not None:
        r = grid[grid.role == "no_substitution_reference"]
        exp0 = int(sum(n_inst[s] for s in r.demand_shape))
        check(f"alpha = 0 solves complete ({exp0} expected)", alpha0_n == exp0, f"{alpha0_n}")


def check_invariants(main_df, demand, W_main, asins, quick):
    total = main_df.demand_id.map(lambda k: demand[k][2].sum())
    gap = (main_df.direct_ref + main_df.switched_ref + main_df.walked_ref - total).abs().max()
    check("every production row: direct + switched + walk-away = total demand (60)", gap <= 1e-9, f"{gap:.3e}")
    neg = main_df.regret.min()
    check("regret >= 0 for every assortment judged by pooled", neg >= -1e-9, f"min regret {neg:.3e}")
    pz = main_df[main_df.condition == REFERENCE].regret.abs().max()
    check("pooled regret = 0", pz <= 1e-12, f"{pz:.3e}")
    if not quick:
        g = main_df.sort_values("capacity").groupby(["demand_id", "alpha", "condition"]).retained_own
        mono = g.apply(lambda s: bool((np.diff(s.to_numpy()) >= -1e-9).all())).all()
        check("optimal retention non-decreasing in C (15 <= 30 <= 45)", mono)
        g = main_df.sort_values("alpha").groupby(["demand_id", "capacity", "condition"]).retained_own
        mono_a = g.apply(lambda s: bool((np.diff(s.to_numpy()) >= -1e-9).all())).all()
        check("optimal retention non-decreasing in alpha (0.45 <= 0.60 <= 0.75)", mono_a)
        # fixed assortment, higher alpha never worse (judged by pooled)
        ok = True
        Wref = W_main[REFERENCE]
        for r in main_df[main_df.condition == REFERENCE].itertuples():
            x = to_x(r.assortment, asins)
            d = demand[r.demand_id][2]
            v = [evaluate(x, d, Wref, a)["retained"] for a in sorted(EXPECTED_ALPHAS)]
            ok &= bool(np.all(np.diff(v) >= -1e-12))
        check("fixed assortment: retention non-decreasing in alpha", ok)


# Central-scenario analyses
def cross_grid(central_x, demand, conditions, W_main):
    sh, a = CENTRAL["demand_shape"], CENTRAL["alpha"]
    raw = []
    for did, draw, d in instances(demand, sh):
        opt = {k: evaluate(central_x[(draw, k)], d, W_main[k], a)["retained"] for k in conditions}
        for m in conditions:
            for k in conditions:
                v = evaluate(central_x[(draw, m)], d, W_main[k], a)["retained"]
                raw.append(dict(draw=draw, decided_with=m, judged_by=k, regret=(opt[k] - v) / opt[k]))
    raw = pd.DataFrame(raw)
    diag = raw[raw.decided_with == raw.judged_by].regret.abs().max()
    check("cross grid: diagonal = 0", diag <= 1e-12, f"{diag:.3e}")
    check("cross grid: every off-diagonal regret >= 0", raw.regret.min() >= -1e-9, f"{raw.regret.min():.3e}")
    g = raw.groupby(["decided_with", "judged_by"]).regret.mean().unstack().loc[conditions, conditions]
    g.index.name = "decided_with \\ judged_by"
    return g, raw


def sensitivities(demand, conditions, W_main, W_raw, ref_mask, strengths, pos, groups, non_coffee,
                  central_x, asins, time_limit):
    sh, a, C = CENTRAL["demand_shape"], CENTRAL["alpha"], CENTRAL["capacity"]
    Wref = W_main[REFERENCE]
    n = len(asins)
    out, n_solves = {}, 0

    # coffee-only: the 5 verified non-coffee SKUs forced out
    rows = []
    for did, draw, d in instances(demand, sh):
        x_ref = central_x[(draw, REFERENCE)]
        r = checked_solve(d, Wref, a, C, groups, forced_out=non_coffee, time_limit=time_limit)
        n_solves += 1
        opt = evaluate(x_ref, d, Wref, a)["retained"]
        rows.append(dict(draw=draw, non_coffee_in_main=int(x_ref[list(non_coffee)].sum()),
                         retained_main=opt, retained_coffee_only=r["objective"],
                         loss_pct=(opt - r["objective"]) / opt,
                         n_diff=int((x_ref.astype(bool) & ~r["x"].astype(bool)).sum())))
    out["coffee_only"] = pd.DataFrame(rows)
    check("coffee-only loss >= 0 (forcing SKUs out cannot beat the pooled optimum)",
          (out["coffee_only"].loss_pct >= -1e-9).all(), f"min {out['coffee_only'].loss_pct.min():.3e}")

    # no coverage constraint, all three capacities
    rows = []
    for cap in NO_COVERAGE_CAPACITIES:
        for did, draw, d in instances(demand, sh):
            with_c = checked_solve(d, Wref, a, cap, groups, time_limit=time_limit)
            free = checked_solve(d, Wref, a, cap, groups, coverage=False, time_limit=time_limit)
            n_solves += 2
            uncovered = [g for g, idx in groups.items() if free["x"][idx].sum() == 0]
            rows.append(dict(capacity=cap, draw=draw, binds=bool(uncovered),
                             groups_left_uncovered="|".join(uncovered),
                             retained_with=with_c["objective"], retained_without=free["objective"],
                             gain_without_pct=(free["objective"] - with_c["objective"]) / with_c["objective"]))
    out["no_coverage"] = pd.DataFrame(rows)
    check("no-coverage optimum >= with-coverage optimum", (out["no_coverage"].gain_without_pct >= -1e-9).all())

    # appendix Stage 08 variants (rescaled over their own allowed sets), judged by pooled main
    rows = []
    check("Stage 08 'main' strengths reproduce Stage 09 pooled weights",
          np.allclose(variant_weights(strengths, "main", pos, n), Wref, atol=1e-9))
    for v in APPENDIX_VARIANTS:
        if v == "dead_end_keep":                                 # candidate-set dead ends keep all 7 LLM shares
            g = strengths[strengths.variant == v]
            Wv = np.zeros((n, n))
            Wv[g.oos_asin.map(pos).to_numpy(), g.candidate_asin.map(pos).to_numpy()] = g.share.to_numpy()
        else:
            Wv = variant_weights(strengths, v, pos, n)
        rs = Wv.sum(axis=1)
        if v == "dead_end_keep":
            check("dead_end_keep: every focal SKU's weights sum to 1 (dead ends keep all 7 LLM shares)",
                  np.allclose(rs, 1.0), f"row sums {rs.min():.6f}-{rs.max():.6f}")
        else:
            check(f"{v}: weights sum to 1 (or 0 for that variant's dead ends)",
                  np.all(np.isclose(rs, 1.0) | (rs == 0)) and (Wv >= 0).all())
        for did, draw, d in instances(demand, sh):
            x_ref = central_x[(draw, REFERENCE)]
            opt = evaluate(x_ref, d, Wref, a)["retained"]
            r = checked_solve(d, Wv, a, C, groups, time_limit=time_limit)
            n_solves += 1
            val = evaluate(r["x"], d, Wref, a)["retained"]
            rows.append(dict(variant=v, draw=draw, regret_vs_main=(opt - val) / opt,
                             n_diff=int((r["x"].astype(bool) & ~x_ref.astype(bool)).sum())))
    out["appendix_variants"] = pd.DataFrame(rows)
    check("appendix-variant regret >= 0 under pooled main weights",
          (out["appendix_variants"].regret_vs_main >= -1e-9).all(),
          f"min {out['appendix_variants'].regret_vs_main.min():.3e}")

    # the 7 conditions WITHOUT rescaling (raw LLM strength kept), judged by pooled no-rescaling
    rows = []
    for did, draw, d in instances(demand, sh):
        sols = {c: checked_solve(d, W_raw[c], a, C, groups, time_limit=time_limit)["x"] for c in conditions}
        n_solves += len(conditions)
        opt = evaluate(sols[REFERENCE], d, W_raw[REFERENCE], a)["retained"]
        x_main_ref = central_x[(draw, REFERENCE)]
        for c, x in sols.items():
            val = evaluate(x, d, W_raw[REFERENCE], a)["retained"]
            rows.append(dict(condition=c, draw=draw, regret=(opt - val) / opt,
                             n_diff=int((x.astype(bool) & ~sols[REFERENCE].astype(bool)).sum()),
                             n_diff_vs_rescaled_pooled=int((x.astype(bool) & ~x_main_ref.astype(bool)).sum())
                             if c == REFERENCE else np.nan))
    out["no_rescaling"] = pd.DataFrame(rows)
    check("no-rescaling regret >= 0", (out["no_rescaling"].regret >= -1e-9).all())
    return out, n_solves


def stability(demand, central_x, W_main, ref_mask, boot, pos, n):
    """Pooled-optimal central assortments scored under the 500 resampled strength sets
    (same allowed mask, rescaled to 1; dead ends stay 0). Full distribution kept."""
    sh, a = CENTRAL["demand_shape"], CENTRAL["alpha"]
    reps = sorted(boot.rep.unique())
    check("bootstrap: exactly 500 resampled strength sets", len(reps) == 500, f"{len(reps)}")
    per_rep = boot.groupby("rep").size()
    arcs = ref_mask.shape[0] * 7
    check("bootstrap: every set covers all 60 SKUs x 7 candidates, no duplicates",
          per_rep.eq(arcs).all() and not boot.duplicated(["rep", "oos_asin", "candidate_asin"]).any()
          and boot.oos_asin.isin(pos.keys()).all() and boot.candidate_asin.isin(pos.keys()).all()
          and np.isfinite(boot.strength).all() and (boot.strength >= 0).all())
    rep_pos = {r: k for k, r in enumerate(reps)}                    # rep id -> array position
    Wb = np.zeros((len(reps), n, n))
    Wb[boot.rep.map(rep_pos).to_numpy(), boot.oos_asin.map(pos).to_numpy(),
       boot.candidate_asin.map(pos).to_numpy()] = boot.strength.to_numpy()
    Wb = Wb * ref_mask[None]
    s = Wb.sum(axis=2, keepdims=True)
    Wb = np.divide(Wb, s, out=np.zeros_like(Wb), where=s > 0)
    raw = []
    for did, draw, d in instances(demand, sh):
        x = central_x[(draw, REFERENCE)]
        base = evaluate(x, d, W_main[REFERENCE], a)["retained"]
        for r in range(len(reps)):
            v = evaluate(x, d, Wb[r], a)["retained"]
            raw.append(dict(draw=draw, rep=reps[r], retained=v, ratio_to_main=v / base))
    raw = pd.DataFrame(raw)
    summ = raw.groupby("draw").agg(p2_5=("ratio_to_main", lambda v: np.percentile(v, 2.5)),
                                   median=("ratio_to_main", "median"),
                                   p97_5=("ratio_to_main", lambda v: np.percentile(v, 97.5))).reset_index()
    summ["range_width_pct"] = summ.p97_5 - summ.p2_5
    return summ, raw


# Summaries, figures, report
def summarise(main_df):
    keys = ["condition", "demand_shape", "alpha", "capacity"]
    agg = dict(regret_mean=("regret", "mean"), regret_median=("regret", "median"),
               regret_max=("regret", "max"), loss_abs_mean=("retained_loss_abs", "mean"),
               n_diff_mean=("n_diff", "mean"), effective_n_diff_mean=("effective_n_diff", "mean"),
               jaccard_mean=("jaccard", "mean"),
               retention_rate_mean=("retention_rate_ref", "mean"),
               non_coffee_mean=("non_coffee_selected", "mean"), tied_share=("tied_optimum", "mean"),
               draws=("draw", "nunique"))
    full = main_df.groupby(keys).agg(**agg).reset_index()
    first10 = main_df[main_df.draw <= 10].groupby(keys).agg(regret_mean_10=("regret", "mean"),
                                                           n_diff_mean_10=("n_diff", "mean")).reset_index()
    conv = full.merge(first10, on=keys)
    conv["regret_change_10_vs_20"] = conv.regret_mean - conv.regret_mean_10
    c = main_df[main_df.central]
    central = c.groupby("condition").agg(
        regret_mean=("regret", "mean"), regret_min=("regret", "min"), regret_max=("regret", "max"),
        n_diff_mean=("n_diff", "mean"), n_diff_min=("n_diff", "min"), n_diff_max=("n_diff", "max"),
        jaccard_mean=("jaccard", "mean"), non_coffee_mean=("non_coffee_selected", "mean")).reset_index()
    return full, conv, central


def figures(full, alpha0, grid_df, conditions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    os.makedirs(FIG_DIR, exist_ok=True)
    shapes = ["equal", "moderate", "strong"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), sharey=True)
    sub = full[(full.alpha == CENTRAL["alpha"]) & (full.condition != REFERENCE)]
    for ax, sh in zip(axes, shapes):
        for c in conditions:
            if c == REFERENCE:
                continue
            s = sub[(sub.demand_shape == sh) & (sub.condition == c)].sort_values("capacity")
            ax.plot(s.capacity, 100 * s.regret_mean, marker="o", label=c)
        ax.set_title(f"{sh} demand, α = {CENTRAL['alpha']:.2f}")
        ax.set_xlabel("shelf size C")
        ax.set_xticks(sorted(full.capacity.unique()))
    axes[0].set_ylabel("retention regret vs pooled (%)")
    axes[-1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "f1_regret_vs_capacity.png"), dpi=150)
    plt.close(fig)
    if alpha0 is not None and len(alpha0):
        fig, axes = plt.subplots(1, 2, figsize=(9, 4), sharey=True)
        for ax, sh in zip(axes, ["moderate", "strong"]):
            s = alpha0[alpha0.demand_shape == sh].groupby(["alpha", "capacity"]).gain_pct.mean().reset_index()
            for a, g in s.groupby("alpha"):
                ax.plot(g.capacity, 100 * g.gain_pct, marker="o", label=f"α = {a:.2f}")
            ax.set_title(f"{sh} concentration")
            ax.set_xlabel("shelf size C")
            ax.set_xticks(sorted(s.capacity.unique()))
        axes[0].set_ylabel("customers gained by planning for substitution (%)")
        axes[-1].legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "f2_value_of_substitution.png"), dpi=150)
        plt.close(fig)
    if grid_df is not None:
        fig, ax = plt.subplots(figsize=(7, 6))
        m = 100 * grid_df.to_numpy(float)
        im = ax.imshow(m, cmap="Blues")
        ax.set_xticks(range(len(conditions)), conditions, rotation=45, ha="right")
        ax.set_yticks(range(len(conditions)), conditions)
        ax.set_xlabel("judged by")
        ax.set_ylabel("decided with")
        for i in range(m.shape[0]):
            for j in range(m.shape[1]):
                ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=7)
        fig.colorbar(im, ax=ax, label="retention regret (%)")
        ax.set_title("Cross-evaluation, central scenario")
        fig.tight_layout()
        fig.savefig(os.path.join(FIG_DIR, "f3_cross_grid.png"), dpi=150)
        plt.close(fig)


def versions() -> str:
    try:
        cbc = subprocess.run([pulp.PULP_CBC_CMD().path, "-quit"], capture_output=True, text=True,
                             timeout=20).stdout
        cbc = next((ln.strip() for ln in cbc.splitlines() if ln.strip().startswith("Version")), "unknown")
    except Exception:  # noqa: BLE001
        cbc = "unknown"
    return (f"Python {platform.python_version()}, numpy {np.__version__}, pandas {pd.__version__}, "
            f"PuLP {pulp.__version__}, CBC {cbc}; seed {RANDOM_SEED}")


def report(val, main_df=None, central=None, conv=None, alpha0=None, grid_df=None, sens=None,
           stab=None, n_solves=None, total_secs=0.0, failed: str | None = None):
    n_solves = n_solves or {}
    n_total = sum(n_solves.values())
    L = ["Stage 10 — substitution-aware assortment experiment", "=" * 64, versions(), ""]
    L.append("CHECKS (must pass)")
    for name, ok, detail in CHECKS:
        if name.startswith("demand ") and ok:
            continue                                             # 41 per-instance lines summarised below
        L.append(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  ({detail})" if detail and not ok else ""))
    n_dem = sum(1 for n, ok, _ in CHECKS if n.startswith("demand ") and ok)
    if n_dem:
        L.append(f"  [PASS] per-instance demand checks: {n_dem} checks over 41 instances (same 60 SKUs, positive, sum 60)")
    if main_df is not None:
        L.append(f"  [PASS] per-solve gates on all {n_total} solves: proven optimal, binary x, exactly C SKUs, "
                 "coverage, objective = evaluator, every flow q_ij/q_i0 = evaluator share")
    if failed:
        L += ["", f"RUN STOPPED: {failed}"]
    if val is not None:
        L += ["", f"Validation: {int(val.match.sum())}/{len(val)} random instances, MILP best = brute force "
              f"({int(val.dead_ends.sum())} dead ends in total; alpha incl. 0; both weight conventions)"]
    if main_df is not None:
        parts = ", ".join(f"{k} {v}" for k, v in n_solves.items())
        L.append(f"Solves: {n_total} ({parts}; the 2,703 in the design = main + alpha=0 references only); "
                 f"main-grid median {main_df.seconds.median():.2f} s, max {main_df.seconds.max():.2f} s; "
                 f"total {total_secs / 60:.1f} min")
        L += ["", f"Central scenario ({CENTRAL}), {main_df[main_df.central].draw.nunique()} draws, "
              "judged by pooled PL (reference, not ground truth):",
              f"  {'condition':18s} {'regret mean':>11s} {'[min, max]':>16s} {'products changed':>17s} {'non-coffee':>10s}"]
        for r in central.itertuples():
            L.append(f"  {r.condition:18s} {100 * r.regret_mean:10.2f}% [{100 * r.regret_min:5.2f}, "
                     f"{100 * r.regret_max:5.2f}]% {r.n_diff_mean:9.1f} ({r.n_diff_min}-{r.n_diff_max}) "
                     f"{r.non_coffee_mean:10.1f}")
        tied = main_df[(main_df.condition != REFERENCE)].tied_optimum.mean()
        L.append("  'products changed' = raw SKU replacements vs the pooled assortment; retention regret is the "
                 "primary metric")
        L.append(f"  Non-reference solves differing only by exact ties (effective change 0): {100 * tied:.1f}%")
        L.append(f"  Draws 1-10 vs 1-20: max |change in mean regret| = "
                 f"{100 * conv.regret_change_10_vs_20.abs().max():.3f} percentage points")
    if alpha0 is not None and len(alpha0):
        L += ["", "Value of planning for substitution (concentrated demand; judged by pooled):"]
        s = alpha0.groupby(["demand_shape", "alpha", "capacity"]).gain_pct.mean().reset_index()
        for r in s.itertuples():
            L.append(f"  {r.demand_shape:9s} α={r.alpha:.2f} C={r.capacity:2d}: +{100 * r.gain_pct:.2f}% customers")
    if grid_df is not None:
        L += ["", "Cross-evaluation (central; row = decided with, column = judged by; mean regret %):",
              (100 * grid_df).round(2).to_string()]
    if sens:
        co = sens["coffee_only"]
        L += ["", f"Coffee-only (non-coffee forced out, central): non-coffee chosen in main "
              f"{co.non_coffee_in_main.mean():.1f} per draw; retention loss {100 * co.loss_pct.mean():.2f}% "
              f"(max {100 * co.loss_pct.max():.2f}%)"]
        nc = sens["no_coverage"].groupby("capacity").agg(binds=("binds", "mean"),
                                                         gain=("gain_without_pct", "mean")).reset_index()
        L.append("No coverage constraint (central demand/alpha):")
        for r in nc.itertuples():
            L.append(f"  C={r.capacity:2d}: coverage binds in {100 * r.binds:.0f}% of draws; "
                     f"dropping it gains {100 * r.gain:.3f}% customers")
        av = sens["appendix_variants"].groupby("variant").agg(regret=("regret_vs_main", "mean"),
                                                              n_diff=("n_diff", "mean")).reset_index()
        L.append("Appendix variants (decided with variant, judged by pooled main):")
        for r in av.itertuples():
            L.append(f"  {r.variant:18s} regret {100 * r.regret:.2f}%, products changed {r.n_diff:.1f}")
        nr = sens["no_rescaling"].groupby("condition").agg(regret=("regret", "mean"),
                                                           n_diff=("n_diff", "mean")).reset_index()
        L.append("Without rescaling (all 7 conditions; judged by pooled without rescaling):")
        for r in nr.itertuples():
            L.append(f"  {r.condition:18s} regret {100 * r.regret:.2f}%, products changed {r.n_diff:.1f}")
        pr = sens["no_rescaling"].query("condition == @REFERENCE").n_diff_vs_rescaled_pooled.mean()
        L.append(f"  Pooled assortment with vs without rescaling: {pr:.1f} products differ on average")
    if stab is not None:
        L += ["", f"Stability (central pooled assortment under 500 resampled strength sets): "
              f"median 95% range width {100 * stab.range_width_pct.median():.2f}% of retained demand"]
    L += ["", "Revenue columns are listing-price revenue (not profit); pack sizes differ across listings."]
    text = "\n".join(L)
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, "assortment_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def run(args):
    os.makedirs(OUT_DIR, exist_ok=True)
    state = {"val": None}
    try:
        print("Hand-calculated evaluator tests...")
        hand_tests()
        print("Random brute-force validation...")
        val = validate(N_VALIDATION, RANDOM_SEED)
        state["val"] = val
        val.to_csv(os.path.join(OUT_DIR, "validation_bruteforce.csv"), index=False)
        print("Input integrity...")
        (products, asins, pos, groups, non_coffee, price, demand, conditions,
         W_main, W_raw, ref_mask, grid) = load_inputs()
        if args.validate_only:
            report(val)
            return

        t0 = time.perf_counter()
        print("Main grid..." if not args.quick else "Central scenario only (--quick)...")
        main_df, central_x = run_main(grid, demand, conditions, W_main, groups, non_coffee, price, asins,
                                      args.time_limit, only_central=args.quick)
        main_df.to_csv(os.path.join(OUT_DIR, "runs_main.csv"), index=False)
        n_solves = {"main": len(main_df)}
        check_completeness(main_df, None, grid, demand, conditions, args.quick)
        check_invariants(main_df, demand, W_main, asins, args.quick)
        full, conv, central = summarise(main_df)
        full.to_csv(os.path.join(OUT_DIR, "summary_by_scenario.csv"), index=False)
        conv.to_csv(os.path.join(OUT_DIR, "convergence_draws.csv"), index=False)
        central.to_csv(os.path.join(OUT_DIR, "summary_central.csv"), index=False)
        grid_df, grid_raw = cross_grid(central_x, demand, conditions, W_main)
        grid_df.to_csv(os.path.join(OUT_DIR, "cross_grid_central.csv"))
        grid_raw.to_csv(os.path.join(OUT_DIR, "cross_grid_central_by_draw.csv"), index=False)
        if args.quick:
            report(val, main_df, central, conv, None, grid_df, None, None, n_solves, time.perf_counter() - t0)
            return

        print("alpha = 0 references...")
        alpha0, n0 = run_alpha0(grid, demand, W_main, groups, main_df, asins, args.time_limit)
        alpha0.to_csv(os.path.join(OUT_DIR, "value_of_substitution.csv"), index=False)
        check_completeness(main_df, n0, grid, demand, conditions, quick=False)
        check("value of planning for substitution >= 0", (alpha0.gain >= -1e-9).all(), f"min {alpha0.gain.min():.3e}")
        n_solves["alpha=0 references"] = n0

        print("Sensitivities (central scenario)...")
        strengths = pd.read_parquet(os.path.join(SUBST_DIR, "strengths.parquet"))
        sens, ns = sensitivities(demand, conditions, W_main, W_raw, ref_mask, strengths, pos, groups,
                                 non_coffee, central_x, asins, args.time_limit)
        for k, df in sens.items():
            df.to_csv(os.path.join(OUT_DIR, f"sensitivity_{k}.csv"), index=False)
        n_solves["sensitivities"] = ns

        stab = None
        boot_path = os.path.join(SUBST_DIR, "bootstrap_strengths.parquet")
        if os.path.exists(boot_path):
            print("Stability under resampled strengths...")
            stab, stab_raw = stability(demand, central_x, W_main, ref_mask, pd.read_parquet(boot_path),
                                       pos, len(asins))
            stab.to_csv(os.path.join(OUT_DIR, "stability_central.csv"), index=False)
            stab_raw.to_csv(os.path.join(OUT_DIR, "stability_central_full.csv"), index=False)
        else:
            print("  bootstrap_strengths.parquet not found (re-run Stage 08) — stability skipped")

        if not args.no_figures:
            figures(full, alpha0, grid_df, conditions)
        print()
        report(val, main_df, central, conv, alpha0, grid_df, sens, stab, n_solves, time.perf_counter() - t0)
    except CheckFailed as e:
        report(state["val"], failed=str(e))
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description="Stage 10 assortment experiment")
    ap.add_argument("--validate-only", action="store_true", help="tests + input checks only")
    ap.add_argument("--quick", action="store_true", help="tests + central scenario only (timing)")
    ap.add_argument("--time-limit", type=int, default=300,
                    help="seconds per solve; a solve not proven optimal stops the run")
    ap.add_argument("--no-figures", action="store_true")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
