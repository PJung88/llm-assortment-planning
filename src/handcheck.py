"""Check the Stage 10 assortment model by hand on 3-4 products.

Two independent calculations are compared with the Stage 10 code itself:

1. a plain, loop-by-loop version of the customer model, written to mirror the formula
   you would use with a calculator (exact fractions when the inputs are fractions);
2. Stage 10's own evaluator (`evaluate`) for every possible shelf;
3. Stage 10's own MILP (`solve` + `verify_solution`), which must pick the best shelf.

The customer model (one round of switching), for a product i that is NOT on the shelf:

    v0  = (1 - alpha) / alpha                 weight of walking away
    S_i = sum of w_ij over i's candidates j that ARE on the shelf
    share of i's customers who buy j  = w_ij / (v0 + S_i)
    share of i's customers who leave  = v0   / (v0 + S_i)      (all leave if S_i = 0)

    retained = sum over stocked i of d_i  +  sum over dropped i of d_i * S_i / (v0 + S_i)

Used by scripts/hand_check.py (terminal) and demo/pages/5_Check_by_hand.py (demo).
"""
from __future__ import annotations

import importlib.util
import itertools
from fractions import Fraction
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"


@lru_cache(maxsize=None)
def stage10():
    """Stage 10 itself (the file name starts with a digit, so it is loaded by path)."""
    path = ROOT / "src" / "pipeline" / "10_assortment_experiment.py"
    spec = importlib.util.spec_from_file_location("stage10", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def worked_example() -> dict:
    """Four made-up ground coffees, a shelf for two, half of displaced shoppers willing to switch.

    C is the only decaf, so its customers have no acceptable substitute (a dead end).
    Weights are already rescaled to sum to 1 over each product's allowed candidates,
    exactly as in Stage 09.
    """
    F = Fraction
    names = ["A  Dark roast", "B  Medium roast", "C  Decaf", "D  Light roast"]
    W = [[F(0), F(3, 4), F(0), F(1, 4)],     # A missing: 3/4 lean to B, 1/4 to D
         [F(3, 4), F(0), F(0), F(1, 4)],     # B missing: 3/4 lean to A, 1/4 to D
         [F(0), F(0), F(0), F(0)],           # C missing: no other decaf -> nobody switches
         [F(1, 2), F(1, 2), F(0), F(0)]]     # D missing: half lean to A, half to B
    return {"names": names, "d": [F(4), F(3), F(2), F(1)], "W": W, "alpha": F(1, 2),
            "capacity": 2, "groups": {}}


def from_data(asins: list[str], condition: str = "pooled_pl", demand: list[float] | None = None,
              alpha: float = 0.6, capacity: int = 2, coverage: bool = False) -> dict:
    """A small instance cut out of the real Stage 09 inputs.

    The weights are the real Stage 09 weights between the chosen products. Candidates that
    are not among the chosen products count as not stocked, which is exactly what Stage 10
    does when they are off the shelf.
    """
    products = pd.read_csv(OUT / "scenarios" / "products.csv").set_index("asin")
    subs = pd.read_csv(OUT / "scenarios" / "substitution_inputs.csv")
    subs = subs[(subs["condition"] == condition) & subs["oos_asin"].isin(asins)
                & subs["candidate_asin"].isin(asins)]
    pos = {a: k for k, a in enumerate(asins)}
    W = [[0.0] * len(asins) for _ in asins]
    for r in subs.itertuples():
        W[pos[r.oos_asin]][pos[r.candidate_asin]] = float(r.weight)
    groups = {}
    if coverage:
        p = products.loc[asins]
        for g in sorted(p.loc[p["counts_for_coverage"], "coverage_group"].unique()):
            groups[g] = [pos[a] for a in asins if p.loc[a, "counts_for_coverage"]
                         and p.loc[a, "coverage_group"] == g]
    names = [f"{chr(65 + k)}  {products.loc[a, 'title'][:40]}" for k, a in enumerate(asins)]
    return {"names": names, "d": list(demand or [1.0] * len(asins)), "W": W, "alpha": alpha,
            "capacity": capacity, "groups": groups, "asins": asins}


def by_hand(x: list[int], d, W, alpha) -> list[dict]:
    """The customer model written out one product at a time, as on paper."""
    rows = []
    v0 = (1 - alpha) / alpha if alpha > 0 else None
    for i in range(len(d)):
        if x[i]:
            rows.append({"product": i, "on_shelf": True, "S": None, "switch_share": None,
                         "kept": d[i], "lost": 0 * d[i]})
            continue
        S = sum(W[i][j] for j in range(len(d)) if x[j])
        share = 0 * S if (alpha == 0 or S == 0) else S / (v0 + S)
        rows.append({"product": i, "on_shelf": False, "S": S, "switch_share": share,
                     "kept": d[i] * share, "lost": d[i] * (1 - share)})
    return rows


def feasible_shelves(n: int, capacity: int, groups: dict) -> list[tuple[int, ...]]:
    """Every shelf with exactly `capacity` products that meets the coverage rule."""
    out = []
    for shelf in itertools.combinations(range(n), capacity):
        if all(any(j in shelf for j in idx) for idx in groups.values()):
            out.append(shelf)
    return out


def check(inst: dict) -> dict:
    """Run all three calculations and report whether they agree."""
    s10 = stage10()
    d, W, alpha, C, groups = inst["d"], inst["W"], inst["alpha"], inst["capacity"], inst["groups"]
    n = len(d)
    d_np = np.array([float(v) for v in d])
    W_np = np.array([[float(v) for v in row] for row in W])
    a_np = float(alpha)

    table = []
    for shelf in feasible_shelves(n, C, groups):
        x = [1 if j in shelf else 0 for j in range(n)]
        hand = by_hand(x, d, W, alpha)
        total = sum(r["kept"] for r in hand)
        code = s10.evaluate(np.array(x), d_np, W_np, a_np)["retained"]
        table.append({"shelf": shelf, "x": x, "hand_rows": hand, "hand_total": total,
                      "stage10_evaluate": code, "agree": abs(float(total) - code) < 1e-9})
    best_hand = max(table, key=lambda t: float(t["hand_total"]))

    r = s10.solve(d_np, W_np, a_np, C, groups)
    s10.verify_solution(r, d_np, W_np, a_np, C, groups)     # Stage 10's own per-solve checks
    milp_shelf = tuple(int(j) for j in np.nonzero(r["x"])[0])
    milp_ok = abs(r["objective"] - float(best_hand["hand_total"])) < 1e-6
    return {"table": table, "best_hand": best_hand, "milp_shelf": milp_shelf,
            "milp_objective": r["objective"], "milp_matches_best": milp_ok,
            "all_agree": all(t["agree"] for t in table) and milp_ok}


def fmt(v) -> str:
    """Fractions as 'a/b (decimal)', everything else as a decimal."""
    if isinstance(v, Fraction):
        return str(v) if v.denominator == 1 else f"{v} ({float(v):.4f})"
    return f"{float(v):.4f}"
