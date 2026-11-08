"""Print a Stage 10 calculation you can check with a calculator.

    python scripts/hand_check.py
        Worked example: 4 made-up coffees, shelf for 2, alpha = 1/2. Exact fractions.

    python scripts/hand_check.py --asins B001ELL4RM B0055ZS98K B082P9S82G --alpha 0.6 --capacity 2
        3-4 real products with their real Stage 09 weights (pooled LLM estimate by default).
        Optional: --demand 3 2 1   --condition gemini_pl   --coverage

For every possible shelf it prints the calculation product by product, then checks that
Stage 10's evaluator gives the same totals and that Stage 10's MILP picks the best shelf.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import handcheck as hc  # noqa: E402


def cell(v) -> str:
    """Table cell: '.' for zero, fractions as written, decimals to 4 places."""
    if not v:
        return "."
    return str(v) if isinstance(v, hc.Fraction) else f"{float(v):.4f}".rstrip("0").rstrip(".")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--asins", nargs="+", help="3-4 product IDs (default: the worked example)")
    ap.add_argument("--condition", default="pooled_pl", help="substitution source (Stage 09 condition)")
    ap.add_argument("--demand", nargs="+", type=float, help="demand per product (default: 1 each)")
    ap.add_argument("--alpha", type=float, default=0.6, help="share of displaced shoppers who switch")
    ap.add_argument("--capacity", type=int, default=2, help="shelf size")
    ap.add_argument("--coverage", action="store_true", help="apply the one-per-format-group rule")
    a = ap.parse_args()

    if a.asins:
        if not 2 <= len(a.asins) <= 6:
            ap.error("use 2-6 products so every shelf can be checked by hand")
        inst = hc.from_data(a.asins, a.condition, a.demand, a.alpha, a.capacity, a.coverage)
    else:
        inst = hc.worked_example()
    names, d, W, alpha = inst["names"], inst["d"], inst["W"], inst["alpha"]
    n = len(d)
    v0 = (1 - alpha) / alpha

    print("INPUTS")
    print(f"  alpha = {hc.fmt(alpha)}   ->   v0 = (1 - alpha) / alpha = {hc.fmt(v0)}")
    print(f"  shelf size C = {inst['capacity']}" + (f";  coverage groups {inst['groups']}" if inst["groups"] else ""))
    print("  demand d and weights w_ij (row = product missing, column = where its switchers lean):")
    letters = [nm.split()[0] for nm in names]
    print("      " + "".join(f"{L:>12}" for L in letters) + f"{'d':>10}")
    for i in range(n):
        print(f"  {letters[i]:<4}" +
              "".join(f"{cell(W[i][j]):>12}" for j in range(n)) + f"{cell(d[i]):>10}")
    for nm in names:
        print(f"    {nm}")
    if a.asins:
        print("  Rows can add up to less than 1: the rest of each product's weight belongs to candidates")
        print("  that are not among these products, and Stage 10 treats those as not stocked.")

    res = hc.check(inst)
    print("\nEVERY POSSIBLE SHELF")
    for t in res["table"]:
        print(f"\n  Shelf {{{', '.join(letters[j] for j in t['shelf'])}}}")
        for r in t["hand_rows"]:
            L = letters[r["product"]]
            if r["on_shelf"]:
                print(f"    {L} on shelf: keeps all {hc.fmt(r['kept'])}")
            elif r["S"] == 0:
                print(f"    {L} missing: S = 0 (no acceptable substitute on the shelf) -> all {hc.fmt(d[r['product']])} leave")
            else:
                print(f"    {L} missing: S = {hc.fmt(r['S'])};  switch = S/(v0+S) = {hc.fmt(r['switch_share'])};"
                      f"  kept = {hc.fmt(d[r['product']])} x {hc.fmt(r['switch_share'])} = {hc.fmt(r['kept'])}")
        flag = "same" if t["agree"] else "DIFFERENT"
        print(f"    retained = {hc.fmt(t['hand_total'])}   |  Stage 10 evaluate(): {t['stage10_evaluate']:.6f}  ({flag})")

    best = res["best_hand"]
    print("\nRESULT")
    print(f"  Best shelf by hand:      {{{', '.join(letters[j] for j in best['shelf'])}}}  retained {hc.fmt(best['hand_total'])}")
    print(f"  Stage 10 MILP chose:     {{{', '.join(letters[j] for j in res['milp_shelf'])}}}  objective {res['milp_objective']:.6f}")
    print("  " + ("ALL AGREE: hand calculation = Stage 10 evaluator on every shelf, and the MILP finds the best one."
                  if res["all_agree"] else "MISMATCH - see lines marked DIFFERENT above."))
    return 0 if res["all_agree"] else 1


if __name__ == "__main__":
    sys.exit(main())
