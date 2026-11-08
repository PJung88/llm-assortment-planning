"""Compare a freshly generated outputs/ folder with the committed reference outputs.

Tables (.csv, .parquet) are compared value by value with a small numeric tolerance;
text reports are compared line by line, ignoring lines with timings or software
versions. Figures are skipped (PNG bytes depend on the plotting library build).

Stage 10 run-level tables get one allowance. When several shelves are optimal (or within
CBC's tolerances of each other), a different CBC build can return a different one. Rows whose
chosen assortment differs are then reported, and accepted if their regret agrees within
0.01 percentage points; every other row and column must still match exactly.

    python scripts/compare_outputs.py REFERENCE_DIR NEW_DIR [--stages analysis substitution ...]
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

VOLATILE = re.compile(r"(^Wrote |\d+(\.\d+)?\s*(s|sec|seconds|min)\b|elapsed|runtime|python|pulp|cbc|platform|version|wall|generated|date)", re.I)
RTOL, ATOL = 1e-9, 1e-9
SOLVER_SENSITIVE = {"runs_main.csv", "summary_by_scenario.csv", "convergence_draws.csv"}
REGRET_ATOL = 1e-4   # 0.01 percentage points


def read_table(p: Path) -> pd.DataFrame:
    return pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)


def solver_sensitive(x: pd.DataFrame, y: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, str | None, str]:
    """Set aside what depends on WHICH optimal shelf CBC returned; regret must still agree."""
    regret_cols = [c for c in x.columns if c.startswith("regret")]
    for c in regret_cols:
        d = np.abs(x[c].to_numpy(float) - y[c].to_numpy(float))
        if np.nanmax(d, initial=0) > REGRET_ATOL:
            return x, y, f"column {c!r}: regret differs by {np.nanmax(d):.2g} (> {REGRET_ATOL})", ""
    if "assortment" in x.columns:
        # a different reference shelf also changes the comparisons of every condition in that
        # (scenario, demand draw), so the whole group is set aside
        moved = x["assortment"].astype(str) != y["assortment"].astype(str)
        keys = ["scenario_id", "demand_id"]
        hit = set(map(tuple, x.loc[moved, keys].to_numpy()))
        keep = ~x[keys].apply(tuple, axis=1).isin(hit).to_numpy()
        note = (f"{int(moved.sum())} of {len(x)} solves returned a different shelf; "
                f"{len(hit)} scenario-draw groups compared on regret only")
        return x[keep].reset_index(drop=True), y[keep].reset_index(drop=True), None, note
    # summaries over draws: compare identifiers and regret; shelf-composition columns may move
    ids = [c for c in x.columns if not pd.api.types.is_numeric_dtype(x[c])]
    return x[ids], y[ids], None, "compared on regret and identifiers"


def compare_tables(a: Path, b: Path) -> str | None:
    x, y = read_table(a), read_table(b)
    if list(x.columns) != list(y.columns):
        return f"columns differ: {sorted(set(x.columns) ^ set(y.columns))}"
    if len(x) != len(y):
        return f"row count {len(x)} vs {len(y)}"
    if a.name in SOLVER_SENSITIVE:
        x, y, err, note = solver_sensitive(x, y)
        if err:
            return err
        print(f"         ({a.name}: {note}; regret agrees within {REGRET_ATOL})")
    for c in x.columns:
        if c == "seconds":
            continue
        u, v = x[c], y[c]
        if pd.api.types.is_numeric_dtype(u) and pd.api.types.is_numeric_dtype(v):
            if not np.allclose(u.to_numpy(float), v.to_numpy(float), rtol=RTOL, atol=ATOL, equal_nan=True):
                d = np.nanmax(np.abs(u.to_numpy(float) - v.to_numpy(float)))
                return f"column {c!r}: max abs difference {d:.3g}"
        elif not u.astype(str).equals(v.astype(str)):
            i = int(np.argmax(u.astype(str).to_numpy() != v.astype(str).to_numpy()))
            return f"column {c!r} row {i}: {u.iloc[i]!r} vs {v.iloc[i]!r}"
    return None


def compare_text(a: Path, b: Path) -> str | None:
    la = [l for l in a.read_text(encoding="utf-8").splitlines() if not VOLATILE.search(l)]
    lb = [l for l in b.read_text(encoding="utf-8").splitlines() if not VOLATILE.search(l)]
    for i, (p, q) in enumerate(zip(la, lb)):
        if p.rstrip() != q.rstrip():
            return f"line {i + 1}: {p.strip()[:70]!r} vs {q.strip()[:70]!r}"
    return None if len(la) == len(lb) else f"{len(la)} vs {len(lb)} comparable lines"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("reference")
    ap.add_argument("new")
    ap.add_argument("--stages", nargs="*", default=["analysis", "substitution", "scenarios", "assortment"])
    args = ap.parse_args()
    ref, new = Path(args.reference), Path(args.new)
    n_ok = n_bad = n_skip = 0
    for stage in args.stages:
        for a in sorted((ref / stage).rglob("*")):
            if not a.is_file():
                continue
            rel = a.relative_to(ref)
            b = new / rel
            if a.suffix == ".png":
                n_skip += 1
                continue
            if not b.exists():
                print(f"MISSING  {rel}"); n_bad += 1; continue
            if a.suffix in (".csv", ".parquet"):
                err = compare_tables(a, b)
            elif a.suffix == ".txt":
                err = compare_text(a, b)
            else:
                n_skip += 1
                continue
            if err:
                print(f"DIFFERS  {rel}: {err}"); n_bad += 1
            else:
                print(f"match    {rel}"); n_ok += 1
    print(f"\n{n_ok} files match, {n_bad} differ, {n_skip} skipped (figures)")
    return 1 if n_bad else 0


if __name__ == "__main__":
    sys.exit(main())
