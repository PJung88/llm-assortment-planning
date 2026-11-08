import sys
from fractions import Fraction
from pathlib import Path

import pandas as pd
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import handcheck as hc  # noqa: E402

from engine import load  # noqa: E402

st.header("🧮 Check it by hand")
st.info(
    "**What you're looking at.** The optimiser behind this study (Stage 10 of the thesis) on a shelf "
    "small enough to check with a calculator. Every possible shelf is worked out step by step, then "
    "compared with **the thesis code itself**: its customer model and its optimiser, not a copy. "
    "If the numbers agree here, you have checked the same formulas that produced the results."
)

with st.expander("The formula (one round of switching)", expanded=True):
    st.markdown(
        r"""
For a coffee $i$ that is **not** on the shelf, with demand $d_i$:

- $v_0 = (1-\alpha)/\alpha$ is the weight of walking away ($\alpha$ = share willing to switch).
- $S_i = \sum_j w_{ij}$ over $i$'s substitutes $j$ that **are** on the shelf.
- Of $i$'s customers, a share $S_i / (v_0 + S_i)$ buys a substitute; the rest leave.
  If $S_i = 0$, all of them leave.

**Customers kept** $= \sum_{i \text{ on shelf}} d_i \;+\; \sum_{i \text{ not on shelf}} d_i \, \dfrac{S_i}{v_0 + S_i}$.
The optimiser picks the shelf that makes this as large as possible.
"""
    )

mode = st.radio("Example", ["Worked example: 4 made-up coffees (exact fractions)",
                            "Pick 3-4 real coffees from the study"], horizontal=True)

if mode.startswith("Worked"):
    inst = hc.worked_example()
    st.caption("A, B and D are regular coffees; C is the only decaf, so its customers have no "
               "acceptable substitute. Half of the customers whose coffee is missing are willing "
               "to switch, and there is room for 2 coffees.")
else:
    products, subs, _, _ = load()
    labels = dict(zip(products["asin"], products["short"] + " (" + products["type"] + ")"))
    default = ["B001ELL4RM", "B0055ZS98K", "B082P9S82G", "B009NVDN7Y"]
    asins = st.multiselect("Coffees (3 or 4)", list(labels), default=default, max_selections=4,
                           format_func=labels.get)
    if len(asins) < 3:
        st.warning("Pick at least 3 coffees.")
        st.stop()
    c1, c2, c3 = st.columns(3)
    alpha = c1.slider("Share willing to switch (alpha)", 0.3, 0.9, 0.6, step=0.05)
    capacity = c2.slider("Shelf size", 1, len(asins) - 1, 2)
    source = c3.selectbox("Substitute estimates from", ["pooled_pl", "gemini_pl", "claude_pl", "qwen_pl",
                                                       "deepseek_pl", "llm_points", "embedding_points"],
                          format_func=lambda c: {"pooled_pl": "All 4 AIs pooled (main)",
                                                 "llm_points": "AI rankings, simple points",
                                                 "embedding_points": "Text similarity only"}.get(
                              c, c.replace("_pl", "").capitalize() + " only"))
    st.markdown("**Demand** (how many customers come for each coffee)")
    cols = st.columns(len(asins))
    demand = [cols[k].number_input(chr(65 + k), 0.0, 100.0, float(len(asins) - k), step=1.0)
              for k in range(len(asins))]
    inst = hc.from_data(asins, source, demand, alpha, capacity, coverage=False)
    st.caption("Weights are the study's real substitute estimates between these coffees. A row can add "
               "up to less than 1: the rest belongs to substitutes not in this small list, which count "
               "as not on the shelf.")

names, d, W, alpha = inst["names"], inst["d"], inst["W"], inst["alpha"]
letters = [nm.split()[0] for nm in names]
v0 = (1 - alpha) / alpha


def show(v):
    if isinstance(v, Fraction):
        return str(v) if v.denominator == 1 else f"{v} ≈ {float(v):.4f}"
    return f"{float(v):.4f}".rstrip("0").rstrip(".")


def short(v):
    """Inside a calculation: fractions as written, decimals to 4 places."""
    return str(v) if isinstance(v, Fraction) else f"{float(v):.4f}".rstrip("0").rstrip(".")


st.subheader("Inputs")
st.markdown(f"alpha = **{show(alpha)}**, so v0 = (1 - alpha) / alpha = **{show(v0)}**. "
            f"Shelf size = **{inst['capacity']}**.")
wtab = pd.DataFrame([[show(W[i][j]) if W[i][j] else "" for j in range(len(d))] + [show(d[i])]
                     for i in range(len(d))],
                    index=[f"{nm}" for nm in names], columns=letters + ["demand d"])
st.markdown("Weights w (row = coffee that is missing, column = where its switchers lean):")
st.dataframe(wtab, width="stretch")

res = hc.check(inst)
best = res["best_hand"]

st.subheader("Every possible shelf")
rows = []
for t in res["table"]:
    rows.append({"Shelf": "{" + ", ".join(letters[j] for j in t["shelf"]) + "}",
                 "Customers kept (by hand)": show(t["hand_total"]),
                 "Thesis code (evaluate)": f"{t['stage10_evaluate']:.6f}",
                 "Same?": "✅" if t["agree"] else "❌",
                 "Best": "⭐" if t["shelf"] == best["shelf"] else ""})
st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")

for t in res["table"]:
    title = "Shelf {" + ", ".join(letters[j] for j in t["shelf"]) + "}"
    with st.expander(f"{title}: step by step" + ("  ⭐" if t["shelf"] == best["shelf"] else "")):
        lines = []
        for r in t["hand_rows"]:
            L, di = letters[r["product"]], d[r["product"]]
            if r["on_shelf"]:
                lines.append(f"- **{L}** on the shelf: keeps all **{show(di)}**")
            elif not r["S"]:
                lines.append(f"- **{L}** missing: S = 0, no acceptable substitute on the shelf, "
                             f"so all {show(di)} leave. Keeps **0**")
            else:
                on = [j for j in t["shelf"] if W[r["product"]][j]]
                terms = " + ".join(f"w({L},{letters[j]}) = {short(W[r['product']][j])}" if len(on) == 1
                                   else short(W[r["product"]][j]) for j in on)
                total = "" if len(on) == 1 else f" = {short(r['S'])}"
                lines.append(f"- **{L}** missing. Substitutes on the shelf: {', '.join(letters[j] for j in on)}, "
                             f"so S = {terms}{total}. Switch share = S / (v0 + S) = {short(r['S'])} / "
                             f"{short(v0 + r['S'])} = {short(r['switch_share'])}. "
                             f"Keeps {short(di)} × {short(r['switch_share'])} = **{show(r['kept'])}**")
        lines.append(f"\n**Total kept = {show(t['hand_total'])}**")
        st.markdown("\n".join(lines))

st.subheader("What the thesis optimiser chose")
milp = "{" + ", ".join(letters[j] for j in res["milp_shelf"]) + "}"
a, b = st.columns(2)
a.metric("Best shelf by hand", "{" + ", ".join(letters[j] for j in best["shelf"]) + "}",
         help="Largest 'customers kept' in the table above")
b.metric("Stage 10 optimiser (MILP)", milp, help="The exact integer program from the thesis, solved with CBC")
if res["all_agree"]:
    st.success(f"Everything agrees: the hand calculation equals the thesis code on all "
               f"{len(res['table'])} shelves, and the optimiser picks the best one "
               f"(customers kept {res['milp_objective']:.4f}). The optimiser's result also passed "
               "the thesis code's own checks: optimal, exactly the shelf size, and every customer flow "
               "matching the formula.")
else:
    st.error("Something disagrees. See the rows marked ❌ above.")

if mode.startswith("Worked"):
    st.caption("Why not the two bestsellers {A, B}? B's customers happily switch to A, but C's decaf "
               "customers have nowhere else to go. Stocking C instead of B keeps 160/21 ≈ 7.62 "
               "customers instead of 15/2 = 7.5.")
st.caption("Same check from a terminal: `python scripts/hand_check.py` (worked example) or "
           "`python scripts/hand_check.py --asins ASIN1 ASIN2 ASIN3 --alpha 0.6`.")
