import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import (SHAPE, SWITCH_HELP, best_shelf, coverage_groups, flows, load, popularity_rank,
                    retained, weights)

products, subs, demand, _ = load()
asins = list(products["asin"])
W = weights(subs, asins)
groups = coverage_groups(products)
name = products["short"]

st.header("🛒 Plan the shelf")
st.info(
    "**What you're looking at.** Two ways to fill the same shelf:\n"
    "- **Bestseller shelf**: stock the most popular coffees. This is what you'd do if you assumed "
    "customers never switch.\n"
    "- **Smart shelf**: the optimiser picks the coffees that keep the most customers once switching "
    "is counted, using the AI substitute estimates from the previous page.\n\n"
    "Both shelves must carry at least one coffee of each main type (ground, whole beans, pods, "
    "instant, ready to drink), and both are scored in the same way. Move the controls to change the "
    "situation."
)

c1, c2, c3 = st.columns(3)
C = c1.slider("Shelf space (how many of the 60 coffees fit)", 15, 45, 30, step=5)
switch = c2.slider("Customers willing to switch (out of 10)", 3, 8, 6, help=SWITCH_HELP)
shape = c3.selectbox("How sales are spread", ["moderate", "strong"], format_func=SHAPE.get,
                     help="Equal sales are not offered here: if every coffee sold the same, there would "
                          "be no bestsellers, so the bestseller shelf would be an arbitrary pick.")
alpha = switch / 10
dm = demand[demand["demand_shape"] == shape]
draw = dm["draw"].min()                                          # one demand scenario for the demo
d = dm[dm["draw"] == draw].set_index("asin").loc[asins, "demand"].to_numpy()
rank = popularity_rank(d)

smart = best_shelf(d, W, alpha, C, groups)
bestsellers = best_shelf(d, W, 0.0, C, groups)                   # ignores switching
r_smart, r_best = retained(smart, d, W, alpha), retained(bestsellers, d, W, alpha)

m1, m2, m3 = st.columns(3)
m1.metric("Bestseller shelf keeps", f"{r_best / d.sum():.1%}", help="Share of all customers who leave with a coffee")
m2.metric("Smart shelf keeps", f"{r_smart / d.sum():.1%}",
          delta=f"{(r_smart - r_best) / r_best:+.1%} more customers")
m3.metric("Coffees that differ between the shelves", int(((smart == 1) & (bestsellers == 0)).sum()))
st.caption("“Keeps” = the share of all customers who leave with a coffee, either their usual one or a "
           f"substitute. Out of every 100 customers, the smart shelf keeps about "
           f"{100 * (r_smart - r_best) / d.sum():.0f} more.")

# Where every customer ends up, bestseller shelf vs smart shelf
OWN, SUB, LOST = "#1baf7a", "#3987e5", "#d95926"   # validated palette; values also printed on bars


def outcome(x):
    """Share of ALL customers who buy their usual coffee / a substitute / nothing."""
    Q, walk = flows(x, W, alpha)
    total = d.sum()
    return 100 * (d @ x) / total, 100 * (d @ Q.sum(axis=1)) / total, 100 * (d @ walk) / total


rows = ["Bestseller shelf", "Smart shelf"]
own, sub, lost = zip(outcome(bestsellers), outcome(smart))
st.subheader("Where every customer ends up")
st.markdown(f"The smart shelf has fewer customers finding their usual coffee ({own[1]:.0f}% vs "
            f"{own[0]:.0f}%), but it stocks the coffees people switch to. So only **{lost[1]:.0f}%** "
            f"walk away instead of **{lost[0]:.0f}%**.")
fig = go.Figure()
for vals, label, colour in [(own, "Bought their usual coffee", OWN),
                            (sub, "Bought a substitute", SUB),
                            (lost, "Walked away", LOST)]:
    fig.add_bar(y=rows, x=vals, name=label, orientation="h", marker_color=colour,
                text=[f"{v:.0f}%" for v in vals], textposition="inside", insidetextanchor="middle",
                textfont=dict(color="white", size=15),
                hovertemplate=f"{label}: %{{x:.1f}}%<extra></extra>")
fig.update_layout(barmode="stack", bargap=0.45, height=230, margin=dict(l=10, r=10, t=40, b=10),
                  xaxis=dict(range=[0, 100], ticksuffix="%", showgrid=False),
                  yaxis=dict(autorange="reversed"),
                  legend=dict(orientation="h", y=1.25, x=0, traceorder="normal"))
st.plotly_chart(fig, width="stretch")

# What the smart shelf does differently
st.subheader("What the smart shelf does differently")
st.markdown("It leaves out a few bestsellers whose customers would mostly switch to something else "
            "on the shelf, and uses the space for coffees that either **catch those switchers** or "
            "**have no replacement** (if they are missing, all their customers leave).")

Q, walk = flows(smart, W, alpha)
caught = (d[:, None] * Q).sum(axis=0)                            # switchers each stocked coffee catches

dropped = np.where((bestsellers == 1) & (smart == 0))[0]
added = np.where((smart == 1) & (bestsellers == 0))[0]
left, right = st.container(), st.container()                     # stacked: full width keeps columns readable
with left:
    st.markdown("**Taken off: bestsellers the smart shelf leaves out**")
    st.dataframe(pd.DataFrame({
        "Coffee": [name[i] for i in dropped],
        "Sales rank": [f"#{rank[i]}" for i in dropped],
        "Its customers who still buy": [f"{100 * Q[i].sum():.0f}%" for i in dropped],
        "Most of them switch to": [name[Q[i].argmax()] if Q[i].sum() > 0 else "nothing (they leave)"
                                   for i in dropped],
    }).sort_values("Sales rank", key=lambda s: s.str[1:].astype(int)),
        hide_index=True, width="stretch")
with right:
    st.markdown("**Put on instead**")

    def reason(j):
        why = []
        n_src = int(((smart == 0) & (Q[:, j] > 0)).sum())
        if caught[j] > 0.005:
            why.append(f"catches customers from {n_src} missing coffee{'s' if n_src != 1 else ''}")
        if not (W[j] > 0).any():
            why.append("nothing can replace it")
        return "; ".join(why).capitalize() or "Its customers have few good alternatives"

    st.dataframe(pd.DataFrame({
        "Coffee": [name[j] for j in added],
        "Sales rank": [f"#{rank[j]}" for j in added],
        "Why it's worth the space": [reason(j) for j in added],
    }).sort_values("Sales rank", key=lambda s: s.str[1:].astype(int)),
        hide_index=True, width="stretch")
st.caption("Sales rank: #1 is the best-selling of the 60 coffees.")

with st.expander("See the full smart shelf"):
    shelf = products.loc[smart == 1, ["short", "type", "price"]].assign(
        rank=rank[smart == 1], also=np.where(bestsellers[smart == 1] == 1, "Yes", "No"))
    st.dataframe(shelf.sort_values("rank").rename(columns={
        "short": "Coffee", "type": "Type", "price": "Price ($)", "rank": "Sales rank",
        "also": "Also on bestseller shelf?"}), hide_index=True, width="stretch")

st.caption("This page uses one simulated sales pattern. The thesis averages over 20 patterns, which "
           "gives about +6% for the default settings.")
