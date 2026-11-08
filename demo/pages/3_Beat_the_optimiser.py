import numpy as np
import streamlit as st

from engine import best_shelf, coverage_groups, load, popularity_rank, retained, weights

products, subs, demand, _ = load()
asins = list(products["asin"])
W = weights(subs, asins)
C, alpha = 30, 0.6
d = demand[(demand["demand_shape"] == "moderate") & (demand["draw"] == 1)].set_index("asin").loc[asins, "demand"].to_numpy()
rank = popularity_rank(d)
name = products["short"]

st.header("🏆 Can you beat the optimiser?")
st.info(
    f"**The game.** Choose **{C}** of the 60 coffees for the shelf. When a customer's coffee is "
    "missing, 6 in 10 will try another one (using the AI substitute estimates) and the rest leave. "
    "Your score is the share of customers who leave with a coffee. Then see how the optimiser does.\n\n"
    "**Tip:** the 30 bestsellers are a good start, but not the best answer. Use the button below to "
    "start from them, then try swapping a few."
)

st.markdown("**The 60 coffees, best-selling first**")
view = products.assign(rank=rank, popularity=d.round(2))[["rank", "short", "type", "price", "popularity"]]
st.dataframe(view.sort_values("rank").rename(columns={
    "rank": "Sales rank", "short": "Coffee", "type": "Type", "price": "Price ($)",
    "popularity": "Popularity"}), hide_index=True, height=250, width="stretch")
st.caption("Popularity: 1 = an average coffee; 2 = sells twice as much as average; 0.5 = half as much. "
           "Sales are simulated.")

top = [int(i) for i in np.argsort(-d)[:C]]
b1, b2, _ = st.columns([2, 1, 2])
b1.button("Start from the 30 bestsellers", on_click=lambda: st.session_state.update(shelf=top))
b2.button("Clear my shelf", on_click=lambda: st.session_state.update(shelf=[]))

chosen = st.multiselect("Your shelf (type to search)", products.index, key="shelf",
                        format_func=lambda k: f"#{rank[k]} {name[k]}", max_selections=C)
if len(chosen) == C:
    x = np.array([1 if k in chosen else 0 for k in range(len(asins))])
    opt = best_shelf(d, W, alpha, C, coverage_groups(products))
    you, best = retained(x, d, W, alpha), retained(opt, d, W, alpha)
    a, b = st.columns(2)
    a.metric("Your shelf keeps", f"{you / d.sum():.1%}")
    b.metric("Optimiser's shelf keeps", f"{best / d.sum():.1%}",
             delta=f"{(best - you) / d.sum() * 100:+.1f} points vs you")
    if you >= best - 1e-9:
        st.success("You matched the optimiser!")
    else:
        st.warning("The optimiser wins. It keeps coffees that catch switchers or have no replacement, "
                   "not just bestsellers.")
        l, r = st.columns(2)
        l.markdown("**On the optimiser's shelf, not yours**")
        l.write("\n".join(f"- #{rank[k]} {name[k]}" for k in np.where((opt == 1) & (x == 0))[0]))
        r.markdown("**On your shelf, not the optimiser's**")
        r.write("\n".join(f"- #{rank[k]} {name[k]}" for k in np.where((opt == 0) & (x == 1))[0]))
else:
    st.caption(f"{len(chosen)} of {C} chosen. Your score appears when you reach {C}.")
