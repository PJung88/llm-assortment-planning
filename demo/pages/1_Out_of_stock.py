import re

import pandas as pd
import streamlit as st

from engine import FORMAT, MODEL, load, load_attributes

products, subs, demand, rankings = load()
attrs = load_attributes()
info = products.set_index("asin")
name = info["short"]

st.header("🤖 Out of stock? Ask the AI")
st.info(
    "**What you're looking at.** Pick a coffee and imagine it has sold out. Below are the 7 coffees "
    "whose descriptions are most similar to it, and what four AI models think its customers would "
    "buy instead. The AIs saw only the product descriptions, never any sales figures."
)

pick = st.selectbox("Pick a coffee that has sold out", products["asin"], format_func=lambda a: name[a],
                    index=int((products["asin"] == "B001ELL4RM").idxmax()))   # a clear example to start with
st.markdown(f"**Sold out:** {info.loc[pick, 'title'][:110]}  \n"
            f"Type: **{info.loc[pick, 'type']}** · Price: **\\${info.loc[pick, 'price']:.2f}**")

CAF = {"decaf": "decaf", "caffeine_free": "caffeine-free", "regular": "regular"}


def why_not(oos, cand):
    """Plain-English reason a substitute is ruled out (from the verified product attributes)."""
    A, B = attrs.loc[oos], attrs.loc[cand]
    if A.format != B.format:
        if B.format == "unresolvable":
            return "❌ No: type not stated"
        if A.format in ("unresolvable", "other"):
            return "❌ No: different kind of product"
        return f"❌ No: {FORMAT.get(B.format, B.format).lower()}, not {FORMAT.get(A.format, A.format).lower()}"
    if A.format == "pod_capsule" and A.pod_system != B.pod_system:
        return "❌ No: fits a different pod machine"
    if A.caffeine in CAF and B.caffeine in CAF and (A.caffeine == "regular") != (B.caffeine == "regular"):
        return f"❌ No: {CAF[B.caffeine]}, not {CAF[A.caffeine]}"
    return "❌ No: ruled out by the rules"


# How each AI model ranked the 7 look-alike products (average position over its 4 answers)
r = rankings[rankings["oos_asin"] == pick]
ranks = {}
for row in r.itertuples():
    for pos, cand in enumerate(row.ranking, start=1):
        ranks.setdefault(cand, {}).setdefault(MODEL[row.model], []).append(pos)
models = ["Gemini", "Claude", "Qwen", "DeepSeek"]
table = pd.DataFrame({c: {m: sum(v) / len(v) for m, v in d.items()} for c, d in ranks.items()}).T[models]

s = subs[(subs["condition"] == "pooled_pl") & (subs["oos_asin"] == pick)].set_index("candidate_asin")
table["Product"] = [name[c] for c in table.index]
table["Type"] = [info.loc[c, "type"] for c in table.index]
table["Can they use it?"] = ["✅ Yes" if s.loc[c, "allowed"] else why_not(pick, c) for c in table.index]
table["Of 100 switchers"] = (s.loc[table.index, "weight"] * 100).round().astype(int)
table = table.sort_values(["Of 100 switchers", *models], ascending=[False, True, True, True, True])

st.subheader("Where its customers would go")
st.dataframe(table[["Product", "Type", "Can they use it?", "Of 100 switchers", *models]].round(1),
             hide_index=True, width="stretch",
             column_config={"Product": st.column_config.TextColumn(width=210),
                            "Type": st.column_config.TextColumn(width=100),
                            "Can they use it?": st.column_config.TextColumn(width=200),
                            "Of 100 switchers": st.column_config.NumberColumn(width=110),
                            **{m: st.column_config.NumberColumn(width=62) for m in models}})
st.markdown(
    "**How to read this table**\n"
    "- **Of 100 switchers**: of every 100 customers who buy a substitute, how many pick this one. "
    "It combines all four AIs' rankings into one estimate.\n"
    "- **Gemini, Claude, Qwen, DeepSeek**: where each AI placed the product, from 1 (best substitute) to 7 (worst). "
    "Each AI was asked 4 times with the list in a different order (similarity order, reversed, and "
    "two random shuffles), and the number is its average. A whole number means it gave the same "
    "answer every time.\n"
    "- **Can they use it?**: a product a customer can't use gets 0, even if an AI ranked it highly."
)

# One plain sentence on whether the AIs agree
best = {m: table[m].idxmin() for m in models}
if len(set(best.values())) == 1:
    st.success(f"All four AIs agree that **{info.loc[next(iter(best.values())), 'title'][:80]}** is the "
               "best substitute.")
else:
    picks = "; ".join(f"{m}: {name[c]}" for m, c in best.items())
    st.warning(f"The AIs disagree on the best substitute ({picks}). Combining them evens out "
               "individual quirks.")

if not s["allowed"].any():
    st.error("None of the 7 similar coffees is usable for this product's customers, so if it sells "
             "out, all of them leave.")

st.subheader("Why? In the AIs' own words")
st.caption("Each AI's explanation for one of its four answers (the one with the list in similarity order).")
code = re.compile(r"\bB0[0-9A-Z]{8}\b")
for row in r[r["perm_index"] == 0].itertuples():
    with st.expander(MODEL[row.model]):
        text = code.sub(lambda m: f"“{name[m.group()]}”" if m.group() in name.index else m.group(),
                        row.reasoning)
        st.markdown(text.replace("$", "\\$"))   # stop prices being read as maths
