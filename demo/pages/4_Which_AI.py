from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from engine import SHAPE, SWITCH_HELP

RESULTS = Path(__file__).resolve().parents[2] / "outputs" / "assortment"
AI, BASELINE, NO_RULES = "#3987e5", "#d95926", "#7a7a7a"     # blue = one AI, orange = simpler methods

LABELS = {                                                    # source -> (label, colour)
    "qwen_pl": ("Qwen only", AI),
    "deepseek_pl": ("DeepSeek only", AI),
    "claude_pl": ("Claude only", AI),
    "gemini_pl": ("Gemini only", AI),
    "llm_points": ("All 4 AIs, rough scoring", BASELINE),
    "embedding_points": ("No AI ranking: description similarity only", BASELINE),
}


@st.cache_data
def load():
    summary = pd.read_csv(RESULTS / "summary_by_scenario.csv")
    variants = pd.read_csv(RESULTS / "sensitivity_appendix_variants.csv")
    return summary, variants


summary, variants = load()

st.header("🧪 Does it matter which AI you use?")
st.info(
    "**The question.** The smart shelf is only as good as the substitute estimates behind it. What if "
    "you had used just one AI, a simpler method, or no AI at all?\n\n"
    "**How it was tested.** Each option below was used to plan a shelf. Every shelf was then scored "
    "against the same reference: the combined estimate of all four AIs. (Real switching behaviour for "
    "these products is unknown, so the combined estimate is the best reference available.) Each bar "
    "shows how many more customers that shelf loses than the shelf planned with all four AIs. "
    "**A shorter bar means a better shelf.**"
)

with st.expander("What the options mean"):
    st.markdown(
        "- **Qwen / DeepSeek / Claude / Gemini only**: substitutes estimated from that one AI's rankings.\n"
        "- **All 4 AIs, rough scoring**: the same rankings, but turned into estimates with a simple points "
        "system (1st place gets the most points) instead of the statistical model used everywhere else.\n"
        "- **No AI ranking: description similarity only**: a text-matching model scores how similar the "
        "descriptions are, and the most similar coffees count as the best substitutes. No AI is asked "
        "which coffee a customer would actually buy.\n"
        "- **All 4 AIs, without the common-sense rules** (default settings only): the AI estimates "
        "are used as they are, without ruling out substitutes a customer can't use (wrong pod machine, "
        "ground vs whole beans, decaf vs regular)."
    )

c1, c2, c3 = st.columns(3)
C = c1.select_slider("Shelf space (coffees)", [15, 30, 45], value=30)
switch = c2.select_slider("Customers willing to switch (out of 10)", [4.5, 6, 7.5], value=6,
                          format_func=lambda v: f"{v:g}", help=SWITCH_HELP)
shape = c3.selectbox("How sales are spread", ["moderate", "strong", "equal"], format_func=SHAPE.get)
alpha = round(switch / 10, 2)

cell = summary[(summary.capacity == C) & (summary.alpha == alpha) & (summary.demand_shape == shape)]
rows = [(LABELS[c][0], 100 * cell.loc[cell.condition == c, "regret_mean"].item(), LABELS[c][1]) for c in LABELS]
central = (C, alpha, shape) == (30, 0.6, "moderate")
if central:                                                   # rules-off variant exists only for this scenario
    rows.append(("All 4 AIs, without the common-sense rules",
                 100 * variants.loc[variants.variant == "no_rules", "regret_vs_main"].mean(), NO_RULES))
rows.sort(key=lambda r: r[1])

fig = go.Figure(go.Bar(
    y=[r[0] for r in rows], x=[r[1] for r in rows], orientation="h",
    marker_color=[r[2] for r in rows], text=[f"{r[1]:.2f}%" for r in rows], textposition="outside",
    hovertemplate="%{y}: %{x:.2f}% more customers lost than the 4-AI shelf<extra></extra>"))
fig.update_layout(height=70 + 48 * len(rows), margin=dict(l=10, r=40, t=10, b=10), bargap=0.35,
                  xaxis=dict(title="extra customers lost compared with the 4-AI shelf (%)", ticksuffix="%",
                             showgrid=False, range=[0, max(r[1] for r in rows) * 1.25 + 0.05]),
                  yaxis=dict(autorange="reversed"))
st.plotly_chart(fig, width="stretch")
worst = max(r[1] for r in rows)
st.caption("Blue: one AI on its own. Orange: simpler methods. Grey: the four AIs without the common-sense "
           "rules (shown for the default settings only). "
           f"To read the numbers: {worst:.2f}% means about {worst * 100:.0f} more customers lost per "
           "10,000. Averages over 20 simulated sales patterns"
           + (" (1 pattern when every coffee sells the same)." if shape == "equal" else "."))

ai_max = max(r[1] for r in rows if r[2] == AI)
emb = next(r[1] for r in rows if r[0].startswith("No AI"))
st.subheader("What this shows")
lines = []
if ai_max < 0.5:
    lines.append(f"- **Any one of the four AIs works almost as well as all four together** (at most "
                 f"{ai_max:.2f}% more customers lost).")
else:
    lines.append(f"- **Single AIs differ more here** (up to {ai_max:.2f}% more customers lost), so "
                 "combining the four is the safer choice.")
if ai_max > 0 and emb > 1.5 * ai_max:
    lines.append(f"- **Description similarity alone does worse** ({emb:.2f}%), about {emb / ai_max:.0f}× "
                 "the worst single AI. The AIs' judgement adds something.")
elif emb >= 0.05:
    lines.append(f"- **Description similarity alone does worse** ({emb:.2f}%) than any single AI.")
else:
    lines.append(f"- **With these settings every option picks nearly the same shelf** (description "
                 f"similarity alone: {emb:.2f}%).")
if central:
    rules = next(r[1] for r in rows if r[2] == NO_RULES)
    lines.append(f"- **The common-sense rules matter more than which AI you pick**: dropping them costs "
                 f"{rules:.2f}%.")
if C == 45:
    lines.append("- **With room for 45 coffees, almost every option ends up with nearly the same shelf**, "
                 "because there is space for most of the coffees that matter.")
else:
    lines.append("- **Try a shelf of 45**: with more room, the choice of method matters much less.")
st.markdown("\n".join(lines))
