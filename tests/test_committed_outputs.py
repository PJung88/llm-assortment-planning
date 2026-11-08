"""The committed results are internally consistent, and the README quotes them correctly."""
import json

import numpy as np
import pandas as pd
import pytest

from conftest import OUTPUTS


def test_960_complete_rankings():
    raw = pd.read_parquet(OUTPUTS / "rankings" / "raw_rankings.parquet")
    assert len(raw) == 960
    assert raw.groupby(["model", "oos_asin"]).size().eq(4).all()
    assert raw.model.nunique() == 4 and raw.oos_asin.nunique() == 60
    for r in raw.itertuples():
        assert sorted(json.loads(r.ranking)) == sorted(json.loads(r.candidate_order))


def test_share_matrix_rows_sum_to_one_or_zero():
    m = pd.read_csv(OUTPUTS / "substitution" / "share_matrix_main.csv", index_col=0)
    s = m.sum(axis=1).to_numpy()
    assert np.all(np.isclose(s, 1) | np.isclose(s, 0))
    assert np.isclose(s, 0).sum() == 6                     # the six dead ends
    assert (np.diag(m.to_numpy()) == 0).all()


def test_stage10_report_says_all_checks_passed():
    text = (OUTPUTS / "assortment" / "assortment_report.txt").read_text(encoding="utf-8")
    assert "[FAIL]" not in text and text.count("[PASS]") > 50


# Numbers quoted in README.md. If a result ever changes, this test fails until the README is updated.
README_REGRET = {"pooled_pl": 0.0, "qwen_pl": 0.15, "deepseek_pl": 0.15, "claude_pl": 0.25,
                 "gemini_pl": 0.26, "llm_points": 0.49, "embedding_points": 1.08}
README_TAU = {"gemini-3.6-flash": 0.88, "claude-sonnet-4-5": 0.68, "qwen3.6:27b": 0.56, "deepseek-r1:32b": 0.47}


@pytest.mark.parametrize("condition,pct", README_REGRET.items())
def test_readme_regret_table(condition, pct):
    c = pd.read_csv(OUTPUTS / "assortment" / "summary_central.csv").set_index("condition")
    assert round(100 * c.loc[condition, "regret_mean"], 2) == pct


@pytest.mark.parametrize("model,tau", README_TAU.items())
def test_readme_stability_numbers(model, tau):
    h = pd.read_csv(OUTPUTS / "analysis" / "headline_metrics.csv")
    row = h[(h.metric == "mean Kendall tau-b across candidate orders") & (h.model_or_set == model)]
    assert round(float(row.estimate.iloc[0]), 2) == tau


def test_readme_value_of_substitution():
    v = pd.read_csv(OUTPUTS / "assortment" / "value_of_substitution.csv")
    g = 100 * v.groupby(["demand_shape", "alpha", "capacity"]).gain_pct.mean()
    assert round(g.loc[("moderate", 0.60, 30)], 1) == 6.0      # central scenario
    assert round(g.min(), 1) == 0.5 and round(g.max(), 1) == 18.0
    assert len(v) == 360 and (v.gain > 0).all()
