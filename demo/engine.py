"""Shared logic for the demo: load data once, score a shelf, find the best shelf.

Same customer model as Stage 10 (one round of switching, walk-away rises as
substitutes disappear), written small so the demo does not depend on the pipeline code.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pulp
import streamlit as st

OUT = Path(__file__).resolve().parents[1] / "outputs"

# Plain-English labels used on every page
FORMAT = {"pod_capsule": "Pods / capsules", "ground": "Ground", "whole_bean": "Whole beans",
          "instant": "Instant", "ready_to_drink": "Ready to drink", "concentrate": "Cold-brew concentrate",
          "unresolvable": "Not stated", "other": "Other"}
MODEL = {"qwen3.6:27b": "Qwen", "deepseek-r1:32b": "DeepSeek",
         "gemini-3.6-flash": "Gemini", "claude-sonnet-4-5": "Claude"}
SHAPE = {"moderate": "Some coffees sell much more than others",
         "strong": "A handful of coffees dominate sales (extreme case)",
         "equal": "Every coffee sells the same"}
SWITCH_HELP = ("When a customer's usual coffee is missing, how many of 10 such customers buy another "
               "coffee instead of leaving? The rest walk away without buying.")


@st.cache_data
def load():
    products = pd.read_csv(OUT / "scenarios" / "products.csv")
    subs = pd.read_csv(OUT / "scenarios" / "substitution_inputs.csv")
    demand = pd.read_csv(OUT / "scenarios" / "demand.csv")
    rankings = pd.read_parquet(OUT / "rankings" / "raw_rankings.parquet")
    rankings["ranking"] = rankings["ranking"].map(json.loads)
    products["short"] = products["title"].str.slice(0, 45)
    products["type"] = products["format"].map(FORMAT).fillna(products["format"])
    return products, subs, demand, rankings


@st.cache_data
def load_attributes():
    """Verified product attributes (Stage 06), used to say why a substitute is ruled out."""
    return pd.read_csv(OUT / "attributes" / "attributes_final.csv").set_index("parent_asin")


def popularity_rank(d):
    """1 = best-selling product."""
    return (-np.asarray(d)).argsort().argsort() + 1


def weights(subs, asins, condition="pooled_pl"):
    """W[i, j] = how strongly customers of i lean towards j (rows sum to 1, or 0)."""
    pos = {a: k for k, a in enumerate(asins)}
    W = np.zeros((len(asins), len(asins)))
    for r in subs[subs["condition"] == condition].itertuples():
        W[pos[r.oos_asin], pos[r.candidate_asin]] = r.weight
    return W


def flows(x, W, alpha):
    """For each dropped product i: share of its customers going to j, and share walking away."""
    x = np.asarray(x, float)
    dropped = 1 - x
    if alpha == 0:
        return np.zeros_like(W), dropped
    v0 = (1 - alpha) / alpha
    S = W @ x
    Q = dropped[:, None] * W * x[None, :] / (v0 + S)[:, None]
    return Q, dropped - Q.sum(axis=1)


def retained(x, d, W, alpha):
    Q, _ = flows(x, W, alpha)
    return float(d @ np.asarray(x, float) + d @ Q.sum(axis=1))


@st.cache_data(show_spinner="Optimising the shelf...")
def best_shelf(d, W, alpha, C, groups):
    """Exact MILP (same formulation as Stage 10). groups = tuple of index tuples (coverage)."""
    n = len(d)
    prob = pulp.LpProblem("shelf", pulp.LpMaximize)
    x = [pulp.LpVariable(f"x{j}", cat="Binary") for j in range(n)]
    terms = [d[i] * x[i] for i in range(n)]
    if alpha > 0:
        v0 = (1 - alpha) / alpha
        for i in range(n):
            arcs = [j for j in range(n) if W[i, j] > 0]
            if not arcs:
                continue
            q0 = pulp.LpVariable(f"q0_{i}", 0, 1)
            q = {j: pulp.LpVariable(f"q_{i}_{j}", 0, 1) for j in arcs}
            prob += q0 + pulp.lpSum(q.values()) == 1 - x[i]
            for j in arcs:
                prob += q[j] <= x[j]
                prob += q[j] <= (W[i, j] / v0) * q0
                terms.append(d[i] * q[j])
    prob += pulp.lpSum(terms)
    prob += pulp.lpSum(x) <= C
    for g in groups:
        prob += pulp.lpSum(x[j] for j in g) >= 1
    prob.solve(pulp.PULP_CBC_CMD(msg=False))
    return np.array([round(v.value()) for v in x])


def coverage_groups(products):
    p = products[products["counts_for_coverage"]]
    return tuple(tuple(p.index[p["coverage_group"] == g]) for g in sorted(p["coverage_group"].unique()))
