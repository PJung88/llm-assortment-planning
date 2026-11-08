"""
Stage 08: Turn the 960 LLM rankings into substitution shares.

Question answered for every product A (treated as out of stock):
    "Of the shoppers who switch, what fraction go to each of A's 7 candidates?"

Nothing else. Walk-away (shoppers who buy nothing) and demand are NOT applied here;
they are Stage 09 assumptions. Keeping them out means everything in this stage's
output comes from the LLM rankings plus the compatibility rules, and nothing else.

Method (per out-of-stock product, separately)
---------------------------------------------
1. Strength scores (Plackett-Luce). Each candidate gets a positive "strength".
   A ranking is read as repeated choices: pick the favourite, then pick again from
   the rest, always with probability proportional to strength. The strengths are
   the values that make the observed rankings most likely, found with Hunter's
   (2004) MM algorithm. Full rankings are used, not just the #1 picks.
2. Smoothing. Before fitting, a small number of "neutral" pseudo-rankings is added:
   the 7 cyclic rotations of the candidate list, in which every candidate takes
   every position exactly once. On their own they imply equal strengths, so they
   pull the fit gently toward "all candidates equal". Their total weight is given
   in units of rankings (default 2, against 16 real rankings). This (a) stops 16
   agreeing rankings producing 100/0 splits and (b) guarantees a finite fit even
   when a candidate is never chosen. The weight is a choice, so it is varied.
3. Compatibility rules (prompt's hard constraints, same definitions as Stage 07,
   imported from 07_ranking_analysis.py so the two stages can never drift apart):
     compatible   -> allowed
     incompatible -> share set to 0
     undetermined -> allowed ONLY if the out-of-stock product itself has an
                     undetermined format (no rule can be applied, so the LLM's
                     judgement stands); otherwise excluded (a candidate whose fit
                     cannot be verified is not trusted).
   Allowed candidates share 100% in proportion to their strengths.
4. Dead ends: products with no allowed candidate. Main rule: their switching
   shoppers are lost (all shares 0). Sensitivity: keep the unfiltered LLM shares.
5. Uncertainty (main version only): resample the rankings (within each model, with
   replacement, so every model keeps equal weight), refit, repeat. Gives a stability
   range for every share (how much it depends on the particular rankings collected;
   NOT a confidence interval for customer behaviour), and the resampled strengths.

Dead-end flags: structural_dead_end = the rules leave no allowed candidate;
dead_end = no allowed candidate in THIS variant (differs only in dead_end_keep).
Every PL fit must converge, otherwise the script stops.

Why strengths are saved, not only shares
----------------------------------------
The shares assume all of A's other candidates are on the shelf. When the
assortment model drops A AND some of its candidates, A's shoppers must be
redistributed over what is still stocked. Plackett-Luce does this consistently:
share_j = strength_j / sum of strengths of allowed, stocked candidates. So the
OR stage needs the strengths and the "allowed" flag, not just one fixed table.

Variants (each tests one assumption; not a full factorial)
----------------------------------------------------------
  main               all 4 models pooled (16 rankings), smoothing 2, rules on, dead ends lost
  smooth_light       smoothing 0.5
  smooth_heavy       smoothing 8
  model_<name>       one model only (4 rankings), smoothing 0.5 -> for cross-model comparison.
                     Smoothing is scaled to keep the same pseudo-to-real ratio as main
                     (2/16 = 0.5/4), so differences reflect the model, not the smoothing.
  caffeine_unstated  "not stated" caffeine treated as caffeinated (Stage 07 sensitivity rule)
  no_rules           compatibility rules off (raw LLM shares)
  dead_end_keep      dead ends keep their unfiltered LLM shares
  points_llm         no model: shares proportional to average points (1st=7 ... 7th=1)
  points_embedding   same points rule applied to the embedding order (no LLM at all)
The two points variants share one transformation, so comparing them isolates the
value of the LLM ordering; comparing main with points_llm isolates the model.

Inputs
------
  outputs/rankings/raw_rankings.parquet        (Stage 05)
  outputs/embeddings/all_pairs_scores.parquet  (Stage 04, candidate sets and order)
  outputs/attributes/attributes_final.parquet  (Stage 06, verified attributes)

Outputs (outputs/substitution/)
-------------------------------
  strengths.parquet          one row per (variant, oos_asin, candidate): strength (sums to 1
                             over the 7), compat_status, allowed, share, dead_end, ...
                             -> the input to Stages 09/10
  strengths.csv              same, for inspection
  share_matrix_main.csv      60 x 60 main-version table (rows = out of stock, cols = goes to)
  bootstrap_strengths.parquet  resampled strengths, main version
  share_intervals_main.csv   95% range of each main share from the resampling
  diagnostics/*.csv          sanity checks (see report)
  substitution_report.txt    the headline checks in one place

Usage
-----
    python src/pipeline/08_substitution_shares.py
    python src/pipeline/08_substitution_shares.py --n-boot 1000
    python src/pipeline/08_substitution_shares.py --n-boot 0      # skip resampling (quick run)
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from config import (  # noqa: E402
    EMBEDDING_TOP_K,
    OUTPUTS_DIR,
    RANDOM_SEED,
    RANKING_MODELS_ALL,
    RANKINGS_FILE,
)

# Stage 07 is the single source of truth for the compatibility rules and the
# embedding order. Importing it runs no analysis (it has a __main__ guard).
_S7_PATH = Path(__file__).resolve().parent / "07_ranking_analysis.py"
_spec = importlib.util.spec_from_file_location("stage07", _S7_PATH)
s7 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(s7)

OUT_DIR = os.path.join(OUTPUTS_DIR, "substitution")
DIAG_DIR = os.path.join(OUT_DIR, "diagnostics")
ATTR_FILE = os.path.join(OUTPUTS_DIR, "attributes", "attributes_final.parquet")

K = EMBEDDING_TOP_K            # 7 candidates per product
EXPECTED_N_SKUS = 60
EXPECTED_PER_MODEL = 4         # candidate orders per (model, product)
SHORT = s7.SHORT               # {"gemini-3.6-flash": "Gemini", ...}

SMOOTHING = {"light": 0.5, "medium": 2.0, "heavy": 8.0}   # pseudo-rankings, total weight
MAIN_SMOOTHING = SMOOTHING["medium"]
# Single-model fits keep main's pseudo-to-real ratio: 2/16 -> 0.5/4.
MODEL_SMOOTHING = MAIN_SMOOTHING * EXPECTED_PER_MODEL / (EXPECTED_PER_MODEL * len(RANKING_MODELS_ALL))
MM_TOL = 1e-10
MM_MAX_ITER = 20000


# Variants
def build_variants() -> list[dict]:
    base = dict(source="pooled", method="pl", smoothing=MAIN_SMOOTHING,
                caffeine_rule=s7.PRIMARY_CAFFEINE_RULE, rules=True, dead_end="lost")
    v = [dict(base, name="main",
              tests="reference version"),
         dict(base, name="smooth_light", smoothing=SMOOTHING["light"],
              tests="smoothing strength (sharper shares)"),
         dict(base, name="smooth_heavy", smoothing=SMOOTHING["heavy"],
              tests="smoothing strength (flatter shares)")]
    for m in RANKING_MODELS_ALL:
        v.append(dict(base, name=f"model_{SHORT[m].lower()}", source=m,
                      smoothing=MODEL_SMOOTHING,
                      tests="one model only (cross-model comparison in the OR stage)"))
    v += [dict(base, name="caffeine_unstated", caffeine_rule="unstated_caffeinated",
               tests="'not stated' caffeine treated as caffeinated"),
          dict(base, name="no_rules", rules=False,
               tests="compatibility rules switched off"),
          dict(base, name="dead_end_keep", dead_end="keep",
               tests="dead ends keep their LLM shares instead of losing shoppers"),
          dict(base, name="points_llm", method="points",
               tests="simple points rule instead of Plackett-Luce"),
          dict(base, name="points_embedding", method="points", source="embedding",
               tests="points rule on the embedding order (no LLM information)")]
    return v


# Loading and validation
def load_inputs():
    for path in (RANKINGS_FILE, ATTR_FILE):
        if not os.path.exists(path):
            raise SystemExit(f"Missing input: {path}")

    emb_order, emb_cos = s7.load_embedding_order()
    attrs = pd.read_parquet(ATTR_FILE).set_index("parent_asin")
    raw = pd.read_parquet(RANKINGS_FILE)

    skus = sorted(raw["oos_asin"].unique())
    problems = []
    if len(skus) != EXPECTED_N_SKUS:
        problems.append(f"{len(skus)} products in rankings, expected {EXPECTED_N_SKUS}")
    if set(raw["model"]) != set(RANKING_MODELS_ALL):
        problems.append(f"models {sorted(set(raw['model']))} != config {RANKING_MODELS_ALL}")
    missing_attr = [a for a in skus if a not in attrs.index]
    if missing_attr:
        problems.append(f"{len(missing_attr)} products missing from attributes: {missing_attr[:5]}")
    candidate_skus = {c for a in skus for c in emb_order.get(a, [])}
    missing_cand_attr = sorted(candidate_skus - set(attrs.index))
    if missing_cand_attr:
        problems.append(f"{len(missing_cand_attr)} candidates missing from attributes: "
                        f"{missing_cand_attr[:5]}")

    # rankings[sku] = list of (model, perm_index, np.array of candidate indices, best first)
    rankings: dict[str, list] = {a: [] for a in skus}
    for row in raw.itertuples(index=False):
        cands = emb_order.get(row.oos_asin)
        if cands is None or len(cands) != K:
            problems.append(f"{row.oos_asin}: no embedding top-{K}")
            continue
        rk = json.loads(row.ranking)
        if sorted(rk) != sorted(cands):
            problems.append(f"{row.oos_asin}/{row.model}/{row.perm_index}: ranking is not a "
                            f"permutation of the embedding top-{K}")
            continue
        pos = {c: i for i, c in enumerate(cands)}
        rankings[row.oos_asin].append((row.model, int(row.perm_index),
                                       np.array([pos[c] for c in rk], dtype=int)))
    for a in skus:
        for m in RANKING_MODELS_ALL:
            n = sum(1 for r in rankings[a] if r[0] == m)
            if n != EXPECTED_PER_MODEL:
                problems.append(f"{a}/{m}: {n} rankings, expected {EXPECTED_PER_MODEL}")
    if problems:
        raise SystemExit("Input validation failed:\n  " + "\n  ".join(problems[:20]))
    return skus, emb_order, emb_cos, attrs, rankings


# Plackett-Luce fit (MM algorithm, Hunter 2004) with neutral pseudo-rankings
def neutral_rankings(k: int) -> np.ndarray:
    """The k cyclic rotations of 0..k-1: every item in every position exactly once."""
    base = np.arange(k)
    return np.array([np.roll(base, -s) for s in range(k)])


def fit_pl(R: np.ndarray, c: np.ndarray, smoothing: float,
           w0: np.ndarray | None = None) -> tuple[np.ndarray, int, bool]:
    """
    R: (n, K) candidate indices, best first.  c: (n,) ranking weights.
    smoothing: total weight of the K neutral pseudo-rankings (must be > 0).
    Returns strengths normalised to sum 1, iterations used, converged flag.
    """
    if smoothing <= 0:
        raise ValueError("smoothing must be > 0 (it also guarantees a finite fit)")
    k = R.shape[1]
    R = np.vstack([R, neutral_rankings(k)])
    c = np.concatenate([c, np.full(k, smoothing / k)])

    # "wins": weighted number of choice stages each item won (the last item is not a choice)
    wins = np.bincount(R[:, :-1].ravel(), weights=np.repeat(c, k - 1), minlength=k)
    w = np.full(k, 1.0 / k) if w0 is None else w0.copy()
    for it in range(1, MM_MAX_ITER + 1):
        wr = w[R]                                         # (n, k) strengths in ranking order
        remaining = np.cumsum(wr[:, ::-1], axis=1)[:, ::-1]   # strength still available at each stage
        inv = c[:, None] / remaining[:, :-1]              # (n, k-1)
        cum = np.cumsum(inv, axis=1)                      # item at position t sat in stages 0..t
        contrib = np.concatenate([cum, cum[:, -1:]], axis=1)
        denom = np.bincount(R.ravel(), weights=contrib.ravel(), minlength=k)
        w_new = wins / denom
        w_new /= w_new.sum()
        if np.max(np.abs(w_new - w)) < MM_TOL:
            return w_new, it, True
        w = w_new
    return w, MM_MAX_ITER, False


def points_scores(R: np.ndarray) -> np.ndarray:
    """Average points (1st = K ... last = 1), normalised to sum 1."""
    k = R.shape[1]
    pts = np.zeros(k)
    for r in R:
        pts[r] += np.arange(k, 0, -1)
    return pts / pts.sum()


def select_rankings(rk_list: list, source: str, k: int) -> np.ndarray:
    if source == "embedding":                 # embedding order = candidate index order
        return np.arange(k)[None, :]
    if source == "pooled":
        return np.array([r[2] for r in rk_list])
    return np.array([r[2] for r in rk_list if r[0] == source])


# Compatibility
STATUS = {True: "compatible", False: "incompatible", None: "undetermined"}


def allowed_mask(src_asin: str, cands: list[str], attrs: pd.DataFrame,
                 caffeine_rule: str) -> tuple[list[str], np.ndarray]:
    src = attrs.loc[src_asin]
    src_undetermined = src["format"] in s7.UNDETERMINED_FORMAT
    status, allowed = [], []
    for cnd in cands:
        ok = s7.compatible(src, attrs.loc[cnd], caffeine_rule)["overall"]
        status.append(STATUS[ok])
        allowed.append(ok is True or (ok is None and src_undetermined))
    return status, np.array(allowed, dtype=bool)


# Build one variant
def run_variant(v: dict, skus, emb_order, emb_cos, attrs, rankings) -> tuple[pd.DataFrame, list]:
    rows, fit_log = [], []
    for a in skus:
        cands = emb_order[a]
        R = select_rankings(rankings[a], v["source"], K)
        if v["method"] == "pl":
            strength, iters, conv = fit_pl(R, np.ones(len(R)), v["smoothing"])
            fit_log.append(dict(variant=v["name"], oos_asin=a, iterations=iters, converged=conv))
        else:
            strength = points_scores(R)

        status, allowed = allowed_mask(a, cands, attrs, v["caffeine_rule"])
        if not v["rules"]:
            allowed = np.ones(K, dtype=bool)
        structural_dead_end = not allowed.any()          # the rules leave nothing
        if structural_dead_end and v["dead_end"] == "keep":
            allowed = np.ones(K, dtype=bool)
        dead_end = not allowed.any()                     # nothing allowed in THIS variant
        share = np.where(allowed, strength, 0.0)
        share = share / share.sum() if share.sum() > 0 else share

        for j, cnd in enumerate(cands):
            rows.append(dict(variant=v["name"], oos_asin=a, candidate_asin=cnd,
                             embedding_rank=j + 1, cosine_sim=emb_cos[(a, cnd)],
                             strength=strength[j], compat_status=status[j],
                             allowed=bool(allowed[j]), share=share[j],
                             structural_dead_end=structural_dead_end,
                             dead_end=dead_end, method=v["method"], source=v["source"],
                             smoothing=v["smoothing"] if v["method"] == "pl" else np.nan,
                             caffeine_rule=v["caffeine_rule"], rules=v["rules"],
                             dead_end_rule=v["dead_end"]))
    return pd.DataFrame(rows), fit_log


# Resampling (main version)
def bootstrap_main(main: pd.DataFrame, skus, emb_order, attrs, rankings,
                   n_boot: int, seed: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(seed)
    boot_rows, share_draws = [], {}
    for a in skus:
        cands = emb_order[a]
        # align explicitly by candidate, so row order can never mismatch strengths
        m_rows = main[main.oos_asin == a].set_index("candidate_asin").reindex(cands)
        if m_rows["strength"].isna().any():
            raise RuntimeError(f"Main rows do not match the candidate list for {a}")
        w_hat = m_rows["strength"].to_numpy()
        allowed = m_rows["allowed"].to_numpy(dtype=bool)
        by_model = [np.array([r[2] for r in rankings[a] if r[0] == m]) for m in RANKING_MODELS_ALL]
        draws = np.empty((n_boot, K))
        for b in range(n_boot):
            R = np.vstack([Rm[rng.integers(0, len(Rm), len(Rm))] for Rm in by_model])
            w, _, conv = fit_pl(R, np.ones(len(R)), MAIN_SMOOTHING, w0=w_hat)
            if not conv:
                raise RuntimeError(f"Resampling PL fit did not converge: {a}, draw {b}")
            draws[b] = w
            boot_rows.extend(dict(rep=b, oos_asin=a, candidate_asin=c, strength=w[j])
                             for j, c in enumerate(cands))
        sh = np.where(allowed, draws, 0.0)
        tot = sh.sum(axis=1, keepdims=True)
        share_draws[a] = np.divide(sh, tot, out=np.zeros_like(sh), where=tot > 0)

    iv = []
    for a in skus:
        cands = emb_order[a]
        lo, med, hi = np.percentile(share_draws[a], [2.5, 50, 97.5], axis=0)
        for j, c in enumerate(cands):
            iv.append(dict(oos_asin=a, candidate_asin=c, share_low=lo[j],
                           share_median=med[j], share_high=hi[j]))
    return pd.DataFrame(boot_rows), pd.DataFrame(iv)


# Diagnostics
def diagnostics(all_v: pd.DataFrame, skus, emb_order, rankings, intervals):
    main = all_v[all_v.variant == "main"]
    out = {}

    # 1. Does the strength order match the plain mean-rank order of the 16 rankings?
    rows = []
    for a in skus:
        R = np.array([r[2] for r in rankings[a]])
        mean_pos = np.array([np.mean(np.where(R == j)[1]) for j in range(K)])
        top1_obs = np.bincount(R[:, 0], minlength=K) / len(R)
        s = main[main.oos_asin == a]["strength"].to_numpy()
        tau = kendalltau(-s, mean_pos).statistic
        rows.append(dict(oos_asin=a, tau_strength_vs_mean_rank=tau,
                         same_top=int(np.argmax(s)) == int(np.argmin(mean_pos)),
                         top1_abs_gap=float(np.mean(np.abs(s - top1_obs)))))
    out["order_check"] = pd.DataFrame(rows)

    # 2. Concentration: how much does the top substitute get? effective number of substitutes
    conc = []
    for (v, a), g in all_v.groupby(["variant", "oos_asin"]):
        sh = g["share"].to_numpy()
        if sh.sum() == 0:
            conc.append(dict(variant=v, oos_asin=a, dead_end=True, top_share=np.nan,
                             effective_n=np.nan, n_allowed=0, removed_strength=np.nan))
            continue
        conc.append(dict(variant=v, oos_asin=a, dead_end=False, top_share=sh.max(),
                         effective_n=1.0 / np.sum(sh ** 2), n_allowed=int(g["allowed"].sum()),
                         removed_strength=float(g.loc[~g["allowed"], "strength"].sum())))
    conc = pd.DataFrame(conc)
    out["concentration"] = conc

    # 3. Hubs: total share each product receives from others (main version)
    inc = main.groupby("candidate_asin")["share"].agg(["sum", lambda s: int((s > 0).sum())])
    inc.columns = ["incoming_share_total", "n_sources_sending"]
    inc = inc.reindex(skus, fill_value=0).sort_values("incoming_share_total", ascending=False)
    inc.index.name = "asin"
    out["incoming"] = inc.reset_index()

    # 4. Each variant vs main
    ref = main.set_index(["oos_asin", "candidate_asin"])["share"]
    ref_top = main.loc[main.groupby("oos_asin")["share"].idxmax()].set_index("oos_asin")["candidate_asin"]
    ref_dead = set(main.loc[main.dead_end, "oos_asin"])
    comp = []
    for v, g in all_v.groupby("variant", sort=False):
        dead = set(g.loc[g.dead_end, "oos_asin"])
        live = [a for a in skus if a not in dead and a not in ref_dead]
        gi = g.set_index(["oos_asin", "candidate_asin"])["share"]
        diff = (gi - ref).abs().groupby(level=0).sum().reindex(live)   # per product, 0..2
        top = g.loc[g.groupby("oos_asin")["share"].idxmax()].set_index("oos_asin")["candidate_asin"]
        c = conc[(conc.variant == v) & (~conc.dead_end)]
        comp.append(dict(variant=v, n_dead_ends=len(dead),
                         median_top_share=c["top_share"].median(),
                         median_effective_n=c["effective_n"].median(),
                         mean_total_abs_diff_vs_main=diff.mean() / 2,   # share of switchers redirected
                         same_top_substitute_as_main=float((top.reindex(live) == ref_top.reindex(live)).mean())))
    out["variant_comparison"] = pd.DataFrame(comp)

    if intervals is not None:
        iv = intervals.merge(main[["oos_asin", "candidate_asin", "share"]], on=["oos_asin", "candidate_asin"])
        top = iv.loc[iv.groupby("oos_asin")["share"].idxmax()]
        top = top[top["share"] > 0]
        out["top_share_intervals"] = top.assign(width=top.share_high - top.share_low)
    return out


def write_report(diag, all_v, fit_log, skus, attrs, n_boot):
    main = all_v[all_v.variant == "main"]
    oc, vc = diag["order_check"], diag["variant_comparison"]
    conc_main = diag["concentration"].query("variant == 'main' and not dead_end")
    dead = main.loc[main.dead_end, "oos_asin"].unique()
    inc = diag["incoming"]
    fl = pd.DataFrame(fit_log)
    L = ["Stage 08 — substitution shares", "=" * 60, ""]
    L.append(f"Products: {len(skus)}; candidates per product: {K}; variants: {all_v.variant.nunique()}")
    L.append(f"PL fits converged: {int(fl.converged.sum())}/{len(fl)} "
             f"(max iterations {int(fl.iterations.max())})")
    L += ["", "Main version (4 models pooled, smoothing "
          f"{MAIN_SMOOTHING:g} pseudo-rankings vs 16 real, rules on, dead ends lost)"]
    L.append(f"  Dead ends (no allowed candidate): {len(dead)}")
    for a in dead:
        L.append(f"    {a}  {attrs.loc[a, 'title'][:60]}")
    L.append(f"  Top substitute's share: median {conc_main.top_share.median():.1%} "
             f"(range {conc_main.top_share.min():.1%}–{conc_main.top_share.max():.1%})")
    L.append(f"  Effective number of substitutes: median {conc_main.effective_n.median():.2f}")
    L.append(f"  Strength removed by the rules (went to incompatible/unverified candidates): "
             f"median {conc_main.removed_strength.median():.1%}, mean {conc_main.removed_strength.mean():.1%}")
    L.append(f"  Products receiving no switched demand: {int((inc.incoming_share_total == 0).sum())}")
    L.append("  Largest receivers (sum of incoming shares):")
    for r in inc.head(5).itertuples():
        L.append(f"    {r.asin}  {r.incoming_share_total:.2f} from {r.n_sources_sending} products  "
                 f"{attrs.loc[r.asin, 'title'][:45]}")
    L += ["", "Sanity: strength order vs plain mean-rank order (16 rankings)"]
    L.append(f"  Kendall tau: median {oc.tau_strength_vs_mean_rank.median():.2f}, "
             f"min {oc.tau_strength_vs_mean_rank.min():.2f}")
    L.append(f"  Same top candidate: {int(oc.same_top.sum())}/{len(oc)}")
    L.append(f"  Mean |strength - observed #1 rate|: {oc.top1_abs_gap.mean():.3f}")
    if "top_share_intervals" in diag:
        t = diag["top_share_intervals"]
        L += ["", f"Resampling ({n_boot} draws): width of 95% range for the top substitute's share"]
        L.append(f"  median {t.width.median():.1%}, 90th percentile {t.width.quantile(.9):.1%}")
    L += ["", "Variants vs main  (redirected = share of switchers sent elsewhere than in main)"]
    L.append(f"  {'variant':20s} {'dead':>4s} {'top share':>9s} {'eff. n':>6s} {'redirected':>10s} {'same #1':>7s}")
    for r in vc.itertuples():
        L.append(f"  {r.variant:20s} {r.n_dead_ends:4d} {r.median_top_share:9.1%} "
                 f"{r.median_effective_n:6.2f} {r.mean_total_abs_diff_vs_main:10.1%} "
                 f"{r.same_top_substitute_as_main:7.0%}")
    text = "\n".join(L)
    with open(os.path.join(OUT_DIR, "substitution_report.txt"), "w", encoding="utf-8") as f:
        f.write(text + "\n")
    print(text)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--n-boot", type=int, default=500, help="resampling draws (0 = skip)")
    ap.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = ap.parse_args()

    os.makedirs(DIAG_DIR, exist_ok=True)
    skus, emb_order, emb_cos, attrs, rankings = load_inputs()

    variants = build_variants()
    parts, fit_log = [], []
    for v in variants:
        df, log = run_variant(v, skus, emb_order, emb_cos, attrs, rankings)
        parts.append(df)
        fit_log += log
        print(f"  {v['name']:20s} done  ({v['tests']})")
    all_v = pd.concat(parts, ignore_index=True)

    fit_df = pd.DataFrame(fit_log)
    fit_df.to_csv(os.path.join(DIAG_DIR, "pl_fit_log.csv"), index=False)
    if not fit_df["converged"].all():
        bad = fit_df.loc[~fit_df["converged"], ["variant", "oos_asin"]].head().to_dict("records")
        raise SystemExit(f"{(~fit_df['converged']).sum()} PL fits did not converge "
                         f"(see diagnostics/pl_fit_log.csv), e.g. {bad}")

    # sanity: every product's shares sum to 1, or 0 for dead ends
    sums = all_v.groupby(["variant", "oos_asin"])["share"].sum()
    assert np.allclose(sums[sums > 0], 1.0), "shares do not sum to 1"

    all_v.to_parquet(os.path.join(OUT_DIR, "strengths.parquet"), index=False)
    all_v.to_csv(os.path.join(OUT_DIR, "strengths.csv"), index=False)
    pd.DataFrame(variants).to_csv(os.path.join(OUT_DIR, "variants.csv"), index=False)
    main_v = all_v[all_v.variant == "main"]
    matrix = (main_v.pivot(index="oos_asin", columns="candidate_asin", values="share")
              .reindex(index=skus, columns=skus).fillna(0.0))
    matrix.to_csv(os.path.join(OUT_DIR, "share_matrix_main.csv"))

    intervals = None
    if args.n_boot > 0:
        print(f"  resampling main version ({args.n_boot} draws)...")
        boot, intervals = bootstrap_main(main_v, skus, emb_order, attrs, rankings,
                                         args.n_boot, args.seed)
        boot.to_parquet(os.path.join(OUT_DIR, "bootstrap_strengths.parquet"), index=False)
        intervals.to_csv(os.path.join(OUT_DIR, "share_intervals_main.csv"), index=False)

    diag = diagnostics(all_v, skus, emb_order, rankings, intervals)
    for name, df in diag.items():
        df.to_csv(os.path.join(DIAG_DIR, f"{name}.csv"), index=False)
    print()
    write_report(diag, all_v, fit_log, skus, attrs, args.n_boot)


if __name__ == "__main__":
    main()
