# Methods

Detailed design of each stage. The README has the overview; the full argument and the
literature are in the thesis.

## Stage 01-04b: sample and candidates

- **Sample (01).** Amazon *Grocery & Gourmet Food* metadata, coffee categories only, then
  field trimming, a price filter and a text-sufficiency filter (at least 295 characters of
  title + features + description): 7,406 products. A stratified sample of 60 is drawn by
  format (ground, whole bean, pods, instant, RTD/cold brew) and price tier (budget,
  mid-range, premium), seed 42. Amazon category labels decide the strata; a title-based
  override corrects obvious mislabels.
- **Model inputs (02, 04b).** Sampling labels (category, stratum, price tier) are removed so
  no model can shortcut on them. Each product is rendered once into a fixed text block that
  every model sees verbatim.
- **Embeddings (04).** `BAAI/bge-large-en-v1.5`, cosine similarity for all 3,540 ordered
  pairs. Each product's 7 nearest neighbours become its substitute candidates.

## Stage 05: ranking elicitation

| Model | Access | Temperature |
|---|---|---|
| `gemini-3.6-flash` | Google API | 0.7 |
| `claude-sonnet-4-5` | Anthropic API | API default (no temperature passed) |
| `qwen3.6:27b` | Ollama, open weights | 0.7 |
| `deepseek-r1:32b` | Ollama, open weights | 0.7 |

Each product is treated as out of stock and each model ranks its 7 candidates **4 times**
with different candidate orders: embedding order, reversed, and two seeded shuffles
(SHA-256 seeded, so the orders are identical across machines). That is 60 x 4 x 4 =
**960 rankings**. The prompt gives an explicit rubric (format and caffeine as hard
constraints, then roast and flavour, price, other attributes) and asks for JSON only.
Output is parsed with a balanced-brace JSON extractor and validated (exactly the 7 IDs,
each once); failed calls are retried up to three attempts in total. Max output tokens 4096;
Ollama is called through its native `/api/chat` with `think: false`, JSON format and
`num_ctx` 8192.

## Stage 06: attribute extraction and manual verification

Gemini (temperature 0) extracts product type, format, pod system, caffeine, roast,
flavoured and pack size from the same text the rankers saw. Every value must carry an
evidence quote that the script checks occurs verbatim in the text; an independent keyword
pass and the Amazon category are shown alongside. Product type, format, pod system and
caffeine were then checked by hand for all 60 products; roast, flavoured and pack size for
a random 20 plus every flagged row, with 61 of those values filled in by a second reviewer
(marked in the `note` column of `outputs/attributes/attributes_verification.xlsx`; summary in
`outputs/attributes/attributes_verification_report.txt`).

## Stage 07: ranking analysis

Unit of analysis = out-of-stock product (n = 60). Aggregation = mean rank (Borda) with
embedding-order tie-break. 95% CIs from a product-level bootstrap within the five format
groups (2,000 replicates). Chance references are simulated through the same pipeline.

- **A1 stability:** Kendall's tau-b between a model's 4 rankings of the same product.
- **A2 position bias:** within (product, candidate), does a later slot get a worse rank?
  Slope plus permutation test.
- **A3 agreement:** pairwise tau-b, Kendall's W, top-3 Jaccard, all-four same top-1.
- **Embedding baseline:** how far the LLM rankings move from the embedding order.
- **A4 rubric compliance:** violations of the format/caffeine constraints, using the
  verified attributes. A caffeine conflict counts only when both products state caffeine
  (sensitivity: unstated = caffeinated).
- **A5:** descriptive subgroup summaries only.

## Stage 08: rankings to substitution shares

For each out-of-stock product: *of the shoppers who switch, what fraction go to each
candidate?* Walk-away and demand are not applied here.

- **Plackett-Luce** strengths fitted on the 16 raw rankings (4 models x 4 orders) with
  Hunter's (2004) MM algorithm.
- **Smoothing:** the 7 cyclic rotations of the candidate list are added as neutral
  pseudo-rankings (total weight 2 against 16 real rankings), so 16 agreeing rankings
  cannot produce a 100/0 split and the fit is always finite. Varied 0.5 / 2 / 8.
- **Compatibility rules** (shared code with Stage 07): incompatible candidates get share 0;
  allowed candidates share 100% in proportion to strength. Six products have no allowed
  candidate (dead ends); their switching shoppers are lost.
- **Variants:** single-model fits, smoothing levels, caffeine rule, rules off, dead ends
  kept, a simple points rule on the LLM rankings and on the embedding order.
- **Stability ranges:** rankings resampled within each model, 500 draws.
- Strengths (not only shares) are saved so the optimiser can redistribute shoppers over
  whatever is still stocked.

## Stage 09: scenario inputs

- **Demand:** same total in every instance; only concentration changes. Equal; moderate
  (Zipf s = 0.5, top 20% of products about 40% of demand); strong (s = 1.0, about 66%).
  Popularity is assigned to products at random, 20 draws per concentrated shape, seed 42.
- **Switching rate alpha** in {0.45, 0.60, 0.75}: the share of displaced shoppers who switch
  when all their allowed candidates are stocked. Outside option v0 = (1 - alpha) / alpha.
- **Shelf size C** in {15, 30, 45} of 60. Central scenario: moderate, alpha = 0.60, C = 30.
- **Coverage:** at least one coffee product from each of the 5 format groups.
- **7 substitution conditions:** pooled PL (reference), each single LLM, LLM points,
  embedding points. Weights rescaled to sum to 1 over allowed candidates, so every
  condition has the same switching rate and differences reflect only *where* shoppers go.

## Stage 10: assortment MILP

Decide x_j in {0,1} for each product. Customers of a dropped product i choose a stocked
allowed candidate j with probability w_ij / (v0 + S_i), or walk away, where
S_i = sum_j w_ij x_j. Because S_i is not rescaled, walk-away rises as substitutes disappear.
One round of switching. Objective: maximise expected retained demand.

Linearised exactly with flow shares q (no big-M, because every q_ij earns the same reward):

```
max  sum_i d_i (x_i + sum_j q_ij)
s.t. q_i0 + sum_j q_ij = 1 - x_i
     q_ij <= x_j
     q_ij <= (w_ij / v0) q_i0
     sum_j x_j <= C,  coverage,  0 <= q <= 1
```

Three valid inequalities tighten the LP relaxation for speed only. Solved with CBC to proven
optimality (`gapRel = 0`). 3,083 solves in total.

**Checks the script must pass or it stops:** hand-calculated cases; MILP = brute force on
20 random small instances; input integrity; for every solve: proven optimal, binary x,
exactly C products, coverage, MILP objective and every flow equal to an independent
evaluator; completeness; invariants (regret >= 0, optimum non-decreasing in C and alpha,
value of substitution >= 0). The list is printed at the top of
`outputs/assortment/assortment_report.txt`.

**Checking it by hand.** `src/handcheck.py` enumerates every shelf of a 3-4 product instance and
computes the customer model with plain loops (exact fractions for the worked example). It then
checks that Stage 10's `evaluate` gives the same total for every shelf and that Stage 10's `solve`
(with `verify_solution`) returns the best one. Worked example: alpha = 1/2 (v0 = 1), shelf of 2,
demand A 4, B 3, C 2, D 1; C has no substitute. Best shelf {A, C} keeps 160/21 = 7.619 customers;
the two bestsellers {A, B} keep 15/2. Run it with `python scripts/hand_check.py`, or open the
demo's *Check it by hand* page.

**Metrics:** retention regret against the pooled optimum (primary), products changed,
7 x 7 cross-evaluation, value of planning for substitution (alpha = 0 plan judged under
pooled weights), sensitivities and resampling stability.

## Limitations

- Demand is synthetic; no sales data was used.
- Pooled PL is a reference specification, not behavioural ground truth. Small regret
  against it does not mean the shares are accurate.
- The LLM conditions agree partly by construction: they rank the same 7 candidates and
  share the same compatibility rules.
- The value of planning for substitution cannot be negative by construction, because
  both plans are judged under the same pooled weights.
- 60 products, one category, one run per candidate order.
- "Optimal" means optimal within CBC's numerical tolerances. Re-running Stage 10 on Linux
  (same PuLP 3.3.2 / CBC 2.10.3) returned a different shelf in 15 of 2,583 main solves: 14 exact
  ties under equal demand, and one shelf 0.05% better than the committed one (strong demand,
  alpha = 0.75, C = 30, Qwen weights, draw 6). Regret changed by at most 0.004 percentage points;
  no summary changes at reported precision.
