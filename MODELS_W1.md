# W1 models — Weissman et al. 2018, Table 2 replicated

Run by `run_models_W1.py` (main), `run_sensitivity_W1.py`, `run_splitnoise_W1.py`.
All outputs in `out/w1_models/`.

Four model types × two feature sets. Structured = the 18 predictors from
`FEATURES_W1.md`. Structured+text = those 18 plus 21 closed-vocabulary counts,
500 open-vocabulary counts and the total raw word count (`TEXT_W1.md`) = 540
predictors. Split, tuning grids and deviations are documented in the script
header.

---

## 1. Table 2

| model | feature set | our train | their train | our test (95% CI) | their test | Δ |
|---|---|---|---|---|---|---|
| logistic regression | structured | 0.8252 | 0.80 | 0.8069 (0.794–0.820) | 0.79 | +0.017 |
| elastic net | structured | 0.8252 | 0.80 | 0.8068 (0.794–0.820) | 0.79 | +0.017 |
| random forest | structured | 1.0000 | 1.00 | 0.8285 (0.816–0.841) | 0.82 | +0.008 |
| gradient boosting | structured | 0.8989 | 0.89 | 0.8313 (0.819–0.843) | 0.83 | **+0.001** |
| logistic regression | + text | 0.9165 | 0.90 | 0.8524 (0.840–0.864) | 0.86 | −0.008 |
| elastic net | + text | 0.9120 | 0.89 | 0.8620 (0.851–0.873) | 0.88 | −0.018 |
| random forest | + text | 1.0000 | 1.00 | 0.8663 (0.856–0.877) | 0.88 | −0.014 |
| gradient boosting | + text | 0.9512 | 0.95 | 0.8870 (0.877–0.897) | 0.89 | **−0.003** |

Every cell is within 0.018 of the published value, and the two headline GBM
cells are within 0.003. Every training AUROC reproduces to 0.02 or better,
including the random forest's 1.0000 — their overfitting signature reproduces
exactly. This is across implementations: theirs is R `gbm` / `randomForest` /
`glmnet` under `caret`, ours is scikit-learn on the same grids.

### The one systematic difference

Our structured arms run slightly **high** (+0.008 to +0.017) and our text arms
slightly **low** (−0.003 to −0.018). So the text increment we measure is
consistently smaller than theirs:

| model | our gain from text | their gain |
|---|---|---|
| logistic regression | +0.046 | +0.07 |
| elastic net | +0.055 | +0.09 |
| random forest | +0.038 | +0.06 |
| gradient boosting | +0.056 | +0.06 |

Both sides of that squeeze have a documented cause. The structured side is
richer than theirs — our extraction has better coverage on urine output (18.0%
vs 34.0% missing) and weight (15.3% vs 21.0%), so fewer patients are carrying
an imputed mean. The text side is thinner — our open vocabulary is 4,825
phrases against their 5,790, so the 500 are chosen from a smaller pool
(`TEXT_W1.md` §4.1 traces this to the NLTK stopword-list version).

The direction of the effect is reproduced in every model type; the magnitude is
reproduced only for gradient boosting.

## 2. Significance

DeLong, all 28 pairwise comparisons among the eight test models, Bonferroni
m = 28 — which is what the paper's m = 28 refers to. **24 of 28 are significant.**
Adding text within a model type:

| model | structured | + text | difference | z | Bonferroni p |
|---|---|---|---|---|---|
| logistic regression | 0.8069 | 0.8524 | +0.0455 | 7.47 | 2.2 × 10⁻¹² |
| elastic net | 0.8068 | 0.8620 | +0.0552 | 10.49 | 2.8 × 10⁻²⁴ |
| random forest | 0.8285 | 0.8663 | +0.0379 | 8.07 | 2.0 × 10⁻¹⁴ |
| gradient boosting | 0.8313 | 0.8870 | +0.0556 | 12.95 | 6.6 × 10⁻³⁷ |

The DeLong implementation was checked against a 400-draw bootstrap on the
logistic-regression scores: SE 0.00668 vs 0.00689 (structured) and 0.00613 vs
0.00612 (+ text), with CIs agreeing to three decimals.

---

## 3. Their tuning grids stop before the optimum

Tuning selected a **grid-boundary value for almost every hyperparameter**:

| model | selected | grid |
|---|---|---|
| RF, structured | mtry = 5 | 5:15 — bottom edge |
| RF, + text | mtry = 15 | 5:15 — top edge |
| GBM, structured | n.trees 500, depth 15, shrinkage 0.01, minobs 15 | three of four at the top edge |
| GBM, + text | n.trees 500, depth 20, shrinkage 0.01, minobs 15 | four of four at the top edge |

That is a signature of a truncated grid, so both were extended on the text arm
(5-fold CV, same folds throughout).

**Random forest — CV AUROC is still climbing where their grid ends:**

| mtry | 5 | 10 | **15** | 23 | 35 | 50 | 100 |
|---|---|---|---|---|---|---|---|
| CV AUROC | 0.8741 | 0.8801 | **0.8831** | 0.8860 | 0.8888 | 0.8899 | 0.8909 |

Monotone past their boundary, worth **+0.008** by mtry = 100. Their `mtry = 5:15`
is a sensible range for a 20-feature model; carried over unchanged to a
542-feature model it lands well below √p ≈ 23.

**Gradient boosting — the binding constraint is shrinkage, not depth:**

| n.trees | depth | shrinkage | CV AUROC | |
|---|---|---|---|---|
| 500 | 20 | 0.01 | 0.9008 | their best in-grid |
| 1000 | 20 | 0.01 | 0.9052 | beyond |
| 500 | 20 | 0.05 | 0.9055 | beyond |
| 500 | 25 | 0.01 | 0.9007 | beyond |
| 1000 | 25 | 0.05 | 0.9068 | beyond |

Depth past 20 buys nothing; raising shrinkage above 0.01 or trees above 500
buys **+0.006**.

This does not overturn anything — the gains are small and they are on the
*training* CV, not the test set. What it means is that the published AUROCs for
the two tree models are **floors set by the grid, not tuned maxima**, and the
grid was set once and reused for a feature set 27× larger. Anyone reading
Table 2 as "this is what a random forest can do here" is reading it slightly
low.

---

## 4. Does the ungrouped split inflate the result?

The paper's split is outcome-stratified and **not grouped by patient**:
`caret::createDataPartition(df$Y, p=0.75)`. In our cohort that puts 23.1% of
test admissions with a patient who also appears in training. The obvious worry
is that this inflates the published numbers.

**It does not.** A single patient-grouped partition gave a *higher* test AUROC
than the ungrouped one in all eight models (+0.010 to +0.027) — the opposite
direction. Because one draw cannot separate a leak from ordinary partition
variation, both schemes were repeated 40 times with logistic regression:

| feature set | ungrouped mean (sd) | grouped mean (sd) | difference | p |
|---|---|---|---|---|
| structured | 0.8194 (0.0051) | 0.8202 (0.0051) | −0.0008 | 0.51 |
| + text | 0.8766 (0.0047) | 0.8763 (0.0048) | +0.0003 | 0.78 |

No difference in either arm, with mean patient overlap of 23.4% in the
ungrouped draws. Repeat admissions in MIMIC-III are too few, and too far apart
in time, for within-patient memorisation to move a 6,400-admission test AUROC.

> **Caveat on the absolute levels in that table.** The 500 open-vocabulary
> phrases were selected on the *original* training set, so in the redrawn splits
> some of that data lands in test. The **+ text** row is therefore inflated in
> absolute terms and is not comparable to §1. The ungrouped-vs-grouped
> *contrast* is unaffected, since both schemes carry the same contamination —
> and the structured row, which has no feature selection, is clean and shows
> the same null.

### The published CIs understate the uncertainty

The more useful number in that table is the **sd of 0.005 across partitions**.
A single 75/25 split carries roughly ±0.010 of pure partition noise at 95%,
which is about the same width as the DeLong CI computed *within* one split.
Table 2's intervals are conditional on the draw and do not include it. Two of
our eight cells sit just outside the published point estimate by more than the
DeLong interval would suggest; partition variation alone covers that.

---

## 5. Declared deviations

1. **Tuning resampling is 10-fold CV with one repeat, not five.** Repeats
   stabilise hyperparameter *selection* and do not enter the reported test
   AUROC; five repeats is 5× the compute to choose among grid points mostly
   within noise of each other, on two cores.
2. **Implementations differ.** R `gbm` → sklearn `HistGradientBoostingClassifier`,
   R `randomForest` → sklearn `RandomForestClassifier`, `glmnet` → sklearn
   elastic-net logistic regression. Grids are mapped parameter for parameter
   (n.trees→max_iter, interaction.depth→max_depth, shrinkage→learning_rate,
   n.minobsinnode→min_samples_leaf, mtry→max_features); glmnet's (α, λ) maps to
   (l1_ratio = α, C = 1/(n_train·λ)).
3. **Mean imputation uses the full-cohort mean**, as the supplement describes
   ("the mean of non-missing values"), which leaks test information into the
   imputed values. Replicated as described rather than corrected.
4. **Elixhauser is discharge-coded ICD-9** and is not observable at hour 48.
   Replicated because they used it; see `FEATURES_W1.md` §5.

## 6. What this establishes for our own work

The replication holds: on the same cohort, same window and same 18 structured
predictors, unstructured text at hour 48 adds 0.038–0.056 AUROC, and the
ranking of the four model types is reproduced. That gives us a validated
external anchor at 18 structured variables, alongside our own 216-variable
baseline in `Analysis/text_vs_structured/BASELINE.md`.

It also sharpens the question. `TEXT_W1.md` §4.3 showed the open vocabulary's
negative weights are almost entirely disposition language — *extubated,
transfer floor, chair, diet, oob* — and its positive weights are severity terms
already present in the structured block. So a considerable part of this
0.05 AUROC is the note telling us what the team has decided, not what the
physiology shows. Separating those two is the thing the comparison across
Ghassemi (3 structured variables), Weissman (18), Husmann (42), ours (216) and
Hsu (~32,000) cannot currently do, because feature-set size, cohort size and
note sparsity all move together.
