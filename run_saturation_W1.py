"""The saturation curve: does text's incremental value shrink as the structured
baseline grows?

Everything is held fixed except the number of structured features -- same cohort
(W1, 25,692 admissions), same 48h window from hospital admission, same composite
outcome, same 75/25 outcome-stratified split, same text block (21 closed-
vocabulary counts + 500 open-vocabulary counts + total word count, selected once
on the original training set), same learners, same tuning protocol.

Five strictly nested tiers, built by build_tiers_W1.py:
  T1   20 cols   Weissman's exact 18 predictors
  T2   54        + min/max/mean/count for the concepts he already used
  T3   72        + the routine vitals he omitted, and body size
  T4  200        + ~30 further routine labs
  T5  222        + fluids, interventions and admission context

At each tier: structured only, and structured + text. Two learners -- gradient
boosting (the paper's best, and our closest reproduction) and logistic
regression (no tuning, so it cannot be confounded by tuning budget).

Reported quantity is the TEXT INCREMENT at each tier: test AUROC(structured +
text) - test AUROC(structured), with a DeLong paired CI. If the increment falls
monotonically with tier size, the cross-study trend has an explanation that does
not require the studies to differ in cohort, outcome or text pipeline.

PRE-SPECIFIED READING. Three outcomes are distinguishable in advance:
  (a) increment falls with tier size  -> baseline richness explains the
      cross-study pattern;
  (b) increment is flat               -> the cross-study pattern is cohort,
      outcome or text-pipeline driven, not baseline richness;
  (c) increment falls but the structured curve also flattens -> both saturate,
      and the interesting quantity is where.
The structured-only curve is reported alongside so (c) is distinguishable.

DECLARED CHOICES
  * Missing values: HistGradientBoosting handles NaN natively and is given the
    raw columns; logistic regression gets mean imputation fit on TRAINING ONLY
    (not the full-cohort mean Weissman used -- that leaks, and here it would
    leak differently at each tier, which would put a spurious trend in the
    curve). T1-GBM here therefore differs slightly from MODELS_W1.md, which
    replicated the published full-cohort imputation; both are reported.
  * Categoricals in T5 are one-hot encoded, levels with < 1% prevalence pooled.
  * Tuning: 5-fold CV on the training set over a reduced GBM grid, re-tuned at
    every tier and arm, so no tier inherits another's hyperparameters.
"""
import os, sys, json, time, itertools, warnings
import numpy as np, pandas as pd
# 2026-09-27: learner "lrcv" = tuned L2 logistic regression from lr_tuned.py.
# Default learners unchanged; run with W1_LEARNERS=gbm,lr,lrcv to add it.
# (Named LEARNERS, not MODELS: fit_bert_W1.py already uses MODELS for encoders.)
import sys as _sys_lrcv
_sys_lrcv.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_tuned import lr_tuned_scores, lr_tuned_fit_predict
LEARNERS = os.environ.get("W1_LEARNERS", "gbm,lr").split(",")
assert set(LEARNERS) <= {"gbm", "lr", "lrcv"}, LEARNERS
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
# W1_DROP lets a variable be removed from every tier at once, so the curve can
# be rerun with one predictor held out. Elixhauser is discharge-coded ICD-9 and
# sits in T1, hence in all five tiers.
DROP = [c.strip() for c in os.environ.get("W1_DROP", "").split(",") if c.strip()]
TAG  = os.environ.get("W1_TAG", "")

# ---- corpus switch (added 2026-09-21 after the corpus audit) ---------------
# v2 is the DEFAULT and the primary corpus: discharge summaries are dropped, so
# no note the model reads was written after the prediction time. v1 keeps them,
# and exists for exactly two jobs -- reproducing Weissman's Table 2, and forming
# the v1 arm of the leak ablation that measures what the leak was worth.
#
# This script had no switch at all before today and was hardcoded to v1, which
# put the leak into F3's curve. The old comment here read "text block is held at
# v1 so that Elixhauser is the ONLY thing that changes" -- true for the W1_DROP
# sensitivity it was written for, and the reason the default was never revisited.
# Pass W1_CORPUS=v1 to reproduce that.
CORPUS = os.environ.get("W1_CORPUS", "v2")
assert CORPUS in ("v1", "v2"), CORPUS
CSUF = "" if CORPUS == "v1" else "_v2"
# ---- specification switch (SPEC_DECISION.md, 2026-09-20) -------------------
# Two feature sets, named for what distinguishes them:
#   observable  (PRIMARY)  every predictor knowable at hour 48. Discharge-coded
#                          elixhauser -> has_prior_admission + prior_elixhauser;
#                          the five context variables windowed to 48h.
#   replication            Weissman's feature set exactly as published, which the
#                          F1 / Table 2 replication needs. "Published" describes
#                          his PREDICTOR LIST, never any result -- no saturation
#                          curve has ever been published; the curve is ours.
# Output directories are suffixed, so the two can never collide.
SPEC = os.environ.get("W1_SPEC", "observable")
SPEC = {"leakfree": "observable", "published": "replication"}.get(SPEC, SPEC)
assert SPEC in ("observable", "replication"), SPEC
SFX  = "" if SPEC == "replication" else "_observable"

OUT = (f"{U}/w1_saturation" + SFX + ("" if CORPUS == "v1" else "_v2")
       + (f"_{TAG}" if TAG else ""))
os.makedirs(OUT, exist_ok=True)
SEED = 20180701

t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong(y, scores):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(scores, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y) - m; k = s.shape[0]
    tx = np.vstack([_midrank(s[r, :m]) for r in range(k)])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(k)])
    tz = np.vstack([_midrank(s[r])     for r in range(k)])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    v01 = (tz[:, :m] - tx)/n; v10 = 1.0 - (tz[:, m:] - ty)/m
    return aucs, np.atleast_2d(np.cov(v01))/m + np.atleast_2d(np.cov(v10))/n

def paired(y, s_struct, s_text):
    a, cov = delong(y, [s_struct, s_text])
    v = cov[0,0] + cov[1,1] - 2*cov[0,1]
    d = a[1] - a[0]; se = float(np.sqrt(max(v, 0)))
    z = d/se if se > 0 else 0.0
    return dict(auc_struct=float(a[0]), auc_text=float(a[1]), increment=float(d),
                inc_lo=float(d-1.96*se), inc_hi=float(d+1.96*se),
                z=float(z), p=float(2*stats.norm.sf(abs(z))))

def auc_ci(y, s):
    a, cov = delong(y, [s]); se = float(np.sqrt(cov[0,0]))
    return float(a[0]), float(a[0]-1.96*se), float(a[0]+1.96*se)

# ---------------------------------------------------------------- data
def load():
    t = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
    tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
    tab = pd.read_parquet(f"{U}/model_tabular_W1{CSUF}.parquet")
    Xo = sparse.load_npz(
        f"{U}/text_open_W1{CSUF}{'_aligned' if CSUF else ''}.npz"
        ).toarray().astype(np.float32)
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    # tiers_W1 and model_tabular_W1 are both keyed on hadm_id; align explicitly
    tab = tab.set_index("hadm_id")
    order = tab.index.to_numpy()
    assert (np.sort(order) == order).all(), "open-vocab npz rows are sorted by hadm_id"
    t = t.set_index("hadm_id").reindex(order)
    TEXT = np.hstack([tab[cvc].to_numpy(np.float32),
                      tab[["n_words_raw"]].to_numpy(np.float32), Xo])
    y = tab.labels.to_numpy()
    tr = (tab.split == "train").to_numpy()
    return t, tiers, TEXT, y, tr

def encode(df, cols, tr):
    """numeric passthrough; categoricals one-hot with rare levels pooled.
    Level set is chosen on TRAINING rows only."""
    num, out, names = [], [], []
    for c in cols:
        s = df[c]
        if not pd.api.types.is_numeric_dtype(s):   # object, category, or arrow string
            v = s.astype(object).where(s.notna(), "__na__")
            keep = v[tr].value_counts()
            keep = set(keep[keep >= 0.01*tr.sum()].index)
            v = v.where(v.isin(keep), "__other__")
            for lev in sorted(keep | {"__other__"}):
                out.append((v == lev).to_numpy(np.float32)); names.append(f"{c}={lev}")
        else:
            out.append(s.to_numpy(np.float32)); names.append(c)
    return np.column_stack(out), names

# ---------------------------------------------------------------- models
GBM_GRID = [dict(max_depth=d, learning_rate=lr, max_iter=500, min_samples_leaf=15)
            for d in (10, 20) for lr in (0.01, 0.05)]

def fit_gbm(Xtr, ytr, X, folds):
    best, bs = None, -1
    for g in GBM_GRID:
        a = []
        for i, b in folds:
            m = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **g)
            m.fit(Xtr[i], ytr[i]); a.append(roc_auc_score(ytr[b], m.decision_function(Xtr[b])))
        if np.mean(a) > bs: bs, best = float(np.mean(a)), g
    m = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **best).fit(Xtr, ytr)
    return m.decision_function(X), best, bs

def fit_lr(Xtr, ytr, X, folds):
    mu = np.nanmean(Xtr, axis=0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    m = LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr)
    return m.decision_function(sc.transform(B)), {}, np.nan

def fit_lrcv(Xtr, ytr, X, folds):
    s, C = lr_tuned_fit_predict(Xtr, ytr, X)
    return s, {"C": C}, np.nan

FITTERS = {"gbm": fit_gbm, "lr": fit_lr, "lrcv": fit_lrcv}

# ---------------------------------------------------------------- main
def main():
    t, tiers, TEXT, y, tr = load()
    log(f"{len(y):,} admissions | train {tr.sum():,} / test {(~tr).sum():,} "
        f"| outcome {y.mean():.4f} | text block {TEXT.shape[1]} cols")
    skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
    rows = []
    if DROP:
        for k in tiers: tiers[k] = [c for c in tiers[k] if c not in DROP]
        log(f"dropped from every tier: {DROP}")
    for tier in ["T1", "T2", "T3", "T4", "T5"]:
        S, names = encode(t, tiers[tier], tr)
        log(f"{tier}: {len(tiers[tier])} variables -> {S.shape[1]} encoded columns")
        for mk in LEARNERS:
            sc = {}
            for arm, X in [("structured", S),
                           ("structured_text", np.hstack([S, TEXT]))]:
                f = f"{OUT}/scores_{tier}_{arm}_{mk}.npy"
                if os.path.exists(f):
                    sc[arm] = np.load(f); continue
                Xtr, ytr = X[tr], y[tr]
                folds = list(skf.split(Xtr, ytr))
                s, best, cv = FITTERS[mk](Xtr, ytr, X, folds)
                np.save(f, s); sc[arm] = s
                log(f"  {tier}/{mk}/{arm}: test AUROC {roc_auc_score(y[~tr], s[~tr]):.4f}"
                    + (f"  [{best}, cv {cv:.4f}]" if best else ""))
            r = paired(y[~tr], sc["structured"][~tr], sc["structured_text"][~tr])
            a, lo, hi = auc_ci(y[~tr], sc["structured"][~tr])
            b, blo, bhi = auc_ci(y[~tr], sc["structured_text"][~tr])
            rows.append(dict(tier=tier, n_vars=len(tiers[tier]), n_cols=S.shape[1],
                             model=mk,
                             struct=round(a,4), struct_ci=f"{lo:.3f}-{hi:.3f}",
                             text=round(b,4), text_ci=f"{blo:.3f}-{bhi:.3f}",
                             increment=round(r["increment"],4),
                             inc_ci=f"{r['inc_lo']:.4f}-{r['inc_hi']:.4f}",
                             z=round(r["z"],2), p=r["p"]))
            log(f"  {tier}/{mk}: struct {a:.4f} -> +text {b:.4f} "
                f"| increment {r['increment']:+.4f} "
                f"[{r['inc_lo']:+.4f},{r['inc_hi']:+.4f}] p={r['p']:.2g}")
            pd.DataFrame(rows).to_csv(f"{OUT}/W1_saturation.csv", index=False)

    res = pd.DataFrame(rows)
    print("\n=== saturation curve ===")
    print(res.to_string(index=False))

    # is the increment trend real? compare T1 vs T5 increments directly.
    print("\n=== does the increment shrink? T1 vs T5 ===")
    for mk in LEARNERS:
        a = res[(res.tier=="T1") & (res.model==mk)].iloc[0]
        b = res[(res.tier=="T5") & (res.model==mk)].iloc[0]
        print(f"  {mk}: T1 increment {a.increment:+.4f}  ->  T5 increment {b.increment:+.4f}"
              f"   ({100*(b.increment-a.increment)/a.increment:+.0f}%)")
    log("done")

if __name__ == "__main__":
    main()
