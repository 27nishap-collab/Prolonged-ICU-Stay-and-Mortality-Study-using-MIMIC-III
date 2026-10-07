"""How much of the saturation curve is one lucky partition? — RUN THIS ON THE MAC.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3 run_repeated_splits_W1.py

Every number in RESULTS_saturation.md, RESULTS_ablation.md, RESULTS_reselect.md,
RESULTS_outcome_split.md, RESULTS_lsa.md and RESULTS_bert.md comes from ONE
75/25 split. MODELS_W1.md section 4 put the partition sd at about 0.005 from a
separate noise study, which is the basis for every "inside the noise band"
caveat in those documents. That estimate was made on a single model, not on the
curve, so the curve's own sampling variability has never been measured.

This measures it. R independent outcome-stratified 75/25 draws, the whole five-
tier curve rebuilt inside each one. The reported quantity is not the increment
at any tier but the COLLAPSE -- increment(T1) - increment(T5), and the ratio --
with its distribution across draws. That is the claim the project rests on, so
that is the thing that needs an interval.

WHAT IS HELD FIXED, AND THE ONE PLACE THAT COSTS SOMETHING
  * The 500 open-vocabulary phrases were L1-selected once, on the ORIGINAL
    training rows. They are NOT re-selected per draw. Under a new split some
    test rows were in the selection's training set, so every increment here is
    biased slightly UP. That bias applies identically at all five tiers, so it
    inflates the level of the curve and not its shape -- and the shape is what
    is being measured. Re-selecting per draw would need the full 4,687-phrase
    DTM and an L1 path per draw; if that is ever wanted it belongs in its own
    run, not folded in here.
  * GBM configuration is FIXED (depth 10, shrinkage 0.05, 500 trees) rather than
    re-tuned per tier and draw, as in every ablation run since MODELS_W1.md.
    Re-tuning would be 5 folds x 4 grid points x 200 cells and would confound
    partition noise with tuning noise.
  * Mean imputation for logistic regression is fit on each draw's own training
    rows, never the full cohort.

Resumable: one row per (draw, tier, learner) is appended to the CSV and a
re-run skips draws already complete. Ctrl-C and restart is safe.

    W1_R=10          number of draws          (default 10)
    W1_CORPUS=v2     v1 or v2                 (default v2 since 2026-09-21)
    W1_DATA=<dir>    input directory          (default <script dir>/out)
    W1_OUTDIR=<dir>  where results are written (default W1_DATA)
    W1_LEARNERS=...  comma list of learners   (default gbm,lr -- unchanged;
                                              W1_MODELS accepted as an alias)

LEARNER lrcv (added 2026-09-27). The `lr` rows use scikit-learn's default C = 1,
which check_lr_tuning_W1.py showed is far too weak once text is added: the T5
increment goes negative and the collapse is inflated. `lrcv` is L2 logistic
regression with C chosen by 5-fold CV inside each draw's training rows, from
lr_tuned.py, fitted separately for the structured-only and structured+text
models. The `lr` rows are kept for provenance, not deleted. Chosen C values go to
W1_repeated_lrcv_C.csv so the CSV schema above is unchanged.

CORPUS, AND WHY THE DEFAULT CHANGED. This defaulted to v1 until 2026-09-21,
because the script was written to put error bars on a headline curve that was
itself v1. v1 keeps the discharge summaries that COALESCE(charttime, chartdate)
pulls inside the 48h window for 5.4% of admissions -- notes written after the
prediction time. That default survived the move to the observable specification
and silently put the leak into the headline collapse figure. v2 is now the
default; v1 remains available for the leak ablation and the Table 2
replication, which are the only two places it belongs.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_tuned import lr_tuned_scores
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
OUTROOT = os.environ.get("W1_OUTDIR", U)
R      = int(os.environ.get("W1_R", 10))
CORPUS = os.environ.get("W1_CORPUS", "v2")
assert CORPUS in ("v1", "v2"), CORPUS
SUF    = "" if CORPUS == "v1" else "_v2"
SEED = 20180701

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

OUT = os.path.join(OUTROOT, f"w1_repeated{SUF}{SFX}"); os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/W1_repeated.csv"
CSV_C = f"{OUT}/W1_repeated_lrcv_C.csv"
# W1_LEARNERS matches the other fitting scripts; W1_MODELS still accepted (2026-09-27 name)
LEARNERS = os.environ.get("W1_LEARNERS", os.environ.get("W1_MODELS", "gbm,lr")).split(",")
assert set(LEARNERS) <= {"gbm", "lr", "lrcv"}, LEARNERS
LAST_C = []

GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
TIERS = ["T1", "T2", "T3", "T4", "T5"]
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
    return aucs, (np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m
                  + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n)

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0,0]+c[1,1]-2*c[0,1]
    d = au[1]-au[0]; se = float(np.sqrt(max(v, 0)))
    return float(au[0]), float(au[1]), float(d), float(se)

# ---------------------------------------------------------------- data
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet")
Xo  = sparse.load_npz(f"{U}/text_open_W1{SUF}{'_aligned' if SUF else ''}.npz"
                      ).toarray().astype(np.float32)
tab = tab.set_index("hadm_id"); order = tab.index.to_numpy()
assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc = [c for c in tab.columns if c.startswith("cv_")]
TEXT = np.hstack([tab[cvc].to_numpy(np.float32),
                  tab[["n_words_raw"]].to_numpy(np.float32), Xo])
y = tab.labels.to_numpy().astype(int)
log(f"corpus {CORPUS} | {len(y):,} admissions | outcome {y.mean():.4f} "
    f"| text block {TEXT.shape[1]} cols | {R} draws")

def draw_split(r):
    """The original procedure -- 25% test, stratified on outcome, not grouped by
    subject -- re-run with a different seed."""
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def encode(cols, tr):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            k = v[tr].value_counts(); lv = set(k[k >= 0.01*tr.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

def score(X, tr, mk):
    Xtr, ytr = X[tr], y[tr]
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(Xtr, ytr).decision_function(X)
    if mk == "lrcv":
        s, C = lr_tuned_scores(X, tr, y); LAST_C.append(C); return s
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr
           ).decision_function(sc.transform(B))

done = set()
if os.path.exists(CSV):
    old = pd.read_csv(CSV)
    done = set(zip(old.draw, old.tier, old.model))
    log(f"resuming: {len(done)} of {R*len(TIERS)*len(LEARNERS)} cells already on disk")

rows = []
for r in range(R):
    if all((r, t, m) in done for t in TIERS for m in LEARNERS):
        continue
    tr = draw_split(r)
    log(f"draw {r}: train {tr.sum():,} / test {(~tr).sum():,} "
        f"| test outcome {y[~tr].mean():.4f}")
    for tier in TIERS:
        S = encode(tiers[tier], tr)
        ST = np.hstack([S, TEXT])
        for mk in LEARNERS:
            if (r, tier, mk) in done: continue
            a, b, d, se = paired(y[~tr], score(S, tr, mk)[~tr], score(ST, tr, mk)[~tr])
            rows.append(dict(draw=r, tier=tier, n_vars=len(tiers[tier]), model=mk,
                             struct=round(a, 4), with_text=round(b, 4),
                             increment=round(d, 4), se=round(se, 4)))
            pd.DataFrame(rows).to_csv(CSV, mode="a", header=not os.path.exists(CSV),
                                      index=False); rows = []
            log(f"  {tier}/{mk}: struct {a:.4f} -> +text {b:.4f}  inc {d:+.4f}")
            if mk == "lrcv":
                pd.DataFrame([dict(draw=r, tier=tier, C_struct=LAST_C[-2], C_text=LAST_C[-1])]
                             ).to_csv(CSV_C, mode="a", header=not os.path.exists(CSV_C),
                                      index=False)
                log(f"         chosen C: struct {LAST_C[-2]:.0e}  text {LAST_C[-1]:.0e}")

# ---------------------------------------------------------------- summary
res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model"])
pd.set_option("display.width", 200)
print(f"\n=== text increment by tier, across {res.draw.nunique()} draws ===")
PRESENT = [m for m in ("gbm", "lr", "lrcv") if m in set(res.model)]
for mk in PRESENT:
    g = res[res.model == mk].groupby("tier").increment
    s = pd.DataFrame({"mean": g.mean().round(4), "sd": g.std().round(4),
                      "min": g.min().round(4), "max": g.max().round(4)}).reindex(TIERS)
    print(f"\n  {mk}\n{s.to_string()}")

print("\n=== the collapse, per draw ===")
for mk in PRESENT:
    p = res[res.model == mk].pivot(index="draw", columns="tier", values="increment")
    if not {"T1", "T5"} <= set(p.columns): continue
    p = p.dropna(subset=["T1", "T5"])
    coll = p.T1 - p.T5; pct = 100 * coll / p.T1
    print(f"  {mk}: T1 {p.T1.mean():+.4f} (sd {p.T1.std():.4f})  "
          f"T5 {p.T5.mean():+.4f} (sd {p.T5.std():.4f})")
    print(f"       collapse {coll.mean():+.4f} (sd {coll.std():.4f}, "
          f"range {coll.min():+.4f} to {coll.max():+.4f})  "
          f"= {pct.mean():.0f}% (range {pct.min():.0f}-{pct.max():.0f}%)")
    print(f"       draws where the increment falls from T1 to T5: "
          f"{int((coll > 0).sum())}/{len(coll)}")
log("done")
