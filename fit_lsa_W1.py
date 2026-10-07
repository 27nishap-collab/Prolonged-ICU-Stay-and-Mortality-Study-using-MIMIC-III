"""Does a dense compression of the vocabulary beat a sparse selection of it?

Text blocks compared. All keep the 21 closed-vocabulary counts and the raw word
count fixed, and swap ONLY the open-vocabulary component:

  bow           500 L1-selected counts        (the block used everywhere else)
  lsa_pool_300  300 SVD components of the SAME 4,687-phrase candidate pool
  lsa_wide_300  300 SVD components of a 6,701-term unigram vocabulary (df>=0.5%)

plus a k-sensitivity at T5 (100 / 300 / 500 components of the pool).

Bases T1 and T5; labels composite and death; GBM and LR. Same split, same
corpus (v2), same fixed GBM configuration as the other ablation runs, so the
text block is the only thing that moves.

Pre-specified bar, from RESULTS_outcome_split.md: at T5 the bow block is worth
+0.0062 (composite) and +0.0032, p=0.26 (death), against a partition-noise sd of
0.005. A representation needs roughly +0.02 at T5 to be a finding rather than a
rounding error.
"""
import os, json, time, warnings
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
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
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

OUT = f"{U}/w1_lsa" + SFX; os.makedirs(OUT, exist_ok=True)
SEED = 20180701

GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def _mid(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong(y, S):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(S, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y)-m; k = s.shape[0]
    tx = np.vstack([_mid(s[r, :m]) for r in range(k)])
    ty = np.vstack([_mid(s[r, m:]) for r in range(k)])
    tz = np.vstack([_mid(s[r])     for r in range(k)])
    a = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return a, np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0,0]+c[1,1]-2*c[0,1]
    d = au[1]-au[0]; se = float(np.sqrt(max(v, 0))); z = d/se if se > 0 else 0.0
    return d, d-1.96*se, d+1.96*se, float(2*stats.norm.sf(abs(z)))

tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order = tab.hadm_id.to_numpy(); T = T.set_index("hadm_id").reindex(order)
tr = (tab.split == "train").to_numpy()
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
Xo = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)
Lp = np.load(f"{U}/lsa_pool_W1.npy"); Lw = np.load(f"{U}/lsa_wide_W1.npy")
log(f"fixed text part {FIX.shape[1]} cols | bow {Xo.shape[1]} | lsa pool {Lp.shape} wide {Lw.shape}")

TEXT = {"bow": np.hstack([FIX, Xo]),
        "lsa_pool_300": np.hstack([FIX, Lp[:, :300]]),
        "lsa_wide_300": np.hstack([FIX, Lw[:, :300]]),
        "lsa_pool_100": np.hstack([FIX, Lp[:, :100]]),
        "lsa_pool_500": np.hstack([FIX, Lp])}
LABELS = {"composite": T.labels.to_numpy().astype(int),
          "death":     T.outcome_death.to_numpy().astype(int)}

def enc(cols):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__"); kc = v[tr].value_counts()
            lv = set(kc[kc >= 0.01*tr.sum()].index); v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)
BASES = {"T1": enc(tiers["T1"]), "T5": enc(tiers["T5"])}

def score(X, y, mk):
    Xtr, ytr = X[tr], y[tr]
    if mk == "lrcv": return lr_tuned_scores(X, tr, y)[0]
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(Xtr, ytr).decision_function(X)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); Bm = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), y[tr]).decision_function(sc.transform(Bm))

def get(base, txt, lab, mk):
    f = f"{OUT}/s_{base}_{txt}_{lab}_{mk}.npy"
    if os.path.exists(f): return np.load(f)
    X = BASES[base] if txt == "none" else np.hstack([BASES[base], TEXT[txt]])
    s = score(X, LABELS[lab], mk); np.save(f, s)
    log(f"  {base}/{txt}/{lab}/{mk}: {X.shape[1]} cols, "
        f"AUROC {roc_auc_score(LABELS[lab][~tr], s[~tr]):.4f}")
    return s

PLAN = [(b, t, l, m) for m in LEARNERS for b in ["T1", "T5"]
        for l in ["composite", "death"] for t in ["bow", "lsa_pool_300", "lsa_wide_300"]]
PLAN += [("T5", t, "composite", m) for m in LEARNERS for t in ["lsa_pool_100", "lsa_pool_500"]]

rows = []
for base, txt, lab, mk in PLAN:
    y = LABELS[lab]
    b = get(base, "none", lab, mk); s = get(base, txt, lab, mk)
    d, lo, hi, p = paired(y[~tr], b[~tr], s[~tr])
    a0 = roc_auc_score(y[~tr], b[~tr])
    rows.append(dict(model=mk, base=base, outcome=lab, text=txt,
                     struct=round(a0, 4), with_text=round(roc_auc_score(y[~tr], s[~tr]), 4),
                     increment=round(d, 4), ci=f"{lo:+.4f},{hi:+.4f}", p=p,
                     pct_headroom=round(100*d/(1-a0), 1)))
    pd.DataFrame(rows).to_csv(f"{OUT}/W1_lsa.csv", index=False)

res = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("\n=== text increment: sparse selection vs dense compression ===")
print(res.to_string(index=False))
print("\n=== head to head ===")
for mk in LEARNERS:
    for base in ["T1", "T5"]:
        for lab in ["composite", "death"]:
            r = res[(res.model==mk)&(res.base==base)&(res.outcome==lab)].set_index("text")
            if len(r) < 3: continue
            print(f"  {mk}/{base}/{lab:9s}  bow {r.loc['bow','increment']:+.4f}   "
                  f"lsa_pool300 {r.loc['lsa_pool_300','increment']:+.4f}   "
                  f"lsa_wide300 {r.loc['lsa_wide_300','increment']:+.4f}")
log("done")
