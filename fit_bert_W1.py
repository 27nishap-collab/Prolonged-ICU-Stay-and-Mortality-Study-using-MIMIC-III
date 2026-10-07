"""Does a contextual encoder buy anything the bag of words does not?

The last open axis. RESULTS_lsa.md ruled out three alternative explanations for
the saturation result -- sparsity, dimensionality, term selection -- by showing
that a dense compression of the same vocabulary collapses from T1 to T5 exactly
as the sparse selection does (five representation/learner combinations, 87-95%).
What LSA cannot supply is contextual meaning: word order, negation scope, which
concept a hedge attaches to. Frozen Bio_ClinicalBERT and PubMedBERT can.

PRE-SPECIFIED BAR, fixed in RESULTS_lsa.md before any embedding was computed:
at T5 the bag-of-words block is worth +0.0062 (composite) and +0.0032, p=0.26
(death), against a partition-noise sd of 0.005. A representation needs roughly
+0.02 at T5 to be a finding rather than a rounding error. The best LSA arm
reaches +0.0081. The T5/death cell is the pre-specified cell of interest.

TWO ENCODERS, AND WHY BOTH. Bio_ClinicalBERT was pretrained ON MIMIC-III notes,
so its encoder has already seen this corpus. That is not label leakage -- the
pretraining objective is masked-token prediction, not outcome -- but it does
mean it cannot be read as an out-of-the-box clinical encoder. PubMedBERT was
pretrained on PubMed abstracts and full text and has never seen a MIMIC note,
so it is the clean comparator. If they disagree, the gap is itself the result.

BLOCKS. All arms keep the 21 closed-vocabulary counts and the raw word count
fixed and swap only the open-vocabulary component, exactly as in fit_lsa_W1.py:

  bow            500 L1-selected counts        (the block used everywhere else)
  <m>_mean       768-d token mean-pool of the encoder's last hidden layer
  <m>_cls        768-d mean of the per-window [CLS] vector
  <m>_mean_300   SVD-300 of <m>_mean, fit on train rows  (dimension-matched to
                 lsa_wide_300, so the encoder and LSA are compared at equal width)
  bow_<m>        bow AND <m>_mean_300 together -- does the encoder add anything
                 ON TOP of the bag of words, rather than instead of it?

TRUNCATION SUBGROUP, pre-specified. The encoder reads at most 4 windows of 510
wordpieces; 45% of admissions (Bio) fit entirely inside that budget and 27% of
all corpus tokens are read. The bag-of-words block reads 100%. So every headline
cell is repeated on the test-set subgroup whose notes fit COMPLETELY (nfull<=4),
where the encoder has no coverage disadvantage at all. If the encoder loses
there too, truncation is not the explanation.

    python fit_bert_W1.py
"""
import os, json, time, glob, warnings
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
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
B = U                                   # frozen-encoder shards: out/bert_<model>_w0-4/
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

OUT = f"{U}/w1_bert_fit" + SFX; os.makedirs(OUT, exist_ok=True)
SEED = 20180701

GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
BAND = "w0-4"
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# ---------- DeLong ----------
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

# ---------- data, identical to fit_lsa_W1.py ----------
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order = tab.hadm_id.to_numpy(); T = T.set_index("hadm_id").reindex(order)
tr = (tab.split == "train").to_numpy()
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
Xo = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)
Lw = np.load(f"{U}/lsa_wide_W1.npy")

TEXT = {"bow": np.hstack([FIX, Xo]), "lsa_wide_300": np.hstack([FIX, Lw[:, :300]])}
MODELS, NFULL = [], {}
for m in ["bio_clinicalbert", "pubmedbert"]:
    d = f"{B}/bert_{m}_{BAND}"
    fs = sorted(glob.glob(f"{d}/b_*.npz"))
    if not fs:
        log(f"skipping {m}: nothing encoded yet"); continue
    g = lambda k: np.concatenate([np.load(f)[k] for f in fs])
    hd, sc, st, nt, nw = g("hadm"), g("s_cls"), g("s_tok"), g("n_tok"), g("n_win")
    pos = pd.Index(hd).get_indexer(order)
    if (pos < 0).any():                      # shard size is not assumed; coverage is checked
        log(f"skipping {m}: {int((pos<0).sum()):,} of {len(order):,} admissions not encoded yet")
        continue
    sc, st, nt, nw = sc[pos], st[pos], nt[pos], nw[pos]
    MEAN = st / np.maximum(nt, 1)[:, None]
    CLSV = sc / np.maximum(nw, 1)[:, None]
    svd = TruncatedSVD(300, random_state=SEED).fit(MEAN[tr])
    M300 = svd.transform(MEAN).astype(np.float32)
    TEXT[f"{m}_mean"]     = np.hstack([FIX, MEAN.astype(np.float32)])
    TEXT[f"{m}_cls"]      = np.hstack([FIX, CLSV.astype(np.float32)])
    TEXT[f"{m}_mean_300"] = np.hstack([FIX, M300])
    TEXT[f"bow_{m}"]      = np.hstack([FIX, Xo, M300])
    z = np.load(f"{U}/tok16_{m}_W1.npz")
    NFULL[m] = pd.Series(z["nfull"], index=z["hadm"]).reindex(order).to_numpy()
    MODELS.append(m)
    log(f"{m}: mean-pool {MEAN.shape}, SVD300 ev {svd.explained_variance_ratio_.sum():.3f}, "
        f"{(NFULL[m] <= 4).mean()*100:.1f}% of admissions fully covered")
assert MODELS, "no encoder finished yet"

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
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr).decision_function(sc.transform(Bm))

def get(base, txt, lab, mk):
    f = f"{OUT}/s_{base}_{txt}_{lab}_{mk}.npy"
    if os.path.exists(f): return np.load(f)
    X = BASES[base] if txt == "none" else np.hstack([BASES[base], TEXT[txt]])
    s = score(X, LABELS[lab], mk); np.save(f, s)
    log(f"  {base}/{txt}/{lab}/{mk}: {X.shape[1]} cols, "
        f"AUROC {roc_auc_score(LABELS[lab][~tr], s[~tr]):.4f}")
    return s

ARMS = ["bow", "lsa_wide_300"] + [f"{m}{s}" for m in MODELS
        for s in ["_mean", "_cls", "_mean_300"]] + [f"bow_{m}" for m in MODELS]
PLAN = [(b, t, l, mk) for mk in LEARNERS for b in ["T1", "T5"]
        for l in ["composite", "death"] for t in ARMS]

rows = []
for base, txt, lab, mk in PLAN:
    y = LABELS[lab]
    b = get(base, "none", lab, mk); s = get(base, txt, lab, mk)
    d, lo, hi, p = paired(y[~tr], b[~tr], s[~tr])
    a0 = roc_auc_score(y[~tr], b[~tr])
    r = dict(model=mk, base=base, outcome=lab, text=txt,
             struct=round(a0, 4), with_text=round(roc_auc_score(y[~tr], s[~tr]), 4),
             increment=round(d, 4), ci=f"{lo:+.4f},{hi:+.4f}", p=p,
             pct_headroom=round(100*d/(1-a0), 1))
    # pre-specified truncation subgroup: test rows whose notes fit in the budget
    for m in MODELS:
        if m in txt or txt in ("bow", "lsa_wide_300"):
            k = (NFULL[m] <= 4)[~tr]
            if k.sum() > 200 and 0 < y[~tr][k].sum() < k.sum():
                dd, _, _, pp = paired(y[~tr][k], b[~tr][k], s[~tr][k])
                r[f"inc_full_{m}"] = round(dd, 4); r[f"p_full_{m}"] = round(pp, 4)
    rows.append(r)
    pd.DataFrame(rows).to_csv(f"{OUT}/W1_bert.csv", index=False)

res = pd.DataFrame(rows); pd.set_option("display.width", 250)
print("\n=== text increment: bag of words vs LSA vs frozen contextual encoder ===")
print(res.to_string(index=False))
print("\n=== the pre-specified cell (T5) ===")
print(res[res.base == "T5"].pivot_table(index="text", columns=["model", "outcome"],
      values="increment", sort=False).to_string())
log("done")
