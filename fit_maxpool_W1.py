"""Does max-pooling rescue the frozen encoder? -- the pre-specified mechanism test.

    W1_SPEC=observable python3 fit_maxpool_W1.py

THE PREDICTION, fixed before the fit (stated in chat 2026-09-21, before the
re-encode finished):

    Mean-pooling over ~8,000 wordpieces buries a handful of decisive tokens AND
    rotates the signal off the coordinate axes, which is near worst-case for the
    axis-parallel splits a tree makes and comparatively harmless to a linear
    model that can take any direction. If that is the mechanism, max-pooling --
    which keeps "did anything here fire on this dimension" -- should recover a
    meaningful part of the GBM deficit at T1 and do comparatively little under
    LR. If both stay flat, the account is wrong and anisotropy has to carry the
    explanation alone.

That asymmetry (GBM gain >> LR gain) is the thing being tested, not the absolute
level. Both outcomes are informative; the flat one is the one that forces a
rewrite of the mechanism paragraph in RESULTS_bert.md.

ARMS ADDED (PubMedBERT only -- the stronger encoder at T1 and the more anisotropic
of the two, cosine 0.995; if max-pooling does not move it, it will not rescue
Bio_ClinicalBERT either):

    pubmedbert_max_tok      element-wise max over every content token    (768)
    pubmedbert_max_win      element-wise max over per-window means       (768)
    pubmedbert_max_tok_300  SVD-300 of max_tok, dimension-matched to LSA (300)

Shares fit_bert_W1.py's cache directory and tag format, so bow / lsa_wide_300 /
pubmedbert_mean / the structured-only bases are REUSED, not refitted -- the
comparison is against the identical score vectors already reported. Only the
three new arms cost anything. Resumable per cell.
"""
import os, sys, json, time, glob, warnings
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
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
B = U                                   # frozen-encoder shards: out/bert_<model>_w0-4/
BAND, SEED = "w0-4", 20180701
GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)

SPEC = os.environ.get("W1_SPEC", "observable")
SPEC = {"leakfree": "observable", "published": "replication"}.get(SPEC, SPEC)
assert SPEC in ("observable", "replication"), SPEC
SFX = "" if SPEC == "replication" else "_observable"

OUT = f"{U}/w1_bert_fit" + SFX          # SAME cache as fit_bert_W1.py
os.makedirs(OUT, exist_ok=True)
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

# ---------- data, byte-for-byte as fit_bert_W1.py builds it ----------
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

def gather(d, keys):
    fs = sorted(glob.glob(f"{B}/{d}/b_*.npz"))
    assert fs, d
    return {k: np.concatenate([np.load(f)[k] for f in fs]) for k in keys}

M = "pubmedbert"
old = gather(f"bert_{M}_{BAND}", ["hadm", "s_tok", "n_tok"])
pos = pd.Index(old["hadm"]).get_indexer(order); assert not (pos < 0).any()
TEXT[f"{M}_mean"] = np.hstack([FIX, (old["s_tok"][pos] /
                    np.maximum(old["n_tok"][pos], 1)[:, None]).astype(np.float32)])

new = gather(f"bertpool_{M}_{BAND}", ["hadm", "s_tok", "n_tok", "m_tok", "m_win"])
p2 = pd.Index(new["hadm"]).get_indexer(order); assert not (p2 < 0).any()
# the re-encode must reproduce the original forward pass exactly, or the new
# arms are not comparable with the reported ones
assert np.array_equal(old["n_tok"][pos], new["n_tok"][p2]), "token counts differ"
dmax = float(np.abs(old["s_tok"][pos] - new["s_tok"][p2]).max())
log(f"re-encode check: max |s_tok difference| = {dmax:.3e}")
assert dmax == 0.0, "re-encode is not bit-identical; arms are not comparable"

MT, MW = new["m_tok"][p2].astype(np.float32), new["m_win"][p2].astype(np.float32)
TEXT[f"{M}_max_tok"] = np.hstack([FIX, MT])
TEXT[f"{M}_max_win"] = np.hstack([FIX, MW])
svd = TruncatedSVD(300, random_state=SEED).fit(MT[tr])
TEXT[f"{M}_max_tok_300"] = np.hstack([FIX, svd.transform(MT).astype(np.float32)])
log(f"max_tok SVD300 explained variance {svd.explained_variance_ratio_.sum():.3f}")

# anisotropy, the quantity the mean-pool explanation rests on, for each pooling
def cosmean(V, n=3000):
    rng = np.random.RandomState(SEED); i = rng.choice(len(V), n, replace=False)
    W = V[i] / np.maximum(np.linalg.norm(V[i], axis=1, keepdims=True), 1e-9)
    G = W @ W.T; iu = np.triu_indices(n, 1); return float(G[iu].mean())
MEANV = (old["s_tok"][pos] / np.maximum(old["n_tok"][pos], 1)[:, None]).astype(np.float32)
for nm, V in [("mean", MEANV), ("max_tok", MT), ("max_win", MW)]:
    log(f"anisotropy {nm:8s} mean pairwise cosine {cosmean(V):.4f}")

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
    log(f"  FIT {base}/{txt}/{lab}/{mk}: {X.shape[1]} cols, "
        f"AUROC {roc_auc_score(LABELS[lab][~tr], s[~tr]):.4f}")
    return s

ARMS = ["bow", "lsa_wide_300", f"{M}_mean",
        f"{M}_max_tok", f"{M}_max_win", f"{M}_max_tok_300"]
PLAN = [(b, t, l, mk) for mk in LEARNERS for b in ["T1", "T5"]
        for l in ["composite", "death"] for t in ARMS]

rows = []
for base, txt, lab, mk in PLAN:
    y = LABELS[lab]
    b = get(base, "none", lab, mk); s = get(base, txt, lab, mk)
    d, lo, hi, p = paired(y[~tr], b[~tr], s[~tr])
    a0 = roc_auc_score(y[~tr], b[~tr])
    rows.append(dict(model=mk, base=base, outcome=lab, text=txt,
                     struct=round(a0, 4), with_text=round(roc_auc_score(y[~tr], s[~tr]), 4),
                     increment=round(d, 4), ci=f"{lo:+.4f},{hi:+.4f}", p=p))
    pd.DataFrame(rows).to_csv(f"{OUT}/W1_maxpool.csv", index=False)

R = pd.DataFrame(rows); pd.set_option("display.width", 220)
print("\n=== increment over structured-only ===")
print(R.pivot_table(index="text", columns=["base", "outcome", "model"],
                    values="increment").to_string())
print("\n=== THE TEST: does max-pooling close the gap to bow, and only under GBM? ===")
for base in ["T1", "T5"]:
    for lab in ["composite", "death"]:
        q = R[(R.base == base) & (R.outcome == lab)].set_index(["model", "text"])
        for mk in LEARNERS:
            bw, mn = q.loc[(mk, "bow"), "increment"], q.loc[(mk, f"{M}_mean"), "increment"]
            mx = max(q.loc[(mk, f"{M}_max_tok"), "increment"],
                     q.loc[(mk, f"{M}_max_win"), "increment"],
                     q.loc[(mk, f"{M}_max_tok_300"), "increment"])
            gap = bw - mn
            closed = 100*(mx - mn)/gap if abs(gap) > 1e-9 else float("nan")
            print(f"  {base}/{lab:9s}/{mk:3s}  bow {bw:+.4f}  mean {mn:+.4f}  "
                  f"best-max {mx:+.4f}   gap {gap:+.4f}  closed {closed:5.0f}%")
log("done")
