"""Attribution net of the 22 fixed text features — the F13 table.

Every arm in fit_bert_W1.py includes FIX (21 closed-vocabulary counts +
n_words_raw) alongside the open-vocabulary block, so the increments there credit
the open-vocab block with work the 22 fixed features did. This measures the
open-vocab block ALONE, against `structured + fix22` as the base.

    W1_SPEC=observable python3 fit_attrib_W1.py

Resumable: one .npy score vector per cell.
"""
import os, sys, json, time, glob, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.decomposition import TruncatedSVD

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
B = U                                   # frozen-encoder shards: out/bert_<model>_w0-4/
BAND = "w0-4"
SEED = 20180701

SPEC = os.environ.get("W1_SPEC", "observable")
SPEC = {"leakfree": "observable", "published": "replication"}.get(SPEC, SPEC)
assert SPEC in ("observable", "replication"), SPEC
SFX = "" if SPEC == "replication" else "_observable"

OUT = f"{U}/w1_attrib" + SFX; os.makedirs(OUT, exist_ok=True)
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
    return float(au[0]), float(au[1]), d, d-1.96*se, d+1.96*se, float(2*stats.norm.sf(abs(z)))

# ---------- data, identical to fit_bert_W1.py ----------
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order = tab.hadm_id.to_numpy(); T = T.set_index("hadm_id").reindex(order)
tr = (tab.split == "train").to_numpy()
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
WC  = tab[["n_words_raw"]].to_numpy(np.float32)
Xo  = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)
Lw  = np.load(f"{U}/lsa_wide_W1.npy")[:, :300].astype(np.float32)
log(f"spec={SPEC}  FIX {FIX.shape[1]} cols (of which {len(cvc)} closed-vocab)  bow {Xo.shape[1]}")

OPEN = {"bow": Xo, "lsa_wide_300": Lw}
for m in ["bio_clinicalbert", "pubmedbert"]:
    fs = sorted(glob.glob(f"{B}/bert_{m}_{BAND}/b_*.npz"))
    g = lambda k: np.concatenate([np.load(f)[k] for f in fs])
    hd, st, nt = g("hadm"), g("s_tok"), g("n_tok")
    pos = pd.Index(hd).get_indexer(order)
    assert not (pos < 0).any(), f"{m}: {int((pos<0).sum())} admissions unencoded"
    MEAN = (st[pos] / np.maximum(nt[pos], 1)[:, None]).astype(np.float32)
    OPEN[f"{m}_mean"] = MEAN
    log(f"{m}_mean {MEAN.shape}")

def encode(cols):
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

S = {t: encode(tiers[t]) for t in ("T1", "T5")}
Y = {"composite": tab.labels.to_numpy().astype(int),
     "death": T.outcome_death.to_numpy().astype(int)}
log(f"outcomes: " + ", ".join(f"{k} {v.mean():.4f}" for k, v in Y.items()))

def fit(tag, X, y):
    f = f"{OUT}/s_{tag}.npy"
    if os.path.exists(f): return np.load(f)
    s = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                       **GBM).fit(X[tr], y[tr]).decision_function(X)
    np.save(f, s); log(f"  {tag}: {X.shape[1]} cols"); return s

rows = []
for tier in ("T1", "T5"):
    for oc, y in Y.items():
        # base 1: structured only            base 2: structured + fix22
        s_struct = fit(f"{tier}_{oc}_struct", S[tier], y)
        s_fix    = fit(f"{tier}_{oc}_fix22", np.hstack([S[tier], FIX]), y)
        s_wc     = fit(f"{tier}_{oc}_wconly", np.hstack([S[tier], WC]), y)
        a, b, d, lo, hi, p = paired(y[~tr], s_struct[~tr], s_fix[~tr])
        rows.append(dict(tier=tier, outcome=oc, base="structured", block="fix22",
                         auc_base=round(a,4), auc_with=round(b,4), increment=round(d,4),
                         ci=f"{lo:+.4f},{hi:+.4f}", p=p))
        a2,b2,d2,lo2,hi2,p2 = paired(y[~tr], s_struct[~tr], s_wc[~tr])
        rows.append(dict(tier=tier, outcome=oc, base="structured", block="n_words_raw_only",
                         auc_base=round(a2,4), auc_with=round(b2,4), increment=round(d2,4),
                         ci=f"{lo2:+.4f},{hi2:+.4f}", p=p2))
        for nm, blk in OPEN.items():
            s = fit(f"{tier}_{oc}_{nm}", np.hstack([S[tier], FIX, blk]), y)
            a, b, d, lo, hi, p = paired(y[~tr], s_fix[~tr], s[~tr])
            rows.append(dict(tier=tier, outcome=oc, base="structured+fix22", block=nm,
                             auc_base=round(a,4), auc_with=round(b,4), increment=round(d,4),
                             ci=f"{lo:+.4f},{hi:+.4f}", p=p))
        log(f"{tier}/{oc} done")

R = pd.DataFrame(rows); R.to_csv(f"{OUT}/W1_attrib.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== open-vocabulary block ALONE, net of the 22 fixed text features (GBM) ===")
o = R[R.base == "structured+fix22"].pivot_table(index="block", columns=["tier","outcome"],
                                                values="increment")
print(o.to_string())
print("\n=== what the 22 fixed features are worth, and how much of that is word count ===")
print(R[R.base == "structured"].to_string(index=False))
log("done")
