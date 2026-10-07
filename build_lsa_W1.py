"""LSA text representations for W1 — is the 500-feature SELECTION the bottleneck,
or is it the information in the notes?

The saturation result (RESULTS_saturation.md) is measured with one text
representation: 21 closed-vocabulary counts + 500 open-vocabulary counts chosen
by L1. TEXT_W1.md section 4.3 shows that block is effectively bag-of-unigrams —
84% of the selected features and 85% of the coefficient weight are single words.
So "text adds little against a rich baseline" is, strictly, a claim about that
representation.

LSA is the cheapest possible probe of the next question. It needs no external
model weights, and it changes exactly one thing: instead of SELECTING 500 terms
it COMPRESSES the whole vocabulary into k dense components. If a dense
representation of the same words behaves like the sparse selection, then
dimensionality and term selection are not what is limiting the text block, and
the case for trying a contextual encoder rests on contextual meaning rather than
on capacity.

TWO VARIANTS
  A. lsa_pool  — the SAME 4,687-phrase candidate pool the L1 selector chose from
     (1-3 grams, df >= 5% in training). Vocabulary held fixed, so this isolates
     dense-compression vs sparse-selection.
  B. lsa_wide  — a much WIDER unigram vocabulary (df >= 0.5% in training,
     ~7x more terms). Tests whether the 5% document-frequency floor, not the
     selection, is what discards the signal. Unigrams only: the apriori bigram
     pass at 0.5% does not fit in this shell's 3.9 GB.

Both are fit on TRAINING ROWS ONLY — vocabulary, IDF weights and the SVD basis —
then applied to all 25,692 admissions. k = 500 components are computed once and
sliced to 100 / 300 / 500 downstream, since TruncatedSVD orders them.

Corpus is v2 (discharge summaries removed). Run:

    python build_lsa_W1.py
"""
import os, sys, time, json
os.environ["W1_TEXT_V2"] = "1"
from collections import Counter
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import TruncatedSVD
import build_text_W1 as B

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
OUT = B.OUT
K = 500
WIDE_DF = 0.005

meta = pd.read_parquet(B.META)
order = np.sort(meta.hadm_id.to_numpy())
train_ids = set(meta.hadm_id[meta.split == "train"])
ntr = len(train_ids)
log(f"{len(order):,} admissions, {ntr:,} training")

def lsa(vocab, tag, k=K):
    vi = {p: i for i, p in enumerate(vocab)}
    log(f"[{tag}] DTM over {len(vocab):,} terms ...")
    X, hs = B._dtm(vi)                      # all rows, counts
    o = np.argsort(hs); X = X[o]; hs = hs[o]
    if len(hs) != len(order) or not (hs == order).all():   # v2 leaves 372 note-less
        pos = {h: i for i, h in enumerate(hs)}
        idx = np.array([pos.get(h, -1) for h in order]); have = idx >= 0
        Z = sparse.lil_matrix((len(order), X.shape[1]), dtype=X.dtype)
        Z[np.flatnonzero(have)] = X[idx[have]]
        X = Z.tocsr()
    tr = np.isin(order, list(train_ids))
    log(f"[{tag}] nnz {X.nnz:,}; fitting TF-IDF and SVD(k={k}) on TRAIN only ...")
    tf = TfidfTransformer(sublinear_tf=True).fit(X[tr])
    Xt = tf.transform(X)
    svd = TruncatedSVD(n_components=k, random_state=B.SEED, algorithm="randomized")
    svd.fit(Xt[tr])
    Z = svd.transform(Xt).astype(np.float32)
    ev = svd.explained_variance_ratio_
    np.save(f"{OUT}/lsa_{tag}_W1.npy", Z)
    log(f"[{tag}] lsa_{tag}_W1.npy {Z.shape}; explained variance "
        f"k=100 {ev[:100].sum():.3f}  k=300 {ev[:300].sum():.3f}  k={k} {ev.sum():.3f}")
    return dict(tag=tag, terms=len(vocab), k=k, nnz=int(X.nnz),
                ev100=float(ev[:100].sum()), ev300=float(ev[:300].sum()),
                ev500=float(ev.sum()))

info = []

# ---- A. the same pool the L1 selector chose from -------------------------
pool = pd.read_csv(B.VOCAB).phrase.astype(str).tolist()
info.append(lsa(pool, "pool"))

# ---- B. a wider unigram vocabulary ---------------------------------------
log(f"building the wide unigram vocabulary (df >= {100*WIDE_DF:.1f}% of train) ...")
df1 = Counter()
for d in B._iter_docs(train_ids):
    for doc in d.doc.to_numpy(): df1.update(set(doc.split()))
thr = WIDE_DF * ntr
wide = sorted(w for w, c in df1.items() if c >= thr)
log(f"  {len(df1):,} distinct unigrams -> {len(wide):,} at df >= {100*WIDE_DF:.1f}% "
    f"(the 5% floor kept {sum(1 for w,c in df1.items() if c >= 0.05*ntr):,})")
del df1
info.append(lsa(wide, "wide"))

json.dump(info, open(f"{OUT}/W1_lsa_meta.json", "w"), indent=2)
pd.DataFrame({"term": wide}).to_csv(f"{OUT}/W1_lsa_wide_vocab.csv", index=False)
print("\n", pd.DataFrame(info).to_string(index=False))
log("done")
