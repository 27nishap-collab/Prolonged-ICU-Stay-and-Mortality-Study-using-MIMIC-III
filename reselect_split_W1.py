"""Check (b): re-select the 500 bag-of-words phrases inside each partition.

    # step 1 -- ON THE MAC, needs the cleaned note text (never leaves the Mac).
    #           Run only when finetune_split_W1.py is NOT running (memory).
    python3.11 reselect_split_W1.py dtm        # -> out/text_full_W1_v2.npz

    # steps 2-3 -- derived counts only; run anywhere (the Mac or the cloud workspace)
    python3.11 reselect_split_W1.py select     # -> out/w1_ftsplit_v2_observable/bow_rs_draw{r}.npz
    W1_SELECT_ON=train python3.11 reselect_split_W1.py select   # -> .../bow_rt_draw{r}.npz
    python3.11 reselect_split_W1.py fit        # -> .../W1_reselect_split.csv

WHY. The 500 phrases in the bag-of-words block were chosen once, by an L1
logistic regression on the ORIGINAL split's 19,269 training admissions
(build_text_W1.py step_open). Every later partition is a fresh 75/25 draw, so
about 75% of each partition's test patients (4,801-4,856 of 6,423 in Design 3)
were among the admissions whose labels chose the phrases. That can only flatter
bag-of-words, by an unmeasured amount -- and Design 3 compares it head to head
with a fine-tuned encoder that saw no test labels at all.

WHAT. Same selector, re-run inside each Design 3 partition on HALF A only -- the
patients the fine-tuned encoder learned from -- so bag-of-words and the encoder
learn their text representation from the same patients, and every downstream
model is fitted on half B and scored on the untouched test set.
  * candidate pool: the same 4,687 phrases (1-3 grams with document frequency
    >= 5% on the original training documents). That threshold uses no labels,
    so the pool itself leaks nothing and is kept fixed.
  * selector: log1p counts, L1 logistic regression (liblinear), C grid
    {0.005, 0.01, 0.05, 0.1} scored by 10-fold CV AUROC, C = the best grid value
    that keeps >= 500 non-zero phrases, then the top 500 by |coefficient|; raw
    counts shipped downstream -- the rule step_open and run_reselect_W1.py use.
  * downstream: bow (original 500) vs bow_rs (re-selected 500), each with the 22
    fixed text features, fixed GBM and tuned LR, at T1, T5 and text only.

SECOND ARM, added 2026-09-29 (W1_SELECT_ON=train -> "bow_rt"). Selecting on half A
changes two things at once against the original phrases: no test labels, AND half
the patients (~9,600 vs 19,269). bow_rt selects on each partition's FULL training
set -- 19,269 admissions, the same number as the original choice, none of them test
patients -- so it changes only the leak. Same selector, same C grid.
  original -> bow_rt   = the leak alone
  bow_rt   -> bow_rs   = the sample-size effect
Downstream is unchanged: fitted on half B, scored on test. (bow_rt's phrases saw
half B's labels, as the original phrases did; bow_rs's did not.)

READ-OFF, fixed before any result: bow_rs minus bow, paired within partition, at
T1 and T5. A drop means the original selection flattered bag-of-words; the
fine-tuned-vs-bag-of-words comparison should then use bow_rs.
"""
import os, sys, json, glob, time, warnings
from collections import Counter
import numpy as np, pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
OUTD = os.environ.get("W1_OUT", f"{U}/w1_ftsplit_v2_observable")
SEED = 20180701
MODE = sys.argv[1] if len(sys.argv) > 1 else ""
assert MODE in ("dtm", "select", "fit"), "mode: dtm | select | fit"
DRAWS = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
SELECT_ON = os.environ.get("W1_SELECT_ON", "A"); assert SELECT_ON in ("A", "train"), SELECT_ON
TAGSEL = {"A": "bow_rs", "train": "bow_rt"}[SELECT_ON]
# corpus switch (2026-09-29): W1_CORPUS=v1 builds the notes-all DTM for the replication-spec
# companion curve. Only the dtm step supports v1; select/fit are Design 3 (v2) only.
CORPUS = os.environ.get("W1_CORPUS", "v2"); assert CORPUS in ("v1", "v2"), CORPUS
CSUF = "_v2" if CORPUS == "v2" else ""
assert CORPUS == "v2" or MODE == "dtm", "only the dtm step supports W1_CORPUS=v1"
FULL = f"{U}/text_full_W1{CSUF}.npz"
SHIPPED = f"{U}/text_open_W1_v2_aligned.npz" if CORPUS == "v2" else f"{U}/text_open_W1.npz"
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)

tab = pd.read_parquet(f"{U}/model_tabular_W1{CSUF}.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
y = tab.labels.to_numpy().astype(int); N = len(y)

# ---------------------------------------------------------------- step 1: full DTM (Mac only)
def dtm():
    """Counts of all 4,687 candidate phrases for every admission, in tabular order.
    Same tokenisation as build_text_W1._dtm: the cleaned documents are already
    space-separated tokens; 1-, 2- and 3-grams are counted against the pool."""
    vocab = pd.read_csv(f"{U}/W1_openvocab_full{CSUF}.csv").phrase.astype(str).tolist()
    vi = {p: i for i, p in enumerate(vocab)}
    parts = sorted(glob.glob(f"{U}/text_parts{CSUF}/*.parquet")); assert parts
    indptr, indices, data, hs = [0], [], [], []
    for f in parts:
        d = pd.read_parquet(f, columns=["hadm_id", "doc"])
        for h, doc in zip(d.hadm_id.to_numpy(), d.doc.to_numpy()):
            tk = doc.split(); c = Counter()
            for j, w in enumerate(tk):
                k = vi.get(w)
                if k is not None: c[k] += 1
                if j + 1 < len(tk):
                    k = vi.get(w + " " + tk[j+1])
                    if k is not None: c[k] += 1
                if j + 2 < len(tk):
                    k = vi.get(w + " " + tk[j+1] + " " + tk[j+2])
                    if k is not None: c[k] += 1
            indices.extend(c.keys()); data.extend(c.values())
            indptr.append(len(indices)); hs.append(h)
        log(f"  {os.path.basename(f)}: {len(hs):,} documents so far")
    X = sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32),
                           np.asarray(indptr, np.int64)), shape=(len(hs), len(vocab)))
    hs = np.asarray(hs)
    # align to the tabular order; admissions with no in-window note get an empty row
    pos = pd.Index(hs).get_indexer(order); have = pos >= 0
    Z = sparse.lil_matrix((N, len(vocab)), dtype=np.float32)
    Z[np.flatnonzero(have)] = X[pos[have]]
    Z = Z.tocsr(); sparse.save_npz(FULL, Z)
    log(f"{FULL}: {Z.shape}, nnz {Z.nnz:,}; {int((~have).sum())} admissions without notes")
    # sanity: the original 500 columns must reproduce text_open_W1_v2_aligned.npz exactly
    old = sparse.load_npz(SHIPPED)
    sel = pd.read_csv(f"{U}/text_open_vocab_W1{CSUF}.csv").phrase.astype(str).tolist()
    diff = abs(Z[:, [vi[p] for p in sel]] - old).max()
    log(f"sanity: max |rebuilt - shipped| over the original 500 columns = {diff}")
    assert diff == 0, "rebuilt counts do not match the shipped bag-of-words -- stop"

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(N, bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def halves(r, tr):
    z = np.load(f"{U}/w1_splitsample_v2_observable/halves_draw{r}.npz"); A, B = z["A"], z["B"]
    assert not (A & B).any() and ((A | B) == tr).all()
    return A, B

# ---------------------------------------------------------------- step 2: select on half A
def select():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    Z = sparse.load_npz(FULL).tocsr()
    vocab = pd.read_csv(f"{U}/W1_openvocab_full_v2.csv").phrase.astype(str).tolist()
    orig = set(pd.read_csv(f"{U}/text_open_vocab_W1_v2.csv").phrase.astype(str))
    CS = [0.005, 0.01, 0.05, 0.1]
    for r in DRAWS:
        f = f"{OUTD}/{TAGSEL}_draw{r}.npz"
        if os.path.exists(f): log(f"draw {r}: {TAGSEL} selection on disk, skipping"); continue
        tr = draw_split(r); A, B = halves(r, tr)
        rows = np.flatnonzero(A if SELECT_ON == "A" else tr)
        assert not (~tr[rows]).any(), "selection rows include test patients"
        log(f"draw {r}: selecting {TAGSEL} on {len(rows):,} admissions")
        Xs = Z[rows].copy(); Xs.data = np.log1p(Xs.data); ya = y[rows]
        skf = StratifiedKFold(10, shuffle=True, random_state=SEED); path = []
        for C in CS:
            au = []
            for a, b in skf.split(Xs, ya):
                m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000).fit(Xs[a], ya[a])
                au.append(roc_auc_score(ya[b], m.decision_function(Xs[b])))
            nz = int((LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
                      .fit(Xs, ya).coef_[0] != 0).sum())
            path.append(dict(C=C, cv_auc=float(np.mean(au)), nonzero=nz))
            log(f"  draw {r} C={C:<6g} cv AUROC {np.mean(au):.4f} nonzero {nz:,}")
        p = pd.DataFrame(path); ok = p[p.nonzero >= 500]
        C = float(ok.loc[ok.cv_auc.idxmax()].C) if len(ok) else float(p.C.max())
        coef = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=4000).fit(Xs, ya).coef_[0]
        keep = np.argsort(-np.abs(coef))[:500]; keep = keep[coef[keep] != 0]
        sel = [vocab[i] for i in keep]
        sparse.save_npz(f, Z[:, keep].tocsr())
        json.dump(dict(draw=r, C=C, n=len(sel), overlap_with_original=len(orig & set(sel)),
                       phrases=sel, path=path), open(f.replace(".npz", ".json"), "w"), indent=1)
        log(f"draw {r}: C={C:g}, {len(sel)} phrases, {len(orig & set(sel))} shared with the original 500")

# ---------------------------------------------------------------- step 3: downstream on half B
def _midrank(x):
    J = np.argsort(x); Zs = x[J]; n = len(x); T_ = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and Zs[j] == Zs[i]: j += 1
        T_[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(n); T2[J] = T_; return T2

def paired(yt, a, b):
    s = np.vstack([a, b]).astype(float); o = np.argsort(-yt, kind="mergesort"); s = s[:, o]
    m = int(yt.sum()); n = len(yt) - m
    tx = np.vstack([_midrank(s[k, :m]) for k in range(2)])
    ty = np.vstack([_midrank(s[k, m:]) for k in range(2)])
    tz = np.vstack([_midrank(s[k]) for k in range(2)])
    au = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    c = np.cov((tz[:, :m]-tx)/n)/m + np.cov(1.0-(tz[:, m:]-ty)/m)/n
    return float(au[0]), float(au[1]), float(au[1]-au[0]), float(np.sqrt(max(c[0,0]+c[1,1]-2*c[0,1], 0)))

def fit():
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.metrics import roc_auc_score
    from lr_tuned import lr_tuned_scores
    GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
    tiers = json.load(open(f"{U}/W1_tier_columns_observable.json"))
    T = pd.read_parquet(f"{U}/tiers_W1_observable.parquet").set_index("hadm_id").reindex(order)
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
    BOW = np.hstack([FIX, sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)])

    def encode_tier(cols, fitr):
        out = []
        for c in cols:
            s = T[c]
            if not pd.api.types.is_numeric_dtype(s):
                v = s.astype(object).where(s.notna(), "__na__")
                k = v[fitr].value_counts(); lv = set(k[k >= 0.01*fitr.sum()].index)
                v = v.where(v.isin(lv), "__other__")
                for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
            else: out.append(s.to_numpy(np.float32))
        return np.column_stack(out)

    def score(X, fitr, mk):
        if mk == "gbm":
            return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **GBM
                                                  ).fit(X[fitr], y[fitr]).decision_function(X)
        return lr_tuned_scores(X, fitr, y)[0]

    CSV = f"{OUTD}/W1_reselect_split.csv"
    COLS = ["draw", "tier", "model", "arm", "auroc", "struct", "with_text", "increment", "se"]
    done = set()
    if os.path.exists(CSV):
        o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.model, o.arm))
    for r in DRAWS:
        tr = draw_split(r); A, B = halves(r, tr); te = ~tr
        ARMS = {"bow": BOW}
        for t_ in ("bow_rs", "bow_rt"):
            if os.path.exists(f"{OUTD}/{t_}_draw{r}.npz"):
                ARMS[t_] = np.hstack([FIX, sparse.load_npz(f"{OUTD}/{t_}_draw{r}.npz").toarray().astype(np.float32)])
        for tier in ("none", "T1", "T5"):
            S = None if tier == "none" else encode_tier(tiers[tier], B)
            for mk in ("gbm", "lrcv"):
                todo = [a for a in ARMS if (r, tier, mk, a) not in done]
                if not todo: continue
                sb = None if S is None else score(S, B, mk)
                for arm, X in ARMS.items():
                    if (r, tier, mk, arm) in done: continue
                    Xf = X if S is None else np.hstack([S, X])
                    s = score(Xf, B, mk)
                    row = dict(draw=r, tier=tier, model=mk, arm=arm)
                    if S is None:
                        row.update(auroc=round(roc_auc_score(y[te], s[te]), 4))
                    else:
                        a, b, d, se = paired(y[te], sb[te], s[te])
                        row.update(struct=round(a, 4), with_text=round(b, 4), increment=round(d, 4), se=round(se, 4))
                    pd.DataFrame([row]).reindex(columns=COLS).to_csv(
                        CSV, mode="a", header=not os.path.exists(CSV), index=False)
                    log(f"  draw {r} {tier:4s} {mk:4s} {arm:6s}: {row}")
    res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"], keep="last")
    print("\n=== text-only AUROC ===")
    print(res[res.tier == "none"].pivot_table(index="arm", columns="model", values="auroc").round(4).to_string())
    q = res[res.tier != "none"].pivot_table(index=["model", "draw", "tier"], columns="arm", values="increment")
    print("\n=== increments, paired within partition ===")
    print(q.round(4).to_string())
    for a_, b_ in (("bow_rt", "bow"), ("bow_rs", "bow_rt"), ("bow_rs", "bow")):
        if a_ in q and b_ in q:
            d_ = (q[a_] - q[b_]).rename(f"{a_} - {b_}")
            print(f"\n{a_} - {b_}:\n" + d_.groupby(level=["model", "tier"]).agg(["mean", "min", "max", "count"]).round(4).to_string())

{"dtm": dtm, "select": select, "fit": fit}[MODE]()
log("done")
