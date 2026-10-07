"""LSA and frozen PubMedBERT on the split-sample design (2026-10-04).

RUN ON THE MAC, native python3.11. In order:

    python3.11 splitsample_more_W1.py check            # ~1 min: what exists, what is missing
    python3.11 splitsample_more_W1.py dtm_wide         # reads the cleaned notes (Mac only) -> out/text_wide_W1_v2.npz
    python3.11 splitsample_more_W1.py lsa              # TF-IDF + SVD per draw, fitted on half A -> lsa_draw{r}.npz
    W1_ARMS=bow_rs,lsa_pool_300,lsa_wide_300 python3.11 splitsample_more_W1.py run 2>&1 | tee ssmore_lsa.log

  PubMedBERT needs every window tokenised first (reads raw notes; stays on the Mac):
    W1_MAXWIN=all W1_TOKMODELS=pubmedbert python3 tokenize_W1.py         # repeat until ALL PARTS DONE
    W1_MAXWIN=all W1_TOKMODELS=pubmedbert python3 tokenize_W1.py merge   # -> out/tokall_pubmedbert_W1.npz
    caffeinate -dis env HF_HUB_OFFLINE=1 W1_SMOKE=1 python3.11 splitsample_more_W1.py encode   # timing, writes nothing
    caffeinate -dis env HF_HUB_OFFLINE=1 python3.11 splitsample_more_W1.py encode 2>&1 | tee ssmore_enc.log
    W1_ARMS=pubmedbert_tok python3.11 splitsample_more_W1.py run 2>&1 | tee ssmore_pmb.log

    python3.11 splitsample_more_W1.py summary

WHY. The table of T1/T5 increments on the split-sample design (finetune_full_W1.py,
out/w1_ftfull_v2_observable/) has bag-of-words, frozen Bio_ClinicalBERT and fine-tuned
Bio_ClinicalBERT, but not LSA or PubMedBERT: those two were only ever fitted on the
original single split (fit_lsa_W1.py, fit_bert_W1.py). This adds them on the same five
partitions, the same halves, the same 22 fixed text features, the same downstream
learners fitted on half B and scored on the untouched test set.

WHAT EACH NEW ARM IS, and what is held fixed against the earlier single-split versions:
  * lsa_pool_300 -- 300 SVD components of the SAME 4,687-phrase candidate pool
    (text_full_W1_v2.npz, the counts reselect_split_W1.py made and checked).
  * lsa_wide_300 -- 300 SVD components of the SAME 6,701-term unigram vocabulary
    (W1_lsa_wide_vocab.csv).
    Both vocabularies were fixed by document frequency on the original training
    documents, without labels, so -- as with the bag-of-words candidate pool -- they
    are kept. What is re-fitted per partition is the TF-IDF weighting and the SVD
    basis, on HALF A (the patients the fine-tuned encoder and bow_rs learn from),
    k = 500 then sliced to 300, exactly as build_lsa_W1.py. W1_LSA_ON=train fits on
    the partition's whole training set instead. Neither uses a label or a test row.
  * pubmedbert_tok -- frozen PubMedBERT, token mean over EVERY 510-wordpiece window,
    the same readout as frozen_tok (Bio_ClinicalBERT) in finetune_full_W1.py. The
    single-split version read at most 4 windows; this one reads all of them, so it is
    like for like with the Bio_ClinicalBERT rows of the table.
  * bow_rs -- refitted as an ANCHOR: it must reproduce W1_ftfull.csv to 4 decimals.
    If it does, the halves, tiers, fixed features and learners are the same as the
    run the new arms are compared with. summary prints the check.

READ-OFF, fixed before any result: T1 and T5 increments and the collapse, 95% t CI over
the five partitions, for each new arm next to bow_rs / frozen_tok / ft_tok from
W1_ftfull.csv; each arm minus bow_rs, paired within partition. Same machine as
W1_ftfull.csv; never compare across machines.

Outputs in out/w1_ssmore_v2_observable/ (gitignored with the rest of out/):
  lsa_draw{r}.npz, frozen_pubmedbert_all.npz, W1_ssmore.csv, W1_ssmore_table.csv
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); U = os.environ.get("W1_DATA", os.path.join(D, "out"))
sys.path.insert(0, D)

MODE = sys.argv[1] if len(sys.argv) > 1 else ""
MODES = ("check", "dtm_wide", "lsa", "encode", "run", "summary")
assert MODE in MODES, f"mode: {' | '.join(MODES)}"

SEED = 20180701
GBM  = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
DRAWS = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
TIERS, LEARNERS = ["T1", "T5"], ["gbm", "lrcv"]
ALL_ARMS = ["bow_rs", "lsa_pool_300", "lsa_wide_300", "pubmedbert_tok"]
ARMS = os.environ.get("W1_ARMS", ",".join(ALL_ARMS)).split(",")
assert set(ARMS) <= set(ALL_ARMS), ARMS
K, KUSE = 500, 300
LSA_ON = os.environ.get("W1_LSA_ON", "A"); assert LSA_ON in ("A", "train"), LSA_ON
TAG, HF = "pubmedbert", "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"
BATCH, CKPT_SEC = 8, 600
SMOKE = os.environ.get("W1_SMOKE") == "1"; SMOKE_ENC = 400

SFX, SUF = "_observable", "_v2"
HALVES = f"{U}/w1_splitsample{SUF}{SFX}"
BOWD   = f"{U}/w1_ftsplit{SUF}{SFX}"
FTFULL = f"{U}/w1_ftfull{SUF}{SFX}/W1_ftfull.csv"
POOL   = f"{U}/text_full_W1{SUF}.npz"
WIDE   = f"{U}/text_wide_W1{SUF}.npz"
WVOCAB = f"{U}/W1_lsa_wide_vocab.csv"
TOKF   = f"{U}/tokall_{TAG}_W1.npz"
OUT    = f"{U}/w1_ssmore{SUF}{SFX}"; os.makedirs(OUT, exist_ok=True)
CSV, TABLE = f"{OUT}/W1_ssmore.csv", f"{OUT}/W1_ssmore_table.csv"
FROZEN, PARTIAL = f"{OUT}/frozen_{TAG}_all.npz", f"{OUT}/frozen_{TAG}_partial.npz"
LSAF = lambda r: f"{OUT}/lsa{'' if LSA_ON == 'A' else '_train'}_draw{r}.npz"
COLS = ["draw", "tier", "n_vars", "model", "arm", "n_fit", "ev_fit", "struct", "with_text", "increment", "se"]

t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/3600:5.2f}h] {m}", flush=True)
def append(path, row):
    pd.DataFrame([row]).reindex(columns=COLS).to_csv(path, mode="a", header=not os.path.exists(path), index=False)

# ---------------------------------------------------------------- DeLong (as finetune_full_W1.py)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; n = len(x); T_ = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
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

# ---------------------------------------------------------------- shared data (as finetune_full_W1.py)
tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
y = tab.labels.to_numpy().astype(int); N = len(y)

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(N, bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def halves(r, tr):
    f = f"{HALVES}/halves_draw{r}.npz"; assert os.path.exists(f), f
    z_ = np.load(f); A, B = z_["A"], z_["B"]
    assert len(A) == N and not (A & B).any() and ((A | B) == tr).all()
    return A, B

def fixed22():
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    return np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])

# ---------------------------------------------------------------- check
def check():
    def have(p, how): log(f"  {'OK     ' if os.path.exists(p) else 'MISSING'} {os.path.relpath(p, U)}"
                          + ("" if os.path.exists(p) else f"   <- {how}"))
    log(f"{N:,} admissions; draws {DRAWS}; arms {ARMS}; LSA fitted on {LSA_ON}")
    for r in DRAWS:
        A, B = halves(r, draw_split(r)); log(f"draw {r}: A {A.sum():,}  B {B.sum():,}  test {N - A.sum() - B.sum():,}")
        have(f"{BOWD}/bow_rs_draw{r}.npz", "reselect_split_W1.py select")
        have(LSAF(r), "splitsample_more_W1.py lsa")
    have(FTFULL, "finetune_full_W1.py run")
    have(POOL, "reselect_split_W1.py dtm"); have(WVOCAB, "build_lsa_W1.py")
    have(WIDE, "splitsample_more_W1.py dtm_wide")
    have(TOKF, "tokenize_W1.py with W1_MAXWIN=all W1_TOKMODELS=pubmedbert, then merge")
    have(os.path.join(D, "hf", TAG, "vocab.txt"), "the PubMedBERT tokenizer files")
    have(FROZEN, "splitsample_more_W1.py encode")
    if os.path.exists(PARTIAL): log(f"  partial encoding on disk -- encode will resume from it")
    if os.path.exists(TOKF):
        z = np.load(TOKF); log(f"  {TAG}: {int(z['nwin'].sum()):,} windows; every window stored: {bool((z['nwin'] == z['nfull']).all())}")
    log("check done")

# ---------------------------------------------------------------- wide unigram counts (Mac only: reads the cleaned notes)
# 2026-10-04: reads out/text_parts_v2/ directly (as reselect_split_W1.py dtm) rather than importing
# build_text_W1, which needs duckdb; same files, same whitespace tokens, unigrams only.
def dtm_wide():
    import glob
    from collections import Counter
    vocab = pd.read_csv(WVOCAB).term.astype(str).tolist(); vi = {w: i for i, w in enumerate(vocab)}
    log(f"unigram counts over {len(vocab):,} terms (the vocabulary build_lsa_W1.py fixed) ...")
    parts = sorted(glob.glob(f"{U}/text_parts{SUF}/*.parquet")); assert parts, "no cleaned note parts"
    indptr, indices, data, hs = [0], [], [], []
    for f in parts:
        d = pd.read_parquet(f, columns=["hadm_id", "doc"])
        for h, doc in zip(d.hadm_id.to_numpy(), d.doc.to_numpy()):
            c = Counter(k for k in map(vi.get, doc.split()) if k is not None)
            indices.extend(c.keys()); data.extend(c.values()); indptr.append(len(indices)); hs.append(h)
        log(f"  {os.path.basename(f)}: {len(hs):,} documents so far")
    X = sparse.csr_matrix((np.asarray(data, np.float32), np.asarray(indices, np.int32),
                           np.asarray(indptr, np.int64)), shape=(len(hs), len(vocab)))
    hs = np.asarray(hs)
    pos = pd.Index(hs).get_indexer(order); have = pos >= 0
    Z = sparse.lil_matrix((N, len(vocab)), dtype=np.float32)
    Z[np.flatnonzero(have)] = X[pos[have]]
    Z = Z.tocsr(); sparse.save_npz(WIDE, Z)
    log(f"{WIDE}: {Z.shape}, nnz {Z.nnz:,}; {int((~have).sum())} admissions without notes "
        f"[build_lsa_W1.py recorded nnz 11,673,075]")

# ---------------------------------------------------------------- LSA per draw
def lsa():
    from sklearn.feature_extraction.text import TfidfTransformer
    from sklearn.decomposition import TruncatedSVD
    for p in (POOL, WIDE): assert os.path.exists(p), f"{p} missing -- see check"
    X = {"pool": sparse.load_npz(POOL).tocsr(), "wide": sparse.load_npz(WIDE).tocsr()}
    for r in DRAWS:
        if os.path.exists(LSAF(r)): log(f"draw {r}: {os.path.basename(LSAF(r))} exists, skipping"); continue
        tr = draw_split(r); A, B = halves(r, tr); fit = A if LSA_ON == "A" else tr
        out = {}
        for tag, M in X.items():
            tf = TfidfTransformer(sublinear_tf=True).fit(M[fit]); Mt = tf.transform(M)
            svd = TruncatedSVD(n_components=K, random_state=SEED, algorithm="randomized").fit(Mt[fit])
            out[tag] = svd.transform(Mt)[:, :KUSE].astype(np.float32)
            ev = svd.explained_variance_ratio_
            log(f"draw {r} {tag}: fitted on {fit.sum():,} rows; explained variance k=300 {ev[:300].sum():.3f}")
        np.savez(LSAF(r), **out); log(f"draw {r}: -> {LSAF(r)}")

# ---------------------------------------------------------------- frozen PubMedBERT, every window, resumable
def encode():
    import torch
    if not torch.backends.mps.is_available(): sys.exit("MPS unavailable. Run with native python3.11 on the Mac.")
    DEV = torch.device("mps")
    from transformers import AutoModel
    assert os.path.exists(TOKF), f"{TOKF} missing -- see check"
    if os.path.exists(FROZEN) and not SMOKE: log(f"{FROZEN} exists, nothing to do"); return
    z = np.load(TOKF); ids, wlen, nwin, nfull, hadm = z["ids"], z["wlen"], z["nwin"], z["nfull"], z["hadm"]
    assert (nwin == nfull).all(), "tokall file does not hold every window"
    woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])
    aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])
    tok_pos = pd.Index(hadm).get_indexer(order); assert not (tok_pos < 0).any()
    vocab = open(os.path.join(D, "hf", TAG, "vocab.txt")).read().split("\n")
    CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")
    rows = np.arange(SMOKE_ENC) if SMOKE else np.arange(N)
    W_all = np.array([(j, i) for i in rows for j in range(aoff[tok_pos[i]], aoff[tok_pos[i]] + int(nwin[tok_pos[i]]))],
                     np.int64).reshape(-1, 2)
    o = np.argsort(wlen[W_all[:, 0]], kind="stable")
    H = 768
    s_tok, s_cls = np.zeros((N, H)), np.zeros((N, H)); n_tok, n_w = np.zeros(N, np.int64), np.zeros(N, np.int64); b0 = 0
    if os.path.exists(PARTIAL) and not SMOKE:
        p = np.load(PARTIAL); s_tok, s_cls, n_tok, n_w, b0 = p["s_tok"], p["s_cls"], p["n_tok"], p["n_w"], int(p["b"])
        assert int(p["nwin_total"]) == len(o); log(f"resuming at window {b0:,}/{len(o):,}")
    model = AutoModel.from_pretrained(HF).to(DEV).eval()
    log(f"frozen {TAG}: {len(o):,} windows to encode" + (" (SMOKE)" if SMOKE else ""))
    ts, last = time.time(), time.time(); done = 0
    with torch.no_grad():
        for b in range(b0, len(o), BATCH):
            k = W_all[o[b:b+BATCH]]
            ws = [ids[woff[j]:woff[j+1]].astype(np.int64) for j in k[:, 0]]
            L = max(len(w) for w in ws) + 2
            X = np.full((len(ws), L), PAD, np.int64); M = np.zeros((len(ws), L), np.int64)
            for q, w in enumerate(ws):
                s = [CLS] + list(w) + [SEP]; X[q, :len(s)] = s; M[q, :len(s)] = 1
            h = model(input_ids=torch.from_numpy(X).to(DEV), attention_mask=torch.from_numpy(M).to(DEV)
                      ).last_hidden_state.float().cpu().numpy()
            for q, (j, i) in enumerate(k):
                m = int(wlen[j]); s_tok[i] += h[q, 1:1+m].sum(0); n_tok[i] += m; s_cls[i] += h[q, 0]; n_w[i] += 1
            done += len(k)
            if time.time() - last > CKPT_SEC and not SMOKE:
                tmp = PARTIAL + ".tmp.npz"
                np.savez(tmp, s_tok=s_tok, s_cls=s_cls, n_tok=n_tok, n_w=n_w, b=b + BATCH, nwin_total=len(o))
                os.replace(tmp, PARTIAL); last = time.time()
                el = time.time() - ts
                log(f"  {b + BATCH:,}/{len(o):,} windows, ~{(len(o) - b - BATCH) * el / done / 60:.0f} min left (checkpointed)")
    if SMOKE:
        z_all = int(nwin.sum()); per = (time.time() - ts) / max(done, 1)
        log(f"SMOKE: {per*1000:.0f} ms per window -> all {z_all:,} windows ~{per * z_all / 3600:.2f} h. Nothing written."); return
    np.savez(FROZEN, tok=(s_tok / np.maximum(n_tok, 1)[:, None]).astype(np.float32),
             cls=(s_cls / np.maximum(n_w, 1)[:, None]).astype(np.float32), n_w=n_w)
    try: os.remove(PARTIAL)
    except OSError: pass
    log(f"{FROZEN} written; {int((n_w == 0).sum())} admissions without text")

# ---------------------------------------------------------------- downstream fits
def run():
    from lr_tuned import lr_tuned_scores
    tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
    T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet").set_index("hadm_id").reindex(order)
    FIX = fixed22()

    def encode_tier(cols, fit):
        out = []
        for c in cols:
            s = T[c]
            if not pd.api.types.is_numeric_dtype(s):
                v = s.astype(object).where(s.notna(), "__na__")
                k = v[fit].value_counts(); lv = set(k[k >= 0.01*fit.sum()].index)
                v = v.where(v.isin(lv), "__other__")
                for L_ in sorted(lv | {"__other__"}): out.append((v == L_).to_numpy(np.float32))
            else: out.append(s.to_numpy(np.float32))
        return np.column_stack(out)

    def score(X, fit, mk):
        if mk == "gbm":
            return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                                  **GBM).fit(X[fit], y[fit]).decision_function(X)
        return lr_tuned_scores(X, fit, y)[0]

    done = set()
    if os.path.exists(CSV):
        o_ = pd.read_csv(CSV); done = set(zip(o_.draw, o_.tier, o_.model, o_.arm))
    PB = None
    if "pubmedbert_tok" in ARMS:
        assert os.path.exists(FROZEN), f"{FROZEN} missing -- run encode first, or drop pubmedbert_tok from W1_ARMS"
        PB = np.load(FROZEN)["tok"]
    for r in DRAWS:
        tr = draw_split(r); A, B = halves(r, tr); te = ~tr
        X = {}
        if "bow_rs" in ARMS:
            X["bow_rs"] = np.hstack([FIX, sparse.load_npz(f"{BOWD}/bow_rs_draw{r}.npz").toarray().astype(np.float32)])
        if {"lsa_pool_300", "lsa_wide_300"} & set(ARMS):
            assert os.path.exists(LSAF(r)), f"{LSAF(r)} missing -- run lsa first"
            Lz = np.load(LSAF(r))
            X["lsa_pool_300"] = np.hstack([FIX, Lz["pool"]]); X["lsa_wide_300"] = np.hstack([FIX, Lz["wide"]])
        if PB is not None: X["pubmedbert_tok"] = np.hstack([FIX, PB])
        log(f"draw {r}: B {B.sum():,} (downstream) | test {te.sum():,}")
        for t in TIERS:
            S = encode_tier(tiers[t], B)
            for mk in LEARNERS:
                todo = [a for a in ARMS if (r, t, mk, a) not in done]
                if not todo: continue
                ts = time.time(); sb = score(S, B, mk)
                for arm in todo:
                    sa = score(np.hstack([S, X[arm]]), B, mk)
                    a, b, d, se = paired(y[te], sb[te], sa[te])
                    append(CSV, dict(draw=r, tier=t, n_vars=len(tiers[t]), model=mk, arm=arm, n_fit=int(B.sum()),
                                     ev_fit=int(y[B].sum()), struct=round(a, 4), with_text=round(b, 4),
                                     increment=round(d, 4), se=round(se, 4)))
                    log(f"  draw {r} {t}/{mk}/{arm:14s}: struct {a:.4f} -> {b:.4f}  inc {d:+.4f}  "
                        f"({(time.time()-ts)/60:.1f} min so far this block)")
    summary()

# ---------------------------------------------------------------- summary
def ci(v):
    v = np.asarray(v, float); n = len(v)
    h = stats.t.ppf(0.975, n-1) * v.std(ddof=1) / np.sqrt(n) if n > 1 else np.nan
    return v.mean(), v.mean() - h, v.mean() + h

def summary():
    if not os.path.exists(CSV): sys.exit("no results yet")
    new = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"], keep="last")
    old = pd.read_csv(FTFULL).drop_duplicates(["draw", "tier", "model", "arm"], keep="last")
    a_new = new[new.arm == "bow_rs"].set_index(["draw", "tier", "model"]).increment
    a_old = old[old.arm == "bow_rs"].set_index(["draw", "tier", "model"]).increment
    j = pd.concat([a_new.rename("new"), a_old.rename("ftfull")], axis=1, join="inner")
    if len(j):
        dmax = (j.new - j.ftfull).abs().max()
        print(f"\nANCHOR bow_rs vs W1_ftfull.csv: {len(j)} cells, max |difference| {dmax:.4f} "
              + ("-- reproduces, same setup" if dmax <= 1e-4 else "-- DOES NOT REPRODUCE: stop and tell Claude"))
    res = pd.concat([old[old.arm.isin(["bow_rs", "frozen_tok", "ft_tok"])],
                     new[new.arm.isin(["lsa_pool_300", "lsa_wide_300", "pubmedbert_tok"])]])
    rows = []
    print("\n=== split-sample, 5 partitions: T1 -> T5 and collapse, 95% t CI ===")
    for mk in LEARNERS:
        for arm in ["bow_rs", "lsa_pool_300", "lsa_wide_300", "frozen_tok", "pubmedbert_tok", "ft_tok"]:
            q = res[(res.model == mk) & (res.arm == arm)].pivot(index="draw", columns="tier", values="increment").dropna()
            if q.empty or not {"T1", "T5"} <= set(q.columns): continue
            c, lo, hi = ci(100 * (q.T1 - q.T5) / q.T1); t1 = ci(q.T1); t5 = ci(q.T5)
            rows.append(dict(model=mk, arm=arm, n=len(q), T1=t1[0], T1_lo=t1[1], T1_hi=t1[2], T5=t5[0], T5_lo=t5[1],
                             T5_hi=t5[2], collapse=c, collapse_lo=lo, collapse_hi=hi, falls=int((q.T1 > q.T5).sum())))
            print(f"  {mk:4s} {arm:15s} n={len(q)}  T1 {t1[0]:+.4f}  T5 {t5[0]:+.4f} [{t5[1]:+.4f}, {t5[2]:+.4f}]"
                  f"  collapse {c:5.1f}% [{lo:5.1f}, {hi:5.1f}]  falls {(q.T1 > q.T5).sum()}/{len(q)}")
    pd.DataFrame(rows).round(4).to_csv(TABLE, index=False)
    print("\n=== each arm minus bow_rs, paired within partition ===")
    for mk in LEARNERS:
        for t in TIERS:
            p = res[(res.model == mk) & (res.tier == t)].pivot(index="draw", columns="arm", values="increment")
            for arm in ["lsa_pool_300", "lsa_wide_300", "pubmedbert_tok"]:
                if arm not in p.columns or "bow_rs" not in p.columns: continue
                d = (p[arm] - p.bow_rs).dropna()
                if len(d) < 2: continue
                m, lo, hi = ci(d)
                print(f"  {mk:4s} {t} {arm:15s} - bow_rs: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  ahead in {(d > 0).sum()}/{len(d)}")
    log(f"table -> {TABLE}; tell Claude and it will read it back")

{"check": check, "dtm_wide": dtm_wide, "lsa": lsa, "encode": encode, "run": run, "summary": summary}[MODE]()
