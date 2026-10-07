"""W1 text pipeline -- Weissman et al. 2018 replication.

    python build_text_W1.py build   # notes -> cleaned doc + closed-vocab counts
    python build_text_W1.py vocab   # 1-3 grams with train doc-freq >= 5%
    python build_text_W1.py open    # 500 phrases by penalised LR, train only
    python build_text_W1.py all

Every step is chunked and resumable: it writes parts under out/ and skips parts
already on disk, so a step can be re-run to continue.

--------------------------------------------------------------------------
SPEC (paper + supplement + authors' released code)
--------------------------------------------------------------------------
Corpus, verbatim: "All clinical text notes from physicians, nurses, and other
clinicians time-stamped during the first 48 hours of the hospitalization were
aggregated into a single document for each admission."

  * Category filter: everything EXCEPT Radiology. This is the corpus definition
    that reproduces Table 1's notes-per-admission exactly -- 6 (4-11) overall
    and 8 (6-12) in the outcome group. It differs from the ELIGIBILITY filter
    used to build the cohort, which also drops ECG and Echo; see
    build_cohort_W1.py and METHODS_weissman.md gap #7.
  * Timestamp: COALESCE(charttime, chartdate), as in the cohort build.
  * iserror IS NULL.

NO DEDUPLICATION. Weissman does not deduplicate and neither do the comparison
studies. Rules A and B live in Analysis/notes_extract and are used only as a
sensitivity test on our own cohorts -- they are deliberately absent here.

preprocess(), from the authors' released code (gweissman/text-pred-icu):
  1. lowercase
  2. DELETE  ! " # $ % & ( ) * + , : ; < = > ? @ [ \\ ] ^ _ ` { | } ~  and all digits
  3. REPLACE WITH SPACE  -  .  /  '
  4. remove NLTK English stopwords
  5. no stemming; whitespace split

Closed vocabulary: 21 terms specified a priori (supplemental table e1).
Open vocabulary, verbatim: "we also built a document-term matrix of 1-, 2-, and
3-word phrases (i.e. n-grams) appearing in at least 5% of all documents in the
training set, and used penalized logistic regression with 10-fold
cross-validation to identify the 500 most predictive phrases." ... "This
variable selection was conducted in the training set only." ... "Extraction of
text features yielded 5,790 unique phrases from which 500 were chosen."

Feature values: "the integer counts of phrases appearing in the aggregated
clinical notes were included as continuous variables". COUNTS are primary;
their released code binarises (x > 0), kept as a sensitivity in the models.
Also: "The total word count of each aggregated set of documents was also
included as a covariate" -- computed on the RAW document (it is already in
cohort_W1.csv, where it reproduces Table 1).

The 75/25 split is created HERE, not in the modelling script, so the train-only
vocabulary selection and the models are guaranteed to use the same partition.
caret::createDataPartition(df$Y, p=0.75) is stratified on the outcome and NOT
grouped by patient; that is replicated, and a patient-grouped split is written
alongside as a declared deviation.

--------------------------------------------------------------------------
WHY THIS IS WRITTEN IN CHUNKS
--------------------------------------------------------------------------
The corpus is 548 MB of text over 235,101 notes, and the shell this runs in has
3.9 GB of RAM and no swap. Materialising the whole corpus -- in DuckDB, in
pandas, or as Python strings -- is an OOM kill, so nothing here ever holds more
than one chunk of raw text.

The same constraint rules out sklearn's CountVectorizer for the open
vocabulary: with ngram_range=(1,3) it builds the complete n-gram vocabulary
before applying min_df, and 1-3 grams over ~90M tokens is hundreds of millions
of distinct trigrams. step_vocab instead prunes level by level using the
downward-closure property -- a bigram with document frequency >= t requires
both of its unigrams to have df >= t, and likewise for trigrams from bigrams.
That is EXACT, not an approximation: it yields the same vocabulary
CountVectorizer(min_df=0.05, ngram_range=(1,3)) would, in bounded memory.
"""
import os, re, sys, time, json
from collections import Counter
import duckdb, numpy as np, pandas as pd
from scipy import sparse

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
DB = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
SEED = 20180701
CHUNK = 1000
MIN_DF = 0.05
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# --- corpus variant -------------------------------------------------------
# v1 reproduces Weissman's notes-per-admission exactly but LEAKS: discharge
# summaries carry a NULL charttime, so COALESCE falls back to chartdate, which
# is midnight of the discharge DAY. A late admission time can therefore pull a
# discharge summary inside [admittime, admittime+48h) while the patient is still
# in hospital. That happens for 1,454 notes across 1,398 admissions (5.4% of the
# cohort), and 1,449 of them belong to admissions discharged AFTER hour 48 -- a
# genuine outcome leak, since the summary states the outcome.
#   v2 drops the Discharge summary category. Both are built so the size of the
# leak is measurable rather than merely corrected.
V2 = "--v2" in sys.argv or os.environ.get("W1_TEXT_V2") == "1"
SUF = "_v2" if V2 else ""
EXTRA_EXCLUDE = ["Discharge summary"] if V2 else []
CAT_SQL = "n.category NOT IN (" + ",".join(
    "'" + c + "'" for c in ["Radiology"] + EXTRA_EXCLUDE) + ")"

PARTS  = os.path.join(OUT, f"text_parts{SUF}")
META   = os.path.join(OUT, f"text_meta_W1{SUF}.parquet")
VOCAB  = os.path.join(OUT, f"W1_openvocab_full{SUF}.csv")
OPENZ  = os.path.join(OUT, f"text_open_W1{SUF}.npz")
OPENV  = os.path.join(OUT, f"text_open_vocab_W1{SUF}.csv")

CLOSED_VOCAB = [
    "code status", "comfort care", "comfortable", "death", "die",
    "doesn't want", "family meeting", "family", "goals of care", "high risk",
    "hospice", "mortality", "mortal", "pall care", "palliation",
    "palliative care", "palliative", "poor prognosis", "survival", "wishes",
    "wouldn't want",
]
assert len(CLOSED_VOCAB) == 21

DELETE = '!"#$%&()*+,:;<=>?@[\\]^_`{|}~' + "0123456789"
SPACE  = "-./'"
_TBL = str.maketrans({**{c: None for c in DELETE}, **{c: " " for c in SPACE}})
_nm  = lambda s: re.sub(r"\W+", "_", s).strip("_")

SWFILE = os.path.join(D, "nltk_stopwords_english.txt")

def _stopwords():
    """NLTK English stopwords, read from a vendored file.

    Vendored rather than downloaded so the list is pinned and the pipeline is
    reproducible offline (this shell cannot reach the NLTK data mirror anyway).
    The file is NLTK 3.10.3's list: 198 words.

    VERSION SENSITIVITY, declared. Weissman's code ran on Python 3.5.1 in
    2016-17, when NLTK's English list had 153 words and did NOT contain the
    contraction fragments ('wouldn', 'doesn', 't', 's', 'd', 'll', ...). Those
    are in the 198-word list, so under this list "wouldn't want" and
    "doesn't want" both reduce to "want", while under theirs they would have
    reduced to "wouldn t want" and "doesn t want". This affects only the
    post-preprocess closed-vocabulary variant (cvp_*) and the open vocabulary;
    the PRIMARY closed-vocabulary features are matched on raw text and are
    unaffected.
    """
    if os.path.exists(SWFILE):
        return set(open(SWFILE).read().split())
    from nltk.corpus import stopwords          # fallback if the file is absent
    return set(stopwords.words("english"))

def preprocess(text, sw):
    return " ".join(w for w in text.lower().translate(_TBL).split() if w not in sw)

# ==========================================================================
def step_build():
    """Stream notes chunk by chunk: aggregate -> closed-vocab counts on the raw
    text -> preprocess -> write the cleaned doc. Raw text is never persisted and
    never held beyond one chunk."""
    os.makedirs(PARTS, exist_ok=True)
    sw = _stopwords()
    coh = pd.read_csv(os.path.join(OUT, "cohort_W1.csv"), parse_dates=["admittime", "t48"])

    con = duckdb.connect(DB, read_only=True)
    con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='1GB'")
    con.register("coh_df", coh[["hadm_id","admittime","t48"]])
    con.execute("CREATE TEMP TABLE coh AS SELECT * FROM coh_df")
    NT = "COALESCE(CAST(n.charttime AS TIMESTAMP), CAST(n.chartdate AS TIMESTAMP))"

    # ---- metadata + the shared 75/25 split ----
    if not os.path.exists(META):
        if V2:
            nt = con.execute(f"""
                SELECT c.hadm_id, COUNT(*) AS n_notes,
                       SUM(LENGTH(n.text) - LENGTH(REPLACE(n.text,' ','')) + 1) AS n_words_raw
                FROM coh c JOIN noteevents n USING (hadm_id)
                WHERE {CAT_SQL} AND n.iserror IS NULL AND n.text IS NOT NULL
                  AND {NT} >= c.admittime AND {NT} < c.t48
                GROUP BY c.hadm_id""").df()
            nt.columns = [x.lower() for x in nt.columns]
            df = coh[["hadm_id","subject_id","labels"]].merge(nt, on="hadm_id", how="left")
            df[["n_notes","n_words_raw"]] = df[["n_notes","n_words_raw"]].fillna(0).astype(int)
        else:
            df = coh[["hadm_id","subject_id","labels","n_notes","raw_word_count"]] \
                    .rename(columns={"raw_word_count": "n_words_raw"}).copy()
        rng = np.random.RandomState(SEED)
        df["split"] = "train"
        for y in (0, 1):
            idx = df.index[df.labels == y].to_numpy()
            rng.shuffle(idx)
            df.loc[idx[: int(round(0.25 * len(idx)))], "split"] = "test"
        subs = df.subject_id.drop_duplicates().to_numpy(); rng.shuffle(subs)
        test_subs = set(subs[: int(round(0.25 * len(subs)))])
        df["split_grouped"] = np.where(df.subject_id.isin(test_subs), "test", "train")
        df.to_parquet(META, index=False)
        log(f"{META}: train {(df.split=='train').sum():,} / test {(df.split=='test').sum():,}"
            f"  outcome rate {df.groupby('split').labels.mean().round(4).to_dict()}")
        leak = df.loc[df.split=="test","subject_id"].isin(
               set(df.loc[df.split=="train","subject_id"])).mean()
        log(f"  ungrouped split: {100*leak:.1f}% of test admissions share a patient with "
            f"train -- what the paper did; split_grouped has 0%")

    pats = {t: re.compile(r"\b" + re.escape(t).replace(r"\ ", r"\s+") + r"\b") for t in CLOSED_VOCAB}
    # post-preprocess variants, for the sensitivity match
    coll = {}
    for t in CLOSED_VOCAB: coll.setdefault(preprocess(t, sw), []).append(t)
    ppats = {p: re.compile(r"\b" + re.escape(p).replace(r"\ ", r"\s+") + r"\b")
             for p in coll if p}

    ids = np.sort(coh.hadm_id.to_numpy())
    n = len(ids)
    log(f"building {n:,} documents in chunks of {CHUNK:,} ...")

    for lo in range(0, n, CHUNK):
        part = os.path.join(PARTS, f"part_{lo:06d}.parquet")
        if os.path.exists(part): continue
        batch = ids[lo:lo + CHUNK]
        con.execute("CREATE OR REPLACE TEMP TABLE b AS SELECT * FROM coh WHERE hadm_id IN "
                    f"({','.join(map(str, batch))})")
        sub = con.execute(f"""
            SELECT c.hadm_id, STRING_AGG(n.text, chr(10) ORDER BY {NT}, n.row_id) AS doc
            FROM b c JOIN noteevents n USING (hadm_id)
            WHERE {CAT_SQL} AND n.iserror IS NULL AND n.text IS NOT NULL
              AND {NT} >= c.admittime AND {NT} < c.t48
            GROUP BY c.hadm_id""").df()
        sub.columns = [c.lower() for c in sub.columns]

        recs = []
        for h, raw in zip(sub.hadm_id.to_numpy(), sub.doc.to_numpy()):
            low = raw.lower()
            r = {"hadm_id": int(h)}
            for t, p in pats.items():  r["cv_"  + _nm(t)] = len(p.findall(low))
            clean = preprocess(raw, sw)
            for p, pc in ppats.items(): r["cvp_" + _nm(p)] = len(pc.findall(clean))
            r["doc"] = clean
            r["n_words_clean"] = clean.count(" ") + 1 if clean else 0
            recs.append(r)
        pd.DataFrame(recs).to_parquet(part, index=False, compression="zstd")
        del sub, recs
        log(f"  {min(lo+CHUNK, n):,}/{n:,}")

    done = sorted(f for f in os.listdir(PARTS) if f.endswith(".parquet"))
    exp = (n + CHUNK - 1) // CHUNK
    if len(done) < exp:
        log(f"INCOMPLETE: {len(done)}/{exp} parts -- re-run this step to continue")
        return
    log(f"all {exp} parts present")

    # ---- report: what preprocess() does to the 21 a-priori terms ----
    print("\n  a-priori term -> after preprocess():")
    for t in CLOSED_VOCAB:
        p = preprocess(t, sw)
        flag = ""
        if not p: flag = "   <-- ERASED"
        elif len(coll[p]) > 1: flag = f"   <-- COLLIDES with {[x for x in coll[p] if x != t]}"
        elif p != t: flag = "   <-- CHANGED"
        print(f"    {t:18s} -> {p!r}{flag}")

    con2 = duckdb.connect(":memory:")
    g = os.path.join(PARTS, "*.parquet")
    cv = [f"cv_{_nm(t)}" for t in CLOSED_VOCAB]
    pres = con2.execute(
        "SELECT " + ", ".join(f"AVG(CASE WHEN {c}>0 THEN 1.0 ELSE 0 END) AS {c}" for c in cv) +
        f" FROM '{g}'").df().iloc[0].sort_values(ascending=False)
    print("\n  prevalence of each a-priori term (raw-text match):")
    for k, v in pres.items(): print(f"    {k[3:]:20s} {100*v:5.1f}%")
    w = con2.execute(f"SELECT MEDIAN(n_words_clean) m, COUNT(*) n FROM '{g}'").df().iloc[0]
    meta = pd.read_parquet(META)
    print(f"\n  documents {int(w.n):,}; median cleaned word count {w.m:.0f} "
          f"(raw median {meta.n_words_raw.median():.0f})")

# ==========================================================================
def _iter_docs(where_ids=None, cols=("hadm_id", "doc")):
    """Stream cleaned documents part by part, so the corpus is never in RAM."""
    for f in sorted(os.listdir(PARTS)):
        if not f.endswith(".parquet"): continue
        d = pd.read_parquet(os.path.join(PARTS, f), columns=list(cols))
        if where_ids is not None:
            d = d[d.hadm_id.isin(where_ids)]
        if len(d): yield d

def step_vocab():
    """Exact min_df=5% vocabulary for 1-3 grams by downward closure."""
    meta = pd.read_parquet(META)
    train = set(meta.hadm_id[meta.split == "train"])
    nt = len(train)
    thr = MIN_DF * nt
    log(f"train documents {nt:,}; document-frequency threshold {thr:.0f}")

    df1 = Counter()
    for d in _iter_docs(train):
        for doc in d.doc.to_numpy(): df1.update(set(doc.split()))
    D1 = {w: c for w, c in df1.items() if c >= thr}; V1 = set(D1)
    log(f"  unigrams: {len(df1):,} distinct -> {len(V1):,} with df >= 5%")
    del df1

    df2 = Counter()
    for d in _iter_docs(train):
        for doc in d.doc.to_numpy():
            tk = doc.split()
            df2.update({(tk[i], tk[i+1]) for i in range(len(tk)-1)
                        if tk[i] in V1 and tk[i+1] in V1})
    D2 = {b: c for b, c in df2.items() if c >= thr}; VB = set(D2)
    log(f"  bigrams:  {len(df2):,} candidates -> {len(VB):,} with df >= 5%")
    del df2

    df3 = Counter()
    for d in _iter_docs(train):
        for doc in d.doc.to_numpy():
            tk = doc.split()
            df3.update({(tk[i], tk[i+1], tk[i+2]) for i in range(len(tk)-2)
                        if (tk[i], tk[i+1]) in VB and (tk[i+1], tk[i+2]) in VB})
    D3 = {g: c for g, c in df3.items() if c >= thr}
    log(f"  trigrams: {len(df3):,} candidates -> {len(D3):,} with df >= 5%")
    del df3

    rows = ([(w, c, 1) for w, c in D1.items()]
          + [(" ".join(b), c, 2) for b, c in D2.items()]
          + [(" ".join(g), c, 3) for g, c in D3.items()])
    v = pd.DataFrame(rows, columns=["phrase", "train_df", "n"]).sort_values(
        ["n", "train_df"], ascending=[True, False])
    v["train_df_pct"] = (100 * v.train_df / nt).round(2)
    v.to_csv(VOCAB, index=False)
    log(f"{VOCAB}: {len(v):,} phrases   [published 5,790]")
    print(v.groupby("n").size().rename("phrases").to_string())

# ==========================================================================
def _dtm(vocab_index, ids=None, ngram_max=3):
    """Stream a count matrix over a fixed vocabulary. Returns (csr, hadm_ids)."""
    indptr, indices, data, hs = [0], [], [], []
    for d in _iter_docs(ids):
        for h, doc in zip(d.hadm_id.to_numpy(), d.doc.to_numpy()):
            tk = doc.split()
            c = Counter()
            for j, w in enumerate(tk):
                k = vocab_index.get(w)
                if k is not None: c[k] += 1
                if ngram_max >= 2 and j + 1 < len(tk):
                    k = vocab_index.get(w + " " + tk[j+1])
                    if k is not None: c[k] += 1
                if ngram_max >= 3 and j + 2 < len(tk):
                    k = vocab_index.get(w + " " + tk[j+1] + " " + tk[j+2])
                    if k is not None: c[k] += 1
            indices.extend(c.keys()); data.extend(c.values())
            indptr.append(len(indices)); hs.append(h)
    X = sparse.csr_matrix((np.asarray(data, np.float32),
                           np.asarray(indices, np.int32),
                           np.asarray(indptr, np.int64)),
                          shape=(len(hs), len(vocab_index)))
    return X, np.asarray(hs)

def step_open():
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score

    meta = pd.read_parquet(META).set_index("hadm_id")
    v = pd.read_csv(VOCAB)
    vocab = v.phrase.astype(str).tolist()
    vi = {p: i for i, p in enumerate(vocab)}
    train = set(meta.index[meta.split == "train"])
    log(f"building train DTM: {len(train):,} docs x {len(vocab):,} phrases ...")
    Xtr, htr = _dtm(vi, train)
    ytr = meta.labels.loc[htr].to_numpy()
    log(f"  nnz {Xtr.nnz:,} ({Xtr.nnz*8/1e6:.0f} MB)")

    # Penalised logistic regression, 10-fold CV, to pick the 500 most
    # predictive phrases. glmnet's lambda path is walked here as an sklearn C
    # grid, scored by CV AUROC. log1p is applied for the SELECTOR only -- the
    # features shipped downstream are raw integer counts, as the paper states.
    Xs = Xtr.copy(); Xs.data = np.log1p(Xs.data)
    CS = [0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5]
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED)
    rows = []
    for C in CS:
        aucs = []
        for a, b in skf.split(Xs, ytr):
            m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
            m.fit(Xs[a], ytr[a])
            aucs.append(roc_auc_score(ytr[b], m.decision_function(Xs[b])))
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000).fit(Xs, ytr)
        nz = int((m.coef_[0] != 0).sum())
        rows.append(dict(C=C, cv_auc=float(np.mean(aucs)), nonzero=nz))
        log(f"  C={C:<8g} cv AUROC {np.mean(aucs):.4f}  nonzero {nz:,}")
    path = pd.DataFrame(rows)
    path.to_csv(os.path.join(OUT, "W1_openvocab_l1_path.csv"), index=False)

    best = path.loc[path.cv_auc.idxmax()]
    usable = path[path.nonzero >= 500]
    C = float(best.C) if best.nonzero >= 500 else (
        float(usable.loc[usable.cv_auc.idxmax()].C) if len(usable) else float(path.C.max()))
    if C != float(best.C):
        log(f"CV-optimal C={best.C:g} keeps only {int(best.nonzero)} phrases; using "
            f"C={C:g} (cv AUROC {float(path.loc[path.C==C,'cv_auc'].iloc[0]):.4f}) to reach 500")
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=4000).fit(Xs, ytr)
    coef = m.coef_[0]
    keep = np.argsort(-np.abs(coef))[:500]
    keep = keep[coef[keep] != 0]
    sel = [vocab[i] for i in keep]
    log(f"selected {len(sel)} phrases at C={C:g}")
    del Xs, Xtr

    log("building full-cohort DTM over the selected phrases ...")
    X, hs = _dtm({p: i for i, p in enumerate(sel)})
    order = np.argsort(hs)
    sparse.save_npz(OPENZ, X[order].tocsr())
    pd.DataFrame({"hadm_id": hs[order]}).to_parquet(
        os.path.join(OUT, f"text_open_rows_W1{SUF}.parquet"), index=False)
    pd.DataFrame({"phrase": sel, "coef": coef[keep],
                  "train_df_pct": v.set_index("phrase").train_df_pct.reindex(sel).to_numpy()}) \
      .to_csv(OPENV, index=False)
    log(f"{OPENZ} ({X.shape[0]:,} x {X.shape[1]}) written")

    o = np.argsort(-coef[keep])
    print("\n  most positive:", ", ".join(f"'{sel[i]}'" for i in o[:15]))
    print("  most negative:", ", ".join(f"'{sel[i]}'" for i in o[-15:]))
    json.dump({"vocab_min_df_5pct": len(vocab), "published_vocab": 5790,
               "selected": len(sel), "C": C, "seed": SEED},
              open(os.path.join(OUT, "W1_openvocab_meta.json"), "w"), indent=2)

# ==========================================================================
STEPS = {"build": step_build, "vocab": step_vocab, "open": step_open}
if __name__ == "__main__":
    want = [a for a in sys.argv[1:] if not a.startswith("--")] or ["all"]
    for s in (list(STEPS) if want == ["all"] else want):
        log(f"=== {s} ===")
        STEPS[s]()
    log("done")
