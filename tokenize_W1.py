"""Wordpiece-tokenise the RAW W1 note corpus on the local machine.

Why here and not in the cloud: the raw notes never leave this machine. What
crosses is a matrix of integer token ids, which is what the frozen encoder
consumes.

Why RAW text and not the Weissman-preprocessed doc: the bag-of-words and LSA
arms are replications of a published pipeline (lowercase, punctuation stripped,
digits deleted, stopwords removed, no order). A contextual encoder is being
tested precisely on what that pipeline throws away -- word order, negation
scope, numbers, which concept a hedge attaches to. Feeding it the stripped text
would test a representation nobody would ever build.

Corpus definition is identical to build_text_W1.py --v2: every note category
except Radiology and Discharge summary, iserror IS NULL, timestamped
COALESCE(charttime, chartdate) inside [admittime, admittime+48h), concatenated
in time order into one document per admission.

WINDOWING. Documents are long and very skewed (v2 raw word count: median 998,
p90 18,983, p99 42,476; the median admission needs 5 windows, p90 needs 39, the
longest 202), and a frozen BERT reads 512 wordpieces at a time. Full coverage of
the corpus is 328,408 forward passes per model, which is ~39 h on this machine's
CPU budget, so windows have to be sampled.

They are sampled EVENLY across the document -- always including the first and
the last -- never as a prefix. Taking a prefix would systematically discard the
end of the 48-hour window, which is both the closest text to the prediction
point and the most likely place for the care team's plan to appear; that would
bias the test against the encoder in exactly the direction the study is trying
to measure. Because mean-pooling is linear, an even sample is an unbiased
estimate of the mean-pooled embedding of the whole document.

Up to 16 windows per admission are stored, IN NESTED PRIORITY ORDER rather than
document order, so that encoding the first k of them is a valid even sample for
any k <= 16 and a larger budget re-uses everything a smaller one computed.
nfull is recorded so truncation can be reported and so the subset of admissions
whose notes fit entirely can be analysed separately.

Output, per model, per part:
  ids   uint16 flat concatenation of all windows   (both vocabs < 65535)
  wlen  int32  length of each window
  nwin  int32  windows kept per admission
  nfull int32  windows the document would have needed
  hadm  int64

Chunked and resumable: parts already on disk are skipped.

    python3 tokenize_W1.py

ALL WINDOWS (2026-09-29, for finetune_full_W1.py). The same windowing with no cap,
for Bio_ClinicalBERT only, into separate files (the 16-window files are untouched):

    W1_MAXWIN=all W1_TOKMODELS=bio_clinicalbert python3 tokenize_W1.py        # repeat until ALL PARTS DONE
    W1_MAXWIN=all W1_TOKMODELS=bio_clinicalbert python3 tokenize_W1.py merge  # -> out/tokall_bio_clinicalbert_W1.npz

Windows stay in nested priority order, so the first k of them are still an even
sample of the document for any k.
"""
import os, sys, time
import numpy as np, pandas as pd, duckdb
from transformers import AutoTokenizer

D = os.path.dirname(os.path.abspath(__file__))
_MW = os.environ.get("W1_MAXWIN", "16")
TOKTAG = "all" if _MW == "all" else str(int(_MW))           # "16" reproduces the original files
OUT = os.path.join(D, "out"); PARTS = os.path.join(OUT, f"tok{TOKTAG}_parts")
DB = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
MODELS = os.environ.get("W1_TOKMODELS", "bio_clinicalbert,pubmedbert").split(",")
W, MAXWIN = 510, (10**9 if TOKTAG == "all" else int(TOKTAG))
os.makedirs(PARTS, exist_ok=True)

if len(sys.argv) > 1 and sys.argv[1] == "merge":
    import glob
    for m in MODELS:
        fs = sorted(glob.glob(f"{PARTS}/{m}_*.npz"))
        zs = [np.load(f) for f in fs]
        hadm = np.concatenate([z["hadm"] for z in zs])
        assert (np.diff(hadm) > 0).all(), "parts out of order or overlapping"
        out = f"{OUT}/tok{TOKTAG}_{m}_W1.npz"
        np.savez(out, **{k: np.concatenate([z[k] for z in zs]) for k in ("ids", "wlen", "nwin", "nfull", "hadm")})
        z = np.load(out)
        print(f"{out}: {len(z['hadm']):,} admissions, {len(z['wlen']):,} windows, "
              f"{int((z['nwin'] == z['nfull']).mean()*100)}% fully stored")
    sys.exit(0)
CHUNK = 500
EXCLUDE = ["Radiology", "Discharge summary"]
CAT_SQL = "n.category NOT IN (" + ",".join("'" + c + "'" for c in EXCLUDE) + ")"
NT = "COALESCE(CAST(n.charttime AS TIMESTAMP), CAST(n.chartdate AS TIMESTAMP))"

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

toks = {m: AutoTokenizer.from_pretrained(os.path.join(D, "hf", m)) for m in MODELS}

coh = pd.read_csv(os.path.join(OUT, "cohort_W1.csv"), parse_dates=["admittime", "t48"])
ids_all = np.sort(coh.hadm_id.to_numpy()); n = len(ids_all)

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=4"); con.execute("PRAGMA memory_limit='1GB'")
con.register("coh_df", coh[["hadm_id", "admittime", "t48"]])
con.execute("CREATE TEMP TABLE coh AS SELECT * FROM coh_df")

def priority(n):
    """Window indices 0..n-1 in NESTED even-spread order: first, last, then
    recursive bisection. Every prefix is a roughly even spread that includes the
    start and the end of the document, and the prefix of length k is contained in
    the prefix of length k' > k. That nesting is what lets the encoder run a small
    window budget first and extend it later without recomputing anything."""
    if n <= 1: return [0]
    out = [0, n - 1]; segs = [(0, n - 1)]
    while len(out) < n and segs:
        nxt = []
        for a, b in segs:
            m = (a + b) // 2
            if m not in (a, b) and m not in out:
                out.append(m); nxt += [(a, m), (m, b)]
        segs = nxt
    return out[:n]

def windows(seq):
    """Up to MAXWIN windows of W tokens, STORED IN PRIORITY ORDER (not document
    order), so the encoder can take the first k of them for any k."""
    if len(seq) == 0: return [], 0
    nfull = (len(seq) + W - 1) // W
    picks = priority(nfull)[:MAXWIN]
    return [seq[i*W:(i+1)*W] for i in picks], nfull

todo = [lo for lo in range(0, n, CHUNK)
        if not all(os.path.exists(f"{PARTS}/{m}_{lo:06d}.npz") for m in MODELS)]
log(f"{n:,} admissions, {len(todo)} of {(n+CHUNK-1)//CHUNK} parts remaining")

for k, lo in enumerate(todo):
    batch = ids_all[lo:lo + CHUNK]
    con.execute("CREATE OR REPLACE TEMP TABLE b AS SELECT * FROM coh WHERE hadm_id IN "
                f"({','.join(map(str, batch))})")
    sub = con.execute(f"""
        SELECT c.hadm_id, STRING_AGG(n.text, chr(10) ORDER BY {NT}, n.row_id) AS doc
        FROM b c JOIN noteevents n USING (hadm_id)
        WHERE {CAT_SQL} AND n.iserror IS NULL AND n.text IS NOT NULL
          AND {NT} >= c.admittime AND {NT} < c.t48
        GROUP BY c.hadm_id""").df()
    sub.columns = [c.lower() for c in sub.columns]
    have = dict(zip(sub.hadm_id.to_numpy(), sub.doc.to_numpy()))
    docs = [have.get(h, "") for h in batch]

    for m in MODELS:
        f = f"{PARTS}/{m}_{lo:06d}.npz"
        if os.path.exists(f): continue
        enc = toks[m](docs, add_special_tokens=False, truncation=False)["input_ids"]
        flat, wlen, nwin, nfull = [], [], [], []
        for seq in enc:
            ws, nf = windows(seq)
            for w in ws: flat.extend(w); wlen.append(len(w))
            nwin.append(len(ws)); nfull.append(nf)
        np.savez_compressed(f,
            ids=np.asarray(flat, np.uint16), wlen=np.asarray(wlen, np.int32),
            nwin=np.asarray(nwin, np.int32), nfull=np.asarray(nfull, np.int32),
            hadm=batch.astype(np.int64))
        del enc, flat
    del sub, have, docs
    log(f"  part {lo:06d}  ({k+1}/{len(todo)})")
    if time.time() - t0 > 140:
        log("time budget reached -- re-run to continue"); sys.exit(0)
log("ALL PARTS DONE")
