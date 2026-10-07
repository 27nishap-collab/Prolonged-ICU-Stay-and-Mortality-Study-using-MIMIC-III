"""Frozen contextual-encoder embeddings for the W1 corpus.

Runs Bio_ClinicalBERT or PubMedBERT over the raw 48-hour note text (tokenised
on the local machine by tokenize_W1.py; only integer ids crossed) and reduces
each admission to a single dense vector.

WHAT IS BEING TESTED
The bag-of-words and LSA arms both order-free. RESULTS_lsa.md removed three
alternative explanations for the saturation result -- it is not sparsity, not
dimensionality, not term selection. The one axis left is CONTEXTUAL MEANING:
word order, negation scope, which concept a hedge attaches to. A frozen
sentence-level encoder is the cheapest representation that has it.

Pre-specified bar, unchanged from RESULTS_lsa.md and fixed before any of these
numbers existed: at T5 the bag-of-words block is worth +0.0062 (composite) and
+0.0032, p=0.26 (death), against a partition-noise sd of 0.005. A representation
needs roughly +0.02 at T5 to be a finding rather than a rounding error. The best
LSA arm reaches +0.0081.

WINDOW BUDGET AND WHY IT IS ADDITIVE
Full coverage is 328k forward passes per model, ~39 h on two CPU cores. Windows
are therefore sampled evenly across each document (see tokenize_W1.py) and
stored in nested priority order. This script encodes the window BAND
[WIN_LO, WIN_HI) and accumulates SUMS, never means:

    sum_cls   sum of the [CLS] vector over the band's windows
    sum_tok   sum of every content token's hidden state over the band
    n_tok     content tokens summed
    n_win     windows encoded

Because these are sums, the band files ADD. Running 0-4 now and 4-8 later and
adding the two gives exactly the k=8 result, with nothing recomputed. Pooled
vectors are formed at the end: mean-of-CLS = sum_cls/n_win, and mean-pool =
sum_tok/n_tok, which is the true token-mean over everything the encoder read.

Precision is fp32 throughout. int8 dynamic quantisation was measured on real
clinical windows from this corpus and agrees with fp32 at cosine 0.974
(min 0.941) -- too coarse to run a test whose whole point is a small increment.
ONNX Runtime was measured at 512 ms/window against PyTorch eager's 424 on this
2-core machine and was dropped.

    W1_MODEL=bio_clinicalbert W1_WIN_LO=0 W1_WIN_HI=4 python encode_bert_W1.py
"""
import os, sys, time
import numpy as np, torch
from transformers import AutoModel

MODEL = os.environ.get("W1_MODEL", "bio_clinicalbert")
LO    = int(os.environ.get("W1_WIN_LO", 0))
HI    = int(os.environ.get("W1_WIN_HI", 4))
BLOCK = 200                      # admissions per shard
BATCH = 8                        # windows per forward pass
U   = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
OUT = f"{U}/bert_{MODEL}_w{LO}-{HI}"   # same layout as encode_bert_local.py
os.makedirs(OUT, exist_ok=True)
torch.set_num_threads(2)

t0 = time.time()
def log(m): print(f"[{time.time()-t0:8.1f}s] {m}", flush=True)

z = np.load(f"{U}/tok16_{MODEL}_W1.npz")
ids, wlen, nwin, nfull, hadm = z["ids"], z["wlen"], z["nwin"], z["nfull"], z["hadm"]
woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])   # into ids
aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])   # into wlen
N = len(hadm)

model = AutoModel.from_pretrained({"bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT", "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}[MODEL]).eval()
H = model.config.hidden_size
CLS = 101 if MODEL == "bio_clinicalbert" else 2   # resolved below from vocab
import json
vocab = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "hf", MODEL, "vocab.txt")).read().split("\n")
CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")
log(f"{MODEL}: hidden {H}, CLS {CLS} SEP {SEP} PAD {PAD}, "
    f"{N:,} admissions, windows [{LO},{HI})")

todo_w = sum(max(0, min(int(nwin[i]), HI) - LO) for i in range(N))
log(f"{todo_w:,} forward passes in this band (~{todo_w*0.424/3600:.1f} h)")

def run_block(lo_a):
    hi_a = min(lo_a + BLOCK, N)
    wins, owner = [], []
    for i in range(lo_a, hi_a):
        k0, k1 = aoff[i] + LO, aoff[i] + min(int(nwin[i]), HI)
        for j in range(k0, k1):
            wins.append(ids[woff[j]:woff[j+1]].astype(np.int64)); owner.append(i - lo_a)
    n = hi_a - lo_a
    s_cls = np.zeros((n, H), np.float64); s_tok = np.zeros((n, H), np.float64)
    n_tok = np.zeros(n, np.int64); n_win = np.zeros(n, np.int64)
    order = np.argsort([len(w) for w in wins])          # length-sorted: less padding
    with torch.no_grad():
        for b in range(0, len(order), BATCH):
            sel = order[b:b+BATCH]
            L = max(len(wins[k]) for k in sel) + 2
            X = np.full((len(sel), L), PAD, np.int64); A = np.zeros((len(sel), L), np.int64)
            for r, k in enumerate(sel):
                s = [CLS] + list(wins[k]) + [SEP]
                X[r, :len(s)] = s; A[r, :len(s)] = 1
            h = model(input_ids=torch.from_numpy(X),
                      attention_mask=torch.from_numpy(A)).last_hidden_state.numpy()
            for r, k in enumerate(sel):
                o = owner[k]; m = len(wins[k])
                s_cls[o] += h[r, 0]
                s_tok[o] += h[r, 1:1+m].sum(0)          # content tokens only
                n_tok[o] += m; n_win[o] += 1
    np.savez(f"{OUT}/b_{lo_a:06d}.npz", hadm=hadm[lo_a:hi_a],
             s_cls=s_cls.astype(np.float32), s_tok=s_tok.astype(np.float32),
             n_tok=n_tok, n_win=n_win)

blocks = [b for b in range(0, N, BLOCK) if not os.path.exists(f"{OUT}/b_{b:06d}.npz")]
log(f"{len(blocks)} of {(N+BLOCK-1)//BLOCK} blocks remaining")
for c, b in enumerate(blocks):
    run_block(b)
    el = time.time() - t0
    log(f"  block {b:06d}  ({c+1}/{len(blocks)})  eta {el/(c+1)*(len(blocks)-c-1)/3600:.1f} h")
log("BAND DONE")
