"""Frozen contextual-encoder embeddings for the W1 corpus -- RUN THIS ON THE MAC.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3 encode_bert_local.py                     # Bio_ClinicalBERT, then PubMedBERT
    python3 encode_bert_local.py bio_clinicalbert    # just one

Needs torch + transformers in your native Python:
    python3 -m pip install torch transformers

Nothing is downloaded and nothing leaves the machine. The notes were tokenised
locally by tokenize_W1.py; this reads those integer ids from out/tok16_*.npz and
the model weights from your existing ~/.cache/huggingface. Set HF_HUB_OFFLINE=1
if you want to be certain it never reaches the network.

WHY LOCALLY. Measured throughput for one BERT-base window of 512 wordpieces:
cloud container (2 Xeon cores) 424 ms, the sandboxed Linux VM (4 aarch64 cores)
273 ms. Either is ~7-10 h for one model, and neither sandbox gets any CPU
between conversation turns, so a long background run makes no progress at all.
Apple Silicon with MPS is roughly an order of magnitude faster and simply runs.

WHAT IT COMPUTES. Each admission's notes are a long document; the 48-hour v2
corpus has a median of 5 windows of 510 wordpieces, p90 of 39, max 202. Windows
were sampled EVENLY across each document (never a prefix -- a prefix would throw
away the end of the window, the text closest to the prediction point and the
likeliest place for the care team's plan) and stored in nested priority order.
This script encodes the first WIN_HI of them and accumulates SUMS, not means:

    s_cls  sum of the [CLS] vector over the windows encoded
    s_tok  sum of every content token's final hidden state
    n_tok  content tokens summed
    n_win  windows encoded

Sums ADD across bands, so raising the budget later re-uses everything already
computed. Pooled vectors are formed at fitting time: s_cls/n_win and s_tok/n_tok.

Resumable: shards of 500 admissions land in out/bert_<model>_w0-<k>/, and a
re-run skips the ones already on disk. Ctrl-C and restart is safe.
"""
import os, sys, time, glob
import numpy as np, torch
from transformers import AutoModel

HF = {"bio_clinicalbert": "emilyalsentzer/Bio_ClinicalBERT",
      "pubmedbert": "microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext"}
WIN_HI = int(os.environ.get("W1_WIN_HI", 4))
BLOCK  = 500
D   = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out")

if torch.backends.mps.is_available():
    DEV, BATCH = torch.device("mps"), 16
else:
    DEV, BATCH = torch.device("cpu"), 8
    torch.set_num_threads(os.cpu_count() or 4)

def run(tag):
    z = np.load(f"{OUT}/tok16_{tag}_W1.npz")
    ids, wlen, nwin, hadm = z["ids"], z["wlen"], z["nwin"], z["hadm"]
    woff = np.concatenate([[0], np.cumsum(wlen.astype(np.int64))])
    aoff = np.concatenate([[0], np.cumsum(nwin.astype(np.int64))])
    N = len(hadm)
    d = f"{OUT}/bert_{tag}_w0-{WIN_HI}"; os.makedirs(d, exist_ok=True)

    repo = HF[tag]
    model = AutoModel.from_pretrained(repo).eval().to(DEV)
    H = model.config.hidden_size
    # [CLS]/[SEP]/[PAD] ids, read from the local vocab rather than assumed
    vocab = open(os.path.join(D, "hf", tag, "vocab.txt")).read().split("\n")
    CLS, SEP, PAD = vocab.index("[CLS]"), vocab.index("[SEP]"), vocab.index("[PAD]")

    todo = [b for b in range(0, N, BLOCK) if not os.path.exists(f"{d}/b_{b:06d}.npz")]
    total = sum(min(int(nwin[i]), WIN_HI) for i in range(N))
    print(f"\n{tag}  on {DEV}  |  {N:,} admissions, {total:,} windows, "
          f"{len(todo)} of {(N+BLOCK-1)//BLOCK} shards to do", flush=True)
    t0 = time.time()

    for c, lo in enumerate(todo):
        hi = min(lo + BLOCK, N)
        wins, owner = [], []
        for i in range(lo, hi):
            for j in range(aoff[i], aoff[i] + min(int(nwin[i]), WIN_HI)):
                wins.append(ids[woff[j]:woff[j+1]].astype(np.int64)); owner.append(i - lo)
        n = hi - lo
        s_cls = np.zeros((n, H), np.float64); s_tok = np.zeros((n, H), np.float64)
        n_tok = np.zeros(n, np.int64); n_win = np.zeros(n, np.int64)
        order = np.argsort([len(w) for w in wins])       # length-sorted: less padding
        with torch.no_grad():
            for b in range(0, len(order), BATCH):
                sel = order[b:b+BATCH]
                L = max(len(wins[k]) for k in sel) + 2
                X = np.full((len(sel), L), PAD, np.int64)
                A = np.zeros((len(sel), L), np.int64)
                for r, k in enumerate(sel):
                    s = [CLS] + list(wins[k]) + [SEP]
                    X[r, :len(s)] = s; A[r, :len(s)] = 1
                h = model(input_ids=torch.from_numpy(X).to(DEV),
                          attention_mask=torch.from_numpy(A).to(DEV)
                          ).last_hidden_state.float().cpu().numpy()
                for r, k in enumerate(sel):
                    o = owner[k]; m = len(wins[k])
                    s_cls[o] += h[r, 0]
                    s_tok[o] += h[r, 1:1+m].sum(0)       # content tokens only
                    n_tok[o] += m; n_win[o] += 1
        np.savez(f"{d}/b_{lo:06d}.npz", hadm=hadm[lo:hi],
                 s_cls=s_cls.astype(np.float32), s_tok=s_tok.astype(np.float32),
                 n_tok=n_tok, n_win=n_win)
        el = time.time() - t0
        print(f"  {c+1:3d}/{len(todo)} shards   {el/60:5.1f} min elapsed   "
              f"{el/(c+1)*(len(todo)-c-1)/60:5.1f} min left", flush=True)
    print(f"{tag} DONE in {(time.time()-t0)/60:.1f} min -> {d}", flush=True)

for tag in (sys.argv[1:] or ["bio_clinicalbert", "pubmedbert"]):
    run(tag)
print("\nAll done. Tell Claude and it will pick the shards up from out/.")
