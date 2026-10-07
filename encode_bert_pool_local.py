"""Max-pooled contextual-encoder embeddings for the W1 corpus -- RUN THIS ON THE MAC.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    HF_HUB_OFFLINE=1 python3 encode_bert_pool_local.py pubmedbert

    # both encoders, ~90 min total:
    HF_HUB_OFFLINE=1 python3 encode_bert_pool_local.py

WHY THIS EXISTS. `encode_bert_local.py` stores only SUMS (s_cls, s_tok, n_tok,
n_win), and a maximum cannot be recovered from a sum. Testing max-pooling
therefore needs a second forward pass -- roughly 46 min per model on MPS, the
same cost as the original encode. This script is that pass. It is a copy of
`encode_bert_local.py` with the accumulators extended; the encoding itself --
same tokenisation, same window budget, same order, same batching -- is
unchanged, so the new arms are directly comparable with the existing ones.

WHAT IT ADDS, on top of everything the original stored:

    m_tok  element-wise MAX over every content token in every window encoded
    m_win  element-wise MAX over the per-WINDOW mean vectors

Two different questions. `m_tok` asks "did any single token anywhere in this
admission's notes fire on this dimension" -- a detector. `m_win` asks "was there
a window whose overall content leaned this way" -- a coarser detector that is
less hostage to one outlier token. The mechanism under test is that mean-pooling
buries a handful of decisive tokens under ~8,000 others AND rotates the signal
off the coordinate axes, which is near worst-case for the axis-parallel splits a
tree makes. If that account is right, max-pooling should recover a good part of
the GBM deficit and do comparatively little under LR. If both stay flat, the
account is wrong and the anisotropy story has to carry the explanation alone.

s_cls / s_tok / n_tok / n_win are carried through unchanged, so these shards are
self-contained: the mean and CLS arms can be rebuilt from them without touching
the originals.

MAXIMA COMBINE ACROSS BANDS, like the sums, because max(A u B) = max(max A,
max B) and the window budgets are nested prefixes. Raising W1_WIN_HI later and
taking the element-wise max of the two bands' m_tok is valid.

Admissions with no in-window notes (372 of them under the v2 corpus) have no
windows at all; their max vectors are written as ZERO, matching what
s_tok/max(n_tok,1) already gives for the mean.

OUTPUT. Shards of 500 admissions in out/bertpool_<model>_w0-<k>/ -- a DIFFERENT
directory from the original bert_<model>_w0-<k>/, which is left untouched.
Resumable: a re-run skips shards already on disk. Ctrl-C and restart is safe.
"""
import os, sys, time
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
    d = f"{OUT}/bertpool_{tag}_w0-{WIN_HI}"; os.makedirs(d, exist_ok=True)

    repo = HF[tag]
    model = AutoModel.from_pretrained(repo).eval().to(DEV)
    H = model.config.hidden_size
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
        n_tok = np.zeros(n, np.int64);        n_win = np.zeros(n, np.int64)
        m_tok = np.full((n, H), -np.inf, np.float64)   # max over content tokens
        m_win = np.full((n, H), -np.inf, np.float64)   # max over per-window means
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
                    tok = h[r, 1:1+m]                    # content tokens only
                    s_cls[o] += h[r, 0]
                    s_tok[o] += tok.sum(0)
                    n_tok[o] += m; n_win[o] += 1
                    np.maximum(m_tok[o], tok.max(0),  out=m_tok[o])
                    np.maximum(m_win[o], tok.mean(0), out=m_win[o])
        empty = n_win == 0                               # no in-window notes at all
        m_tok[empty] = 0.0; m_win[empty] = 0.0
        np.savez(f"{d}/b_{lo:06d}.npz", hadm=hadm[lo:hi],
                 s_cls=s_cls.astype(np.float32), s_tok=s_tok.astype(np.float32),
                 m_tok=m_tok.astype(np.float32), m_win=m_win.astype(np.float32),
                 n_tok=n_tok, n_win=n_win)
        el = time.time() - t0
        print(f"  {c+1:3d}/{len(todo)} shards   {el/60:5.1f} min elapsed   "
              f"{el/(c+1)*(len(todo)-c-1)/60:5.1f} min left", flush=True)
    print(f"{tag} DONE in {(time.time()-t0)/60:.1f} min -> {d}", flush=True)

for tag in (sys.argv[1:] or ["bio_clinicalbert", "pubmedbert"]):
    run(tag)
print("\nAll done. Tell Claude and it will pick the shards up from out/.")
