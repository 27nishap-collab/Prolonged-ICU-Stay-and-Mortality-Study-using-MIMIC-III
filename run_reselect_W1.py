"""Round 1 of the redundancy test: re-select the 500 phrases from a vocabulary
with the disposition terms already removed.

WHY THE EXISTING `nodisp` ARM IS THE WEAK VERSION OF THIS TEST
--------------------------------------------------------------
In run_ablation_W1.py, `nodisp` deletes 23 disposition phrases from a set of 500
that had ALREADY been selected with those phrases present. The L1 selector chose
those 500 to work together; the 477 survivors were picked for what they added
*alongside* `extubated`, never *instead of* it. So "removing them costs only 8%"
is measured with the remaining vocabulary hamstrung.

This script removes the disposition terms from the 4,687-phrase CANDIDATE pool
BEFORE selection, and lets the L1 selector pick a fresh 500 from what is left --
actively hunting for substitutes. Same corpus (v2, no discharge summaries), same
training rows, same C grid, same everything else.

READING THE RESULT
  * fresh 500 recovers nearly the full increment  -> the disposition signal is
    distributed through the corpus and cannot be isolated by vocabulary; the 8%
    from `nodisp` understates how much of the signal is care-trajectory-shaped.
  * fresh 500 falls well short                    -> disposition terms carry
    something the rest of the vocabulary cannot reconstruct, and 8% was
    misleadingly small for the reason above.

The substitution list -- what the selector reaches for instead -- is printed.
NOTE ON CIRCULARITY: those substitutes are "phrases that predict the way
disposition phrases do", which is NOT the same as "phrases that ARE
disposition". They must not be fed back into the lexicon and re-run; a lexicon
has to be fixed from clinical definition, independently of the outcome.

    python run_reselect_W1.py            # writes out/text_open_W1_v3*.{npz,csv}
"""
import os, re, sys, json, time
os.environ["W1_TEXT_V2"] = "1"          # bind build_text_W1 to the v2 corpus
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score
import build_text_W1 as B

t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

# the same lexicon used in run_ablation_W1.py, applied to the CANDIDATE pool
DISP = re.compile(
    r"(extubat|transfer|floor|chair|oob|out of bed|ambulat|diet|\bpo\b|home|rehab|"
    r"discharg|wean|dnr|\bcmo\b|comfort|hospice|palliat|goals|family meeting|"
    r"code status|wishes|plan|tolerat|valuable|belongings|sign out|call out|"
    r"step down|stepdown|d/c|\bdc\b|want)")

OUT = B.OUT
meta = pd.read_parquet(B.META)
train = set(meta.hadm_id[meta.split == "train"])
v = pd.read_csv(B.VOCAB)
v["is_disp"] = v.phrase.astype(str).str.contains(DISP, regex=True, na=False)
log(f"candidate pool {len(v):,} phrases; {int(v.is_disp.sum())} match the lexicon "
    f"-> {int((~v.is_disp).sum()):,} remain")

vocab = v.loc[~v.is_disp, "phrase"].astype(str).tolist()
vi = {p: i for i, p in enumerate(vocab)}
log("building train DTM over the disposition-free pool ...")
Xtr, htr = B._dtm(vi, train)
ytr = meta.set_index("hadm_id").labels.loc[htr].to_numpy()
log(f"  {Xtr.shape[0]:,} x {Xtr.shape[1]:,}, nnz {Xtr.nnz:,}")

Xs = Xtr.copy(); Xs.data = np.log1p(Xs.data)
CS = [0.005, 0.01, 0.05, 0.1]           # same grid, trimmed to the useful range
skf = StratifiedKFold(10, shuffle=True, random_state=B.SEED)
rows = []
for C in CS:
    a = []
    for i, b in skf.split(Xs, ytr):
        m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
        m.fit(Xs[i], ytr[i]); a.append(roc_auc_score(ytr[b], m.decision_function(Xs[b])))
    m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000).fit(Xs, ytr)
    nz = int((m.coef_[0] != 0).sum())
    rows.append(dict(C=C, cv_auc=float(np.mean(a)), nonzero=nz))
    log(f"  C={C:<7g} cv AUROC {np.mean(a):.4f}  nonzero {nz:,}")
path = pd.DataFrame(rows); path.to_csv(f"{OUT}/W1_openvocab_l1_path_v3.csv", index=False)

best = path.loc[path.cv_auc.idxmax()]
usable = path[path.nonzero >= 500]
C = float(best.C) if best.nonzero >= 500 else float(usable.loc[usable.cv_auc.idxmax()].C)
m = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=4000).fit(Xs, ytr)
coef = m.coef_[0]
keep = np.argsort(-np.abs(coef))[:500]; keep = keep[coef[keep] != 0]
sel = [vocab[i] for i in keep]
log(f"selected {len(sel)} phrases at C={C:g} (cv AUROC "
    f"{float(path.loc[path.C==C,'cv_auc'].iloc[0]):.4f})")

log("building full-cohort DTM over the re-selected phrases ...")
X, hs = B._dtm({p: i for i, p in enumerate(sel)})
o = np.argsort(hs)
# align to all 25,692 cohort rows (v2 leaves 372 admissions with no in-window note)
order = np.sort(meta.hadm_id.to_numpy())
pos = {h: i for i, h in enumerate(hs[o])}
idx = np.array([pos.get(h, -1) for h in order]); have = idx >= 0
Z = sparse.lil_matrix((len(order), X.shape[1]), dtype=X.dtype)
Z[np.flatnonzero(have)] = X[o][idx[have]]
sparse.save_npz(f"{OUT}/text_open_W1_v3.npz", Z.tocsr())
pd.DataFrame({"phrase": sel, "coef": coef[keep]}).to_csv(
    f"{OUT}/text_open_vocab_W1_v3.csv", index=False)
log(f"text_open_W1_v3.npz {Z.shape} written")

old = set(pd.read_csv(f"{OUT}/text_open_vocab_W1_v2.csv").phrase.astype(str))
new_only = [p for p in sel if p not in old]
print(f"\n  {len(new_only)} of the re-selected 500 were NOT in the original 500")
srt = sorted(zip(coef[keep], sel))
print("\n  substitutes pulled in, most negative first:")
print("   ", ", ".join(f"{p} ({c:+.2f})" for c, p in srt if p in set(new_only))[:1400])
json.dump({"pool": int(len(v)), "disp_removed": int(v.is_disp.sum()),
           "pool_after": len(vocab), "selected": len(sel), "C": C,
           "new_vs_v2": len(new_only)},
          open(f"{OUT}/W1_reselect_meta.json", "w"), indent=2)
log("done")
