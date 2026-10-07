"""How contaminated is the fine-tuned embedding? -- CPU only, cached embeddings.

    python3 diag_contamination_W1.py 0        # one draw (resumable, appends)
    python3 diag_contamination_W1.py summary  # after all five

--------------------------------------------------------------------------
THE WORRY. finetune_bert_W1.py fine-tuned Bio_ClinicalBERT on each draw's
TRAINING admissions using the outcome, then encoded everyone. So the embedding
of a training admission was produced by an encoder that had partly memorised
that admission's label. On training rows the text looks more predictive than it
really is; the downstream model learns to over-trust it; on test rows it
underdelivers. That would bias the fine-tuned arm's test increments DOWNWARD,
most at T5.

THE NAIVE CHECK IS WRONG. Fitting a model on training rows and comparing its
training AUROC with its test AUROC mixes two things: the ENCODER having seen the
training rows (the thing we want) and the DOWNSTREAM model overfitting its own
training rows (which happens to every embedding). The second would show up for
the frozen embedding too and swamp the signal.

WHAT THIS DOES INSTEAD. Text-only logistic regression on the 768-dim embedding.

  test         fit on all training rows, score the test rows.
               Neither the encoder nor the downstream model saw them.

  oof_train    5-fold cross-fitting WITHIN the training rows: every training
               row is scored by a model that did not see it. Downstream
               overfitting is removed. The only remaining reason oof_train can
               beat test is that the ENCODER saw these rows.

  inflation = oof_train - test

For the FROZEN embedding the encoder saw nobody, so inflation should sit near
zero (slightly negative, even: the OOF models train on 80% of the rows the test
model gets). It is the control. The contamination estimate is

  inflation(finetuned) - inflation(frozen),  paired within draw.

A second, independent view from the SAME out-of-fold scores:

  oof_seen     training rows the encoder was fine-tuned on
  oof_unseen   the 5% validation slice finetune_bert_W1.py carved out of the
               training rows and never tuned on -- reconstructed here from the
               same seed

Same downstream models, same scoring pass; the only difference between the two
groups is whether the encoder saw them. The unseen slice is small (~960
admissions, ~200 events), so this view is noisier; it is a corroboration, not
the primary number.

Read with care: this measures inflation of TEXT-ONLY signal. The downstream
effect in the paper runs through a model that also has 21 or 223 structured
columns, and how much a given inflation costs there is not the same number.
What this can settle is whether contamination is negligible or material.
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
U = os.path.join(D, "out")
SEED, TAG, VAL_FRAC, K = 20180701, "bio_clinicalbert", 0.05, 5
FT  = os.path.join(U, "w1_finetune_v2_observable")
CSV = os.path.join(FT, "W1_contamination.csv")

tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
y = tab.labels.to_numpy().astype(int); N = len(y)

# ---- partitions and the validation slice, verbatim from finetune_bert_W1.py
def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(N, bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def val_slice(r, tr):
    rng = np.random.RandomState(SEED + r)          # same seed, same first draw
    tr_idx = np.nonzero(tr)[0]; rng.shuffle(tr_idx)
    v = np.zeros(N, bool); v[tr_idx[: int(round(VAL_FRAC * len(tr_idx)))]] = True
    return v

def frozen():
    fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-4/b_*.npz")); assert fs
    g = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("hadm", "s_tok", "n_tok")}
    p = pd.Index(g["hadm"]).get_indexer(order); assert not (p < 0).any()
    return (g["s_tok"][p] / np.maximum(g["n_tok"][p], 1)[:, None]).astype(np.float32)

def lr_scores(Xtr, ytr, Xte):
    sc = StandardScaler().fit(Xtr)
    m = LogisticRegression(max_iter=5000).fit(sc.transform(Xtr), ytr)
    return m.decision_function(sc.transform(Xte))

def run(r):
    if os.path.exists(CSV) and r in set(pd.read_csv(CSV).draw):
        print(f"draw {r}: already on disk"); return
    t0 = time.time()
    tr = draw_split(r); val = val_slice(r, tr)
    EMB = {"frozen": frozen(),
           "finetuned": np.load(f"{FT}/emb_ft_{TAG}_draw{r}.npy")}
    tri = np.nonzero(tr)[0]; tei = np.nonzero(~tr)[0]
    rows = []
    for name, E in EMB.items():
        test = roc_auc_score(y[tei], lr_scores(E[tri], y[tri], E[tei]))
        oof = np.empty(N)
        for a, b in StratifiedKFold(K, shuffle=True, random_state=SEED + r).split(tri, y[tri]):
            oof[tri[b]] = lr_scores(E[tri[a]], y[tri[a]], E[tri[b]])
        o_all  = roc_auc_score(y[tri], oof[tri])
        seen   = tr & ~val; unseen = tr & val
        o_seen = roc_auc_score(y[seen], oof[seen])
        o_uns  = roc_auc_score(y[unseen], oof[unseen])
        rows.append(dict(draw=r, emb=name, test=round(test, 4), oof_train=round(o_all, 4),
                         inflation=round(o_all - test, 4),
                         oof_seen=round(o_seen, 4), oof_unseen=round(o_uns, 4),
                         seen_minus_unseen=round(o_seen - o_uns, 4),
                         n_unseen=int(unseen.sum()), ev_unseen=int(y[unseen].sum())))
        print(f"draw {r} {name:9s} test {test:.4f}  oof_train {o_all:.4f}  "
              f"inflation {o_all-test:+.4f}   seen {o_seen:.4f} unseen {o_uns:.4f} "
              f"({unseen.sum()} adm, {y[unseen].sum()} events)", flush=True)
    pd.DataFrame(rows).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
    print(f"draw {r} done in {time.time()-t0:.0f}s")

def summary():
    d = pd.read_csv(CSV).drop_duplicates(["draw", "emb"])
    print(d.to_string(index=False))
    for col in ("inflation", "seen_minus_unseen"):
        p = d.pivot(index="draw", columns="emb", values=col)
        diff = p.finetuned - p.frozen
        se = diff.std(ddof=1) / np.sqrt(len(diff))
        print(f"\n{col}: frozen {p.frozen.mean():+.4f}  finetuned {p.finetuned.mean():+.4f}"
              f"  |  finetuned - frozen {diff.mean():+.4f}  "
              f"95% CI {diff.mean()-2.776*se:+.4f} to {diff.mean()+2.776*se:+.4f}  "
              f"positive in {(diff > 0).sum()}/{len(diff)} draws")

arg = sys.argv[1] if len(sys.argv) > 1 else "summary"
summary() if arg == "summary" else run(int(arg))
