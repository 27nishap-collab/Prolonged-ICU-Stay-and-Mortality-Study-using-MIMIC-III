"""Does truncation explain where the fine-tuned encoder trails bag-of-words? -- 2026-09-29.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3.11 coverage_split_W1.py            # ~15-30 min, CPU only; safe alongside reselect

WHY. The fine-tuned encoder reads at most 4 windows of 510 wordpieces per admission
(45% of all note text; 45% of admissions are read in full), and was fine-tuned on
at most 2 (22%). Bag-of-words counts phrases over ALL the text. If the missing
text is why the encoder trails, it should trail more on admissions it only partly
read than on admissions it read in full. F13 ran a related test on the FROZEN
encoder (median word-count split, vs LSA, old default LR) and found no length
effect; this is the direct version for the fine-tuned arm.

DESIGN. Design 3 partitions 0-4; every model fitted on half B, scored on test,
exactly as in finetune_split_W1.py. Test patients are split by coverage:
    full     the admission has <= 4 windows -- the encoder read all of it
    partial  > 4 windows -- some text was never read
    (admissions with no in-window note are left out of both groups)
Arms, each with the 22 fixed text features (which include total word count):
    ft_cls      fine-tuned [CLS] readout, the pre-registered primary
    frozen_cls  the frozen encoder's [CLS] readout (same 4-window truncation)
    bow_rs      500 phrases re-chosen on half A (no test labels; same patients the
                fine-tuned encoder learned from) -- the fair bag-of-words
Settings: text only, T1 + text, T5 + text; learners GBM (fixed) and tuned LR.

WHAT IS MEASURED. Within each coverage group, the paired AUROC difference between
two arms' with-text models (DeLong). Within a setting, both arms share the same
structured model, so this equals the difference in their increments. Then
    interaction = (arm - bow_rs) on partial  minus  (arm - bow_rs) on full.
The two groups are disjoint patients, so its variance is the sum of the two.

READ-OFF, fixed before any result:
  * truncation predicts a NEGATIVE interaction for ft_cls - bow_rs: the encoder
    does relatively worse where it read less. Primary cells: text only and T1,
    both learners; mean over 5 partitions with a 95% t CI (4 df).
  * CI below zero in the primary cells -> truncation costs the encoder; the
    fine-tuned numbers understate a full-text model, and the ranking claim needs
    that caveat.
  * CI covering zero -> no detectable truncation effect at this power; say so,
    with the CI width.
  * frozen_cls - bow_rs is the same test for the frozen encoder (extends F13).
Subgroup AUROCs are not comparable across groups in absolute terms (the groups
differ in severity and event rate); only the within-group paired differences are.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from lr_tuned import lr_tuned_scores
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
OUTD = f"{U}/w1_ftsplit_v2_observable"
CSV = f"{OUTD}/W1_coverage_split.csv"
SEED, TAG, ENC_BAND = 20180701, "bio_clinicalbert", 4
GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
DRAWS = [int(x) for x in os.environ.get("W1_DRAWS", "0,1,2,3,4").split(",")]
ARMS_ = ["ft_cls", "frozen_cls", "bow_rs"]; REF = "bow_rs"
COLS = ["draw", "setting", "model", "arm", "group", "n", "events", "auc_arm", "auc_ref", "diff", "se"]
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; n = len(x); T_ = np.zeros(n); i = 0
    while i < n:
        j = i
        while j < n and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(n); T2[J] = T_; return T2

def paired(yt, a, b):
    """AUROC of a, of b, their difference, and its DeLong se."""
    s = np.vstack([a, b]).astype(float); o = np.argsort(-yt, kind="mergesort"); s = s[:, o]
    m = int(yt.sum()); n = len(yt) - m
    tx = np.vstack([_midrank(s[k, :m]) for k in range(2)])
    ty = np.vstack([_midrank(s[k, m:]) for k in range(2)])
    tz = np.vstack([_midrank(s[k]) for k in range(2)])
    au = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    c = np.cov((tz[:, :m]-tx)/n)/m + np.cov(1.0-(tz[:, m:]-ty)/m)/n
    return float(au[0]), float(au[1]), float(au[0]-au[1]), float(np.sqrt(max(c[0,0]+c[1,1]-2*c[0,1], 0)))

# ---------------------------------------------------------------- data (as finetune_split_W1.py)
tiers = json.load(open(f"{U}/W1_tier_columns_observable.json"))
T   = pd.read_parquet(f"{U}/tiers_W1_observable.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
y = tab.labels.to_numpy().astype(int); N = len(y)

z = np.load(f"{U}/tok16_{TAG}_W1.npz")
tp = pd.Index(z["hadm"]).get_indexer(order); assert not (tp < 0).any()
nwin = z["nwin"].astype(int)[tp]
GROUP = np.where(nwin == 0, "none", np.where(nwin <= ENC_BAND, "full", "partial"))
log(f"coverage: full {(GROUP=='full').mean():.1%}, partial {(GROUP=='partial').mean():.1%}, "
    f"no notes {(GROUP=='none').mean():.1%} | event rate full {y[GROUP=='full'].mean():.3f}, "
    f"partial {y[GROUP=='partial'].mean():.3f}")

import glob
fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-{ENC_BAND}/b_*.npz")); assert fs
g = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("hadm", "s_cls", "n_win")}
gp = pd.Index(g["hadm"]).get_indexer(order); assert not (gp < 0).any()
FROZEN_CLS = np.hstack([FIX, (g["s_cls"][gp] / np.maximum(g["n_win"][gp], 1)[:, None]).astype(np.float32)])
del g

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

# ---------------------------------------------------------------- run
done = set()
if os.path.exists(CSV):
    o = pd.read_csv(CSV); done = set(zip(o.draw, o.setting, o.model))
for r in DRAWS:
    tr = draw_split(r); A, B = halves(r, tr); te = ~tr
    E = np.load(f"{OUTD}/emb_draw{r}.npz")
    ARMS = {"ft_cls": np.hstack([FIX, E["ft_cls"]]), "frozen_cls": FROZEN_CLS,
            "bow_rs": np.hstack([FIX, sparse.load_npz(f"{OUTD}/bow_rs_draw{r}.npz").toarray().astype(np.float32)])}
    for setting in ("text", "T1", "T5"):
        S = None if setting == "text" else encode_tier(tiers[setting], B)
        for mk in ("gbm", "lrcv"):
            if (r, setting, mk) in done: continue
            sc = {a: score(ARMS[a] if S is None else np.hstack([S, ARMS[a]]), B, mk) for a in ARMS_}
            rows = []
            for arm in ("ft_cls", "frozen_cls"):
                for grp in ("full", "partial", "all"):
                    m = te & (GROUP != "none") if grp == "all" else te & (GROUP == grp)
                    a1, a2, d, se = paired(y[m], sc[arm][m], sc[REF][m])
                    rows.append(dict(draw=r, setting=setting, model=mk, arm=arm, group=grp, n=int(m.sum()),
                                     events=int(y[m].sum()), auc_arm=round(a1, 4), auc_ref=round(a2, 4),
                                     diff=round(d, 4), se=round(se, 4)))
            pd.DataFrame(rows).reindex(columns=COLS).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
            q = {(x["arm"], x["group"]): x["diff"] for x in rows}
            log(f"draw {r} {setting:4s} {mk:4s}: ft_cls-bow_rs full {q[('ft_cls','full')]:+.4f} "
                f"partial {q[('ft_cls','partial')]:+.4f} | frozen_cls-bow_rs full {q[('frozen_cls','full')]:+.4f} "
                f"partial {q[('frozen_cls','partial')]:+.4f}")

# ---------------------------------------------------------------- summary
res = pd.read_csv(CSV).drop_duplicates(["draw", "setting", "model", "arm", "group"], keep="last")
print("\n=== interaction = (arm - bow_rs) on partially read  minus  on fully read ===")
print("    negative = the arm does relatively worse where it read less (what truncation predicts)\n")
for arm in ("ft_cls", "frozen_cls"):
    for setting in ("text", "T1", "T5"):
        for mk in ("gbm", "lrcv"):
            q = res[(res.arm == arm) & (res.setting == setting) & (res.model == mk)]
            p = q.pivot(index="draw", columns="group", values="diff")
            s = q.pivot(index="draw", columns="group", values="se")
            if not {"full", "partial"} <= set(p.columns): continue
            it = p.partial - p.full; k = len(it)
            h = stats.t.ppf(0.975, k-1) * it.std(ddof=1) / np.sqrt(k) if k > 1 else np.nan
            zs = it / np.sqrt(s.partial**2 + s.full**2)
            print(f"  {arm:10s} {setting:4s} {mk:4s}  full {p.full.mean():+.4f}  partial {p.partial.mean():+.4f}"
                  f"  interaction {it.mean():+.4f} [{it.mean()-h:+.4f}, {it.mean()+h:+.4f}]"
                  f"  negative in {(it < 0).sum()}/{k}; per-draw |z|>1.96 in {(zs.abs() > 1.96).sum()}")
g_ = res[(res.draw == res.draw.min()) & (res.setting == "text") & (res.model == "gbm") & (res.arm == "ft_cls")]
print("\n  group sizes (test patients, one partition):",
      dict(zip(g_.group, zip(g_.n, g_.events))))
log("done -- tell Claude and it will read the CSV back")
