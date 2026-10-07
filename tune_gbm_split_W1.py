"""Is the bag-of-words advantage under GBM a tuning artifact? -- check (a), 2026-09-28.

    python3 tune_gbm_split_W1.py            # draws from W1_DRAWS (default 0)
    W1_SELECT=cv3 W1_DRAWS=0,1,2,3,4 python3 tune_gbm_split_W1.py   # full version

WHY. In Design 3 partition 0, GBM ranked bag-of-words well above the fine-tuned
encoder (T1 +0.072 vs +0.052), while tuned LR had them level (+0.075 each). And
GBM loses signal on the simplest input there is: the fine-tuned model's own
probability scores 0.850 on the test patients directly, 0.852 through tuned LR,
but 0.836 through our fixed GBM (depth 10, 500 rounds, no early stopping). So
the fixed configuration may overfit on half-size data with dense inputs.

WHAT. Same partitions, halves and arms as finetune_split_W1.py. Every model is
fitted on half B and scored on the test patients. For each feature set -- text
only, T1 + text, T5 + text, and structured-only at T1 and T5 -- two GBMs:
  fixed  the project configuration, exactly as in finetune_split_W1.py
  tuned  chosen on half B only, from 6 configurations: max_depth {3, 6, 10} x
         max_features {0.3, 1.0} (share of columns each split may consider),
         learning rate 0.05, min_samples_leaf 15, up to 1,000 rounds with early
         stopping on an internal 10% slice. Choice by inner AUROC: one
         stratified 80/20 split of half B (W1_SELECT=holdout, the pilot) or
         3-fold CV (W1_SELECT=cv3, the full version).
Increments compare tuned-with-text against tuned-structured-only (each tuned
separately), exactly as the tuned LR does.

READ-OFF, fixed before any result:
  * does tuning close the bow - ft_cls gap at T1 and T5 (fixed vs tuned)?
  * does tuned GBM recover the fine-tuned probability's direct AUROC (0.850)?
  * does the collapse of each arm move? (It is the paper's claim; the ranking is not.)
Fixed and tuned are always compared within one environment. Never difference
these numbers against the Mac's W1_ftsplit.csv (GBM differs slightly by machine).

Inputs are derived features only (tiers, word counts, embeddings) -- no note text.
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
SEED = 20180701
FIXED = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
GRID = [dict(max_depth=d, max_features=f) for d in (3, 6, 10) for f in (0.3, 1.0)]
DRAWS = [int(x) for x in os.environ.get("W1_DRAWS", "0").split(",")]
SELECT = os.environ.get("W1_SELECT", "holdout"); assert SELECT in ("holdout", "cv3")
TAG, SFX, SUF = "bio_clinicalbert", "_observable", "_v2"
ARM_NAMES = ["bow", "frozen_tok", "frozen_cls", "ft_cls", "ft_tok", "ft_prob"]
OUT = os.environ.get("W1_OUT", f"{U}/w1_ftsplit{SUF}{SFX}")
os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/W1_gbmtune_{SELECT}.csv"
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)

# ---------------------------------------------------------------- DeLong (as elsewhere)
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T_ = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T_[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T_; return T2

def delong(y, scores):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(scores, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y) - m
    tx = np.vstack([_midrank(s[r, :m]) for r in range(s.shape[0])])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(s.shape[0])])
    tz = np.vstack([_midrank(s[r])     for r in range(s.shape[0])])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return aucs, (np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m
                  + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n)

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0, 0]+c[1, 1]-2*c[0, 1]
    return float(au[0]), float(au[1]), float(au[1]-au[0]), float(np.sqrt(max(v, 0)))

# ---------------------------------------------------------------- data
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1{SUF}.parquet")
Xo  = sparse.load_npz(f"{U}/text_open_W1{SUF}_aligned.npz").toarray().astype(np.float32)
tab = tab.set_index("hadm_id"); order = tab.index.to_numpy()
assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
y = tab.labels.to_numpy().astype(int); N = len(y)
fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-4/b_*.npz")); assert len(fs) == 52, len(fs)
g = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("hadm", "s_tok", "n_tok", "s_cls", "n_win")}
gp = pd.Index(g["hadm"]).get_indexer(order); assert not (gp < 0).any()
BASE = {"bow": np.hstack([FIX, Xo]),
        "frozen_tok": np.hstack([FIX, (g["s_tok"][gp] / np.maximum(g["n_tok"][gp], 1)[:, None]).astype(np.float32)]),
        "frozen_cls": np.hstack([FIX, (g["s_cls"][gp] / np.maximum(g["n_win"][gp], 1)[:, None]).astype(np.float32)])}
del g

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(N, bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def halves(r, tr):
    z = np.load(f"{U}/w1_splitsample{SUF}{SFX}/halves_draw{r}.npz"); A, B = z["A"], z["B"]
    assert not (A & B).any() and ((A | B) == tr).all()
    return A, B

def encode_tier(cols, fit):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            k = v[fit].value_counts(); lv = set(k[k >= 0.01*fit.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

# ---------------------------------------------------------------- the two GBMs
def gbm_fixed(X, fit):
    return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **FIXED
                                          ).fit(X[fit], y[fit]).decision_function(X)

def _tuned_model(cfg):
    return HistGradientBoostingClassifier(learning_rate=0.05, max_iter=1000, min_samples_leaf=15,
                                          early_stopping=True, validation_fraction=0.1,
                                          n_iter_no_change=30, random_state=SEED, **cfg)

def gbm_tuned(X, fit):
    """Choose a configuration inside `fit` (half B), refit on all of it, score every row."""
    rows = np.flatnonzero(fit); yy = y[rows]
    if SELECT == "holdout":
        a, b = train_test_split(np.arange(len(rows)), test_size=0.2, stratify=yy, random_state=SEED)
        folds = [(a, b)]
    else:
        folds = list(StratifiedKFold(3, shuffle=True, random_state=SEED).split(rows, yy))
    res = []
    for cfg in GRID:
        au = [roc_auc_score(yy[b], _tuned_model(cfg).fit(X[rows[a]], yy[a]).decision_function(X[rows[b]]))
              for a, b in folds]
        res.append(np.mean(au))
    best = GRID[int(np.argmax(res))]
    m = _tuned_model(best).fit(X[rows], yy)
    return m.decision_function(X), best, float(max(res)), int(m.n_iter_)

# ---------------------------------------------------------------- run
done = set()
if os.path.exists(CSV):
    o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.arm, o.gbm))
COLS = ["draw", "tier", "arm", "gbm", "select", "cfg", "inner_auc", "n_iter",
        "auroc", "struct", "with_text", "increment", "se"]
def emit(row):
    pd.DataFrame([row]).reindex(columns=COLS).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)

for r in DRAWS:
    tr = draw_split(r); A, B = halves(r, tr); test = ~tr
    E = np.load(f"{U}/w1_ftsplit{SUF}{SFX}/emb_draw{r}.npz")
    ARMS = dict(BASE, **{k: np.hstack([FIX, E[k]]) for k in ("ft_cls", "ft_tok", "ft_prob")})
    for t_ in ("bow_rs", "bow_rt"):            # check (b)'s re-selected phrases, if present
        rs = f"{OUT}/{t_}_draw{r}.npz"
        if os.path.exists(rs):
            ARMS[t_] = np.hstack([FIX, sparse.load_npz(rs).toarray().astype(np.float32)])
    arm_names = [a for a in ARM_NAMES + ["bow_rs", "bow_rt"] if a in ARMS]
    log(f"draw {r}: B {B.sum():,} fit rows, test {test.sum():,} | select={SELECT}")
    log(f"  ft_prob scored directly on test: {roc_auc_score(y[test], np.nan_to_num(E['ft_prob'][test, 0])):.4f}")
    for tier in ("none", "T1", "T5"):
        S = None if tier == "none" else encode_tier(tiers[tier], B)
        struct = {}
        for kind in ("fixed", "tuned"):
            if S is None: continue
            ts = time.time()
            if kind == "fixed": struct[kind] = (gbm_fixed(S, B), None, None, None)
            else: struct[kind] = gbm_tuned(S, B)
            log(f"  {tier} struct-only {kind}: test AUROC {roc_auc_score(y[test], struct[kind][0][test]):.4f}"
                f" {'' if kind == 'fixed' else struct[kind][1:]} ({time.time()-ts:.0f}s)")
        for arm in arm_names:
            X = ARMS[arm] if S is None else np.hstack([S, ARMS[arm]])
            for kind in ("fixed", "tuned"):
                if (r, tier, arm, kind) in done: continue
                ts = time.time()
                if kind == "fixed": s, cfg, cv, it = gbm_fixed(X, B), None, None, FIXED["max_iter"]
                else: s, cfg, cv, it = gbm_tuned(X, B)
                row = dict(draw=r, tier=tier, arm=arm, gbm=kind, select=SELECT,
                           cfg=json.dumps(cfg) if cfg else "fixed", inner_auc=cv, n_iter=it)
                if S is None:
                    row.update(auroc=round(roc_auc_score(y[test], s[test]), 4))
                    msg = f"text-only AUROC {row['auroc']:.4f}"
                else:
                    a, b, d, se = paired(y[test], struct[kind][0][test], s[test])
                    row.update(struct=round(a, 4), with_text=round(b, 4), increment=round(d, 4), se=round(se, 4))
                    msg = f"{a:.4f} -> {b:.4f} inc {d:+.4f}"
                emit(row)
                log(f"  {tier:4s} {arm:10s} {kind}: {msg}  cfg={row['cfg']} iters={it} ({time.time()-ts:.0f}s)")

# ---------------------------------------------------------------- summary
res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "arm", "gbm"], keep="last")
pd.set_option("display.width", 200)
print("\n=== text-only AUROC ===")
print(res[res.tier == "none"].pivot_table(index="arm", columns="gbm", values="auroc").round(4).to_string())
inc = res[res.tier != "none"].pivot_table(index="arm", columns=["gbm", "tier"], values="increment")
print("\n=== increments (mean over draws run) ===")
print(inc.round(4).to_string())
for kind in ("fixed", "tuned"):
    if (kind, "T1") in inc and (kind, "T5") in inc:
        c = 100 * (inc[(kind, "T1")] - inc[(kind, "T5")]) / inc[(kind, "T1")]
        print(f"\ncollapse, {kind} GBM:\n" + c.round(1).to_string())
log("done")
