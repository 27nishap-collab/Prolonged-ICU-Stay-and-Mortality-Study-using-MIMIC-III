"""Default-penalty logistic regression with the phrases chosen INSIDE each draw (2026-10-05).

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3.11 run_lr_default_reselect_W1.py 2>&1 | tee lr_default_reselect.log   # CPU, est. 10-30 min
    python3.11 run_lr_default_reselect_W1.py summary

WHY. Supplement Table S1 compares the default L2 penalty (C = 1) with the CV-tuned
penalty, but over the ten draws with ONE phrase selection reused across draws
(out/observable_v2/W1_repeated.csv, models 'lr' and 'lrcv'). The slimmed S2 no
longer reports the reused selection, so Table S1 is rebuilt on the per-draw
selection that the main text uses. The tuned row already exists
(W1_repeated_reselect.csv, model 'lrcv', arm 'rechosen' = Table 3, 97.1%). Only the
default row is missing; this script fits it.

WHAT. Identical to run_repeated_reselect_W1.py (observable spec, notes-no-discharge
corpus, same draw_split seeds, same tiers and encoding, same paired DeLong) except:
  * learner = the pre-2026-09-27 default LR, copied from run_repeated_splits_W1.py:
    train-mean imputation, StandardScaler, LogisticRegression(max_iter=5000), C = 1;
  * NO phrase selection: every draw must already have
    out/observable_v2/reselect/phrases_draw{r}.npz (asserted, never re-chosen);
  * both arms are fitted: 'rechosen' (the new Table S1 row) and 'orig' (the reused
    500 phrases), the latter ONLY as an anchor -- it must reproduce the 'lr' rows of
    W1_repeated.csv (Table S1's old default row, 99.2%). Machine differences of a
    few 1e-4 are expected; the summary prints the largest.
Nothing in the main text is refitted or overwritten.

OUTPUT (new file; existing files are only read):
  out/observable_v2/W1_repeated_reselect_lrdefault.csv   one row per draw x tier x arm

READ-OFF, fixed before any result: for each arm, mean gain at T1 and T5, collapse
(T1-T5)/T1 with a 95% t CI (9 df), and the number of draws with gain at T5 > 0;
plus default minus tuned gain at T5, paired within draw, against the 'lrcv'
rechosen rows. The main-text Methods sentence ("a penalty suited to 21 columns ...
reduces the measured gain from text") predicts default < tuned at T5.
"""
import os, sys, json, time, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__))
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
OUT = f"{U}/observable_v2"; SELD = f"{OUT}/reselect"
ORIG = f"{OUT}/W1_repeated.csv"                     # 'lr' rows = old Table S1 default row
TUNED = f"{OUT}/W1_repeated_reselect.csv"           # 'lrcv' rechosen rows = Table 3 / new tuned row
CSV = f"{OUT}/W1_repeated_reselect_lrdefault.csv"
SEED = 20180701
R = int(os.environ.get("W1_R", 10))                  # a smoke test can set W1_R=1
TIERS = ["T1", "T2", "T3", "T4", "T5"]
ARMS_ = ["rechosen", "orig"]
COLS = ["draw", "tier", "n_vars", "model", "arm", "struct", "with_text", "increment", "se"]
MODE = sys.argv[1] if len(sys.argv) > 1 else "run"; assert MODE in ("run", "summary")
t0 = time.time()
def log(m): print(f"[{(time.time()-t0)/60:6.1f}m] {m}", flush=True)

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

def ci(v):
    v = np.asarray(v, float); h = stats.t.ppf(0.975, len(v)-1) * v.std(ddof=1) / np.sqrt(len(v))
    return v.mean(), v.mean() - h, v.mean() + h

def summary():
    res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "arm"], keep="last")
    print(f"\n=== default-penalty LR (C = 1), {res.draw.nunique()} draws ===")
    piv = {a: res[res.arm == a].pivot(index="draw", columns="tier", values="increment") for a in ARMS_}
    for a, p in piv.items():
        if not {"T1", "T5"} <= set(p.columns): continue
        c = 100 * (p.T1 - p.T5) / p.T1; m, lo, hi = ci(c)
        print(f"  {a:9s} gain T1 {p.T1.mean():+.4f}  T5 {p.T5.mean():+.4f}  "
              f"decline {m:5.1f}% ({lo:.1f}-{hi:.1f})  T5 > 0 in {int((p.T5 > 0).sum())}/{len(p)}")
    if os.path.exists(ORIG):                          # anchor: orig arm vs the existing 'lr' rows
        f0 = pd.read_csv(ORIG); f0 = f0[f0.model == "lr"].set_index(["draw", "tier"])
        j = res[res.arm == "orig"].set_index(["draw", "tier"]).join(f0[["struct", "increment"]], rsuffix="_file", how="inner")
        if len(j):
            print(f"\n  anchor vs W1_repeated.csv 'lr': {len(j)} cells, max |struct diff| "
                  f"{(j.struct - j.struct_file).abs().max():.4f}, max |increment diff| "
                  f"{(j.increment - j.increment_file).abs().max():.4f}")
    if os.path.exists(TUNED):                         # the comparison Table S1 needs
        tu = pd.read_csv(TUNED); tu = tu[(tu.model == "lrcv") & (tu.arm == "rechosen")]
        pt = tu.pivot(index="draw", columns="tier", values="increment")
        pd_ = piv["rechosen"].reindex(pt.index)
        if {"T1", "T5"} <= set(pt.columns) and pd_.notna().all().all():
            c = 100 * (pt.T1 - pt.T5) / pt.T1; m, lo, hi = ci(c)
            print(f"  tuned     gain T1 {pt.T1.mean():+.4f}  T5 {pt.T5.mean():+.4f}  "
                  f"decline {m:5.1f}% ({lo:.1f}-{hi:.1f})  T5 > 0 in {int((pt.T5 > 0).sum())}/{len(pt)}  [existing file]")
            for t in ("T1", "T5"):
                m, lo, hi = ci(pd_[t] - pt[t])
                print(f"  default - tuned, gain at {t}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]  "
                      f"default lower in {int((pd_[t] < pt[t]).sum())}/{len(pt)}")
    log("done -- tell Claude and it will read the CSV back")

if MODE == "summary":
    summary(); sys.exit(0)

# ---------------------------------------------------------------- data (as run_repeated_reselect_W1.py, observable/v2)
tiers = json.load(open(f"{U}/W1_tier_columns_observable.json"))
T   = pd.read_parquet(f"{U}/tiers_W1_observable.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
y = tab.labels.to_numpy().astype(int); N = len(y)
XO = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)
assert XO.shape[0] == N
missing = [r for r in range(R) if not os.path.exists(f"{SELD}/phrases_draw{r}.npz")]
assert not missing, f"no saved per-draw phrases for draws {missing}; run run_repeated_reselect_W1.py first"
log(f"observable / v2 | {N:,} admissions | {R} draws | default LR (C = 1) | -> {CSV}")

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(N, bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

def encode(cols, tr):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            k = v[tr].value_counts(); lv = set(k[k >= 0.01*tr.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

def score(X, tr):                                     # = run_repeated_splits_W1.py, model 'lr'
    Xtr, ytr = X[tr], y[tr]
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr).decision_function(sc.transform(B))

done = set()
if os.path.exists(CSV):
    o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.arm))
    log(f"resuming: {len(done)} of {R*len(TIERS)*len(ARMS_)} cells on disk")
for r in range(R):
    if all((r, t, a) in done for t in TIERS for a in ARMS_): continue
    tr = draw_split(r); te = ~tr
    j = json.load(open(f"{SELD}/phrases_draw{r}.json")); assert j["draw"] == r
    XR = sparse.load_npz(f"{SELD}/phrases_draw{r}.npz").toarray().astype(np.float32); assert XR.shape == (N, 500)
    TEXTS = {"rechosen": np.hstack([FIX, XR]), "orig": np.hstack([FIX, XO])}
    log(f"draw {r}: train {tr.sum():,} / test {te.sum():,}")
    for tier in TIERS:
        if all((r, tier, a) in done for a in ARMS_): continue
        S = encode(tiers[tier], tr); sb = score(S, tr); got = {}
        for arm in ARMS_:
            if (r, tier, arm) in done: continue
            a, b, d, se = paired(y[te], sb[te], score(np.hstack([S, TEXTS[arm]]), tr)[te]); got[arm] = d
            pd.DataFrame([dict(draw=r, tier=tier, n_vars=len(tiers[tier]), model="lr", arm=arm,
                               struct=round(a, 4), with_text=round(b, 4), increment=round(d, 4),
                               se=round(se, 4))]).reindex(columns=COLS).to_csv(
                CSV, mode="a", header=not os.path.exists(CSV), index=False)
        log(f"  {tier}: struct {a:.4f} | " + "  ".join(f"{k} {v:+.4f}" for k, v in got.items()))
summary()
