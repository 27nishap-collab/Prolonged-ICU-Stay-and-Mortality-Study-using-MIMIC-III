"""Is the logistic-regression problem an untuned penalty? -- one draw, CPU only.

    python3 check_lr_tuning_W1.py run        # resumable; stops at W1_BUDGET seconds
    python3 check_lr_tuning_W1.py summary

--------------------------------------------------------------------------
WHY. Every logistic regression behind a reported result uses scikit-learn's
default L2 penalty, C = 1, for every model regardless of size -- 21 columns or
~1,000. Two symptoms say that is too weak once text is added:
  * bag-of-words increments go NEGATIVE under LR at T5, and turn positive when
    the same information is compressed to 300 LSA dimensions (2026-09-19);
  * halving the downstream training set pushes the LR collapse up 20-29 points,
    while gradient boosting does not move (check_halfsize_W1.py, 2026-09-26).
If that diagnosis is right, the current LR arm FLATTERS the collapse hypothesis,
which is the wrong direction for a robustness check.

WHAT THIS DOES. Draw 0 only. For full-size and half-size (half B from
out/w1_splitsample_v2_observable/halves_draw0.npz, the same halves Design 3
would use), T1 and T5, bag-of-words and frozen:

  lr_default   LogisticRegression(max_iter=5000)   -- C = 1, as now
  lr_tuned     L2 penalty, C chosen by 5-fold stratified CV inside the rows the
               model is fitted on, over 13 values from 1e-4 to 1e2, scored by
               AUROC, then refit on all of those rows.

The structured-only baseline is tuned the same way, separately. So the tuned
increment compares the best LR with text against the best LR without it. No
test patient influences any choice of C.

One simplification, deliberate: imputation means and scaling are fitted on all
fit rows before the inner CV, so they see the inner validation folds. That can
only nudge which C is chosen, never the test evaluation, and at thousands of
rows the effect on a mean and a standard deviation is negligible.

DECISION RULE, fixed before any result (2026-09-27), as stated in chat:
  DIAGNOSIS CONFIRMED if tuned LR, on this draw,
    (a) keeps the full-size T5 increment at or above zero for bow and frozen
        (point estimate >= 0; SE shown alongside), and
    (b) shrinks the half-size distortion to within the bound gradient boosting
        met: bow collapse shift <= 5 points, frozen T5 increment shift <= 0.005.
  DIAGNOSIS WRONG if tuning barely changes anything: tuned and default T5
    increments within 0.003 of each other and chosen C close to 1.
  Otherwise PARTIAL -- say which part held.

ONE DRAW IS A POINT ESTIMATE. Confirmation licenses the full re-run of every LR
number; it is not itself a reportable result.
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(D, "out")
SEED, TAG, SFX, SUF, R = 20180701, "bio_clinicalbert", "_observable", "_v2", 0
CS = np.logspace(-4, 2, 13)
N_JOBS = int(os.environ.get("W1_JOBS", 4))
HALVES = f"{U}/w1_splitsample{SUF}{SFX}/halves_draw{R}.npz"
OUT = f"{U}/w1_lrtune{SUF}{SFX}"; os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/W1_lrtune.csv"
BUDGET = float(os.environ.get("W1_BUDGET", 60))
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.0f}s] {m}", flush=True)

# ---------------------------------------------------------------- DeLong
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
    m = int(y.sum()); n = len(y) - m; k = s.shape[0]
    tx = np.vstack([_midrank(s[r, :m]) for r in range(k)])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(k)])
    tz = np.vstack([_midrank(s[r])     for r in range(k)])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return aucs, (np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m
                  + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n)

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0,0]+c[1,1]-2*c[0,1]
    d = au[1]-au[0]; se = float(np.sqrt(max(v, 0)))
    return float(au[0]), float(au[1]), float(d), float(se)

# ---------------------------------------------------------------- data (verbatim)
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
fs = sorted(glob.glob(f"{U}/bert_{TAG}_w0-4/b_*.npz")); assert fs
g = {k: np.concatenate([np.load(f)[k] for f in fs]) for k in ("hadm", "s_tok", "n_tok")}
gp = pd.Index(g["hadm"]).get_indexer(order); assert not (gp < 0).any()
ARMS = {"bow":    np.hstack([FIX, Xo]),
        "frozen": np.hstack([FIX, (g["s_tok"][gp] /
                             np.maximum(g["n_tok"][gp], 1)[:, None]).astype(np.float32)])}

def draw_split(r):
    rng = np.random.RandomState(SEED + 1000 * (r + 1))
    tr = np.ones(len(y), bool)
    for lab in (0, 1):
        idx = np.flatnonzero(y == lab); rng.shuffle(idx)
        tr[idx[: int(round(0.25 * len(idx)))]] = False
    return tr

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

def score(X, fit, pen):
    Xtr, ytr = X[fit], y[fit]
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A); As, Bs = sc.transform(A), sc.transform(B)
    if pen == "default":
        return LogisticRegression(max_iter=5000).fit(As, ytr).decision_function(Bs), 1.0
    m = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="roc_auc", penalty="l2", max_iter=5000,
                             n_jobs=N_JOBS).fit(As, ytr)
    return m.decision_function(Bs), float(m.C_[0])

tr = draw_split(R)
assert os.path.exists(HALVES), "run check_halfsize_W1.py first -- Design 3's halves live there"
z = np.load(HALVES); A_, B_ = z["A"], z["B"]
assert not (A_ & B_).any() and ((A_ | B_) == tr).all()
FIT = {"full": tr, "half": B_}

def base(size, tier, pen, S):
    f = f"{OUT}/s_{size}_{tier}_{pen}_struct.npz"
    if os.path.exists(f):
        q = np.load(f); return q["s"], float(q["C"])
    s, C = score(S, FIT[size], pen); np.savez(f, s=s, C=C); return s, C

def run():
    done = set()
    if os.path.exists(CSV):
        o = pd.read_csv(CSV); done = set(zip(o["size"], o.tier, o.arm, o.penalty))
    todo = [(z_, t, a, p) for p in ("default", "tuned") for z_ in ("full", "half")
            for t in ("T1", "T5") for a in ("bow", "frozen") if (z_, t, a, p) not in done]
    log(f"{len(done)} cells on disk, {len(todo)} to go")
    for i, (size, tier, arm, pen) in enumerate(todo):
        if time.time() - t0 > BUDGET:
            log(f"budget reached; {len(todo) - i} cells left -- rerun"); return
        fit = FIT[size]; S = encode_tier(tiers[tier], fit)
        sb, Cb = base(size, tier, pen, S)
        st, Ct = score(np.hstack([S, ARMS[arm]]), fit, pen)
        a, b, d, se = paired(y[~tr], sb[~tr], st[~tr])
        pd.DataFrame([dict(size=size, tier=tier, arm=arm, penalty=pen, n_fit=int(fit.sum()),
                           C_struct=Cb, C_text=Ct, struct=round(a, 4), with_text=round(b, 4),
                           increment=round(d, 4), se=round(se, 4))]
                     ).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
        log(f"{pen:7s} {size} {tier}/{arm:6s}  C struct {Cb:.0e} text {Ct:.0e}  "
            f"struct {a:.4f} -> {b:.4f}  inc {d:+.4f} (se {se:.4f})")
    log("all cells done")

def summary():
    R_ = pd.read_csv(CSV).drop_duplicates(["size", "tier", "arm", "penalty"])
    assert len(R_) == 16, f"incomplete: {len(R_)} of 16 cells"
    edge = R_[(R_.penalty == "tuned") & ((R_[["C_struct", "C_text"]].min(1) <= CS[0]) |
                                         (R_[["C_struct", "C_text"]].max(1) >= CS[-1]))]
    print(R_.sort_values(["penalty", "arm", "size", "tier"]).to_string(index=False))
    if len(edge): print(f"\n!! {len(edge)} tuned fits chose C at the edge of the grid")
    print("\n=== draw 0: collapse and the half-size shift ===")
    v = {}
    for pen in ("default", "tuned"):
        for arm in ("bow", "frozen"):
            q = R_[(R_.penalty == pen) & (R_.arm == arm)].set_index(["size", "tier"]).increment
            coll = {s: 100*(q[(s, "T1")] - q[(s, "T5")]) / q[(s, "T1")] for s in ("full", "half")}
            v[(pen, arm)] = dict(t5_full=q[("full", "T5")], shift=coll["half"] - coll["full"],
                                 t5_shift=q[("half", "T5")] - q[("full", "T5")])
            print(f"  {pen:7s} {arm:6s}  T1 {q[('full','T1')]:+.4f}  T5 {q[('full','T5')]:+.4f}  "
                  f"collapse full {coll['full']:6.1f}%  half {coll['half']:6.1f}%  "
                  f"shift {coll['half']-coll['full']:+6.1f} pts   T5 shift {v[(pen, arm)]['t5_shift']:+.4f}")
    t = {a: v[("tuned", a)] for a in ("bow", "frozen")}
    a_ok = t["bow"]["t5_full"] >= 0 and t["frozen"]["t5_full"] >= 0
    b_ok = abs(t["bow"]["shift"]) <= 5 and abs(t["frozen"]["t5_shift"]) <= 0.005
    same = all(abs(v[("tuned", a)]["t5_full"] - v[("default", a)]["t5_full"]) < 0.003
               for a in ("bow", "frozen"))
    print(f"\n  (a) tuned full-size T5 >= 0 for both arms:            {a_ok}")
    print(f"  (b) tuned half-size shift inside the GBM bounds:        {b_ok}")
    print(f"      tuned and default T5 within 0.003 (diagnosis wrong?): {same}")
    print("\nVERDICT:", "CONFIRMED" if a_ok and b_ok else
          "WRONG (tuning changes little)" if same else "PARTIAL")

{"run": run, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
