"""Does halving the downstream training set distort the collapse? -- CPU only.

    python3 check_halfsize_W1.py verify     # reproduce 2 full-size cells first
    python3 check_halfsize_W1.py run        # resumable; stops at W1_BUDGET seconds
    python3 check_halfsize_W1.py summary

--------------------------------------------------------------------------
WHY. Design 3 (split-sample) fine-tunes the encoder on half A of each draw's
training patients and fits the downstream model on half B only. Before spending
GPU hours, this checks the downstream half of that design using the two arms
that need no fine-tuning -- bag-of-words and the frozen encoder -- refit on
half B, scored on the SAME test patients, and compared with their full-size
results on the same draws.

The worry is directional. With ~2,000 events and up to ~1,000 columns at T5, a
half-sized model may use text columns less well against 223 structured
variables than against 21. That would shrink T5 more than T1 and EXAGGERATE the
collapse -- a bias toward our own hypothesis.

The FROZEN arm is the one to watch: Design 3's fine-tuned embedding has the
same shape (768 dense dims), so if halving hurts a model's use of dense
embeddings, frozen shows it. Its full-size T5 increment is ~0, so its collapse
percentage is insensitive; read its T5 INCREMENT shift instead.

DECISION RULE, fixed before any result (2026-09-26):
  Paired within draw, half-size minus full-size, 95% CI on 5 draws (t, 4 df).
  * bow GBM collapse shifts by <= 5 percentage points, and
  * frozen GBM T5 increment shifts by <= 0.005
    -> halving does not materially distort; Design 3 is sound on the
       downstream side.
  * either bound exceeded with its CI excluding zero
    -> Design 3 would distort the collapse; Design 2 earns its cost.
  * in between (point estimate over the bound, CI covering zero)
    -> inconclusive at 5 draws; say so rather than pick.

VERIFY FIRST. Full-size results come from finetune_bert_W1.py, which ran in
your native Python; this runs in the Cowork VM (scikit-learn 1.7.2). If the two
environments disagree, half-vs-full would compare across library versions. So
`verify` refits two full-size cells here and checks them against the CSV. If
they match to rounding, the CSV's full-size rows are reused; if not, stop.

  RESULT (2026-09-26): MISMATCH. Draw 0, T1, GBM: structured-only 0.8371 here
  vs 0.8364 in the CSV; with bow 0.9039 vs 0.9034. HistGradientBoosting differs
  slightly between the two environments (~0.0007 AUROC) -- negligible for any
  reported result, not negligible for a half-vs-full SHIFT. So full-size is
  RECOMPUTED here too, and the comparison runs entirely within this environment.
  The fine-tuning CSV is not used by `run` or `summary`.

HALVES. Within each draw's training patients, stratified on the outcome, seed
SEED + 5000 + draw, saved to halves_draw{r}.npz so Design 3 can reuse them
exactly. All train-only preprocessing (rare-level collapsing, imputation,
scaling) uses half B, as Design 3's downstream would.

Functions copied verbatim from finetune_bert_W1.py, with `tr` replaced by the
rows each model is fitted on.
"""
import os, sys, json, glob, time, warnings
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); U = os.path.join(D, "out")
SEED, TAG = 20180701, "bio_clinicalbert"
GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
SFX, SUF = "_observable", "_v2"
DRAWS, TIERS, ARMS_ = range(5), ["T1", "T5"], ["bow", "frozen"]
FULL = f"{U}/w1_finetune{SUF}{SFX}/W1_finetune.csv"
OUT = f"{U}/w1_splitsample{SUF}{SFX}"; os.makedirs(OUT, exist_ok=True)
CSV = f"{OUT}/W1_halfsize.csv"
BUDGET = float(os.environ.get("W1_BUDGET", 110))
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

def halves(r, tr):
    f = f"{OUT}/halves_draw{r}.npz"
    if os.path.exists(f):
        z = np.load(f); A, B = z["A"], z["B"]
    else:
        rng = np.random.RandomState(SEED + 5000 + r); A = np.zeros(N, bool)
        for lab in (0, 1):
            idx = np.flatnonzero(tr & (y == lab)); rng.shuffle(idx)
            A[idx[: len(idx) // 2]] = True
        B = tr & ~A; np.savez(f, A=A, B=B)
    assert not (A & B).any() and ((A | B) == tr).all() and not (A & ~tr).any()
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

def score(X, fit, mk):
    Xtr, ytr = X[fit], y[fit]
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(Xtr, ytr).decision_function(X)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr
           ).decision_function(sc.transform(B))

def cached(tag, fn):
    f = f"{OUT}/s_{tag}.npy"
    if os.path.exists(f): return np.load(f)
    s = fn(); np.save(f, s); return s

# ---------------------------------------------------------------- modes
def verify():
    full = pd.read_csv(FULL).drop_duplicates(["draw", "tier", "model", "arm"])
    r = 0; tr = draw_split(r); S = encode_tier(tiers["T1"], tr)
    for mk in ("gbm", "lr"):
        a, b, d, _ = paired(y[~tr], score(S, tr, mk)[~tr],
                            score(np.hstack([S, ARMS["bow"]]), tr, mk)[~tr])
        ref = full.query("draw==0 and tier=='T1' and model==@mk and arm=='bow'").iloc[0]
        log(f"verify draw0/T1/{mk}/bow  here struct {a:.4f} +text {b:.4f} inc {d:+.4f}"
            f"   CSV struct {ref.struct:.4f} +text {ref.with_text:.4f} inc {ref.increment:+.4f}")
        ok = abs(a-ref.struct) < 6e-5 and abs(b-ref.with_text) < 6e-5
        if not ok: sys.exit(f"MISMATCH on {mk}: environments disagree -- stop, do not reuse the CSV")
    log("verify OK: this environment reproduces the full-size CSV to rounding")

def run():
    done = set()
    if os.path.exists(CSV):
        o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.model, o.arm, o["size"]))
    todo = [(r, t, m, a, z) for r in DRAWS for z in ("half", "full") for t in TIERS
            for m in ("gbm", "lr") for a in ARMS_ if (r, t, m, a, z) not in done]
    log(f"{len(done)} cells on disk, {len(todo)} to go")
    for i, (r, tier, mk, arm, size) in enumerate(todo):
        if time.time() - t0 > BUDGET:
            log(f"budget reached; {len(todo) - i} cells left -- rerun"); return
        tr = draw_split(r); A, B = halves(r, tr)
        fit = B if size == "half" else tr
        S = encode_tier(tiers[tier], fit)
        sb = cached(f"d{r}_{size}_{tier}_{mk}_struct", lambda: score(S, fit, mk))
        a, b, d, se = paired(y[~tr], sb[~tr], score(np.hstack([S, ARMS[arm]]), fit, mk)[~tr])
        pd.DataFrame([dict(draw=r, size=size, tier=tier, model=mk, arm=arm, n_fit=int(fit.sum()),
                           ev_fit=int(y[fit].sum()), struct=round(a, 4), with_text=round(b, 4),
                           increment=round(d, 4), se=round(se, 4))]
                     ).to_csv(CSV, mode="a", header=not os.path.exists(CSV), index=False)
        log(f"draw {r} {size:4s} {tier}/{mk}/{arm:6s}  n={fit.sum():,}  "
            f"struct {a:.4f} -> {b:.4f}  inc {d:+.4f}")
    log("all cells done")

def summary():
    R = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm", "size"])
    n = R.groupby(["draw", "size", "tier", "model"]).size()
    assert len(R) == 80 and (n == 2).all(), f"incomplete: {len(R)} of 80 cells"
    for z in ("full", "half"):
        q = R[R["size"] == z]
        print(f"{z}-size fits on {q.n_fit.min():,}-{q.n_fit.max():,} admissions "
              f"({q.ev_fit.min():,}-{q.ev_fit.max():,} events)")
    print()
    print("=== structured-only test AUROC (how much the baseline weakens) ===")
    print(R.pivot_table(index=["model", "tier"], columns="size", values="struct").round(4).to_string())
    print("\n=== text increment, mean over draws ===")
    print(R.pivot_table(index=["model", "arm", "tier"], columns="size",
                        values="increment").round(4).to_string())
    t = 2.776
    def ci(x): m = x.mean(); s = x.std(ddof=1)/np.sqrt(len(x)); return m, m-t*s, m+t*s
    print("\n=== paired within draw: half minus full ===")
    for mk in ("gbm", "lr"):
        for arm in ARMS_:
            q = R[(R.model == mk) & (R.arm == arm)].pivot_table(
                index="draw", columns=["size", "tier"], values="increment")
            coll = {s: 100*(q[(s, "T1")] - q[(s, "T5")]) / q[(s, "T1")] for s in ("half", "full")}
            dc = ci(coll["half"] - coll["full"])
            d5 = ci(q[("half", "T5")] - q[("full", "T5")])
            d1 = ci(q[("half", "T1")] - q[("full", "T1")])
            print(f"  {mk:3s}/{arm:6s}  collapse full {coll['full'].mean():5.1f}%  half {coll['half'].mean():5.1f}%"
                  f"  shift {dc[0]:+5.1f} pts [{dc[1]:+.1f}, {dc[2]:+.1f}]"
                  f"   T1 shift {d1[0]:+.4f}   T5 shift {d5[0]:+.4f} [{d5[1]:+.4f}, {d5[2]:+.4f}]")
    print("\nDECISION RULE (fixed beforehand): bow GBM collapse shift <= 5 pts AND frozen GBM "
          "T5 shift <= 0.005 -> no material distortion; exceeded with CI excluding 0 -> distorts; "
          "otherwise inconclusive.")

{"verify": verify, "run": run, "summary": summary}[sys.argv[1] if len(sys.argv) > 1 else "summary"]()
