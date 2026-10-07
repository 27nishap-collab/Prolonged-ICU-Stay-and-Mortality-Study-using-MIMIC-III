"""The ten-draw headline curve with the bag-of-words phrases chosen INSIDE each draw.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3.11 run_repeated_reselect_W1.py 2>&1 | tee repeated_reselect.log     # ~1.5-2 h, CPU
    python3.11 run_repeated_reselect_W1.py summary

WHY. run_repeated_splits_W1.py (F14, Figure 1) reuses ONE set of 500 phrases,
chosen by L1 logistic regression on the ORIGINAL split's training admissions, in
all ten draws. Each draw is a fresh 75/25 split, so ~75% of every draw's test
patients helped choose the phrases. Check (b) (reselect_split_W1.py, 2026-09-29)
measured what that does, on Design 3's half-B downstream fits: choosing on each
partition's own training set instead (same 19,269 patients, no test labels) moves
the bag-of-words increment by
    GBM       T1 -0.005, T5 -0.001   -> collapse 81% -> 81%
    tuned LR  T1 -0.013, T5 -0.009   -> collapse 88% -> 99%
So the headline needs recomputing with the choice made inside each draw.

WHAT. Identical to run_repeated_splits_W1.py -- same ten draws (draw_split, seed
SEED + 1000*(r+1)), observable specification, notes-no-discharge corpus, five tiers,
GBM at the fixed configuration and tuned LR (lr_tuned.py), all train-only
preprocessing -- except the 500 open-vocabulary columns of the text block:
  * chosen per draw from the same 4,687-phrase candidate pool
    (out/text_full_W1_v2.npz, built by reselect_split_W1.py dtm), on that draw's
    19,269 training admissions only, by the same selector: log1p counts, L1
    logistic regression (liblinear), C grid {0.005, 0.01, 0.05, 0.1} by 10-fold CV
    AUROC, best C keeping >= 500 phrases, top 500 by |coefficient|, raw counts.
  * draws 0-4: reselect_split_W1.py already made exactly this choice on exactly
    these training sets (bow_rt_draw{r}); it is reused, after checking the rows.
The 22 fixed text features (21 terms + word count) need no choosing and are unchanged.

OUTPUT, next to the original (which is not touched):
  out/observable_v2/W1_repeated_reselect.csv       one row per draw x tier x learner
  out/observable_v2/W1_repeated_reselect_C.csv     tuned LR's chosen C
  out/observable_v2/reselect/phrases_draw{r}.npz   the per-draw choice (+ .json)

BOTH ARMS ARE REFIT HERE. The original file's rows were produced partly in other
environments (the tuned-LR rows in the cloud workspace), and GBM/LR results differ
slightly between machines. So this script refits the ORIGINAL 500 phrases too, in
the same run: "orig" and "rechosen" share every structured model, and their
difference is exact. The structured-only AUROC is also compared with the original
file and the largest discrepancy reported -- as a note, not a stop.

READ-OFF, fixed before any result: per learner, the ten-draw curve, collapse
(T1-T5)/T1 with a 95% t CI (9 df), and new minus original increment at each tier,
paired within draw. The new file replaces the old one as the headline.
"""
import os, sys, json, time, shutil, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.ensemble import HistGradientBoostingClassifier

warnings.filterwarnings("ignore")
D = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, D)
from lr_tuned import lr_tuned_scores
U = os.environ.get("W1_DATA", os.path.join(D, "out"))
# spec/corpus switch (2026-09-29). Default = the headline (observable spec, notes-no-discharge).
#   W1_SPEC=replication W1_CORPUS=v1  -> Figure 1's companion curve (Weissman's predictor
#   list, notes-all corpus); needs `W1_CORPUS=v1 reselect_split_W1.py dtm` first.
#   W1_SPEC=replication W1_CORPUS=v2  -> (2026-10-03) the same curve on the clean
#   notes-no-discharge corpus, written to out/w1_repeated_v2/. The phrase choice depends only
#   on corpus, training rows and labels, so it reuses observable_v2/reselect/ (checked).
SPEC = os.environ.get("W1_SPEC", "observable"); CORPUS = os.environ.get("W1_CORPUS", "v2")
assert (SPEC, CORPUS) in (("observable", "v2"), ("replication", "v1"), ("replication", "v2")), (SPEC, CORPUS)
SFX = "_observable" if SPEC == "observable" else ""; CSUF = "_v2" if CORPUS == "v2" else ""
OUT = (f"{U}/observable_v2" if SPEC == "observable" else
       f"{U}/w1_repeated" if CORPUS == "v1" else f"{U}/w1_repeated_v2")
SELD = f"{OUT}/reselect"; os.makedirs(SELD, exist_ok=True)
ORIG = f"{OUT}/W1_repeated.csv"
ORIG_LRCV = f"{OUT}/W1_repeated_lrcv.csv"       # replication spec keeps its tuned-LR rows apart
CSV, CSV_C = f"{OUT}/W1_repeated_reselect.csv", f"{OUT}/W1_repeated_reselect_C.csv"
FTS = f"{U}/w1_ftsplit_v2_observable"
SEED = 20180701
R = int(os.environ.get("W1_R", 10))                  # a smoke test can set fewer draws
GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
TIERS = ["T1", "T2", "T3", "T4", "T5"]
LEARNERS = os.environ.get("W1_LEARNERS", "gbm,lrcv").split(",")
CS = [0.005, 0.01, 0.05, 0.1]
COLS = ["draw", "tier", "n_vars", "model", "arm", "struct", "with_text", "increment", "se"]
ARMS_ = ["orig", "rechosen"]
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
    res = pd.read_csv(CSV).drop_duplicates(["draw", "tier", "model", "arm"], keep="last")
    new, old = res[res.arm == "rechosen"], res[res.arm == "orig"]
    fs = [f for f in (ORIG, ORIG_LRCV) if os.path.exists(f)]
    f0 = (pd.concat([pd.read_csv(f) for f in fs]) if fs else
          pd.DataFrame(columns=["draw", "tier", "model", "struct", "increment"])
          ).drop_duplicates(["draw", "tier", "model"]).set_index(["draw", "tier", "model"])
    j = old.set_index(["draw", "tier", "model"]).join(f0[["struct", "increment"]], rsuffix="_file", how="inner")
    if len(j):
        print(f"\n  vs the original W1_repeated.csv: max |structured-only AUROC diff| "
              f"{(j.struct - j.struct_file).abs().max():.4f}, max |orig-arm increment diff| "
              f"{(j.increment - j.increment_file).abs().max():.4f} (machine differences; compare within this file)")
    print(f"\n=== ten-draw curve: original phrases vs phrases chosen inside each draw ({new.draw.nunique()} draws) ===")
    for mk in LEARNERS:
        pn = new[new.model == mk].pivot(index="draw", columns="tier", values="increment")
        po = old[old.model == mk].pivot(index="draw", columns="tier", values="increment").reindex(pn.index)
        if len(f0):
            pf = f0.xs(mk, level="model").increment.unstack("tier")
            c0 = 100 * (pf.T1 - pf.T5) / pf.T1
            print(f"\n  {mk}: for reference, the original file's collapse is {c0.mean():.1f}%")
        if pn.empty: continue
        print(f"\n  {mk}")
        print("    tier   original   re-chosen   new - original [95% CI]   new lower in")
        for t in TIERS:
            if t not in pn: continue
            m, lo, hi = ci(pn[t] - po[t])
            print(f"    {t}    {po[t].mean():+.4f}    {pn[t].mean():+.4f}    {m:+.4f} [{lo:+.4f}, {hi:+.4f}]"
                  f"    {int(((pn[t] - po[t]) < 0).sum())}/{len(pn)}")
        if {"T1", "T5"} <= set(pn.columns):
            for lab, p in (("original ", po), ("re-chosen", pn)):
                c = 100 * (p.T1 - p.T5) / p.T1; m, lo, hi = ci(c)
                mono = int((p[TIERS].diff(axis=1).iloc[:, 1:] < 0).all(axis=1).sum()) if set(TIERS) <= set(p.columns) else -1
                print(f"    collapse, {lab}: {m:5.1f}% [{lo:5.1f}, {hi:5.1f}]  range {c.min():.0f}-{c.max():.0f}%"
                      f"  falls in {int((p.T1 > p.T5).sum())}/{len(p)}  monotone in {mono}/{len(p)}")
    log("done -- tell Claude and it will read the CSV back")

if MODE == "summary":
    summary(); sys.exit(0)

# ---------------------------------------------------------------- data (as run_repeated_splits_W1.py)
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T   = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1{CSUF}.parquet").set_index("hadm_id")
order = tab.index.to_numpy(); assert (np.sort(order) == order).all()
T = T.set_index("hadm_id").reindex(order)
cvc = [c for c in tab.columns if c.startswith("cv_")]
FIX = np.hstack([tab[cvc].to_numpy(np.float32), tab[["n_words_raw"]].to_numpy(np.float32)])
y = tab.labels.to_numpy().astype(int); N = len(y)
Z = sparse.load_npz(f"{U}/text_full_W1{CSUF}.npz").tocsr(); assert Z.shape[0] == N
vocab = pd.read_csv(f"{U}/W1_openvocab_full{CSUF}.csv").phrase.astype(str).tolist()
assert Z.shape[1] == len(vocab)
XO = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz" if CORPUS == "v2" else f"{U}/text_open_W1.npz"
                     ).toarray().astype(np.float32)
assert XO.shape[0] == N
log(f"spec {SPEC}, corpus {CORPUS} -> {OUT} | {N:,} admissions | candidate pool {Z.shape[1]:,} phrases | {R} draws")

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

LAST_C = []
def score(X, tr, mk):
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **GBM
                                              ).fit(X[tr], y[tr]).decision_function(X)
    s, C = lr_tuned_scores(X, tr, y); LAST_C.append(C); return s

def choose(r, tr):
    """The 500 phrases for draw r, chosen on its training admissions only."""
    f = f"{SELD}/phrases_draw{r}.npz"
    if os.path.exists(f): return sparse.load_npz(f).tocsr()
    prev = f"{FTS}/bow_rt_draw{r}.npz"
    if SPEC == "observable" and r < 5 and os.path.exists(prev):          # same draw_split, same rows, same selector
        j = json.load(open(prev.replace(".npz", ".json")))
        assert j["draw"] == r and len(j["phrases"]) == 500
        idx = [vocab.index(p) for p in j["phrases"]]
        X = sparse.load_npz(prev).tocsr()
        assert abs(X - Z[:, idx]).max() == 0, "bow_rt does not match the candidate pool"
        shutil.copy2(prev, f); shutil.copy2(prev.replace(".npz", ".json"), f.replace(".npz", ".json"))
        log(f"draw {r}: reusing reselect_split's bow_rt choice (C={j['C']})"); return X
    prev = f"{U}/observable_v2/reselect/phrases_draw{r}.npz"
    if SPEC == "replication" and CORPUS == "v2" and os.path.exists(prev):  # same corpus, rows, labels, selector
        j = json.load(open(prev.replace(".npz", ".json")))
        assert j["draw"] == r and len(j["phrases"]) == 500
        idx = [vocab.index(p) for p in j["phrases"]]
        X = sparse.load_npz(prev).tocsr()
        assert abs(X - Z[:, idx]).max() == 0, "observable_v2 choice does not match the candidate pool"
        shutil.copy2(prev, f); shutil.copy2(prev.replace(".npz", ".json"), f.replace(".npz", ".json"))
        log(f"draw {r}: reusing the headline curve's choice (C={j['C']})"); return X
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold
    from sklearn.metrics import roc_auc_score
    rows = np.flatnonzero(tr); Xs = Z[rows].copy(); Xs.data = np.log1p(Xs.data); ya = y[rows]
    skf = StratifiedKFold(10, shuffle=True, random_state=SEED); path = []
    for C in CS:
        au = [roc_auc_score(ya[b], LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
                            .fit(Xs[a], ya[a]).decision_function(Xs[b])) for a, b in skf.split(Xs, ya)]
        nz = int((LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=2000)
                  .fit(Xs, ya).coef_[0] != 0).sum())
        path.append(dict(C=C, cv_auc=float(np.mean(au)), nonzero=nz))
        log(f"  draw {r} C={C:<6g} cv AUROC {np.mean(au):.4f} nonzero {nz:,}")
    p = pd.DataFrame(path); ok = p[p.nonzero >= 500]
    C = float(ok.loc[ok.cv_auc.idxmax()].C) if len(ok) else float(p.C.max())
    coef = LogisticRegression(penalty="l1", solver="liblinear", C=C, max_iter=4000).fit(Xs, ya).coef_[0]
    keep = np.argsort(-np.abs(coef))[:500]; keep = keep[coef[keep] != 0]
    X = Z[:, keep].tocsr(); sparse.save_npz(f, X)
    json.dump(dict(draw=r, C=C, n=len(keep), rows="draw training set", phrases=[vocab[i] for i in keep],
                   path=path), open(f.replace(".npz", ".json"), "w"), indent=1)
    log(f"draw {r}: chose {len(keep)} phrases at C={C:g}"); return X

done = set()
if os.path.exists(CSV):
    o = pd.read_csv(CSV); done = set(zip(o.draw, o.tier, o.model, o.arm))
    log(f"resuming: {len(done)} of {R*len(TIERS)*len(LEARNERS)*2} cells on disk")
for r in range(R):
    if all((r, t, m, a) in done for t in TIERS for m in LEARNERS for a in ARMS_): continue
    tr = draw_split(r); te = ~tr
    TEXTS = {"orig": np.hstack([FIX, XO]),
             "rechosen": np.hstack([FIX, choose(r, tr).toarray().astype(np.float32)])}
    log(f"draw {r}: train {tr.sum():,} / test {te.sum():,}")
    for tier in TIERS:
        S = encode(tiers[tier], tr)
        for mk in LEARNERS:
            if all((r, tier, mk, a) in done for a in ARMS_): continue
            sb = score(S, tr, mk); Cs = LAST_C[-1] if mk == "lrcv" else None
            got = {}
            for arm in ARMS_:
                if (r, tier, mk, arm) in done: continue
                a, b, d, se = paired(y[te], sb[te], score(np.hstack([S, TEXTS[arm]]), tr, mk)[te])
                got[arm] = d
                pd.DataFrame([dict(draw=r, tier=tier, n_vars=len(tiers[tier]), model=mk, arm=arm,
                                   struct=round(a, 4), with_text=round(b, 4), increment=round(d, 4),
                                   se=round(se, 4))]).reindex(columns=COLS).to_csv(
                    CSV, mode="a", header=not os.path.exists(CSV), index=False)
                if mk == "lrcv":
                    pd.DataFrame([dict(draw=r, tier=tier, arm=arm, C_struct=Cs, C_text=LAST_C[-1])]).to_csv(
                        CSV_C, mode="a", header=not os.path.exists(CSV_C), index=False)
            log(f"  {tier}/{mk}: struct {a:.4f} | " + "  ".join(f"{k} {v:+.4f}" for k, v in got.items()))
summary()
