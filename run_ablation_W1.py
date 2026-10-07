"""Two questions, one run.

A. HOW BIG IS THE DISCHARGE-SUMMARY LEAK?
   Discharge summaries carry a NULL charttime, so COALESCE(charttime, chartdate)
   stamps them at midnight of the discharge DAY. A late admission time pulls the
   summary inside [admittime, admittime+48h) while the patient is still in
   hospital: 1,454 notes across 1,398 admissions (5.4% of the cohort), 1,449 of
   them on admissions discharged after hour 48. The summary states the outcome.
   v1 = the corpus as originally built (reproduces Weissman's notes-per-admission
   exactly). v2 = the same corpus with Discharge summary dropped. Same split,
   same pipeline, vocabulary re-selected on the same training rows.

   Note which admissions the leak touched: the 372 that lose ALL in-window text
   under v2 have a 6.7% outcome rate against 21.0% cohort-wide -- short stays
   with good outcomes, whose only "note" in the window was the document saying
   they went home.

B. IS THE SATURATION EFFECT A CARE-PROCESS EFFECT?

   TWO DISTINCT GROUPS, deliberately named apart:
     * DISPOSITION TEXT FEATURES -- 35 of the 521 text features (23 open-vocab
       phrases + 12 closed-vocab terms). Words in the notes about care
       trajectory. Matched by the DISP regex below. Build `nodisp`/`disponly`.
     * CARE-PROCESS VARIABLES -- the 8 structured columns in TREAT below,
       derived from chartevents/inputevents, recording what the team DID rather
       than what the body did. vasopressor and rrt are escalation, not
       disposition, which is why "care-process" and not "disposition". Build
       T5_phys = T5 minus these 8.
   `extubated` appears in both: the same event reaches the model once as a word
   and once as a structured flag, and measuring that overlap is the point.
   The open vocabulary's strongest negative weights are care-trajectory language
   (extubated, transfer floor, chair, diet, oob), and T5 of the saturation curve
   adds structured treatment/de-escalation facts (vent, extubated, vasopressor,
   RRT) over the same window. So text may stop adding at T5 because disposition
   became available structurally -- not because the model got better at
   physiology. Crossing text content against structured content separates them:

     structured bases          text blocks
       T1       Weissman 18      none
       T5       222              full      21 closed + 500 open + word count
       T5_phys  T5 minus the     nodisp    full minus disposition features
                treatment /      disponly  only the disposition features
                disposition                (+ word count)
                variables

   Prediction if the disposition story is right: the text increment should be
   much larger over T5_phys than over T5, and `disponly` should recover most of
   the increment that `full` achieves over T1.

DECLARED: GBM uses one fixed configuration everywhere (depth 10, shrinkage 0.05,
500 trees, min_samples_leaf 15 -- the T5 CV winner from the saturation run)
rather than re-tuning each arm. The comparisons here are between text blocks
within a structured base, of near-identical dimensionality, so per-arm tuning
would add variance rather than remove bias. Logistic regression carries no
tuning at all and is reported alongside as the check.
"""
import os, re, json, time, warnings
import numpy as np, pandas as pd
# 2026-09-27: learner "lrcv" = tuned L2 logistic regression from lr_tuned.py.
# Default learners unchanged; run with W1_LEARNERS=gbm,lr,lrcv to add it.
# (Named LEARNERS, not MODELS: fit_bert_W1.py already uses MODELS for encoders.)
import sys as _sys_lrcv
_sys_lrcv.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lr_tuned import lr_tuned_scores, lr_tuned_fit_predict
LEARNERS = os.environ.get("W1_LEARNERS", "gbm,lr").split(",")
assert set(LEARNERS) <= {"gbm", "lr", "lrcv"}, LEARNERS
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
# ---- specification switch (SPEC_DECISION.md, 2026-09-20) -------------------
# Two feature sets, named for what distinguishes them:
#   observable  (PRIMARY)  every predictor knowable at hour 48. Discharge-coded
#                          elixhauser -> has_prior_admission + prior_elixhauser;
#                          the five context variables windowed to 48h.
#   replication            Weissman's feature set exactly as published, which the
#                          F1 / Table 2 replication needs. "Published" describes
#                          his PREDICTOR LIST, never any result -- no saturation
#                          curve has ever been published; the curve is ours.
# Output directories are suffixed, so the two can never collide.
SPEC = os.environ.get("W1_SPEC", "observable")
SPEC = {"leakfree": "observable", "published": "replication"}.get(SPEC, SPEC)
assert SPEC in ("observable", "replication"), SPEC
SFX  = "" if SPEC == "replication" else "_observable"

OUT = f"{U}/w1_ablation" + SFX; os.makedirs(OUT, exist_ok=True)
SEED = 20180701

GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

# DISPOSITION TEXT FEATURES: care-trajectory lexicon, applied to open-vocabulary
# phrases and closed-vocabulary term names. Distinct from TREAT below.
DISP = re.compile(
    r"(extubat|transfer|floor|chair|oob|out of bed|ambulat|diet|\bpo\b|home|rehab|"
    r"discharg|wean|dnr|\bcmo\b|comfort|hospice|palliat|goals|family meeting|"
    r"code status|wishes|plan|tolerat|valuable|belongings|sign out|call out|"
    r"step down|stepdown|d/c|\bdc\b|want)")

# ---------------------------------------------------------------- DeLong
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong(y, scores):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(scores, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y) - m; k = s.shape[0]
    tx = np.vstack([_midrank(s[r, :m]) for r in range(k)])
    ty = np.vstack([_midrank(s[r, m:]) for r in range(k)])
    tz = np.vstack([_midrank(s[r])     for r in range(k)])
    aucs = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    v01 = (tz[:, :m]-tx)/n; v10 = 1.0-(tz[:, m:]-ty)/m
    return aucs, np.atleast_2d(np.cov(v01))/m + np.atleast_2d(np.cov(v10))/n

def paired(y, a, b):
    au, cov = delong(y, [a, b])
    v = cov[0,0]+cov[1,1]-2*cov[0,1]; d = au[1]-au[0]; se = float(np.sqrt(max(v,0)))
    z = d/se if se > 0 else 0.0
    return d, d-1.96*se, d+1.96*se, float(2*stats.norm.sf(abs(z)))

# ---------------------------------------------------------------- data
tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab1 = pd.read_parquet(f"{U}/model_tabular_W1.parquet")
tab2 = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order = tab1.hadm_id.to_numpy()
assert (np.sort(order) == order).all()
assert (tab2.hadm_id.to_numpy() == order).all()
T = T.set_index("hadm_id").reindex(order)
y = tab1.labels.to_numpy(); tr = (tab1.split == "train").to_numpy()
log(f"{len(y):,} admissions | train {tr.sum():,} / test {(~tr).sum():,}")

def text_blocks(tab, npz, tag):
    cvc = [c for c in tab.columns if c.startswith("cv_")]
    Xo = sparse.load_npz(npz).toarray().astype(np.float32)
    voc = pd.read_csv(f"{U}/text_open_vocab_W1{tag}.csv").phrase.astype(str).to_numpy()
    assert Xo.shape[1] == len(voc)
    d_open = np.array([bool(DISP.search(p)) for p in voc])
    d_clos = np.array([bool(DISP.search(c[3:].replace("_", " "))) for c in cvc])
    C = tab[cvc].to_numpy(np.float32); W = tab[["n_words_raw"]].to_numpy(np.float32)
    return {
      "full":     np.hstack([C, W, Xo]),
      "nodisp":   np.hstack([C[:, ~d_clos], W, Xo[:, ~d_open]]),
      "disponly": np.hstack([C[:,  d_clos], W, Xo[:,  d_open]]),
    }, int(d_open.sum()), int(d_clos.sum()), voc[d_open]

TXT2, nd_o, nd_c, disp_phrases = text_blocks(tab2, f"{U}/text_open_W1_v2_aligned.npz", "_v2")
TXT1, _, _, _ = text_blocks(tab1, f"{U}/text_open_W1.npz", "")
log(f"v2 disposition features: {nd_o}/500 open phrases, {nd_c}/21 closed terms")
log("  " + ", ".join(disp_phrases))

def encode(cols):
    out, keep = [], []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__")
            kc = v[tr].value_counts(); lv = set(kc[kc >= 0.01*tr.sum()].index)
            v = v.where(v.isin(lv), "__other__")
            for lev in sorted(lv | {"__other__"}):
                out.append((v == lev).to_numpy(np.float32))
        else:
            out.append(s.to_numpy(np.float32))
    return np.column_stack(out)

# CARE-PROCESS VARIABLES: structured columns recording what the team did.
# Removing them from T5 gives T5_phys. Distinct from the DISP text lexicon.
TREAT = ["mech_vent","vent_flag","vent_span_h","vent_n","extubated","intubated",
         "vasopressor","rrt"]
BASES = {
  "T1":      encode(tiers["T1"]),
  "T5":      encode(tiers["T5"]),
  "T5_phys": encode([c for c in tiers["T5"] if c not in TREAT]),
}
log("bases: " + ", ".join(f"{k} {v.shape[1]} cols" for k, v in BASES.items()))

# ---------------------------------------------------------------- fit
def score(X, mk):
    Xtr, ytr = X[tr], y[tr]
    if mk == "lrcv": return lr_tuned_scores(X, tr, y)[0]
    if mk == "gbm":
        m = HistGradientBoostingClassifier(early_stopping=False, random_state=SEED, **GBM)
        return m.fit(Xtr, ytr).decision_function(X)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr).decision_function(sc.transform(B))

S = {}
def get(base, txt, corpus, mk):
    k = (base, txt, corpus, mk)
    f = f"{OUT}/s_{base}_{txt}_{corpus}_{mk}.npy"
    if k in S: return S[k]
    if os.path.exists(f): S[k] = np.load(f); return S[k]
    X = BASES[base] if txt == "none" else np.hstack(
        [BASES[base], (TXT1 if corpus == "v1" else TXT2)[txt]])
    s = score(X, mk); np.save(f, s); S[k] = s
    log(f"  {base}/{txt}/{corpus}/{mk}: {X.shape[1]} cols, test AUROC {roc_auc_score(y[~tr], s[~tr]):.4f}")
    return s

rows = []
def add(kind, base, txt, corpus, mk):
    b = get(base, "none", corpus, mk); t = get(base, txt, corpus, mk)
    d, lo, hi, p = paired(y[~tr], b[~tr], t[~tr])
    rows.append(dict(analysis=kind, base=base, text=txt, corpus=corpus, model=mk,
                     auc_base=round(roc_auc_score(y[~tr], b[~tr]), 4),
                     auc_text=round(roc_auc_score(y[~tr], t[~tr]), 4),
                     increment=round(d, 4), ci=f"{lo:+.4f},{hi:+.4f}", p=p))
    pd.DataFrame(rows).to_csv(f"{OUT}/W1_ablation.csv", index=False)

log("=== A. discharge-summary leak ===")
for mk in LEARNERS:
    for base in ["T1", "T5"]:
        for corpus in ["v1", "v2"]:
            add("leak", base, "full", corpus, mk)

log("=== B. disposition ablation (v2 corpus) ===")
for mk in LEARNERS:
    for base in ["T1", "T5_phys", "T5"]:
        for txt in ["full", "nodisp", "disponly"]:
            add("disposition", base, txt, "v2", mk)

res = pd.DataFrame(rows)
pd.set_option("display.width", 250)
print("\n=== A. leak: text increment under v1 (leaky) vs v2 (clean) ===")
print(res[res.analysis == "leak"].to_string(index=False))
print("\n=== B. disposition ablation, v2 corpus ===")
print(res[res.analysis == "disposition"].to_string(index=False))
log("done")
