"""Does the text advantage track how much the CARE TEAM controls the outcome?

THE LOGIC
---------
Two accounts of where text's predictive value comes from:
  A. patient state    -- notes carry clinical observation the structured fields
                         miss, so text adds evidence about the patient.
  B. clinician forecast -- notes record the team's decisions and plans, so the
                         model learns to read a prediction rather than make one.

Vocabulary ablation cannot separate them (RESULTS_reselect.md: the concept is
lexically open). This design separates them a different way, using a fact about
the OUTCOMES rather than about the text:

  **ICU length of stay is partly a team decision. Death is much less so.**

The team decides when a patient leaves the ICU; nobody decides that a patient
dies. So under (B), text's advantage should be systematically LARGER for the
LOS outcome than for mortality -- because the outcome is partly the same
decision the notes are recording. Under (A) there is no reason for that
ordering; if anything, nuanced clinical observation should help most with death.

Nothing is rebuilt: same 48h window, same features, same split, same v2 text
block. Only the LABEL moves.

  death        in-hospital mortality                  7.2% prevalence
  los7         ICU LOS >= 7 days                     17.3%
  composite    either (Weissman's primary outcome)   21.0%

PUBLISHED ANCHORS (Weissman supplement, GBM, structured -> +text):
  composite      0.83 -> 0.89   (+0.06)
  death only     0.85 -> 0.89   (+0.04)
  LOS >= 21 d    0.84 -> 0.89   (+0.05)
His gain is smallest for the outcome the team least controls and largest for
the one it most controls. He reported these as robustness checks without
remarking on the ordering.

TWO THINGS THIS CANNOT DO, stated up front
  1. AUROC is not strictly comparable across labels of different prevalence and
     difficulty. The increment is therefore ALSO reported as a fraction of the
     structured model's remaining headroom, increment / (1 - AUROC_struct),
     which partially normalises. The honest read is the ORDERING and whether it
     holds at both T1 and T5, not the absolute gaps.
  2. The LOS label has a competing risk: a patient who dies on day 3 has
     los7 = 0, so that label conflates "recovered quickly" with "died quickly".
     A death-free-survivor restriction is reported alongside as a check.
"""
import os, json, time, warnings
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

OUT = f"{U}/w1_outcome" + SFX; os.makedirs(OUT, exist_ok=True)
SEED = 20180701

GBM = dict(max_depth=10, learning_rate=0.05, max_iter=500, min_samples_leaf=15)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

def _mid(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N); i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5*(i+j-1)+1; i = j
    T2 = np.empty(N); T2[J] = T; return T2

def delong(y, S):
    y = np.asarray(y); s = np.atleast_2d(np.asarray(S, float))
    o = np.argsort(-y, kind="mergesort"); s = s[:, o]
    m = int(y.sum()); n = len(y)-m; k = s.shape[0]
    tx = np.vstack([_mid(s[r, :m]) for r in range(k)])
    ty = np.vstack([_mid(s[r, m:]) for r in range(k)])
    tz = np.vstack([_mid(s[r])     for r in range(k)])
    a = tz[:, :m].sum(1)/m/n - (m+1.0)/2.0/n
    return a, np.atleast_2d(np.cov((tz[:, :m]-tx)/n))/m + np.atleast_2d(np.cov(1.0-(tz[:, m:]-ty)/m))/n

def paired(y, a, b):
    au, c = delong(y, [a, b]); v = c[0,0]+c[1,1]-2*c[0,1]
    d = au[1]-au[0]; se = float(np.sqrt(max(v, 0))); z = d/se if se > 0 else 0.0
    return d, d-1.96*se, d+1.96*se, float(2*stats.norm.sf(abs(z)))

tiers = json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T = pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab = pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order = tab.hadm_id.to_numpy(); T = T.set_index("hadm_id").reindex(order)
tr = (tab.split == "train").to_numpy()
cvc = [c for c in tab.columns if c.startswith("cv_")]
Xo = sparse.load_npz(f"{U}/text_open_W1_v2_aligned.npz").toarray().astype(np.float32)
TEXT = np.hstack([tab[cvc].to_numpy(np.float32),
                  tab[["n_words_raw"]].to_numpy(np.float32), Xo])

LABELS = {"death": T.outcome_death.to_numpy().astype(int),
          "los7":  T.outcome_icu_los_7d.to_numpy().astype(int),
          "composite": T.labels.to_numpy().astype(int)}
for k, v in LABELS.items():
    log(f"{k}: prevalence {100*v.mean():.1f}%  (test {100*v[~tr].mean():.1f}%)")

def enc(cols):
    out = []
    for c in cols:
        s = T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v = s.astype(object).where(s.notna(), "__na__"); kc = v[tr].value_counts()
            lv = set(kc[kc >= 0.01*tr.sum()].index); v = v.where(v.isin(lv), "__other__")
            for L in sorted(lv | {"__other__"}): out.append((v == L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)
BASES = {"T1": enc(tiers["T1"]), "T5": enc(tiers["T5"])}

def score(X, y, mk, mask=None):
    m_tr = tr if mask is None else (tr & mask)
    if mk == "lrcv": return lr_tuned_scores(X, m_tr, y)[0]
    Xtr, ytr = X[m_tr], y[m_tr]
    if mk == "gbm":
        return HistGradientBoostingClassifier(early_stopping=False, random_state=SEED,
                                              **GBM).fit(Xtr, ytr).decision_function(X)
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A), ytr).decision_function(sc.transform(B))

PUB = {("composite","T1"): (0.83, 0.89), ("death","T1"): (0.85, 0.89)}
rows = []
for mk in LEARNERS:
    for base in ["T1", "T5"]:
        for lab, y in LABELS.items():
            sc = {}
            for arm, X in [("structured", BASES[base]),
                           ("structured_text", np.hstack([BASES[base], TEXT]))]:
                f = f"{OUT}/s_{lab}_{base}_{arm}_{mk}.npy"
                sc[arm] = np.load(f) if os.path.exists(f) else score(X, y, mk)
                if not os.path.exists(f): np.save(f, sc[arm])
            a = roc_auc_score(y[~tr], sc["structured"][~tr])
            b = roc_auc_score(y[~tr], sc["structured_text"][~tr])
            d, lo, hi, p = paired(y[~tr], sc["structured"][~tr], sc["structured_text"][~tr])
            pub = PUB.get((lab, base), ("", ""))
            rows.append(dict(model=mk, base=base, outcome=lab,
                             prevalence=round(100*y[~tr].mean(), 1),
                             struct=round(a, 4), text=round(b, 4),
                             increment=round(d, 4), ci=f"{lo:+.4f},{hi:+.4f}", p=p,
                             pct_of_headroom=round(100*d/(1-a), 1),
                             pub_struct=pub[0], pub_text=pub[1]))
            log(f"  {mk}/{base}/{lab}: {a:.4f} -> {b:.4f}  increment {d:+.4f} "
                f"[{lo:+.4f},{hi:+.4f}]  ({100*d/(1-a):.1f}% of headroom)")
            pd.DataFrame(rows).to_csv(f"{OUT}/W1_outcome_split.csv", index=False)

# ---- competing-risk check: LOS>=7d among patients who did NOT die in hospital
log("=== competing-risk check: LOS>=7d among in-hospital survivors ===")
surv = LABELS["death"] == 0
for mk in LEARNERS:
    for base in ["T1", "T5"]:
        y = LABELS["los7"]
        sc = {}
        for arm, X in [("structured", BASES[base]),
                       ("structured_text", np.hstack([BASES[base], TEXT]))]:
            sc[arm] = score(X, y, mk, mask=surv)
        te = (~tr) & surv
        a = roc_auc_score(y[te], sc["structured"][te]); b = roc_auc_score(y[te], sc["structured_text"][te])
        d, lo, hi, p = paired(y[te], sc["structured"][te], sc["structured_text"][te])
        rows.append(dict(model=mk, base=base, outcome="los7_survivors",
                         prevalence=round(100*y[te].mean(), 1),
                         struct=round(a,4), text=round(b,4), increment=round(d,4),
                         ci=f"{lo:+.4f},{hi:+.4f}", p=p,
                         pct_of_headroom=round(100*d/(1-a),1), pub_struct="", pub_text=""))
        log(f"  {mk}/{base}/los7|survivors: {a:.4f} -> {b:.4f}  increment {d:+.4f}")

res = pd.DataFrame(rows); res.to_csv(f"{OUT}/W1_outcome_split.csv", index=False)
pd.set_option("display.width", 220)
print("\n=== text increment by outcome ===")
print(res.to_string(index=False))
print("\n=== the ordering test: is the increment larger for the team-controlled outcome? ===")
for mk in LEARNERS:
    for base in ["T1", "T5"]:
        r = res[(res.model==mk)&(res.base==base)].set_index("outcome")
        print(f"  {mk}/{base}:  death {r.loc['death','increment']:+.4f}   "
              f"composite {r.loc['composite','increment']:+.4f}   "
              f"los7 {r.loc['los7','increment']:+.4f}   "
              f"| headroom-normalised: {r.loc['death','pct_of_headroom']:.1f}% "
              f"{r.loc['composite','pct_of_headroom']:.1f}% {r.loc['los7','pct_of_headroom']:.1f}%")
log("done")
