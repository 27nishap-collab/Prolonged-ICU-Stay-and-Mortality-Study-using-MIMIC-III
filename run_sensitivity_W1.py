"""W1 sensitivities.

A. PATIENT-GROUPED SPLIT. The paper's 75/25 split is stratified on the outcome
   and not grouped by patient, so 23.1% of test admissions belong to a patient
   who also appears in training. Every model is refit on the grouped partition
   and re-evaluated. Hyperparameters are carried over from the ungrouped tuning
   rather than re-tuned -- declared, and conservative in the direction that
   matters (it gives the grouped split the benefit of the ungrouped choice).

B. IS THE RANDOM-FOREST GRID TRUNCATED? Their mtry grid is 5:15. Tuning
   selected mtry=5 for the 20-feature arm (the bottom of the grid) and mtry=15
   for the 542-feature arm (the top). Neither brackets an interior optimum, so
   the grid is extended upward on the text arm to see whether CV AUROC is still
   climbing where their grid stops.

C. IS THE GBM GRID TRUNCATED? Tuning selected n.trees=500 (top), shrinkage=0.01
   (top) and n.minobsinnode=15 (top) on both arms, with depth 15 and 20 of a
   grid ending at 20. Same question, same method.

B and C use 5-fold CV rather than 10 -- these are boundary probes, not a
re-tuning, and the comparison is against the same folds throughout.
"""
import os, json, itertools, warnings, time
import numpy as np, pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
OUT = f"{U}/w1_models"
SEED = 20180701
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

CONT = ["age","urine_cc_kg_hr","elixhauser","creatinine_max","wbc_max","platelets_min",
        "bilirubin_max","pao2_min","gcs_min","sbp_min","hr_min","temp_min","potassium_max"]
BINF = ["male","adm_emergency","adm_elective","adm_urgent",
        "mech_vent","icu_admission","cardiac_arrest"]

t = pd.read_parquet(f"{U}/model_tabular_W1.parquet")
Xo = sparse.load_npz(f"{U}/text_open_W1.npz").toarray().astype(np.float32)
cvc = [c for c in t.columns if c.startswith("cv_")]
S = t[CONT + BINF].to_numpy(np.float32)
T = np.hstack([S, t[cvc].to_numpy(np.float32),
               t[["n_words_raw"]].to_numpy(np.float32), Xo])
XS = {"structured": S, "structured_text": T}
y = t.labels.to_numpy()

def scale(a, b):
    sc = StandardScaler().fit(a); return sc.transform(a), sc.transform(b)

def fit_predict(mk, X, tr, p):
    Xtr, ytr = X[tr], y[tr]
    if mk == "lr":
        a, b = scale(Xtr, X); return LogisticRegression(max_iter=5000).fit(a, ytr).decision_function(b)
    if mk == "enet":
        a, b = scale(Xtr, X)
        return LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=p["alpha"],
                                  C=1.0/(len(ytr)*p["lam"]), max_iter=4000, tol=1e-4
                                  ).fit(a, ytr).decision_function(b)
    if mk == "rf":
        return RandomForestClassifier(n_estimators=500, max_features=int(p["mtry"]),
                                      n_jobs=-1, random_state=SEED).fit(Xtr, ytr).predict_proba(X)[:, 1]
    return HistGradientBoostingClassifier(max_iter=int(p["n_trees"]), max_depth=int(p["depth"]),
            learning_rate=float(p["shrinkage"]), min_samples_leaf=int(p["min_obs"]),
            early_stopping=False, random_state=SEED).fit(Xtr, ytr).decision_function(X)

# ---------------------------------------------------------------- A
log("=== A. patient-grouped split ===")
rows = []
gtr = (t.split_grouped == "train").to_numpy()
utr = (t.split == "train").to_numpy()
for arm in ["structured", "structured_text"]:
    for mk in ["lr", "enet", "rf", "gbm"]:
        p = json.load(open(f"{OUT}/best_{arm}__{mk}.json"))
        s = fit_predict(mk, XS[arm], gtr, p)
        g_te = roc_auc_score(y[~gtr], s[~gtr])
        u_te = roc_auc_score(y[~utr], np.load(f"{OUT}/scores_{arm}__{mk}.npy")[~utr])
        rows.append(dict(arm=arm, model=mk, ungrouped_test=round(u_te, 4),
                         grouped_test=round(g_te, 4), delta=round(g_te - u_te, 4)))
        log(f"  {arm}/{mk}: ungrouped {u_te:.4f} -> grouped {g_te:.4f} ({g_te-u_te:+.4f})")
A = pd.DataFrame(rows); A.to_csv(f"{OUT}/W1_sens_grouped_split.csv", index=False)

# ---------------------------------------------------------------- B
log("=== B. random-forest mtry beyond their grid (text arm) ===")
X = XS["structured_text"]; Xtr, ytr = X[utr], y[utr]
skf = StratifiedKFold(5, shuffle=True, random_state=SEED)
folds = list(skf.split(Xtr, ytr))
rows = []
for mt in [5, 10, 15, 23, 35, 50, 100]:
    a = [roc_auc_score(ytr[b], RandomForestClassifier(n_estimators=500, max_features=mt,
            n_jobs=-1, random_state=SEED).fit(Xtr[i], ytr[i]).predict_proba(Xtr[b])[:, 1])
         for i, b in folds]
    rows.append(dict(mtry=mt, cv_auc=round(float(np.mean(a)), 4),
                     in_their_grid=(5 <= mt <= 15)))
    log(f"  mtry={mt:<4d} cv AUROC {np.mean(a):.4f}{'' if 5<=mt<=15 else '   (beyond their grid)'}")
B = pd.DataFrame(rows); B.to_csv(f"{OUT}/W1_sens_rf_mtry.csv", index=False)

# ---------------------------------------------------------------- C
log("=== C. GBM beyond their grid (text arm) ===")
rows = []
for d, sh in itertools.product([20, 25], [0.01, 0.05]):
    m_by_nt = {}
    accs = {nt: [] for nt in [500, 1000]}
    for i, b in folds:
        m = HistGradientBoostingClassifier(max_depth=d, learning_rate=sh, min_samples_leaf=15,
                early_stopping=False, warm_start=True, random_state=SEED, max_iter=500)
        for nt in [500, 1000]:
            m.max_iter = nt; m.fit(Xtr[i], ytr[i])
            accs[nt].append(roc_auc_score(ytr[b], m.decision_function(Xtr[b])))
    for nt in [500, 1000]:
        inside = (nt == 500 and d == 20 and sh == 0.01)
        rows.append(dict(n_trees=nt, depth=d, shrinkage=sh, min_obs=15,
                         cv_auc=round(float(np.mean(accs[nt])), 4), in_their_grid=inside))
        log(f"  n.trees={nt:<5d} depth={d:<3d} shrink={sh:<6g} cv AUROC {np.mean(accs[nt]):.4f}"
            f"{'' if inside else '   (beyond their grid)'}")
C = pd.DataFrame(rows); C.to_csv(f"{OUT}/W1_sens_gbm_grid.csv", index=False)
log("done")
