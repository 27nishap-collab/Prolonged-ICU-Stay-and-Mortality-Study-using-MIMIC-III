"""W1 models -- Weissman et al. 2018, Table 2.

Four model types (logistic regression, elastic net, random forest, gradient
boosting), each fitted twice: structured only, then structured + text.

    python run_models_W1.py            # everything, resumable
    python run_models_W1.py lr enet    # named models only

Resumable: each (arm, model) writes its tuning grid and its fitted scores as
soon as they are done, and completed pieces are skipped on re-run.

--------------------------------------------------------------------------
PROTOCOL
--------------------------------------------------------------------------
Split: the 75/25 outcome-stratified, patient-UNGROUPED partition created in
build_text_W1.py, which is what caret::createDataPartition(df$Y, p=0.75) does.
23.1% of test admissions share a patient with training; that is the paper's
protocol and it is primary here. The patient-grouped split is refitted at the
end as a declared deviation, carrying the tuned hyperparameters over.

Feature sets:
  structured        18 predictors (design_W1.parquet)
  structured+text   18 + 21 closed-vocabulary counts + 500 open-vocabulary
                    counts + total raw word count = 540

Tuning grids are the authors' own, from sample_model_building.r:
  random forest   mtry = 5:15
  gbm             n.trees {100,250,500} x interaction.depth {5,10,15,20}
                  x shrinkage {0.005,0.01} x n.minobsinnode {5,10,15}
  elastic net     alpha = seq(0.1,1,0.1), lambda = {0.001 ... 0.030}
  logistic        no tuning

DECLARED DEVIATIONS FROM THE PUBLISHED PROTOCOL
  1. Tuning resampling is 10-fold cross-validation with ONE repeat, not five.
     Repeats stabilise hyperparameter SELECTION; they do not enter the reported
     test AUROC. Five repeats is 5x the compute for a choice among grid points
     that are mostly within noise of each other, and this runs on two cores.
  2. R's gbm is replaced by sklearn's HistGradientBoostingClassifier, and R's
     randomForest by sklearn's RandomForestClassifier. The grids are mapped
     parameter for parameter (n.trees -> max_iter, interaction.depth ->
     max_depth, shrinkage -> learning_rate, n.minobsinnode -> min_samples_leaf,
     mtry -> max_features). Implementations differ in split-finding, so exact
     agreement with their numbers is not expected; the comparison that matters
     is structured vs structured+text within each model type.
  3. glmnet's (alpha, lambda) is mapped to sklearn as
     (l1_ratio=alpha, C=1/(n_train * lambda)), which is the standard
     correspondence between the two objectives.

Inference: DeLong for the variance of each test AUROC and for every paired
comparison, Bonferroni-corrected over the 28 pairwise comparisons among the 8
test models -- which is what m=28 in the paper refers to.
"""
import os, sys, json, time, itertools, warnings
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
OUT = f"{U}/w1_models"
os.makedirs(OUT, exist_ok=True)
SEED = 20180701
NFOLD = 10
t0 = time.time()
def log(m): print(f"[{time.time()-t0:7.1f}s] {m}", flush=True)

CONT = ["age","urine_cc_kg_hr","elixhauser","creatinine_max","wbc_max","platelets_min",
        "bilirubin_max","pao2_min","gcs_min","sbp_min","hr_min","temp_min","potassium_max"]
BINF = ["male","adm_emergency","adm_elective","adm_urgent",
        "mech_vent","icu_admission","cardiac_arrest"]

# ==========================================================================
# DeLong (Sun & Xu 2014 fast algorithm)
# ==========================================================================
def _midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N); T2[J] = T
    return T2

def delong(y, scores):
    """scores: (k, n) array of k score vectors. Returns (aucs, kxk covariance)."""
    y = np.asarray(y); scores = np.atleast_2d(np.asarray(scores, float))
    order = np.argsort(-y, kind="mergesort")
    s = scores[:, order]
    m = int(y.sum()); n = len(y) - m
    pos, neg = s[:, :m], s[:, m:]
    k = s.shape[0]
    tx = np.vstack([_midrank(pos[r]) for r in range(k)])
    ty = np.vstack([_midrank(neg[r]) for r in range(k)])
    tz = np.vstack([_midrank(s[r])   for r in range(k)])
    aucs = tz[:, :m].sum(1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    sx = np.atleast_2d(np.cov(v01)); sy = np.atleast_2d(np.cov(v10))
    return aucs, sx / m + sy / n

def auc_ci(y, s):
    a, cov = delong(y, [s])
    se = float(np.sqrt(cov[0, 0]))
    return float(a[0]), max(0.0, a[0] - 1.96*se), min(1.0, a[0] + 1.96*se), se

# ==========================================================================
def load():
    t = pd.read_parquet(f"{U}/model_tabular_W1.parquet")
    Xo = sparse.load_npz(f"{U}/text_open_W1.npz").toarray().astype(np.float32)
    cvc = [c for c in t.columns if c.startswith("cv_")]
    S = t[CONT + BINF].to_numpy(np.float32)
    T = np.hstack([S, t[cvc].to_numpy(np.float32),
                   t[["n_words_raw"]].to_numpy(np.float32), Xo])
    names = {"structured": CONT + BINF,
             "structured_text": CONT + BINF + cvc + ["n_words_raw"]
                                + [f"open_{i}" for i in range(Xo.shape[1])]}
    return t, {"structured": S, "structured_text": T}, names

# ==========================================================================
# model families. each returns list of (params_dict, fitted_scores_fn)
# ==========================================================================
ENET_ALPHA  = [round(a, 1) for a in np.arange(0.1, 1.01, 0.1)]
ENET_LAMBDA = [0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030]
RF_MTRY     = list(range(5, 16))
GBM_TREES   = [100, 250, 500]
GBM_DEPTH   = [5, 10, 15, 20]
GBM_SHRINK  = [0.005, 0.01]
GBM_MINOBS  = [5, 10, 15]

def _scale(Xa, Xb):
    sc = StandardScaler().fit(Xa)
    return sc.transform(Xa), sc.transform(Xb)

def tune_lr(Xtr, ytr, folds):
    return [{}]                                   # no tuning parameters

def tune_enet(Xtr, ytr, folds):
    """glmnet (alpha, lambda) -> sklearn (l1_ratio, C=1/(n*lambda)).
    The lambda path is walked from most to least penalised with warm starts,
    which is how glmnet itself computes a path."""
    rows = {}
    for f, (a, b) in enumerate(folds):
        Xa, Xb = _scale(Xtr[a], Xtr[b])
        n = len(a)
        for al in ENET_ALPHA:
            m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=al,
                                   C=1.0, max_iter=1500, tol=1e-3, warm_start=True)
            for lam in sorted(ENET_LAMBDA, reverse=True):
                m.C = 1.0 / (n * lam)
                m.fit(Xa, ytr[a])
                rows.setdefault((al, lam), []).append(
                    roc_auc_score(ytr[b], m.decision_function(Xb)))
        log(f"    enet fold {f+1}/{len(folds)}")
    return [dict(alpha=k[0], lam=k[1], cv_auc=float(np.mean(v))) for k, v in rows.items()]

def fit_enet(Xtr, ytr, Xall, p):
    Xa, Xb = _scale(Xtr, Xall)
    m = LogisticRegression(penalty="elasticnet", solver="saga", l1_ratio=p["alpha"],
                           C=1.0/(len(ytr)*p["lam"]), max_iter=4000, tol=1e-4).fit(Xa, ytr)
    return m.decision_function(Xb)

def fit_lr(Xtr, ytr, Xall, p):
    Xa, Xb = _scale(Xtr, Xall)
    return LogisticRegression(max_iter=5000).fit(Xa, ytr).decision_function(Xb)

def tune_rf(Xtr, ytr, folds):
    rows = {}
    for f, (a, b) in enumerate(folds):
        for mt in RF_MTRY:
            m = RandomForestClassifier(n_estimators=500, max_features=mt,
                                       n_jobs=-1, random_state=SEED).fit(Xtr[a], ytr[a])
            rows.setdefault(mt, []).append(roc_auc_score(ytr[b], m.predict_proba(Xtr[b])[:, 1]))
        log(f"    rf fold {f+1}/{len(folds)}")
    return [dict(mtry=k, cv_auc=float(np.mean(v))) for k, v in rows.items()]

def fit_rf(Xtr, ytr, Xall, p):
    m = RandomForestClassifier(n_estimators=500, max_features=int(p["mtry"]),
                               n_jobs=-1, random_state=SEED).fit(Xtr, ytr)
    return m.predict_proba(Xall)[:, 1]

def tune_gbm(Xtr, ytr, folds):
    """n.trees is walked with warm starts instead of refitting, so the 72-point
    grid costs the same as its 24 (depth, shrinkage, minobs) combinations."""
    rows = {}
    for f, (a, b) in enumerate(folds):
        for d, sh, mo in itertools.product(GBM_DEPTH, GBM_SHRINK, GBM_MINOBS):
            m = HistGradientBoostingClassifier(
                max_depth=d, learning_rate=sh, min_samples_leaf=mo,
                early_stopping=False, warm_start=True, random_state=SEED,
                max_iter=GBM_TREES[0])
            for nt in GBM_TREES:
                m.max_iter = nt
                m.fit(Xtr[a], ytr[a])
                rows.setdefault((nt, d, sh, mo), []).append(
                    roc_auc_score(ytr[b], m.decision_function(Xtr[b])))
        log(f"    gbm fold {f+1}/{len(folds)}")
    return [dict(n_trees=k[0], depth=k[1], shrinkage=k[2], min_obs=k[3],
                 cv_auc=float(np.mean(v))) for k, v in rows.items()]

def fit_gbm(Xtr, ytr, Xall, p):
    m = HistGradientBoostingClassifier(
        max_iter=int(p["n_trees"]), max_depth=int(p["depth"]),
        learning_rate=float(p["shrinkage"]), min_samples_leaf=int(p["min_obs"]),
        early_stopping=False, random_state=SEED).fit(Xtr, ytr)
    return m.decision_function(Xall)

MODELS = {"lr":   (tune_lr,   fit_lr,   []),
          "enet": (tune_enet, fit_enet, ["alpha","lam"]),
          "rf":   (tune_rf,   fit_rf,   ["mtry"]),
          "gbm":  (tune_gbm,  fit_gbm,  ["n_trees","depth","shrinkage","min_obs"])}

# ==========================================================================
def main(which):
    t, Xs, names = load()
    y = t.labels.to_numpy()
    tr = (t.split == "train").to_numpy()
    log(f"train {tr.sum():,} / test {(~tr).sum():,}; outcome {y.mean():.4f}")
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=SEED)

    for arm in ["structured", "structured_text"]:
        X = Xs[arm]; Xtr, ytr = X[tr], y[tr]
        folds = list(skf.split(Xtr, ytr))
        for mk in which:
            tag = f"{arm}__{mk}"
            sfile = f"{OUT}/scores_{tag}.npy"
            if os.path.exists(sfile):
                log(f"{tag}: done, skipping"); continue
            tune, fit, keys = MODELS[mk]
            gfile = f"{OUT}/tuning_{tag}.csv"
            if os.path.exists(gfile):
                grid = pd.read_csv(gfile)
            else:
                log(f"{tag}: tuning ({X.shape[1]} features) ...")
                grid = pd.DataFrame(tune(Xtr, ytr, folds))
                grid.to_csv(gfile, index=False)
            best = grid.loc[grid.cv_auc.idxmax()].to_dict() if len(keys) else {}
            if keys:
                log(f"{tag}: best {({k: best[k] for k in keys})} cv AUROC {best['cv_auc']:.4f}")
            log(f"{tag}: refitting on the full training set ...")
            s = fit(Xtr, ytr, X, best)
            np.save(sfile, s)
            json.dump({k: best.get(k) for k in keys} | {"cv_auc": best.get("cv_auc")},
                      open(f"{OUT}/best_{tag}.json", "w"), default=float)
            atr = roc_auc_score(ytr, s[tr]); ate = roc_auc_score(y[~tr], s[~tr])
            log(f"{tag}: train AUROC {atr:.4f}   test AUROC {ate:.4f}")

    report(t, y, tr)

# ==========================================================================
PUB = {("structured","lr"):(0.80,0.79), ("structured_text","lr"):(0.90,0.86),
       ("structured","enet"):(0.80,0.79), ("structured_text","enet"):(0.89,0.88),
       ("structured","rf"):(1.00,0.82), ("structured_text","rf"):(1.00,0.88),
       ("structured","gbm"):(0.89,0.83), ("structured_text","gbm"):(0.95,0.89)}

def report(t, y, tr):
    cells, scores = [], {}
    for arm in ["structured", "structured_text"]:
        for mk in ["lr","enet","rf","gbm"]:
            f = f"{OUT}/scores_{arm}__{mk}.npy"
            if not os.path.exists(f): continue
            s = np.load(f); scores[(arm, mk)] = s
            a_tr = roc_auc_score(y[tr], s[tr])
            a_te, lo, hi, se = auc_ci(y[~tr], s[~tr])
            p_tr, p_te = PUB[(arm, mk)]
            cells.append(dict(arm=arm, model=mk,
                              train_auc=round(a_tr,4), published_train=p_tr,
                              test_auc=round(a_te,4),
                              test_ci=f"{lo:.3f}-{hi:.3f}", published_test=p_te,
                              delta_vs_published=round(a_te - p_te, 3)))
    res = pd.DataFrame(cells)
    res.to_csv(f"{OUT}/W1_results.csv", index=False)
    print("\n=== Table 2 ===")
    print(res.to_string(index=False))

    if len(scores) == 8:
        keys = list(scores)
        S = np.vstack([scores[k][~tr] for k in keys])
        aucs, cov = delong(y[~tr], S)
        rows = []
        for i, j in itertools.combinations(range(len(keys)), 2):
            v = cov[i,i] + cov[j,j] - 2*cov[i,j]
            z = (aucs[i]-aucs[j]) / np.sqrt(v) if v > 0 else 0.0
            p = 2*stats.norm.sf(abs(z))
            rows.append(dict(a=f"{keys[i][0]}/{keys[i][1]}", b=f"{keys[j][0]}/{keys[j][1]}",
                             auc_a=round(aucs[i],4), auc_b=round(aucs[j],4),
                             diff=round(aucs[i]-aucs[j],4), z=round(z,2),
                             p=p, p_bonferroni=min(1.0, p*28)))
        cmp = pd.DataFrame(rows)
        cmp.to_csv(f"{OUT}/W1_comparisons.csv", index=False)
        print("\n=== structured vs +text, within model type (DeLong, Bonferroni m=28) ===")
        within = cmp[cmp.a.str.split("/").str[1] == cmp.b.str.split("/").str[1]]
        print(within.to_string(index=False))

if __name__ == "__main__":
    which = [a for a in sys.argv[1:] if a in MODELS] or ["lr","enet","rf","gbm"]
    main(which)
    log("done")
