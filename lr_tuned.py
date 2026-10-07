"""Tuned L2 logistic regression -- the ONE shared definition (2026-09-27).

Every script that reports a logistic-regression result should call this rather
than carry its own copy, so that a future change happens in one place.

WHY IT EXISTS. Until 2026-09-27 every LR in this project was
LogisticRegression(max_iter=5000): scikit-learn's default L2 penalty at C = 1,
for every model regardless of size -- 21 columns or ~1,000. check_lr_tuning_W1.py
showed that was far too weak once text columns are added: cross-validation picks
C between 0.001 and 0.03 for every model with text (30-1,000x stronger), the T5
text increments flip from negative to clearly positive, and the half-size
distortion (20-29 points of spurious extra collapse) disappears. The untuned LR
had been flattering the collapse hypothesis.

WHAT IT DOES. Mean imputation and standardisation fitted on the fit rows (as
before), then C chosen by 5-fold stratified CV INSIDE the fit rows over 13
values from 1e-4 to 1e2, scored by AUROC, and the model refit on all fit rows at
that C. Call it separately for the structured-only model and the
structured-plus-text model, so each gets its own best C: the increment then
compares the best LR with text against the best LR without it.

Deliberate simplification: imputation and scaling see the inner validation
folds. That can only nudge which C is chosen, never the held-out evaluation.

L2, not L1: our features come in correlated groups (overlapping n-grams, 768
near-parallel BERT dimensions, min/max/mean/count of one vital sign). L1 keeps
one member of each group more or less arbitrarily and its choice flips between
draws; L2 spreads weight across the group.

`penalty` is not passed: it is deprecated in scikit-learn 1.8 and L2 is the
default in both 1.7 and 1.8.
"""
import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

CS = np.logspace(-4, 2, 13)
SEED = 20180701


def lr_tuned_scores(X, fit_rows, y, n_jobs=None):
    """Decision scores for EVERY row of X, from a model fitted on X[fit_rows].

    Returns (scores, chosen_C). Rows outside fit_rows are never seen by the
    fit, the imputation, the scaling or the choice of C.
    """
    return lr_tuned_fit_predict(X[fit_rows], y[fit_rows], X, n_jobs)


def lr_tuned_fit_predict(Xtr, ytr, X, n_jobs=None):
    """Same model, for callers that already hold the training slice.

    Fits on (Xtr, ytr); returns (decision scores for every row of X, chosen_C).
    lr_tuned_scores is a thin wrapper around this, so both are one definition.
    """
    mu = np.nanmean(Xtr, 0); mu = np.where(np.isnan(mu), 0.0, mu)
    A = np.where(np.isnan(Xtr), mu, Xtr); B = np.where(np.isnan(X), mu, X)
    sc = StandardScaler().fit(A)
    m = LogisticRegressionCV(Cs=CS, cv=StratifiedKFold(5, shuffle=True, random_state=SEED),
                             scoring="roc_auc", max_iter=5000, n_jobs=n_jobs
                             ).fit(sc.transform(A), ytr)
    return m.decision_function(sc.transform(B)), float(m.C_[0])
