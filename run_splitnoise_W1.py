"""Does the paper's ungrouped split inflate its test AUROC?

A single patient-grouped partition gave a HIGHER test AUROC than the paper's
ungrouped one (+0.010 to +0.027 across the eight models), which is the opposite
of what a leak should do. One draw cannot separate a leak effect from ordinary
partition-to-partition variation, so this repeats both partition schemes 40
times with the same model and compares the distributions.

Logistic regression only -- it has no tuning parameters, so every draw is a
clean refit, and it is the one model type where 80 fits is cheap. The question
is about the SPLIT, not about the learner.

If the ungrouped scheme inflates, its distribution should sit ABOVE the grouped
one. The width of each distribution is the honest error bar on any single
published split.
"""
import os, warnings, time
import numpy as np, pandas as pd
from scipy import sparse, stats
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

warnings.filterwarnings("ignore")
U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
OUT = f"{U}/w1_models"
NREP = 40
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
subj = t.subject_id.to_numpy()
n = len(y)

def draw(rng, grouped):
    te = np.zeros(n, bool)
    if grouped:
        s = np.unique(subj); rng.shuffle(s)
        te = np.isin(subj, s[: int(round(0.25 * len(s)))])
    else:
        for lab in (0, 1):
            idx = np.flatnonzero(y == lab); rng.shuffle(idx)
            te[idx[: int(round(0.25 * len(idx)))]] = True
    return ~te, te

rows = []
for arm in ["structured", "structured_text"]:
    X = XS[arm]
    for grouped in (False, True):
        rng = np.random.RandomState(7)
        for r in range(NREP):
            tr, te = draw(rng, grouped)
            sc = StandardScaler().fit(X[tr])
            m = LogisticRegression(max_iter=5000).fit(sc.transform(X[tr]), y[tr])
            a = roc_auc_score(y[te], m.decision_function(sc.transform(X[te])))
            ov = np.isin(subj[te], np.unique(subj[tr])).mean()
            rows.append(dict(arm=arm, grouped=grouped, rep=r, test_auc=a, patient_overlap=ov))
        d = pd.DataFrame(rows).query("arm==@arm and grouped==@grouped").test_auc
        log(f"  {arm:16s} grouped={grouped}: mean {d.mean():.4f}  sd {d.std():.4f}  "
            f"range {d.min():.4f}-{d.max():.4f}")

df = pd.DataFrame(rows)
df.to_csv(f"{OUT}/W1_sens_split_noise_raw.csv", index=False)
out = []
for arm in ["structured", "structured_text"]:
    u = df.query("arm==@arm and not grouped").test_auc
    g = df.query("arm==@arm and grouped").test_auc
    tt = stats.ttest_ind(u, g, equal_var=False)
    out.append(dict(arm=arm,
                    ungrouped_mean=round(u.mean(),4), ungrouped_sd=round(u.std(),4),
                    grouped_mean=round(g.mean(),4),  grouped_sd=round(g.std(),4),
                    diff=round(u.mean()-g.mean(),4), p=float(tt.pvalue),
                    mean_patient_overlap_ungrouped=round(
                        df.query("arm==@arm and not grouped").patient_overlap.mean(), 4)))
res = pd.DataFrame(out); res.to_csv(f"{OUT}/W1_sens_split_noise.csv", index=False)
print("\n=== ungrouped vs patient-grouped, 40 draws each, logistic regression ===")
print(res.to_string(index=False))
log("done")
