"""Fit the re-selected text block and place it beside `full` and `nodisp`."""
import os, re, json, warnings
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

U = os.environ.get("W1_DATA", os.path.join(os.path.dirname(os.path.abspath(__file__)), "out"))   # data folder; was the cloud-workspace copy
OUT = f"{U}/w1_ablation" + SFX
SEED=20180701; GBM=dict(max_depth=10,learning_rate=0.05,max_iter=500,min_samples_leaf=15)

DISP=re.compile(r"(extubat|transfer|floor|chair|oob|out of bed|ambulat|diet|\bpo\b|home|rehab|"
 r"discharg|wean|dnr|\bcmo\b|comfort|hospice|palliat|goals|family meeting|code status|wishes|"
 r"plan|tolerat|valuable|belongings|sign out|call out|step down|stepdown|d/c|\bdc\b|want)")
def _mid(x):
    J=np.argsort(x);Z=x[J];N=len(x);T=np.zeros(N);i=0
    while i<N:
        j=i
        while j<N and Z[j]==Z[i]: j+=1
        T[i:j]=0.5*(i+j-1)+1; i=j
    T2=np.empty(N);T2[J]=T;return T2
def delong(y,S):
    y=np.asarray(y);s=np.atleast_2d(np.asarray(S,float));o=np.argsort(-y,kind="mergesort");s=s[:,o]
    m=int(y.sum());n=len(y)-m;k=s.shape[0]
    tx=np.vstack([_mid(s[r,:m]) for r in range(k)]);ty=np.vstack([_mid(s[r,m:]) for r in range(k)])
    tz=np.vstack([_mid(s[r]) for r in range(k)])
    a=tz[:,:m].sum(1)/m/n-(m+1.0)/2.0/n
    return a, np.atleast_2d(np.cov((tz[:,:m]-tx)/n))/m + np.atleast_2d(np.cov(1.0-(tz[:,m:]-ty)/m))/n
def paired(y,a,b):
    au,c=delong(y,[a,b]);v=c[0,0]+c[1,1]-2*c[0,1];d=au[1]-au[0];se=float(np.sqrt(max(v,0)))
    z=d/se if se>0 else 0.0
    return d,d-1.96*se,d+1.96*se,float(2*stats.norm.sf(abs(z)))

tiers=json.load(open(f"{U}/W1_tier_columns{SFX}.json"))
T=pd.read_parquet(f"{U}/tiers_W1{SFX}.parquet")
tab=pd.read_parquet(f"{U}/model_tabular_W1_v2.parquet")
order=tab.hadm_id.to_numpy(); T=T.set_index("hadm_id").reindex(order)
y=tab.labels.to_numpy(); tr=(tab.split=="train").to_numpy()
cvc=[c for c in tab.columns if c.startswith("cv_")]
d_clos=np.array([bool(DISP.search(c[3:].replace("_"," "))) for c in cvc])
C=tab[cvc].to_numpy(np.float32); W=tab[["n_words_raw"]].to_numpy(np.float32)
X3=sparse.load_npz(f"{U}/text_open_W1_v3.npz").toarray().astype(np.float32)
RESEL=np.hstack([C[:,~d_clos], W, X3])
print(f"reselect block: {RESEL.shape[1]} cols ({int((~d_clos).sum())} closed + 1 wc + {X3.shape[1]} open)")

def enc(cols):
    out=[]
    for c in cols:
        s=T[c]
        if not pd.api.types.is_numeric_dtype(s):
            v=s.astype(object).where(s.notna(),"__na__"); kc=v[tr].value_counts()
            lv=set(kc[kc>=0.01*tr.sum()].index); v=v.where(v.isin(lv),"__other__")
            for L in sorted(lv|{"__other__"}): out.append((v==L).to_numpy(np.float32))
        else: out.append(s.to_numpy(np.float32))
    return np.column_stack(out)
BASES={"T1":enc(tiers["T1"]),"T5":enc(tiers["T5"])}

def score(X,mk):
    Xtr,ytr=X[tr],y[tr]
    if mk == "lrcv": return lr_tuned_scores(X, tr, y)[0]
    if mk=="gbm":
        return HistGradientBoostingClassifier(early_stopping=False,random_state=SEED,**GBM).fit(Xtr,ytr).decision_function(X)
    mu=np.nanmean(Xtr,0); mu=np.where(np.isnan(mu),0.0,mu)
    A=np.where(np.isnan(Xtr),mu,Xtr); Bm=np.where(np.isnan(X),mu,X)
    sc=StandardScaler().fit(A)
    return LogisticRegression(max_iter=5000).fit(sc.transform(A),y[tr]).decision_function(sc.transform(Bm))

rows=[]
for mk in LEARNERS:
    for base in ["T1","T5"]:
        b=np.load(f"{OUT}/s_{base}_none_v2_{mk}.npy")
        f=f"{OUT}/s_{base}_reselect_v2_{mk}.npy"
        s=np.load(f) if os.path.exists(f) else score(np.hstack([BASES[base],RESEL]),mk)
        if not os.path.exists(f): np.save(f,s)
        d,lo,hi,p=paired(y[~tr],b[~tr],s[~tr])
        rows.append(dict(base=base,text="reselect",model=mk,
            auc_base=round(roc_auc_score(y[~tr],b[~tr]),4),
            auc_text=round(roc_auc_score(y[~tr],s[~tr]),4),
            increment=round(d,4),ci=f"{lo:+.4f},{hi:+.4f}",p=p))
        print(f"  {base}/{mk}: increment {d:+.4f} [{lo:+.4f},{hi:+.4f}] p={p:.3g}", flush=True)
R=pd.DataFrame(rows); R.to_csv(f"{OUT}/W1_reselect.csv",index=False)

old=pd.read_csv(f"{OUT}/W1_ablation.csv")
old=old[(old.analysis=="disposition")&(old.base.isin(["T1","T5"]))]
comb=pd.concat([old[["base","text","model","auc_base","auc_text","increment","ci","p"]],R])
comb["text"]=pd.Categorical(comb.text,["full","nodisp","reselect","disponly"],ordered=True)
comb=comb.sort_values(["model","base","text"])
pd.set_option("display.width",200)
print("\n=== full vs nodisp vs reselect vs disponly ===")
print(comb.to_string(index=False))
comb.to_csv(f"{OUT}/W1_ablation_with_reselect.csv",index=False)
