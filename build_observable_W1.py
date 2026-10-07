"""Rebuild the W1 structured tiers under the LEAK-FREE specification.

    cd /Users/Shared/MIMIC_git/Analysis/replicate_weissman
    python3 build_observable_W1.py

Decided in SPEC_DECISION.md on 2026-09-20, before any observable number existed:

    A predictor enters the analysis only if its value is knowable at hour 48 of
    the admission being predicted.

`AUDIT_48h.md` found four violations among the 222 T5 variables. This script
fixes all four and writes a parallel set of outputs; the published-spec files are
left untouched, because the F1 / Table 2 replication must keep reproducing
Weissman's actual feature set.

  IN   out/tiers_W1.parquet, out/W1_tier_columns.json   (replication specification)
  OUT  out/tiers_W1_observable.parquet
       out/W1_tier_columns_observable.json
       out/W1_observable_manifest.json    what changed, and by how much

THE FOUR FIXES

1. `elixhauser` -> `has_prior_admission` + `prior_elixhauser`
   The published feature is SID30-weighted Elixhauser from THIS admission's
   discharge ICD-9 coding: not knowable at hour 48, for anybody. Replaced by the
   same index computed over the patient's EARLIER admissions (dischtime strictly
   before this admittime), which was finalised before this admission began.
   Only 21.8% of W1 has an earlier admission in MIMIC-III, so the score is NaN
   for the rest and `has_prior_admission` carries the indicator separately —
   collapsing them into one zero-filled column would silently conflate "no
   comorbidities" with "no prior record".

2. `n_icu_stays` — counted stays across the whole admission (differs from its
   48h-observable value in 6.3% of admissions). Now counts stays beginning
   before hour 48.

3. `first_careunit`, `pre_icu_hours` — took the earliest ICU stay whenever it
   occurred (unobservable in 0.2%). Now the earliest stay beginning before
   hour 48; NaN if there is none.

4. `first_service`, `surgical_service` — took the earliest service transfer
   whenever it occurred (unobservable in 0.05%). Now the earliest transfer
   before hour 48.

The other 199 variables (152 labs, 28 vitals, 7 GCS, 7 interventions, 3 in/out,
2 body) and the remaining 19 T1 predictors are already windowed and are copied
through unchanged.
"""
import os, json, re, sys
import numpy as np, pandas as pd, duckdb

D   = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(D, "out")
DB  = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
MC  = os.path.expanduser("~/mc/mimic-iii/concepts/comorbidity")
SID30 = {"aids":7,"alcohol_abuse":-3,"blood_loss_anemia":-3,"cardiac_arrhythmias":5,
    "congestive_heart_failure":9,"chronic_pulmonary":3,"coagulopathy":11,
    "deficiency_anemias":-1,"depression":-5,"diabetes_complicated":-1,
    "diabetes_uncomplicated":0,"drug_abuse":-8,"fluid_electrolyte":10,
    "hypertension":-2,"hypothyroidism":0,"liver_disease":6,"lymphoma":7,
    "metastatic_cancer":13,"other_neurological":5,"obesity":-5,"paralysis":4,
    "peripheral_vascular":3,"peptic_ulcer":0,"psychoses":-5,
    "pulmonary_circulation":5,"rheumatoid_arthritis":0,"renal_failure":6,
    "solid_tumor":10,"valvular_disease":0,"weight_loss":9}
SURG = ["CSURG","SURG","NSURG","TSURG","VSURG","ORTHO","PSURG","TRAUM","GU","ENT","OBS"]

def log(m): print(m, flush=True)

coh = pd.read_csv(f"{OUT}/cohort_W1.csv", parse_dates=["admittime", "t48"])
con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=4")
con.register("coh_df", coh[["hadm_id", "subject_id", "admittime", "t48"]])
con.execute("CREATE TEMP TABLE coh AS SELECT * FROM coh_df")
man = {"decided": "SPEC_DECISION.md 2026-09-20", "fixes": {}}

# ---- 1. prior-admission Elixhauser -----------------------------------------
sql = open(os.path.join(MC, "elixhauser_quan.sql")).read()
sql = re.sub(r"^\s*--.*$", "", sql, flags=re.M).strip().rstrip(";")
sql = sql.replace("`physionet-data.mimiciii_clinical.diagnoses_icd`", "diagnoses_icd")
sql = sql.replace("`physionet-data.mimiciii_clinical.admissions`", "admissions")
eli = con.execute(sql).df(); eli.columns = [c.lower() for c in eli.columns]
FLAGS = [c for c in eli.columns if c != "hadm_id"]
eli[FLAGS] = eli[FLAGS].fillna(0).astype(int)
log(f"elixhauser flags for {len(eli):,} admissions, {len(FLAGS)} flags")

con.register("eli_df", eli)
# every (cohort admission, earlier admission of the same patient) pair
prior = con.execute("""
  SELECT c.hadm_id,
         COUNT(DISTINCT a.hadm_id) AS n_prior_adm,
         """ + ", ".join(f"MAX(e.{f}) AS {f}" for f in FLAGS) + """
  FROM coh c
  JOIN admissions a ON a.subject_id = c.subject_id AND a.dischtime < c.admittime
  JOIN eli_df   e ON e.hadm_id = a.hadm_id
  GROUP BY c.hadm_id""").df()
prior.columns = [c.lower() for c in prior.columns]
prior["prior_elixhauser"] = sum(w * prior[k] for k, w in SID30.items() if k in prior.columns)
prior = prior[["hadm_id", "n_prior_adm", "prior_elixhauser"]]

P = coh[["hadm_id"]].merge(prior, on="hadm_id", how="left")
P["has_prior_admission"] = P.n_prior_adm.notna().astype(int)
P.loc[P.has_prior_admission == 0, "prior_elixhauser"] = np.nan   # absent, not zero
cov = P.has_prior_admission.mean()
log(f"prior_elixhauser: {P.has_prior_admission.sum():,}/{len(P):,} = {100*cov:.1f}% have an earlier admission")
man["fixes"]["elixhauser"] = dict(
    replaced_by=["has_prior_admission", "prior_elixhauser"],
    coverage=round(float(cov), 4),
    mean_prior_score=round(float(P.prior_elixhauser.mean()), 3))

# ---- 2-4. context tier, windowed to 48h ------------------------------------
ctx = con.execute("""
SELECT c.hadm_id, a.admission_location, a.insurance, a.marital_status,
       a.ethnicity, a.language, a.religion,
       s.curr_service AS first_service,
       i.first_careunit, COALESCE(i.n_icu_stays, 0) AS n_icu_stays,
       DATE_DIFF('minute', c.admittime, i.first_intime)/60.0 AS pre_icu_hours
FROM coh c JOIN admissions a USING (hadm_id)
LEFT JOIN (SELECT c.hadm_id,
                  ARG_MIN(i.first_careunit, i.intime) AS first_careunit,
                  MIN(i.intime)                       AS first_intime,
                  COUNT(*)                            AS n_icu_stays
           FROM coh c JOIN icustays i USING (hadm_id)
           WHERE i.intime < c.t48                    -- FIX 2 and 3
           GROUP BY c.hadm_id) i USING (hadm_id)
LEFT JOIN (SELECT c.hadm_id, ARG_MIN(s.curr_service, s.transfertime) AS curr_service
           FROM coh c JOIN services s USING (hadm_id)
           WHERE s.transfertime < c.t48              -- FIX 4
           GROUP BY c.hadm_id) s USING (hadm_id)""").df()
ctx.columns = [x.lower() for x in ctx.columns]
eth = ctx.ethnicity.fillna("").str.upper()
ctx["ethnicity_grp"] = np.select(
    [eth.str.contains("WHITE"), eth.str.contains("BLACK"),
     eth.str.contains("HISPANIC|LATINO"), eth.str.contains("ASIAN")],
    ["white", "black", "hispanic", "asian"], "other")
ctx = ctx.drop(columns=["ethnicity"])
ctx["surgical_service"] = ctx.first_service.isin(SURG).astype(int)

old = pd.read_parquet(f"{OUT}/tier_context_W1.parquet")
cmp = old[["hadm_id", "n_icu_stays", "first_careunit", "pre_icu_hours", "first_service"]].merge(
      ctx[["hadm_id", "n_icu_stays", "first_careunit", "pre_icu_hours", "first_service"]],
      on="hadm_id", suffixes=("_pub", "_lf"))
for c in ["n_icu_stays", "first_careunit", "pre_icu_hours", "first_service"]:
    a, b = cmp[f"{c}_pub"], cmp[f"{c}_lf"]
    # to string first: fillna("~") on a nullable Int64 column raises
    sa = a.astype(object).where(a.notna(), "~").astype(str)
    sb = b.astype(object).where(b.notna(), "~").astype(str)
    d = (sa != sb).mean()
    log(f"  {c:16s} changed in {100*d:5.2f}% of admissions")
    man["fixes"][c] = dict(changed_pct=round(100*float(d), 3))

# ---- assemble ---------------------------------------------------------------
T = pd.read_parquet(f"{OUT}/tiers_W1.parquet")
cols = json.load(open(f"{OUT}/W1_tier_columns.json"))
ctx_cols = [c for c in ctx.columns if c != "hadm_id"]
T = T.drop(columns=[c for c in ctx_cols if c in T.columns] + ["elixhauser"], errors="ignore")
T = T.merge(ctx, on="hadm_id", how="left").merge(
    P[["hadm_id", "has_prior_admission", "prior_elixhauser"]], on="hadm_id", how="left")

new = {}
for k, v in cols.items():
    v2 = [c for c in v if c != "elixhauser"]
    v2 = ["has_prior_admission", "prior_elixhauser"] + v2      # T1..T5 all inherit
    missing = [c for c in v2 if c not in T.columns]
    assert not missing, f"{k}: missing {missing}"
    new[k] = v2
for a, b in zip(["T1","T2","T3","T4"], ["T2","T3","T4","T5"]):
    assert set(new[a]) <= set(new[b]), f"nesting broken at {a} -> {b}"
log("\ntier sizes  " + "  ".join(f"{k} {len(v)}" for k, v in new.items()) + "   (nesting verified)")

T.to_parquet(f"{OUT}/tiers_W1_observable.parquet", index=False)
json.dump(new, open(f"{OUT}/W1_tier_columns_observable.json", "w"), indent=1)
man["tier_sizes"] = {k: len(v) for k, v in new.items()}
json.dump(man, open(f"{OUT}/W1_observable_manifest.json", "w"), indent=2)
log(f"\nwrote out/tiers_W1_observable.parquet  ({T.shape[0]:,} x {T.shape[1]})")
log("      out/W1_tier_columns_observable.json")
log("      out/W1_observable_manifest.json")
log("\nPublished-spec files untouched — the F1 replication still needs them.")
