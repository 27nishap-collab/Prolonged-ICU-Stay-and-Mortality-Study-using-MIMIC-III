"""W1 structured feature set -- the 18 Weissman et al. 2018 predictors.

Specification: recovered Supplemental Methods ("Structured Data Sources").
See METHODS_weissman.md section 3 for the verbatim text.

Window for EVERY variable: [admittime, admittime + 48h) -- anchored on HOSPITAL
admission, not ICU intime.

--------------------------------------------------------------------------
WHAT THE VALIDATION PASS SETTLED (run 1, 2026-09-13)
--------------------------------------------------------------------------
Targets: supplemental table e4 (missingness % and observed mean, 12 variables)
and Table 1 (median and IQR).

RESOLVED -- the paper's variable labels are WRONG for two of the 18:
  * heart rate.   Paper says "highest". Table 1 prints 69 (60-78) and e4 gives
    mean 69.1. Our MAX is mean 106.0 / median 104; our MIN is mean 69.07 /
    median 68. Their variable is the MINIMUM.
  * temperature.  Paper says "highest". Table 1 prints 36.0 (35.6-36.4).
    Our MAX is mean 37.67 / median 37.60; our MIN is mean 36.00 / median 36.06.
    Their variable is the MINIMUM.
  Both are implemented as minima here, with the maxima retained as *_alt.

RESOLVED -- PaO2 must be arterial-only. All-specimen 50821 gives mean 94.1 at
  37.5% missing; restricting to specimens where labevents itemid 50800 = 'ART'
  at the same subject/charttime gives mean 102.1 at 45.4% missing, against
  their 103.0 at 54.0%. Arterial-only adopted.

RESOLVED -- "modified Elixhauser score" is a weighted score, not a count.
  Published median 8 (2-15). Quan flag COUNT gives 3 (2-4); van Walraven gives
  6 (0-12); AHRQ SID30 gives 8 (0-16). SID30 adopted.

MATCHED without adjustment: creatinine max (1.57 vs 1.66), WBC max (13.47 vs
  14.4), platelets min (191.9 vs 190.3), potassium max (4.65 vs 4.68),
  SBP min (91.2 vs 86.5; median 90 vs 89), GCS min (9.68 vs 9.61).

NOT RESOLVED -- declared deviations:
  * bilirubin coverage. Ours 57.3% missing / mean 1.75 vs their 44.1% / 1.54.
    Adding chartevents 4948+225690 and joining labs by subject_id+time instead
    of hadm_id both fail to close it. Their extraction sees ~13pp more patients.
  * urine output coverage. Ours 18.0% missing vs their 34.0%.
  * mechanical ventilation. Target 4,512 (17.4%). NOTHING reproduces it:
    mimic-code vent settings 47.3%; vent mode/type only 47.0%; CareVue 720
    alone 36.2%; CPT 94002-94657 39.8%; ICD-9 96.7x 27.4%; ICD-9 96.71 alone
    14.9%; CareVue 640 intubated/extubated 15.1%; MetaVision procedure 225792
    9.2%. The mimic-code flowsheet definition is used and the gap declared.
  * cardiac arrest. Target 206 (0.8%). The nursing flowsheet is far too sparse:
    CareVue 640 = 'Cardiac Arrest'/'Defibrillation' plus MetaVision 225466/
    225475 gives ~0.2%. ICD-9 procedure 99.60/99.63 (CPR) gives 231 (0.9%),
    the only close match -- but that is billing, not a flowsheet, and carries
    no timestamp. Flowsheet version is primary; ICD version kept as *_alt.

Output: out/features_W1.parquet          all extracted variants
        out/design_W1.parquet            the 18-predictor design matrix,
                                         mean-imputed, ready to model
        out/W1_feature_validation.csv    vs table e4
        out/W1_feature_validation_table1.csv  vs Table 1
"""
import duckdb, pandas as pd, numpy as np, os, re, time

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
DB = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
MC = os.path.expanduser("~/mc/mimic-iii/concepts/comorbidity")
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)

con = duckdb.connect(DB, read_only=True)
con.execute("PRAGMA threads=8")

coh = pd.read_csv(os.path.join(OUT, "cohort_W1.csv"),
                  parse_dates=["admittime", "dischtime", "t48"])
log(f"cohort W1: {len(coh):,} admissions")
con.register("coh_df", coh[["hadm_id", "subject_id", "admittime", "t48"]])
con.execute("CREATE OR REPLACE TEMP TABLE coh AS SELECT * FROM coh_df")

# ---------------------------------------------------------------- 1. LABS
con.execute("""
CREATE OR REPLACE TEMP TABLE labs AS
WITH pvt AS (
  SELECT c.hadm_id,
         CASE le.itemid WHEN 50912 THEN 'creatinine'
                        WHEN 51300 THEN 'wbc'  WHEN 51301 THEN 'wbc'
                        WHEN 51265 THEN 'platelets'
                        WHEN 50885 THEN 'bilirubin'
                        WHEN 50822 THEN 'potassium' WHEN 50971 THEN 'potassium'
         END AS lab,
         CASE WHEN le.itemid = 50912 AND le.valuenum > 150   THEN NULL
              WHEN le.itemid IN (51300,51301) AND le.valuenum > 1000 THEN NULL
              WHEN le.itemid = 51265 AND le.valuenum > 10000 THEN NULL
              WHEN le.itemid = 50885 AND le.valuenum > 150   THEN NULL
              WHEN le.itemid IN (50822,50971) AND le.valuenum > 30 THEN NULL
              ELSE le.valuenum END AS v
  FROM coh c JOIN labevents le USING (hadm_id)
  WHERE le.itemid IN (50912,51300,51301,51265,50885,50822,50971)
    AND le.valuenum IS NOT NULL AND le.valuenum > 0
    AND le.charttime >= c.admittime AND le.charttime < c.t48
)
SELECT hadm_id,
       MAX(CASE WHEN lab='creatinine' THEN v END) AS creatinine_max,
       MAX(CASE WHEN lab='wbc'        THEN v END) AS wbc_max,
       MIN(CASE WHEN lab='platelets'  THEN v END) AS platelets_min,
       MAX(CASE WHEN lab='bilirubin'  THEN v END) AS bilirubin_max,
       MAX(CASE WHEN lab='potassium'  THEN v END) AS potassium_max
FROM pvt GROUP BY hadm_id
""")
# PaO2: arterial specimens only. labevents 50800 SPECIMEN TYPE = 'ART' tags the
# blood-gas panel; match on (subject_id, charttime), which is how panel rows are
# keyed. Mean 102.1 vs their 103.0 -- the all-specimen version is 94.1.
con.execute("""
CREATE OR REPLACE TEMP TABLE pao2 AS
WITH art AS (SELECT DISTINCT subject_id, charttime FROM labevents
             WHERE itemid = 50800 AND value = 'ART')
SELECT c.hadm_id,
       MIN(le.valuenum) FILTER (WHERE a.subject_id IS NOT NULL) AS pao2_min,
       MIN(le.valuenum)                                         AS pao2_min_allspecimen
FROM coh c JOIN labevents le USING (hadm_id)
LEFT JOIN art a ON a.subject_id = le.subject_id AND a.charttime = le.charttime
WHERE le.itemid = 50821 AND le.valuenum > 0 AND le.valuenum < 800
  AND le.charttime >= c.admittime AND le.charttime < c.t48
GROUP BY c.hadm_id
""")
log("labs done")

# ---------------------------------------------------------------- 2. VITALS
# SBP minimum. HR and temperature: MINIMUM is primary (see header), maximum
# retained as *_alt so the choice stays auditable.
con.execute("""
CREATE OR REPLACE TEMP TABLE vitals AS
WITH pvt AS (
  SELECT c.hadm_id,
         CASE WHEN ce.itemid IN (211,220045) AND ce.valuenum > 0 AND ce.valuenum < 300 THEN 'hr'
              WHEN ce.itemid IN (51,442,455,6701,220179,220050) AND ce.valuenum > 0 AND ce.valuenum < 400 THEN 'sbp'
              WHEN ce.itemid IN (223761,678) AND ce.valuenum > 70 AND ce.valuenum < 120 THEN 'temp'
              WHEN ce.itemid IN (223762,676) AND ce.valuenum > 10 AND ce.valuenum < 50  THEN 'temp'
         END AS vit,
         CASE WHEN ce.itemid IN (223761,678) THEN (ce.valuenum - 32)/1.8 ELSE ce.valuenum END AS v
  FROM coh c JOIN chartevents ce USING (hadm_id)
  WHERE ce.itemid IN (211,220045,51,442,455,6701,220179,220050,223761,678,223762,676)
    AND ce.valuenum IS NOT NULL AND (ce.error IS NULL OR ce.error = 0)
    AND ce.charttime >= c.admittime AND ce.charttime < c.t48
)
SELECT hadm_id,
       MIN(CASE WHEN vit='sbp'  THEN v END) AS sbp_min,
       MIN(CASE WHEN vit='hr'   THEN v END) AS hr_min,
       MAX(CASE WHEN vit='hr'   THEN v END) AS hr_max_alt,
       MIN(CASE WHEN vit='temp' THEN v END) AS temp_min,
       MAX(CASE WHEN vit='temp' THEN v END) AS temp_max_alt
FROM pvt WHERE vit IS NOT NULL GROUP BY hadm_id
""")
log("vitals done")

# ---------------------------------------------------------------- 3. GCS
# Components pivoted per charttime; verbal 'ET/Trach' scored 0 (no sedation
# correction -- the paper describes none). Strict = all three present.
con.execute("""
CREATE OR REPLACE TEMP TABLE gcs AS
WITH ce AS (
  SELECT c.hadm_id, ce.charttime,
         CASE WHEN ce.itemid IN (723,223900) THEN 'verbal'
              WHEN ce.itemid IN (454,223901) THEN 'motor'
              WHEN ce.itemid IN (184,220739) THEN 'eyes'
              WHEN ce.itemid = 198           THEN 'total' END AS part,
         CASE WHEN ce.itemid = 723    AND ce.value = '1.0 ET/Trach'    THEN 0
              WHEN ce.itemid = 223900 AND ce.value = 'No Response-ETT' THEN 0
              ELSE ce.valuenum END AS v
  FROM coh c JOIN chartevents ce USING (hadm_id)
  WHERE ce.itemid IN (184,454,723,220739,223901,223900,198)
    AND (ce.error IS NULL OR ce.error = 0)
    AND ce.charttime >= c.admittime AND ce.charttime < c.t48
), wide AS (
  SELECT hadm_id, charttime,
         MAX(CASE WHEN part='motor'  THEN v END) AS m,
         MAX(CASE WHEN part='verbal' THEN v END) AS vb,
         MAX(CASE WHEN part='eyes'   THEN v END) AS e,
         MAX(CASE WHEN part='total'  THEN v END) AS tot
  FROM ce WHERE part IS NOT NULL GROUP BY hadm_id, charttime
)
SELECT hadm_id,
       MIN(CASE WHEN m IS NOT NULL AND vb IS NOT NULL AND e IS NOT NULL
                THEN m + vb + e END) AS gcs_min,
       MIN(CASE WHEN COALESCE(m,vb,e) IS NOT NULL
                THEN COALESCE(m,0)+COALESCE(vb,0)+COALESCE(e,0) END) AS gcs_min_lenient_alt
FROM wide GROUP BY hadm_id
""")
log("gcs done")

# ---------------------------------------------------------------- 4. WEIGHT
con.execute("""
CREATE OR REPLACE TEMP TABLE wt AS
WITH w AS (
  SELECT c.hadm_id, ce.charttime,
         CASE WHEN ce.itemid IN (3581,226531) THEN ce.valuenum * 0.45359237
              WHEN ce.itemid = 3582           THEN ce.valuenum * 0.0283495
              ELSE ce.valuenum END AS kg,
         (ce.charttime >= c.admittime AND ce.charttime < c.t48) AS in48
  FROM coh c JOIN chartevents ce USING (hadm_id)
  WHERE ce.itemid IN (762,763,3580,3693,3581,3582,226512,224639,226531)
    AND ce.valuenum IS NOT NULL AND ce.valuenum > 0
    AND (ce.error IS NULL OR ce.error = 0)
), f AS (SELECT * FROM w WHERE kg BETWEEN 20 AND 400)
SELECT hadm_id,
       MIN_BY(kg, charttime) FILTER (WHERE in48) AS weight_kg,
       MIN_BY(kg, charttime)                     AS weight_kg_any_alt
FROM f GROUP BY hadm_id
""")
log("weight done")

# ---------------------------------------------------------------- 5. URINE
con.execute("""
CREATE OR REPLACE TEMP TABLE uo AS
SELECT c.hadm_id,
       SUM(CASE WHEN oe.itemid = 227488 AND oe.value > 0 THEN -1*oe.value ELSE oe.value END) AS urine_cc
FROM coh c JOIN outputevents oe USING (hadm_id)
WHERE oe.itemid IN (40055,43175,40069,40094,40715,40473,40085,40057,40056,40405,
                    40428,40086,40096,40651,226559,226560,226561,226584,226563,
                    226564,226565,226567,226557,226558,227488,227489)
  AND oe.charttime >= c.admittime AND oe.charttime < c.t48
GROUP BY c.hadm_id
""")
log("urine done")

# ---------------------------------------------------------------- 6. EVENTS
VENT = ("720,223849,223848,445,448,449,450,1340,1486,1600,224687,639,654,681,682,683,684,"
        "224685,224684,224686,218,436,535,444,224697,224695,224696,224746,224747,221,1,"
        "1211,1655,2000,226873,224738,224419,224750,227187,543,5865,5866,224707,224709,"
        "224705,224706,60,437,505,506,686,220339,224700,3459,501,502,503,224702,223,667,"
        "668,669,670,671,672,224701")
con.execute(f"""
CREATE OR REPLACE TEMP TABLE ev AS
WITH icu AS (
  SELECT c.hadm_id, MAX(CASE WHEN i.intime < c.t48 THEN 1 ELSE 0 END) AS icu_48h
  FROM coh c LEFT JOIN icustays i USING (hadm_id) GROUP BY c.hadm_id
), vent AS (
  SELECT c.hadm_id, 1 AS vent_flowsheet
  FROM coh c JOIN chartevents ce USING (hadm_id)
  WHERE ce.itemid IN ({VENT}) AND ce.value IS NOT NULL
    AND (ce.error IS NULL OR ce.error = 0)
    AND ce.charttime >= c.admittime AND ce.charttime < c.t48
  GROUP BY c.hadm_id
), vent_icd AS (
  SELECT DISTINCT hadm_id FROM procedures_icd WHERE icd9_code LIKE '967%'
), arrest_cv AS (
  SELECT DISTINCT c.hadm_id FROM coh c JOIN chartevents ce USING (hadm_id)
  WHERE ((ce.itemid = 640 AND ce.value IN ('Cardiac Arrest','Defibrillation'))
      OR (ce.itemid = 4545 AND lower(ce.value) NOT IN ('none','no')))
    AND ce.value IS NOT NULL
    AND ce.charttime >= c.admittime AND ce.charttime < c.t48
), arrest_mv AS (
  SELECT DISTINCT c.hadm_id FROM coh c JOIN procedureevents_mv pe USING (hadm_id)
  WHERE pe.itemid IN (225466,225475)
    AND pe.starttime >= c.admittime AND pe.starttime < c.t48
), arrest_icd AS (
  SELECT DISTINCT hadm_id FROM procedures_icd WHERE icd9_code IN ('9960','9963')
)
SELECT c.hadm_id,
       COALESCE(icu.icu_48h, 0)             AS icu_48h,
       COALESCE(v.vent_flowsheet, 0)        AS vent_flowsheet,
       CASE WHEN vi.hadm_id  IS NOT NULL THEN 1 ELSE 0 END AS vent_icd_alt,
       CASE WHEN acv.hadm_id IS NOT NULL OR amv.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS cardiac_arrest,
       CASE WHEN aic.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS cardiac_arrest_icd_alt
FROM coh c
LEFT JOIN icu USING (hadm_id)
LEFT JOIN vent v       ON v.hadm_id   = c.hadm_id
LEFT JOIN vent_icd vi  ON vi.hadm_id  = c.hadm_id
LEFT JOIN arrest_cv acv  ON acv.hadm_id = c.hadm_id
LEFT JOIN arrest_mv amv  ON amv.hadm_id = c.hadm_id
LEFT JOIN arrest_icd aic ON aic.hadm_id = c.hadm_id
""")
log("events done")

# ---------------------------------------------------------------- 7. ELIXHAUSER
# LEAKAGE NOTE: ICD-9 codes are assigned at DISCHARGE, so an admission-level
# comorbidity index is not observable at hour 48. Weissman used one; we
# replicate it. See Analysis/availability_leak and METHODS_TRAPS.md.
# Their supplemental secondary analysis (GBM, structured only) drops 0.83 ->
# 0.81 without it, and the text-inclusive model is unaffected.
sql = open(os.path.join(MC, "elixhauser_quan.sql")).read()
sql = re.sub(r"^\s*--.*$", "", sql, flags=re.M).strip().rstrip(";")
sql = sql.replace("`physionet-data.mimiciii_clinical.diagnoses_icd`", "diagnoses_icd")
sql = sql.replace("`physionet-data.mimiciii_clinical.admissions`", "admissions")
eli = con.execute(sql).df()
eli.columns = [c.lower() for c in eli.columns]
FLAGS = [c for c in eli.columns if c != "hadm_id"]
eli[FLAGS] = eli[FLAGS].fillna(0).astype(int)
VW = {"aids":0,"alcohol_abuse":0,"blood_loss_anemia":-2,"congestive_heart_failure":7,
      "chronic_pulmonary":3,"coagulopathy":3,"deficiency_anemias":-2,"depression":-3,
      "diabetes_complicated":0,"diabetes_uncomplicated":0,"drug_abuse":-7,
      "fluid_electrolyte":5,"hypertension":0,"hypothyroidism":0,"liver_disease":11,
      "lymphoma":9,"metastatic_cancer":12,"other_neurological":6,"obesity":-4,
      "paralysis":7,"peripheral_vascular":2,"peptic_ulcer":0,"psychoses":0,
      "pulmonary_circulation":4,"rheumatoid_arthritis":0,"renal_failure":5,
      "solid_tumor":4,"valvular_disease":-1,"weight_loss":6}
SID30 = {"aids":7,"alcohol_abuse":-3,"blood_loss_anemia":-3,"cardiac_arrhythmias":5,
      "congestive_heart_failure":9,"chronic_pulmonary":3,"coagulopathy":11,
      "deficiency_anemias":-1,"depression":-5,"diabetes_complicated":-1,
      "diabetes_uncomplicated":0,"drug_abuse":-8,"fluid_electrolyte":10,
      "hypertension":-2,"hypothyroidism":0,"liver_disease":6,"lymphoma":7,
      "metastatic_cancer":13,"other_neurological":5,"obesity":-5,"paralysis":4,
      "peripheral_vascular":3,"peptic_ulcer":0,"psychoses":-5,
      "pulmonary_circulation":5,"rheumatoid_arthritis":0,"renal_failure":6,
      "solid_tumor":10,"valvular_disease":0,"weight_loss":9}
eli["elix_count_alt"]       = eli[FLAGS].sum(axis=1)
eli["elix_vanwalraven_alt"] = sum(w * eli[k] for k, w in VW.items()    if k in eli.columns)
eli["elixhauser"]           = sum(w * eli[k] for k, w in SID30.items() if k in eli.columns)
log(f"elixhauser: {len(eli):,} admissions, {len(FLAGS)} flags (SID30 adopted)")

# ---------------------------------------------------------------- ASSEMBLE
f = coh[["hadm_id","subject_id","age","gender","admission_type",
         "outcome_death","outcome_icu_los_7d","labels"]].copy()
for tbl in ["labs","pao2","vitals","gcs","wt","uo","ev"]:
    f = f.merge(con.execute(f"SELECT * FROM {tbl}").df(), on="hadm_id", how="left")
f = f.merge(eli[["hadm_id","elixhauser","elix_count_alt","elix_vanwalraven_alt"] + FLAGS],
            on="hadm_id", how="left")
for c in ["icu_48h","vent_flowsheet","vent_icd_alt","cardiac_arrest","cardiac_arrest_icd_alt"]:
    f[c] = f[c].fillna(0).astype(int)
COMO = ["elixhauser","elix_count_alt","elix_vanwalraven_alt"]
f[COMO] = f[COMO].fillna(0)

f["urine_cc_kg_hr"] = (f.urine_cc / f.weight_kg / 48.0).where(lambda s: s >= 0)
f.to_parquet(os.path.join(OUT, "features_W1.parquet"), index=False)
log(f"features_W1.parquet: {f.shape[0]:,} x {f.shape[1]}")

# ---------------------------------------------------------------- VALIDATION
E4 = {"pao2_min": (54.0, 103.0), "bilirubin_max": (44.1, 1.54),
      "urine_cc_kg_hr": (34.0, 0.85), "gcs_min": (26.8, 9.61),
      "weight_kg": (21.0, 83.3), "temp_min": (2.0, 35.5), "hr_min": (1.6, 69.1),
      "sbp_min": (1.6, 86.5), "potassium_max": (0.4, 4.68), "wbc_max": (0.3, 14.4),
      "platelets_min": (0.2, 190.3), "creatinine_max": (0.2, 1.66)}
T1MED = {"creatinine_max":"1.10 (0.80-1.60)", "wbc_max":"13.0 (9.70-17.10)",
         "platelets_min":"176 (125-235)", "bilirubin_max":"1.5 (0.6-1.5)*",
         "pao2_min":"103 (93-103)*", "potassium_max":"4.5 (4.2-4.9)",
         "urine_cc_kg_hr":"0.85 (0.57-0.90)*", "gcs_min":"9 (7-14)*",
         "sbp_min":"89 (79-101)", "hr_min":"69 (60-78)", "temp_min":"36.0 (35.6-36.4)"}
ALT = {"gcs_min":["gcs_min_lenient_alt"], "temp_min":["temp_max_alt"],
       "hr_min":["hr_max_alt"], "weight_kg":["weight_kg_any_alt"],
       "pao2_min":["pao2_min_allspecimen"]}
rows = []
for col, (pm, mu) in E4.items():
    for c in [col] + ALT.get(col, []):
        s = f[c]; q = s.quantile([.25,.5,.75])
        rows.append(dict(variable=c, primary=(c == col),
            ours_missing_pct=round(100*s.isna().mean(),1),
            weissman_missing_pct=pm if c == col else "",
            ours_mean=round(float(s.mean()),2),
            weissman_mean=mu if c == col else "",
            ours_median_iqr=f"{q[.5]:.2f} ({q[.25]:.2f}-{q[.75]:.2f})",
            weissman_median_iqr=T1MED.get(col,"") if c == col else ""))
val = pd.DataFrame(rows)

t1 = pd.DataFrame([
  dict(variable="mechanical ventilation (flowsheet, PRIMARY)",
       ours=f"{f.vent_flowsheet.sum():,} ({100*f.vent_flowsheet.mean():.1f})", published="4,512 (17.4)"),
  dict(variable="mechanical ventilation (ICD-9 96.7x, alt)",
       ours=f"{f.vent_icd_alt.sum():,} ({100*f.vent_icd_alt.mean():.1f})", published="4,512 (17.4)"),
  dict(variable="cardiac arrest (flowsheet, PRIMARY)",
       ours=f"{f.cardiac_arrest.sum():,} ({100*f.cardiac_arrest.mean():.1f})", published="206 (0.8)"),
  dict(variable="cardiac arrest (ICD-9 99.60/99.63 CPR, alt)",
       ours=f"{f.cardiac_arrest_icd_alt.sum():,} ({100*f.cardiac_arrest_icd_alt.mean():.1f})", published="206 (0.8)"),
  dict(variable="ICU transfer within 48h",
       ours=f"{f.icu_48h.sum():,} ({100*f.icu_48h.mean():.1f})", published="25,614 (98.7)"),
])
for c, nm in [("elixhauser","Elixhauser SID30 (PRIMARY)"),
              ("elix_vanwalraven_alt","Elixhauser van Walraven (alt)"),
              ("elix_count_alt","Elixhauser flag count (alt)")]:
    q = f[c].quantile([.25,.5,.75])
    t1.loc[len(t1)] = dict(variable=nm, ours=f"{q[.5]:.0f} ({q[.25]:.0f} - {q[.75]:.0f})",
                           published="8 (2 - 15)")
val.to_csv(os.path.join(OUT, "W1_feature_validation.csv"), index=False)
t1.to_csv(os.path.join(OUT, "W1_feature_validation_table1.csv"), index=False)

# ---------------------------------------------------------------- DESIGN MATRIX
# The 18 predictors, mean-imputed exactly as the supplement specifies
# ("replacing them with the mean of non-missing values"). The mean here is
# computed on the FULL cohort, matching their description; the observable
# training-fold mean is applied inside the modelling script as a sensitivity.
CONT = ["age","urine_cc_kg_hr","elixhauser","creatinine_max","wbc_max",
        "platelets_min","bilirubin_max","pao2_min","gcs_min","sbp_min",
        "hr_min","temp_min","potassium_max"]
d = f[["hadm_id","subject_id","labels","outcome_death","outcome_icu_los_7d"]].copy()
for c in CONT:
    d[c] = f[c]
    d[c + "_imputed"] = f[c].isna().astype(int)
d["male"] = (f.gender == "M").astype(int)
at = f.admission_type.str.upper()
d["adm_emergency"] = (at == "EMERGENCY").astype(int)
d["adm_elective"]  = (at == "ELECTIVE").astype(int)
d["adm_urgent"]    = (at == "URGENT").astype(int)
d["mech_vent"]      = f.vent_flowsheet
d["icu_admission"]  = f.icu_48h
d["cardiac_arrest"] = f.cardiac_arrest
imp = {c: float(f[c].mean()) for c in CONT}
for c in CONT: d[c] = d[c].fillna(imp[c])
d.to_parquet(os.path.join(OUT, "design_W1.parquet"), index=False)
pd.Series(imp).to_csv(os.path.join(OUT, "W1_imputation_means.csv"), header=["mean"])

print("\n=== table e4: missingness, observed mean, median (IQR) ===")
print("    * = their published median is pinned by mean imputation, not comparable")
print(val.to_string(index=False))
print("\n=== Table 1: events and comorbidity ===")
print(t1.to_string(index=False))
print(f"\ndesign_W1: {d.shape[0]:,} rows x {d.shape[1]} cols "
      f"({len(CONT)} continuous + gender + 3 admission-type + 3 event flags = 18 predictors; "
      f"outcome prevalence {100*d.labels.mean():.1f}%)")
log("done")
