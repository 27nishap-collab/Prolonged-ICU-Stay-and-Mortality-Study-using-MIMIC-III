"""Nested structured feature tiers on the W1 cohort -- the saturation curve.

    python build_tiers_W1.py vitals | gcs | body | labs | io | intervention | context | assemble | all

THE QUESTION
------------
Across the published MIMIC-III ICU text studies the measured value of notes
falls as the structured baseline grows -- Ghassemi (~3 variables), Weissman
(18), Husmann (42 encoded dims), ours (216), Hsu (~32,000). But feature-set
size, cohort size, note sparsity, outcome and text pipeline all move together
across those papers, so the trend is uninterpretable.

This holds EVERYTHING fixed except the number of structured features: same
cohort (W1, 25,692 admissions), same 48h window from hospital admission, same
composite outcome, same 75/25 split, same text block (21 closed-vocabulary +
500 open-vocabulary counts + word count, selected once on the original training
set), same model family. Only the structured block changes.

WHY A NEW EXTRACTOR
-------------------
Our existing 216-feature set (structured_data_processing/) is MetaVision-only
itemids, which is fine for A3 (99.6% MetaVision) and useless here: W1 is
**72.6% CareVue** (18,660 admissions vs 6,897 MetaVision). Every chart-derived
concept below therefore pools CareVue and MetaVision itemids. Labs come from
labevents and are source-independent.

THE TIERS -- strictly nested, T1 subset of T2 subset of ... subset of T5
-----------------------------------------------------------------------
T1  ~18   Weissman's exact 18 predictors (design_W1.parquet).
T2  ~48   + min/max/mean/count for the concepts Weissman already used, instead
          of his single "most severe" value. Isolates summary richness with the
          concept set held fixed.
T3  ~68   + the routine vitals Weissman omitted (DBP, MBP, respiratory rate,
          SpO2) and body size (weight, height).
T4  ~190  + ~30 further routine labs, each min/max/mean/count.
T5  ~215  + fluids, interventions (ventilation, intubation/extubation,
          vasopressors, renal replacement) and admission context.

Each tier is a feature set someone could plausibly have built, and each is a
superset of the one before, so the curve reads as "more structure", not
"different structure".

Aggregations are computed over [admittime, admittime+48h) exactly as in
build_features_W1.py. Labs do NOT look back before admittime -- the Weissman
window definition is kept rather than our own 6h lookback, so T1 stays nested.

DECLARED LIMITATION
-------------------
Elixhauser (in T1, so in every tier) is discharge-coded ICD-9 and is not
observable at hour 48. It is carried through all tiers so the comparison is
internally consistent; the leak is constant across the curve and therefore
cannot produce a trend in it. See FEATURES_W1.md section 5.
"""
import os, sys, json, time
import duckdb, numpy as np, pandas as pd

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
DB = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
os.makedirs(OUT, exist_ok=True)
t0 = time.time()
def log(m): print(f"[{time.time()-t0:6.1f}s] {m}", flush=True)
P = lambda b: os.path.join(OUT, f"tier_{b}_W1.parquet")

def con_():
    c = duckdb.connect(DB, read_only=True)
    c.execute("PRAGMA threads=4"); c.execute("PRAGMA memory_limit='1500MB'")
    coh = pd.read_csv(os.path.join(OUT, "cohort_W1.csv"), parse_dates=["admittime", "t48"])
    c.register("coh_df", coh[["hadm_id", "subject_id", "admittime", "t48"]])
    c.execute("CREATE TEMP TABLE coh AS SELECT * FROM coh_df")
    return c, coh

# ---------------------------------------------------------------- vitals
# CareVue + MetaVision itemids pooled per concept; ranges are mimic-code's.
VITALS = {
  "hr":    ([211, 220045], 0, 300),
  "sbp":   ([51, 442, 455, 6701, 220179, 220050], 0, 400),
  "dbp":   ([8368, 8440, 8441, 8555, 220180, 220051], 0, 300),
  "mbp":   ([456, 52, 6702, 443, 220052, 220181, 225312], 0, 300),
  "rr":    ([615, 618, 220210, 224690], 0, 70),
  "spo2":  ([646, 220277], 0, 100),
}
TEMP_F = [223761, 678]; TEMP_C = [223762, 676]

def step_vitals():
    c, _ = con_()
    ids = sorted({i for v, _, _ in VITALS.values() for i in v} | set(TEMP_F) | set(TEMP_C))
    case = "\n".join(
        f"WHEN ce.itemid IN ({','.join(map(str,v))}) AND ce.valuenum > {lo} AND ce.valuenum < {hi} THEN '{k}'"
        for k, (v, lo, hi) in VITALS.items())
    agg = ", ".join(
        f"{f}(CASE WHEN vit='{k}' THEN v END) AS {k}_{s}"
        for k in list(VITALS) + ["temp_c"] for f, s in
        [("MIN","min"),("MAX","max"),("AVG","mean"),("COUNT","n")])
    c.execute(f"""
    CREATE OR REPLACE TEMP TABLE v AS
    WITH pvt AS (
      SELECT c.hadm_id,
             CASE {case}
                  WHEN ce.itemid IN ({','.join(map(str,TEMP_F))}) AND ce.valuenum > 70  AND ce.valuenum < 120 THEN 'temp_c'
                  WHEN ce.itemid IN ({','.join(map(str,TEMP_C))}) AND ce.valuenum > 10  AND ce.valuenum < 50  THEN 'temp_c'
             END AS vit,
             CASE WHEN ce.itemid IN ({','.join(map(str,TEMP_F))}) THEN (ce.valuenum-32)/1.8
                  ELSE ce.valuenum END AS v
      FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid IN ({','.join(map(str,ids))}) AND ce.valuenum IS NOT NULL
        AND (ce.error IS NULL OR ce.error = 0)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    )
    SELECT hadm_id, {agg} FROM pvt WHERE vit IS NOT NULL GROUP BY hadm_id""")
    c.execute(f"SELECT * FROM v").df().to_parquet(P("vitals"), index=False)
    log(f"vitals -> {P('vitals')}")

# ---------------------------------------------------------------- gcs
def step_gcs():
    c, _ = con_()
    c.execute("""
    CREATE OR REPLACE TEMP TABLE g AS
    WITH ce AS (
      SELECT c.hadm_id, ce.charttime,
             CASE WHEN ce.itemid IN (723,223900) THEN 'verbal'
                  WHEN ce.itemid IN (454,223901) THEN 'motor'
                  WHEN ce.itemid IN (184,220739) THEN 'eyes' END AS part,
             CASE WHEN ce.itemid = 723    AND ce.value = '1.0 ET/Trach'    THEN 0
                  WHEN ce.itemid = 223900 AND ce.value = 'No Response-ETT' THEN 0
                  ELSE ce.valuenum END AS v
      FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid IN (184,454,723,220739,223901,223900)
        AND (ce.error IS NULL OR ce.error = 0)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    ), wide AS (
      SELECT hadm_id, charttime,
             MAX(CASE WHEN part='motor'  THEN v END) AS m,
             MAX(CASE WHEN part='verbal' THEN v END) AS vb,
             MAX(CASE WHEN part='eyes'   THEN v END) AS e
      FROM ce WHERE part IS NOT NULL GROUP BY hadm_id, charttime
    ), tot AS (
      SELECT *, CASE WHEN m IS NOT NULL AND vb IS NOT NULL AND e IS NOT NULL
                     THEN m+vb+e END AS g FROM wide
    )
    SELECT hadm_id,
           MIN(g) AS gcs_min, MAX(g) AS gcs_max, AVG(g) AS gcs_mean, COUNT(g) AS gcs_n,
           MIN(m) AS gcs_motor_min, MIN(vb) AS gcs_verbal_min, MIN(e) AS gcs_eyes_min
    FROM tot GROUP BY hadm_id""")
    c.execute("SELECT * FROM g").df().to_parquet(P("gcs"), index=False)
    log(f"gcs -> {P('gcs')}")

# ---------------------------------------------------------------- body
def step_body():
    c, _ = con_()
    c.execute("""
    CREATE OR REPLACE TEMP TABLE b AS
    WITH w AS (
      SELECT c.hadm_id, ce.charttime,
             CASE WHEN ce.itemid IN (3581,226531) THEN ce.valuenum*0.45359237
                  WHEN ce.itemid = 3582          THEN ce.valuenum*0.0283495
                  ELSE ce.valuenum END AS kg
      FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid IN (762,763,3580,3693,3581,3582,226512,224639,226531)
        AND ce.valuenum > 0 AND (ce.error IS NULL OR ce.error = 0)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    ), h AS (
      SELECT c.hadm_id,
             CASE WHEN ce.itemid IN (1394,226707) THEN ce.valuenum*2.54
                  ELSE ce.valuenum END AS cm
      FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid IN (920,1394,4187,3486,3485,4188,226707,226730)
        AND ce.valuenum > 0 AND (ce.error IS NULL OR ce.error = 0)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    )
    SELECT COALESCE(w.hadm_id, h.hadm_id) AS hadm_id, w.weight_kg, h.height_cm FROM
      (SELECT hadm_id, MIN_BY(kg, charttime) AS weight_kg FROM w WHERE kg BETWEEN 20 AND 400 GROUP BY hadm_id) w
    FULL JOIN
      (SELECT hadm_id, MEDIAN(cm) AS height_cm FROM h WHERE cm BETWEEN 100 AND 250 GROUP BY hadm_id) h
    USING (hadm_id)""")
    c.execute("SELECT * FROM b").df().to_parquet(P("body"), index=False)
    log(f"body -> {P('body')}")

# ---------------------------------------------------------------- labs
# name: (itemids, lo, hi). The first six are Weissman's; the rest are T4.
LABS = {
 "creatinine":  ([50912], 0, 150),   "wbc":       ([51300,51301], 0, 1000),
 "platelets":   ([51265], 0, 10000), "bilirubin": ([50885], 0, 150),
 "pao2":        ([50821], 0, 800),   "potassium": ([50822,50971], 0, 30),
 "sodium":      ([50824,50983], 0, 200),  "chloride":  ([50806,50902], 0, 10000),
 "bicarbonate": ([50882], 0, 10000),      "bun":       ([51006], 0, 300),
 "glucose":     ([50809,50931], 0, 10000),"hematocrit":([50810,51221], 0, 100),
 "hemoglobin":  ([50811,51222], 0, 50),   "lactate":   ([50813], 0, 50),
 "inr":         ([51237], 0, 50),         "pt":        ([51274], 0, 150),
 "ptt":         ([51275], 0, 150),        "aniongap":  ([50868], 0, 10000),
 "albumin":     ([50862], 0, 10),         "bands":     ([51144], 0, 100),
 "calcium":     ([50893], 0, 30),         "magnesium": ([50960], 0, 20),
 "phosphate":   ([50970], 0, 30),         "ph":        ([50820], 6, 8),
 "pco2":        ([50818], 0, 200),        "baseexcess":([50802], -50, 50),
 "alt":         ([50861], 0, 20000),      "ast":       ([50878], 0, 20000),
 "alkphos":     ([50863], 0, 5000),       "troponin":  ([51003], 0, 100),
 "ck":          ([50910], 0, 100000),     "ckmb":      ([50911], 0, 2000),
 "fibrinogen":  ([51214], 0, 2000),       "lymphocytes":([51244], 0, 100),
 "neutrophils": ([51256], 0, 100),        "rdw":       ([51277], 0, 50),
 "mcv":         ([51250], 0, 200),        "wbc_urine": ([51516], 0, 10000),
}
def step_labs():
    c, _ = con_()
    ids = sorted({i for v, _, _ in LABS.values() for i in v})
    case = "\n".join(
        f"WHEN le.itemid IN ({','.join(map(str,v))}) AND le.valuenum > {lo} AND le.valuenum < {hi} THEN '{k}'"
        for k, (v, lo, hi) in LABS.items())
    agg = ", ".join(f"{f}(CASE WHEN lab='{k}' THEN v END) AS {k}_{s}"
                    for k in LABS for f, s in
                    [("MIN","min"),("MAX","max"),("AVG","mean"),("COUNT","n")])
    c.execute(f"""
    CREATE OR REPLACE TEMP TABLE l AS
    WITH pvt AS (
      SELECT c.hadm_id, CASE {case} END AS lab, le.valuenum AS v
      FROM coh c JOIN labevents le USING (hadm_id)
      WHERE le.itemid IN ({','.join(map(str,ids))}) AND le.valuenum IS NOT NULL
        AND le.charttime >= c.admittime AND le.charttime < c.t48
    )
    SELECT hadm_id, {agg} FROM pvt WHERE lab IS NOT NULL GROUP BY hadm_id""")
    c.execute("SELECT * FROM l").df().to_parquet(P("labs"), index=False)
    log(f"labs -> {P('labs')} ({len(LABS)} analytes)")

# ---------------------------------------------------------------- io
URINE = [40055,43175,40069,40094,40715,40473,40085,40057,40056,40405,40428,40086,
         40096,40651,226559,226560,226561,226584,226563,226564,226565,226567,
         226557,226558,227488,227489]
def step_io():
    c, _ = con_()
    c.execute(f"""
    CREATE OR REPLACE TEMP TABLE io AS
    SELECT c.hadm_id,
           SUM(CASE WHEN oe.itemid = 227488 AND oe.value > 0 THEN -oe.value
                    WHEN oe.itemid IN ({','.join(map(str,URINE))}) THEN oe.value END) AS urine_ml,
           SUM(oe.value)   AS output_total_ml,
           COUNT(*)        AS output_n
    FROM coh c JOIN outputevents oe USING (hadm_id)
    WHERE oe.value IS NOT NULL
      AND oe.charttime >= c.admittime AND oe.charttime < c.t48
    GROUP BY c.hadm_id""")
    c.execute("SELECT * FROM io").df().to_parquet(P("io"), index=False)
    log(f"io -> {P('io')}")

# ---------------------------------------------------------------- interventions
VENT = ("720,223849,223848,445,448,449,450,1340,1486,1600,224687,639,654,681,682,683,684,"
        "224685,224684,224686,218,436,535,444,224697,224695,224696,224746,224747,221,1,"
        "1211,1655,2000,226873,224738,224419,224750,227187,543,5865,5866,224707,224709,"
        "224705,224706,60,437,505,506,686,220339,224700,3459,501,502,503,224702,223,667,"
        "668,669,670,671,672,224701")
VASO_CV = [30047,30120,30043,30307,30051,42273,42802,30044,30119,30309,30042,30306,30127,30128]
VASO_MV = [221906,221662,222315,221749,221289,221653]
RRT_CHART = [152, 148, 149, 150]
RRT_OUT   = [40386,40425,40426,40507,40613,40624,40690,40745,40881,40910,41016,
             41250,41374,41417,41500,41527,41635,41750]
RRT_MV    = [225802,225803,225805,225809,225955]

def step_intervention():
    c, _ = con_()
    c.execute(f"""
    CREATE OR REPLACE TEMP TABLE iv AS
    WITH vent AS (
      SELECT c.hadm_id, 1 AS vent_flag,
             DATE_DIFF('minute', MIN(ce.charttime), MAX(ce.charttime))/60.0 AS vent_span_h,
             COUNT(*) AS vent_n
      FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid IN ({VENT}) AND ce.value IS NOT NULL
        AND (ce.error IS NULL OR ce.error = 0)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
      GROUP BY c.hadm_id
    ), extub AS (
      SELECT DISTINCT c.hadm_id FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ((ce.itemid = 640 AND ce.value IN ('Extubated','Self Extubation'))
          OR ce.itemid = 227194)
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    ), intub AS (
      SELECT DISTINCT c.hadm_id FROM coh c JOIN chartevents ce USING (hadm_id)
      WHERE ce.itemid = 640 AND ce.value = 'Intubated'
        AND ce.charttime >= c.admittime AND ce.charttime < c.t48
    ), vcv AS (
      SELECT DISTINCT c.hadm_id FROM coh c JOIN inputevents_cv i USING (hadm_id)
      WHERE i.itemid IN ({','.join(map(str,VASO_CV))}) AND i.amount > 0
        AND i.charttime >= c.admittime AND i.charttime < c.t48
    ), vmv AS (
      SELECT DISTINCT c.hadm_id FROM coh c JOIN inputevents_mv i USING (hadm_id)
      WHERE i.itemid IN ({','.join(map(str,VASO_MV))}) AND i.amount > 0
        AND (i.statusdescription IS NULL OR i.statusdescription <> 'Rewritten')
        AND i.starttime >= c.admittime AND i.starttime < c.t48
    ), rrt AS (
      SELECT DISTINCT hadm_id FROM (
        SELECT c.hadm_id FROM coh c JOIN chartevents ce USING (hadm_id)
        WHERE ce.itemid IN ({','.join(map(str,RRT_CHART))}) AND ce.value IS NOT NULL
          AND ce.charttime >= c.admittime AND ce.charttime < c.t48
        UNION ALL
        SELECT c.hadm_id FROM coh c JOIN outputevents oe USING (hadm_id)
        WHERE oe.itemid IN ({','.join(map(str,RRT_OUT))})
          AND oe.charttime >= c.admittime AND oe.charttime < c.t48
        UNION ALL
        SELECT c.hadm_id FROM coh c JOIN procedureevents_mv p USING (hadm_id)
        WHERE p.itemid IN ({','.join(map(str,RRT_MV))})
          AND p.starttime >= c.admittime AND p.starttime < c.t48)
    )
    SELECT c.hadm_id,
           COALESCE(v.vent_flag,0) AS vent_flag,
           COALESCE(v.vent_span_h,0) AS vent_span_h,
           COALESCE(v.vent_n,0) AS vent_n,
           CASE WHEN e.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS extubated,
           CASE WHEN n.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS intubated,
           CASE WHEN cv.hadm_id IS NOT NULL OR mv.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS vasopressor,
           CASE WHEN r.hadm_id IS NOT NULL THEN 1 ELSE 0 END AS rrt
    FROM coh c LEFT JOIN vent v USING (hadm_id)
      LEFT JOIN extub e ON e.hadm_id=c.hadm_id LEFT JOIN intub n ON n.hadm_id=c.hadm_id
      LEFT JOIN vcv cv ON cv.hadm_id=c.hadm_id LEFT JOIN vmv mv ON mv.hadm_id=c.hadm_id
      LEFT JOIN rrt r ON r.hadm_id=c.hadm_id""")
    c.execute("SELECT * FROM iv").df().to_parquet(P("intervention"), index=False)
    log(f"intervention -> {P('intervention')}")

# ---------------------------------------------------------------- context
def step_context():
    c, _ = con_()
    d = c.execute("""
    SELECT c.hadm_id, a.admission_location, a.insurance, a.marital_status,
           a.ethnicity, a.language, a.religion,
           s.curr_service AS first_service,
           i.first_careunit, i.n_icu_stays,
           DATE_DIFF('minute', c.admittime, i.first_intime)/60.0 AS pre_icu_hours
    FROM coh c JOIN admissions a USING (hadm_id)
    LEFT JOIN (SELECT hadm_id, ARG_MIN(first_careunit, intime) AS first_careunit,
                      MIN(intime) AS first_intime, COUNT(*) AS n_icu_stays
               FROM icustays GROUP BY hadm_id) i USING (hadm_id)
    LEFT JOIN (SELECT hadm_id, ARG_MIN(curr_service, transfertime) AS curr_service
               FROM services GROUP BY hadm_id) s USING (hadm_id)""").df()
    d.columns = [x.lower() for x in d.columns]
    eth = d.ethnicity.fillna("").str.upper()
    d["ethnicity_grp"] = np.select(
        [eth.str.contains("WHITE"), eth.str.contains("BLACK"),
         eth.str.contains("HISPANIC|LATINO"), eth.str.contains("ASIAN")],
        ["white", "black", "hispanic", "asian"], "other")
    d = d.drop(columns=["ethnicity"])
    d["surgical_service"] = d.first_service.isin(
        ["CSURG","SURG","NSURG","TSURG","VSURG","ORTHO","PSURG","TRAUM","GU","ENT","OBS"]
    ).astype(int)
    d.to_parquet(P("context"), index=False)
    log(f"context -> {P('context')}")

# ---------------------------------------------------------------- assemble
# T1 is Weissman's 18 (20 columns: admission type is three indicators). Every
# continuous member is taken RAW here, not from design_W1.parquet -- that file is
# mean-imputed, and mixing imputed T1 columns with raw T2+ columns would make the
# tiers non-comparable. Imputation is applied once, uniformly, at model time.
#   temp_c_min IS Weissman's temperature variable (his label says "highest"; the
#   published values are minima -- FEATURES_W1.md 3.1), and pao2_min is the
#   arterial-only version, so both come from the blocks that define them.
T1 = ["age","elixhauser","male","adm_emergency","adm_elective","adm_urgent",
      "mech_vent","icu_admission","cardiac_arrest",
      "urine_cc_kg_hr_w","pao2_min",
      "creatinine_max","wbc_max","platelets_min","bilirubin_max","potassium_max",
      "gcs_min","sbp_min","hr_min","temp_c_min"]
W_LABS  = ["creatinine","wbc","platelets","bilirubin","pao2","potassium"]
W_VITAL = ["hr","sbp","temp_c"]
EXTRA_VITAL = ["dbp","mbp","rr","spo2"]
STAT = ["min","max","mean","n"]

def step_assemble():
    keys = ["hadm_id","subject_id","labels","outcome_death","outcome_icu_los_7d"]
    d = pd.read_parquet(os.path.join(OUT, "design_W1.parquet"))
    raw = pd.read_parquet(os.path.join(OUT, "features_W1.parquet"))
    # no-missingness members + the two T1 variables defined only in features_W1
    f = d[keys + ["age","elixhauser","male","adm_emergency","adm_elective",
                  "adm_urgent","mech_vent","icu_admission","cardiac_arrest"]].copy()
    f = f.merge(raw[["hadm_id","urine_cc_kg_hr","pao2_min"]]
                   .rename(columns={"urine_cc_kg_hr":"urine_cc_kg_hr_w"}),
                on="hadm_id", how="left")
    for b in ["vitals","gcs","body","labs","io","intervention","context"]:
        f = f.merge(pd.read_parquet(P(b)), on="hadm_id", how="left",
                    suffixes=("", f"_{b}"))
    f = f.loc[:, ~f.columns.duplicated()]
    # urine per kg per hour, recomputed on the richer weight
    f["urine_ml_kg_hr"] = (f.urine_ml / f.weight_kg / 48.0).where(lambda s: s >= 0)

    # pao2 from the labs block is all-specimen; T1's pao2_min is arterial-only.
    # Both are kept from T2 on -- they are different measurements, not duplicates.
    t2 = T1 + [f"{k}_{s}" for k in W_LABS + W_VITAL for s in STAT] \
            + ["pao2_min_labs",
               "gcs_max","gcs_mean","gcs_n","gcs_motor_min","gcs_verbal_min","gcs_eyes_min"]
    t3 = t2 + [f"{k}_{s}" for k in EXTRA_VITAL for s in STAT] + ["weight_kg","height_cm"]
    other_labs = [k for k in LABS if k not in W_LABS]
    t4 = t3 + [f"{k}_{s}" for k in other_labs for s in STAT]
    t5 = t4 + ["urine_ml","urine_ml_kg_hr","output_total_ml","output_n",
               "vent_flag","vent_span_h","vent_n","extubated","intubated","vasopressor","rrt",
               "admission_location","insurance","marital_status","language","religion",
               "first_service","surgical_service","first_careunit","n_icu_stays","pre_icu_hours",
               "ethnicity_grp"]
    tiers = {}
    for name, cols in [("T1",T1),("T2",t2),("T3",t3),("T4",t4),("T5",t5)]:
        seen, keep = set(), []
        for c in cols:
            if c in f.columns and c not in seen: seen.add(c); keep.append(c)
        missing = [c for c in cols if c not in f.columns]
        if missing: log(f"  {name}: MISSING {missing[:6]}{'...' if len(missing)>6 else ''}")
        tiers[name] = keep
    for a, b in [("T1","T2"),("T2","T3"),("T3","T4"),("T4","T5")]:
        assert set(tiers[a]) <= set(tiers[b]), f"{a} not nested in {b}"
    f[keys + tiers["T5"]].to_parquet(os.path.join(OUT, "tiers_W1.parquet"), index=False)
    json.dump(tiers, open(os.path.join(OUT, "W1_tier_columns.json"), "w"), indent=1)
    log("tiers_W1.parquet written")
    print("\n  tier   n features   added")
    prev = 0
    for k in ["T1","T2","T3","T4","T5"]:
        print(f"  {k}     {len(tiers[k]):>6}      {len(tiers[k])-prev:+d}"); prev = len(tiers[k])
    cov = f[tiers["T5"]].notna().mean().sort_values()
    print(f"\n  least-populated features: "
          f"{', '.join(f'{k} {100*v:.0f}%' for k,v in cov.head(8).items())}")

STEPS = {"vitals":step_vitals,"gcs":step_gcs,"body":step_body,"labs":step_labs,
         "io":step_io,"intervention":step_intervention,"context":step_context,
         "assemble":step_assemble}
if __name__ == "__main__":
    want = sys.argv[1:] or ["all"]
    for s in (list(STEPS) if want == ["all"] else want):
        log(f"=== {s} ==="); STEPS[s]()
    log("done")
