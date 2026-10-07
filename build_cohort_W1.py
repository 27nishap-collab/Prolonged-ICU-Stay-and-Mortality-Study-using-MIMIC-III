"""W1 -- Weissman et al. 2018 cohort replication.

Anchor is HOSPITAL admission (admittime), not ICU intime. See METHODS_weissman.md.

Published flow to reproduce:
    58,976 admissions -> 43,426 (73.6%) "met clinical criteria for eligibility"
                      -> 25,947 (59.7%) "had complete documentation"

That ordering tells us the decomposition: the clinical criteria are age, hospital
LOS and the life-sustaining-therapy exclusion; "complete documentation" is the
requirement for a clinical encounter note inside the 48h window. The paper does
not define which NOTEEVENTS categories count as "clinical encounter notes" -- only
that radiology notes do not -- so this script tries several definitions and reports
which reproduces 25,947.
"""
import duckdb, pandas as pd, os, sys

D = os.path.dirname(os.path.abspath(__file__)); OUT = os.path.join(D, "out")
DB = os.environ.get("MIMIC3_DB", os.path.expanduser("~/mnt/MIMIC_III_DB/mimic3.duckdb"))
os.makedirs(OUT, exist_ok=True)
con = duckdb.connect(DB, read_only=True)

# Note-category definitions to test. Radiology is excluded in all of them: the
# paper's Limitations say patients "with only radiology notes rather than clinical
# encounter notes" were excluded, so radiology is not a clinical encounter note.
NOTE_DEFS = {
    "clinician_only":   ['Nursing', 'Nursing/other', 'Physician '],
    "exclude_rad_ecg_echo": None,      # everything except Radiology, ECG, Echo
    "exclude_rad_only":     None,      # everything except Radiology
    "all_categories":       None,      # no exclusion (for reference)
}
EXCL = {"exclude_rad_ecg_echo": ['Radiology', 'ECG', 'Echo'],
        "exclude_rad_only": ['Radiology'],
        "all_categories": []}

con.execute("""
CREATE OR REPLACE TEMP TABLE base AS
SELECT a.hadm_id, a.subject_id, a.admittime, a.dischtime, a.hospital_expire_flag,
       a.admission_type, a.ethnicity, p.gender,
       -- fractional age; MIMIC-III shifts DOB back 300 y for patients >89,
       -- which the standard convention re-codes to 90
       CASE WHEN DATE_DIFF('day', p.dob, a.admittime)/365.242 > 120 THEN 90
            ELSE DATE_DIFF('day', p.dob, a.admittime)/365.242 END AS age,
       DATE_DIFF('minute', a.admittime, a.dischtime)/60.0 AS hosp_los_h,
       a.admittime + INTERVAL 48 HOUR AS t48
FROM admissions a JOIN patients p USING (subject_id)
""")

# life-sustaining therapy limitation, mimic-code code_status logic (itemid 128 / 223758)
con.execute("""
CREATE OR REPLACE TEMP TABLE lst AS
SELECT DISTINCT b.hadm_id
FROM base b JOIN chartevents c USING (hadm_id)
WHERE c.itemid IN (128, 223758)
  AND c.value IS NOT NULL AND c.value <> 'Other/Remarks'
  AND (c.error IS NULL OR c.error = 0)
  AND c.charttime >= b.admittime AND c.charttime < b.t48
  AND c.value NOT IN ('Full Code','Full code')
""")

n_all = con.execute("SELECT COUNT(*) FROM base").fetchone()[0]
steps = [("1. All hospital admissions", "TRUE", 58976),
         ("2. Adults (age >= 18)", "age >= 18", None),
         ("3. Hospital LOS >= 48 h", "age >= 18 AND hosp_los_h >= 48", None),
         ("4. No limitation of life-sustaining therapy in 48 h",
          "age >= 18 AND hosp_los_h >= 48 AND hadm_id NOT IN (SELECT hadm_id FROM lst)", 43426)]
rows = []
for name, pred, target in steps:
    n = con.execute(f"SELECT COUNT(*) FROM base WHERE {pred}").fetchone()[0]
    rows.append(dict(step=name, n=n, published=target,
                     pct_of_published=(round(100*n/target, 1) if target else None)))
elig = steps[-1][1]

print("=== FUNNEL: clinical eligibility ===")
print(pd.DataFrame(rows).to_string(index=False))

print("\n=== step 5: which note definition reproduces 25,947? ===")
res = []
for key in ["clinician_only", "exclude_rad_ecg_echo", "exclude_rad_only", "all_categories"]:
    if key == "clinician_only":
        cond = "n.category IN ('Nursing','Nursing/other','Physician ')"
    else:
        ex = EXCL[key]
        cond = ("TRUE" if not ex else
                "n.category NOT IN (" + ",".join(f"'{c}'" for c in ex) + ")")
    q = f"""SELECT COUNT(*) FROM base b WHERE {elig} AND EXISTS (
              SELECT 1 FROM noteevents n
              WHERE n.hadm_id = b.hadm_id AND {cond}
                AND n.iserror IS NULL AND n.text IS NOT NULL
                AND COALESCE(CAST(n.charttime AS TIMESTAMP),
                             CAST(n.chartdate AS TIMESTAMP)) >= b.admittime
                AND COALESCE(CAST(n.charttime AS TIMESTAMP),
                             CAST(n.chartdate AS TIMESTAMP)) <  b.t48)"""
    n = con.execute(q).fetchone()[0]
    res.append(dict(note_definition=key, n=n, published=25947,
                    pct=round(100*n/25947, 1), diff=n-25947))
r = pd.DataFrame(res).sort_values("diff", key=abs)
print(r.to_string(index=False))
r.to_csv(f"{OUT}/W1_note_definition_scan.csv", index=False)
pd.DataFrame(rows).to_csv(f"{OUT}/W1_funnel_clinical.csv", index=False)
print("\nwrote out/W1_funnel_clinical.csv and out/W1_note_definition_scan.csv")

# ---------------------------------------------------------------------------
# Final cohort. "exclude_rad_ecg_echo" is adopted: it reproduces the published
# 25,947 to within 0.9%, and the paper's Limitations state that radiology notes
# are not clinical encounter notes. ECG and Echo are machine-generated reports
# of the same character.
# ---------------------------------------------------------------------------
NOTE_COND = "n.category NOT IN ('Radiology','ECG','Echo')"
NT = ("COALESCE(CAST(n.charttime AS TIMESTAMP), CAST(n.chartdate AS TIMESTAMP))")

con.execute(f"""
CREATE OR REPLACE TEMP TABLE w1 AS
SELECT b.* FROM base b
WHERE b.age >= 18 AND b.hosp_los_h >= 48
  AND b.hadm_id NOT IN (SELECT hadm_id FROM lst)
  AND EXISTS (SELECT 1 FROM noteevents n WHERE n.hadm_id = b.hadm_id
              AND {NOTE_COND} AND n.iserror IS NULL AND n.text IS NOT NULL
              AND {NT} >= b.admittime AND {NT} < b.t48)
""")

# ICU LOS: the paper does not say whether this is the FIRST stay or the SUM of
# stays in the admission. Both are computed; the one matching Table 1 is adopted.
con.execute("""
CREATE OR REPLACE TEMP TABLE icu AS
SELECT hadm_id,
       SUM(los) AS icu_los_sum,
       MIN(intime) AS first_intime,
       ARG_MIN(los, intime) AS icu_los_first,
       ARG_MIN(first_careunit, intime) AS initial_icu_type,
       COUNT(*) AS n_icu_stays
FROM icustays GROUP BY hadm_id
""")

d = con.execute("""
SELECT w.*, i.icu_los_sum, i.icu_los_first, i.initial_icu_type, i.n_icu_stays
FROM w1 w LEFT JOIN icu i USING (hadm_id)
""").df()
d.columns = [c.lower() for c in d.columns]
print(f"\n=== W1 final: {len(d):,} admissions ({100*len(d)/25947:.1f}% of published 25,947), "
      f"{d.subject_id.nunique():,} patients ===")

for k, col in [("first ICU stay", "icu_los_first"), ("summed ICU stays", "icu_los_sum")]:
    los7 = (d[col] >= 7).fillna(False)
    death = d.hospital_expire_flag == 1
    comp = los7 | death
    print(f"  ICU LOS = {k:17} : LOS>=7d {los7.sum():>6,} ({100*los7.mean():.1f}%) "
          f"| death {death.sum():>5,} ({100*death.mean():.2f}%) "
          f"| composite {comp.sum():>6,} ({100*comp.mean():.1f}%)")
print(f"  published                    : LOS>=7d  4,564 (17.6%) | death 1,861 (7.17%) "
      f"| composite  5,504 (21.2%)")

d["outcome_death"] = (d.hospital_expire_flag == 1).astype(int)
# SUMMED ICU LOS is adopted: it gives 17.3% vs the published 17.6% against 15.3%
# for first-stay, and composite 21.0% vs the published 21.2%. This resolves an
# ambiguity the paper leaves open (it reports a single "ICU LOS" per admission).
d["outcome_icu_los_7d"] = (d.icu_los_sum >= 7).fillna(False).astype(int)
d["labels"] = ((d.outcome_death == 1) | (d.outcome_icu_los_7d == 1)).astype(int)
d.to_csv(f"{OUT}/cohort_W1.csv", index=False)
print(f"\nwrote out/cohort_W1.csv  ({len(d):,} rows, {d.labels.mean()*100:.1f}% composite)")

# ---- note counts and raw word counts inside the window (Table 1 rows) --------
# NB two different note sets are in play, and the published numbers require both:
#   ELIGIBILITY  needs a "clinical encounter note" = NOT Radiology/ECG/Echo
#                -> N = 25,713 vs published 25,947 (99.1%)
#   THE CORPUS   is every note except Radiology
#                -> notes/admission 6 (4-11) and 8 (6-12) in the outcome group,
#                   both EXACT matches to Table 1; excluding ECG/Echo gives
#                   5 (3-9) / 7 (5-10), which does not match.
# The paper does not distinguish the two sets; this pairing is what reproduces
# its numbers. See METHODS_weissman.md gap #7.
CORPUS_COND = "n.category <> 'Radiology'"
nt = con.execute(f"""
SELECT b.hadm_id, COUNT(*) AS n_notes,
       SUM(LENGTH(n.text) - LENGTH(REPLACE(n.text,' ','')) + 1) AS raw_word_count
FROM w1 b JOIN noteevents n USING (hadm_id)
WHERE {CORPUS_COND} AND n.iserror IS NULL AND n.text IS NOT NULL
  AND {NT} >= b.admittime AND {NT} < b.t48
GROUP BY 1""").df()
nt.columns = [c.lower() for c in nt.columns]
d = d.drop(columns=[c for c in ("n_notes","raw_word_count") if c in d]).merge(nt, on="hadm_id", how="left")
d.to_csv(f"{OUT}/cohort_W1.csv", index=False)
q = lambda s: f"{s.median():.0f} ({s.quantile(.25):.0f}-{s.quantile(.75):.0f})"
yy = d.labels == 1
print(f"\n  notes/admission : {q(d.n_notes)}  [published 6 (4-11)]   "
      f"outcome-yes {q(d.n_notes[yy])}  [published 8 (6-12)]")
print(f"  raw word count  : {q(d.raw_word_count)}  [published 1,023 (562-2,266)]   "
      f"outcome-yes {q(d.raw_word_count[yy])}  [published 1,410 (956-2,292)]")
