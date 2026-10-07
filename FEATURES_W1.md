# W1 structured feature set — extraction, adjudication, deviations

Built by `build_features_W1.py`. Inputs: `out/cohort_W1.csv` (25,692 admissions).
Outputs: `out/features_W1.parquet` (all variants), `out/design_W1.parquet`
(the 18-predictor matrix, mean-imputed), `out/W1_feature_validation*.csv`.

Window for every variable: `[admittime, admittime + 48h)` — hospital admission,
not ICU intime.

---

## 1. How the spec was pinned down

The paper names "18 variables" but never lists them. The Supplemental Methods
do, and supplemental tables e3/e4 plus Table 1 give per-variable missingness,
means, medians and IQRs. That turns feature extraction from guesswork into a
checkable target: every variable below is reported against a published number,
and any mismatch is either fixed or declared.

---

## 2. Table e4 / Table 1 comparison — as built

Their median/IQR is only comparable for variables with low missingness; where
imputation is heavy (marked `*`) their median is pinned to the imputed mean.

| variable | ours missing % | theirs | ours mean | theirs | ours median (IQR) | theirs |
|---|---|---|---|---|---|---|
| heart rate (min) | 0.5 | 1.6 | 69.07 | 69.1 | 68 (60–78) | 69 (60–78) |
| temperature (min) | 0.8 | 2.0 | 36.00 | 35.5 | 36.06 (35.61–36.44) | 36.0 (35.6–36.4) |
| systolic BP (min) | 0.5 | 1.6 | 91.23 | 86.5 | 90 (81–101) | 89 (79–101) |
| platelets (min) | 0.5 | 0.2 | 191.9 | 190.3 | 177 (126–237) | 176 (125–235) |
| potassium (max) | 0.3 | 0.4 | 4.65 | 4.68 | 4.40 (4.10–5.00) | 4.5 (4.2–4.9) |
| creatinine (max) | 0.3 | 0.2 | 1.57 | 1.66 | 1.00 (0.80–1.50) | 1.10 (0.80–1.60) |
| WBC (max) | 0.6 | 0.3 | 13.47 | 14.4 | 12.10 (9.00–16.10) | 13.0 (9.70–17.10) |
| GCS (min) | 0.6 | 26.8 | 9.68 | 9.61 | 12 (5–15) | 9 (7–14)* |
| PaO2 (min, arterial) | 45.4 | 54.0 | 102.09 | 103.0 | 88 (70–118) | 103 (93–103)* |
| weight (kg) | 15.3 | 21.0 | 81.61 | 83.3 | 78.5 (66.0–92.6) | — |
| urine output (cc/kg/hr) | 18.0 | 34.0 | 0.93 | 0.85 | 0.74 (0.44–1.15) | 0.85 (0.57–0.90)* |
| total bilirubin (max) | 57.3 | 44.1 | 1.75 | 1.54 | 0.70 (0.40–1.40) | 1.5 (0.6–1.5)* |

| event / index | ours | published |
|---|---|---|
| mechanical ventilation (flowsheet) | 12,143 (47.3%) | 4,512 (17.4%) |
| mechanical ventilation (ICD-9 96.7x) | 7,048 (27.4%) | 4,512 (17.4%) |
| cardiac arrest (flowsheet) | 78 (0.3%) | 206 (0.8%) |
| cardiac arrest (ICD-9 99.60/99.63) | 231 (0.9%) | 206 (0.8%) |
| ICU transfer within 48h | 25,582 (99.6%) | 25,614 (98.7%) |
| Elixhauser, AHRQ SID30 | 8 (0–16) | 8 (2–15) |
| Elixhauser, van Walraven | 6 (0–12) | 8 (2–15) |
| Elixhauser, Quan flag count | 3 (2–4) | 8 (2–15) |

---

## 3. Three things the validation pass resolved

### 3.1 Heart rate and temperature are MINIMA, not maxima

The Supplemental Methods say "heart rate (beats per minute; **highest**)" and
"body temperature (Celsius; **highest**)". Both are wrong, and the paper's own
numbers prove it:

| | our max | our min | Weissman |
|---|---|---|---|
| heart rate, mean | 105.99 | **69.07** | 69.1 (table e4) |
| heart rate, median (IQR) | 104 (91–118) | **68 (60–78)** | 69 (60–78) (Table 1) |
| temperature, mean | 37.67 | **36.00** | 35.5 (table e4) |
| temperature, median (IQR) | 37.60 (37.11–38.11) | **36.06 (35.61–36.44)** | 36.0 (35.6–36.4) (Table 1) |

A 48-hour *maximum* heart rate with median 69 and IQR 60–78 is not physiologically
possible in an ICU cohort, and a maximum temperature of 36.0 °C is below normal.
The minima match to within one unit on every statistic. **Implemented as minima**;
the maxima are retained as `hr_max_alt` / `temp_max_alt`.

This is a label error in two of eighteen predictors, and it propagates: anyone
reading the paper and building "highest HR, highest temperature" is not
reproducing what was actually fitted.

### 3.2 PaO2 must be restricted to arterial specimens

`labevents` itemid 50821 (PO2, blood gas) does not distinguish arterial from
venous. Taking a *minimum* over all specimens drags the value down, because
venous PO2 is much lower.

| | missing % | mean | median (IQR) |
|---|---|---|---|
| all specimens | 37.5 | 94.09 | 82 (61–113) |
| arterial only | **45.4** | **102.09** | 88 (70–118) |
| Weissman | 54.0 | 103.0 | — |

Arterial is identified by `labevents` itemid 50800 (SPECIMEN TYPE) = `'ART'` at
the same `(subject_id, charttime)`. The mean lands on theirs almost exactly.
**Arterial-only adopted.**

### 3.3 "Modified Elixhauser score" is AHRQ SID30, not a count

Published median 8 (2–15) rules out a count of conditions.

| weighting | ours |
|---|---|
| Quan flag count (0–30) | 3 (2–4) |
| van Walraven | 6 (0–12) |
| **AHRQ SID30** | **8 (0–16)** |

SID30 reproduces the median exactly and the IQR to within two points.
**SID30 adopted**, from `mimic-code` `elixhauser_quan.sql` flags + the SID30
weights in `elixhauser_score_quan.sql`.

---

## 4. Declared deviations — what did NOT reconcile

### 4.1 Mechanical ventilation: 47.3% vs a published 17.4%

No operationalisation available in MIMIC-III reproduces 17.4%. Everything tried:

| definition | ours |
|---|---|
| mimic-code ventilator settings (broad) | 47.3% |
| ventilator mode/type items only | 47.0% |
| CareVue 720 "Ventilator Mode" alone | 36.2% |
| CPT 94002–94657 (vent management) | 39.8% |
| ICD-9 96.7x (mechanical ventilation) | 27.4% |
| ICD-9 96.71 alone (<96 h) | 14.9% |
| CareVue 640 = Intubated/Extubated | 15.1% |
| MetaVision procedure 225792 (invasive vent) | 9.2% |
| **published** | **17.4%** |

The two definitions that bracket 17.4% — ICD-9 96.71 alone and CareVue
intubation events alone — are each internally incoherent (96.71 excludes
long-run ventilation; 640 exists only in CareVue, covering roughly half the
cohort). The standard mimic-code flowsheet definition is used, and the
discrepancy is declared rather than tuned away.

### 4.2 Cardiac arrest: the flowsheet cannot produce 0.8%

The supplement says "presence in the nursing flowsheet". The flowsheet is far
too sparse for that to be true: CareVue `640` = `'Cardiac Arrest'`/`'Defibrillation'`
has 370 rows in all of MIMIC-III, and MetaVision `225466` has 238. Restricted to
the cohort and the 48h window that is 78 admissions (0.3%).

ICD-9 procedure 99.60/99.63 (CPR) gives 231 (0.9%) against their 206 (0.8%) —
the only close match, but it is a billing code with no timestamp, so it cannot
be restricted to the first 48 hours. The flowsheet flag is primary
(`cardiac_arrest`); the ICD version is kept as `cardiac_arrest_icd_alt`.

### 4.3 Bilirubin and urine-output coverage

Bilirubin: we are missing 57.3% where they are missing 44.1%. Adding chartevents
`4948` / `225690` and joining labs by `subject_id` + time instead of `hadm_id`
both fail to close the gap (56.9% and 57.2%). Their extraction sees roughly 13
percentage points more patients than any route we can construct.

Urine output: we have *better* coverage than they do (18.0% vs 34.0% missing),
most likely because our weight extraction is more generous (15.3% vs 21.0%
missing) and urine output is missing whenever weight is.

Neither gap has a plausible fix that does not amount to degrading our extraction
until it matches theirs, so both are declared.

---

## 5. Known leakage in the replicated specification

The Elixhauser score is built from `diagnoses_icd`, which is assigned at
**discharge**. It is not observable at hour 48. This is the availability leak
measured in `Analysis/availability_leak` and catalogued in `METHODS_TRAPS.md`.

We replicate it because Weissman used it, and their own supplement quantifies
its contribution: dropping the comorbidity index costs the structured GBM
0.83 → 0.81 and costs the text-inclusive model nothing (0.89 → 0.89). For our
own project the observable structured baseline is the one in
`Analysis/text_vs_structured/BASELINE.md`, which contains no comorbidity index.

---

## 6. The design matrix

`out/design_W1.parquet` — 25,692 × 38. The 18 predictors:

**Continuous (13):** age, urine output (cc/kg/hr), Elixhauser SID30, creatinine
(max), WBC (max), platelets (min), total bilirubin (max), PaO2 (min, arterial),
GCS (min), systolic BP (min), heart rate (min), temperature (min), potassium (max).

**Categorical / binary (5):** male; admission type as three indicators
(emergency / elective / urgent); mechanical ventilation, ICU admission, cardiac
arrest.

Each continuous variable also carries a `*_imputed` indicator. Weissman did not
include missingness indicators — they are here only so the effect of imputation
can be audited, and they are excluded from the primary model.

Imputation follows the supplement: "replacing them with the mean of non-missing
values", computed on the full cohort. Imputed means are written to
`out/W1_imputation_means.csv`. Computing the mean on the full cohort leaks
test-set information; the modelling script also fits a training-fold-mean
version as a sensitivity, and the difference will be reported.
