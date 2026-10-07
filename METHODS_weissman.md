# Weissman et al. 2018 — extracted specification

> Weissman GE, Hubbard RA, Ungar LH, Harhay MO, Greene CS, Himes BE, Halpern SD.
> Inclusion of Unstructured Clinical Text Improves Early Prediction of Death or
> Prolonged ICU Stay. *Critical Care Medicine*. 2018;46(7):1125–1132.
> doi:10.1097/CCM.0000000000003148. PMID 29629986. PMCID PMC6005735.

Full text recovered via NCBI BioC (Europe PMC has no OA record; PMC web is
CAPTCHA-walled; the journal page is paywalled). **The supplement HAS been obtained** (2026-09-13):
`https://cdn-links.lww.com/permalink/ccm/d/ccm_46_7_2018_03_15_weissman_ccmed-d-17-02102_sdc1.pdf`
— reached via the SDC short link `links.lww.com/CCM/D511`, found in the author's
own free copy of the published PDF at
`https://gweissman.github.io/papers/weissman_text_preds_ccm_2018.pdf`.
(`journals.lww.com` and `links.lww.com` are bot-blocked; `cdn-links.lww.com` is not.) Authors' released code:
https://github.com/gweissman/text-pred-icu (BSD-3, Zenodo 10.5281/zenodo.826438)
— note this is a **synthetic-data demonstration**, not the analysis pipeline; it
documents the text preprocessing and tuning grids but contains no cohort code.

---

## 1. Cohort — W1

Source: MIMIC-III, BIDMC 2001–2012 (version not stated in the paper).
Unit of analysis: **the hospital admission**.

**Anchor is HOSPITAL admission, not ICU admission.** Stated three ways:
- "using data from the first 48 hours of hospitalization"
- "in the first 48 hours of hospital admission (**regardless of how many of those
  hours were spent in an ICU**)"
- Table 1 caption: "All clinical variables were measured during the first 48
  hours of the hospital admission."

Exclusions, verbatim:
> "We excluded patients less than 18 years of age, admissions without any
> clinical encounter notes recorded in the first 48 hours of hospital admission
> (regardless of how many of those hours were spent in an ICU), admissions with
> total hospital LOS < 48 hours."
> "Those with a documented limitation on life-sustaining therapy within the first
> 48 hours were also excluded to improve the relevance of the predictions."

Published flow:
> "There were 58,976 hospital admissions in the dataset, of which 43,426 (73.6%)
> met clinical criteria for eligibility, and 25,947 of these (59.7%) had complete
> documentation required for the analyses."

**Target: N = 25,947.** Per-criterion attrition is in Figure 1 and not in the text.

## 2. Outcome

> "The primary outcome for the main analysis was a composite of in-hospital death
> or ICU LOS ≥ 7 days."

Published: composite **5,504 (21.2%)**; in-hospital death **1,861**; ICU LOS ≥ 7 d
**4,564 (17.6%)**.

⚠️ Table 1 prints in-hospital death as 7.8%, but 1,861/25,947 = **7.17%**. The
17.6 / 33.8 / 82.9% figures all reconcile exactly, so 7.8% looks like a typo.

## 3. Structured variables

> "All models included 18 variables commonly found in the EHR or used in other ICU
> mortality and LOS prediction models. These included lab values, vital signs,
> clinical events, demographics, and comorbidity burden."

**RESOLVED from the supplement.** Supplemental Methods, "Structured Data Sources",
verbatim:

> "Models included continuous variables for age (years), urine output (cc/kg/hr),
> and modified Elixhauser score, and the most severe value within the first 48
> hours of hospital admission of creatinine (mg/dL; highest), white blood cell
> count (K/ul; highest), platelet count (K/ul; lowest), total bilirubin (mg/dL;
> highest), arterial partial pressure of oxygen (mmHg; lowest), Glasgow coma scale
> (lowest), systolic blood pressure (mmHg; lowest), heart rate (beats per minute;
> highest), and body temperature (Celsius; highest) were included as continuous
> variables.
>
> Categorical variables included gender, admission type (emergency, elective, or
> urgent), and the presence in the nursing flowsheet within the first 48 hours of
> mechanical ventilation, ICU admission, and cardiac arrest."

### THE 18 PREDICTORS

| # | variable | type | summary |
|---|---|---|---|
| 1 | age (years) | continuous | baseline |
| 2 | urine output (cc/kg/hr) | continuous | — |
| 3 | modified Elixhauser score | continuous | — |
| 4 | creatinine (mg/dL) | continuous | **highest** |
| 5 | white blood cell count (K/uL) | continuous | **highest** |
| 6 | platelet count (K/uL) | continuous | **lowest** |
| 7 | total bilirubin (mg/dL) | continuous | **highest** |
| 8 | PaO2 (mmHg) | continuous | **lowest** |
| 9 | Glasgow coma scale | continuous | **lowest** |
| 10 | systolic blood pressure (mmHg) | continuous | **lowest** |
| 11 | heart rate (bpm) | continuous | **highest** |
| 12 | body temperature (C) | continuous | **highest** |
| 13 | **potassium** | continuous | **highest** |
| 14 | gender | categorical | — |
| 15 | admission type (emergency/elective/urgent) | categorical | — |
| 16 | mechanical ventilation | binary | any in nursing flowsheet, 48 h |
| 17 | ICU admission / transfer | binary | any in nursing flowsheet, 48 h |
| 18 | cardiac arrest | binary | any in nursing flowsheet, 48 h |

⚠️ The prose enumerates only **17**. **Potassium** appears in supplemental tables
e3 and e4 but is omitted from the paragraph; including it gives exactly 18.

**Race/ethnicity and initial ICU type are NOT predictors** — they appear in Table 1
as descriptive rows only. That resolves the 20-vs-18 discrepancy.

**Weight (kg)** appears in table e4 but is the denominator for urine output
(cc/kg/hr), not a standalone predictor.

For reference, the Table 1 rows that led to the 20-candidate count:

| domain | variable | summary over the 48 h window |
|---|---|---|
| demographics | age | baseline |
| demographics | male | — |
| demographics | race/ethnicity (5 levels) | — |
| admission | admission type (Emergency/Elective/Urgent) | — |
| admission | initial ICU type (6 levels) | — |
| comorbidity | modified Elixhauser score | — |
| events | mechanical ventilation | any in 48 h |
| events | cardiac arrest | any in 48 h |
| events | ICU transfer | any in 48 h |
| labs | creatinine | **highest** |
| labs | WBC | **highest** |
| labs | platelets | **lowest** |
| labs | total bilirubin | **highest** |
| labs | PaO2 | **lowest** |
| labs | potassium | **highest** |
| vitals | urine output (cc/kg/hr) | (no qualifier printed) |
| vitals | Glasgow coma scale | **lowest** |
| vitals | systolic blood pressure | **lowest** |
| vitals | heart rate | **highest** |
| vitals | temperature | **highest** |

### Missingness and imputation — RESOLVED

> "Imputation of missing vitals and lab values was performed by replacing them
> with the **mean of non-missing values** rather than excluding them entirely to
> preserve as many subjects as possible while not biasing the distribution of
> these variables toward extreme values."

Single unconditional mean imputation. Supplemental table e4 gives the exact
missingness and imputed mean for each variable — these are direct validation
targets for our extraction:

| variable | missing n (%) | imputed mean |
|---|---|---|
| PaO2 | 14,021 (54.0) | 103.0 |
| bilirubin | 11,444 (44.1) | 1.54 |
| urine output | 8,826 (34.0) | 0.85 |
| Glasgow coma scale | 6,964 (26.8) | 9.61 |
| weight (kg) | 5,422 (21.0) | 83.3 |
| temperature (C) | 515 (2.0) | 35.5 |
| heart rate (bpm) | 418 (1.6) | **69.1** |
| systolic blood pressure | 418 (1.6) | 86.5 |
| potassium | 115 (0.4) | 4.68 |
| white blood cells | 84 (0.3) | 14.4 |
| platelets | 57 (0.2) | 190.3 |
| creatinine | 40 (0.2) | 1.66 |

This explains the imputation artifacts in Table 1: bilirubin at 44% missing and
PaO2 at 54% missing have medians pinned to their imputed means (1.5 and 103).

## 4. Text processing

> "All clinical text notes from physicians, nurses, and other clinicians
> time-stamped during the first 48 hours of the hospitalization were aggregated
> into a single document for each admission."

Radiology notes do **not** count as "clinical encounter notes" for eligibility —
stated only obliquely in the Limitations. No explicit category list is given.

**Closed vocabulary** — 21 a-priori terms, matched whole-word. The list is in
Supplemental table e1, but their released code contains exactly 21 terms:
`comfort care, wishes, goals of care, family, wouldn't want, doesn't want,
poor prognosis, high risk, code status, family meeting, hospice, palliative,
palliation, palliative care, pall care, death, die, survival, mortal, mortality,
comfortable`

**Open vocabulary**, verbatim:
> "we also built a document-term matrix of 1-, 2-, and 3-word phrases (i.e.
> n-grams) appearing in at least 5% of all documents in the training set, and
> used penalized logistic regression with 10-fold cross-validation to identify
> the 500 most predictive phrases."
> "This variable selection was conducted in the training set only."
> "Extraction of text features yielded 5,790 unique phrases from which 500 were
> chosen for inclusion based on penalized regression."

> "the integer counts of phrases appearing in the aggregated clinical notes were
> included as continuous variables"

⚠️ **Paper says counts; their released code binarises** (`x > 0`, commented
"transform word counts per document to binary presence/absence"). We implement
counts as primary, binary as a sensitivity.

Also: "The total word count of each aggregated set of documents was also included
as a covariate."

**Preprocessing** is not described in the paper. From their `preprocess()`:
lowercase; **delete** ``!"#$%&()*+,:;<=>?@[\]^_`{|}~`` and **all digits**;
**replace with space** `- . / '`; NLTK English stopwords removed; **no stemming**;
whitespace split, then scikit-learn `CountVectorizer(min_df=0.05, ngram_range=(1,3))`.
Nothing is said about MIMIC `[** … **]` de-identification placeholders — the regex
destroys the scaffolding but leaves residual words like "name", "hospital".

## 5. Models and evaluation

Four types: logistic regression, elastic net, random forests, gradient boosting.
Each fitted twice — structured only, then structured + text.

> "we divided our dataset randomly into a training sample, consisting of 75% of
> encounters, and a testing sample, consisting of the remaining 25%"
> "Tuning parameters for each model were estimated using 5-times repeated,
> 10-fold cross-validation on the training sample"

**The split is not grouped by patient**, and the paper never mentions repeat
admissions. Their code uses `caret::createDataPartition(df$Y, p=0.75)` — stratified
on the outcome, ungrouped. Comparisons by DeLong test, Bonferroni m=28.

Tuning grids are in Supplemental table e5; the synthetic-demo code uses
RF `mtry=5:15`; GBM `n.trees={100,250,500}`, `interaction.depth={5,10,15,20}`,
`shrinkage={0.005,0.01}`, `n.minobsinnode={5,10,15}`; elastic net
`alpha=seq(0.1,1,0.1)`, `lambda={0.001…0.030}`.

### Table 2 — published AUCs (95% CI)

| model | structured, train | structured, test | + text, train | + text, test |
|---|---|---|---|---|
| logistic regression | 0.80 (0.79–0.81) | 0.79 (0.78–0.81) | 0.90 (0.89–0.90) | 0.86 (0.85–0.88) |
| elastic net | 0.80 (0.79–0.81) | 0.79 (0.78–0.81) | 0.89 (0.88–0.89) | 0.88 (0.86–0.89) |
| random forests | 1.00 (1.00–1.00) | 0.82 (0.81–0.83) | 1.00 (1.00–1.00) | 0.88 (0.87–0.89) |
| gradient boosting | 0.89 (0.89–0.90) | **0.83 (0.81–0.84)** | 0.95 (0.95–0.96) | **0.89 (0.88–0.90)** |

⚠️ The published first column-group header reads "Unstructured data only", which
contradicts the caption and the Results text — it must mean structured only.

---

## 6. What is underspecified — decisions we must make and declare

| # | Gap | Our decision |
|---|---|---|
| 1 | ~~The 18 variables~~ **RESOLVED** | Named in Supplemental Methods; race and initial ICU type are descriptive only; potassium is the 18th |
| 2 | "Limitation on life-sustaining therapy" — no field named | mimic-code `code_status` logic: `chartevents` itemid 128 / 223758, any CMO, DNR, DNI or CPR-not-indicated value in the window |
| 3 | ~~Missingness / imputation~~ **RESOLVED** | Mean of non-missing values, imputed means given per variable in table e4 |
| 4 | In-hospital death field | `HOSPITAL_EXPIRE_FLAG` (admission-level, matches "in-hospital") |
| 5 | ICU LOS first-stay or summed | Test both; "Initial ICU type" in Table 1 hints at first-stay |
| 6 | Repeat admissions / patient grouping | Follow the paper: ungrouped, outcome-stratified. **Also report a patient-grouped split as a deviation**, since ungrouped leaks |
| 7 | Note category list | Include Nursing, Nursing/other, Physician, and other clinician categories; exclude Radiology, ECG, Echo. Report sensitivity |
| 8 | Counts vs binary text features | Counts primary (paper), binary sensitivity (their code) |
| 9 | Urine output cc/kg/hr normalisation | Weight from chartevents; document the itemids |
| 10 | HR and temperature "highest" | **RESOLVED — the paper is wrong.** Table 1 prints HR 69 (60-78) and temperature 36.0 (35.6-36.4); our minima reproduce both to within one unit (69.07 / 36.06) while our maxima are 106.0 / 37.67. Both variables are MINIMA. See FEATURES_W1.md 3.1 |
| 11 | PaO2 arterial vs all specimens | **RESOLVED.** Arterial-only (labevents 50800 = 'ART') gives mean 102.1 vs their 103.0; all-specimen gives 94.1. See FEATURES_W1.md 3.2 |
| 12 | "Modified Elixhauser score" | **RESOLVED.** AHRQ SID30 weights give median 8 (0-16) vs published 8 (2-15); van Walraven 6 (0-12); flag count 3 (2-4). See FEATURES_W1.md 3.3 |
| 13 | Mechanical ventilation 17.4% | **NOT RESOLVED.** No definition in MIMIC-III reproduces it (9.2%-47.3% across eight tried). Declared. See FEATURES_W1.md 4.1 |
| 14 | Cardiac arrest 0.8% | **NOT RESOLVED.** The nursing flowsheet yields 0.3%; ICD-9 99.60/99.63 yields 0.9% but is untimestamped billing. Declared. See FEATURES_W1.md 4.2 |


---

## 7. Tuning grids (supplement + released code)

Supplemental table e5 names the tuned parameters but gives no numeric grids; the
grids come from `sample_model_building.r` in the authors' repo:

- CV: `trainControl(method="repeatedcv", number=10, repeats=5)` — matches the paper
- random forest: `mtry = 5:15`
- GBM: `n.trees = {100, 250, 500}`, `interaction.depth = {5,10,15,20}`,
  `shrinkage = {0.005, 0.01}`, `n.minobsinnode = {5,10,15}`
- elastic net: `alpha = seq(0.1, 1, 0.1)`,
  `lambda = {0.001, 0.003, 0.005, 0.010, 0.015, 0.020, 0.025, 0.030}`
- Software: R 3.3.2 (data.table, ggplot2, caret, glmnet, randomForest, gbm);
  Python 3.5.1 (NLTK, scikit-learn)

⚠️ The repo's text step keeps the **top 20 non-zero coefficients** and binarises
counts, while the paper says **500 phrases** and integer counts. The repo is a
synthetic demo; follow the paper and treat the repo as a sensitivity.

## 8. Secondary-analysis targets (supplement)

Structured-only / structured+text, GBM testing AUC:

| secondary analysis | structured | + text |
|---|---|---|
| 24-hour horizon | 0.82 (0.81–0.84) | 0.88 (0.87–0.89) |
| in-hospital death only | 0.85 (0.84–0.87) | 0.89 (0.88–0.90) |
| death only, 24-hour horizon | 0.85 (0.83–0.86) | 0.89 (0.87–0.90) |
| death or ICU LOS >= 21 d | 0.84 (0.83–0.86) | 0.89 (0.88–0.90) |
| **without Elixhauser** | **0.81 (0.80–0.83)** | **0.89 (0.88–0.90)** |

The last row is worth noting for our own project: removing comorbidity costs the
structured model 0.02 AUROC but costs the text-inclusive model nothing.

## 9. The 21 closed-vocabulary terms (Supplemental table e1, verbatim)

code status · comfort care · comfortable · death · die · doesn't want ·
family meeting · family · goals of care · high risk · hospice · mortality ·
mortal · pall care · palliation · palliative care · palliative · poor prognosis ·
survival · wishes · wouldn't want


---

## 10. Feature extraction

Built by `build_features_W1.py`. The adjudication of every ambiguous variable against table e4 and Table 1, and the four declared deviations, are in **`FEATURES_W1.md`**.
