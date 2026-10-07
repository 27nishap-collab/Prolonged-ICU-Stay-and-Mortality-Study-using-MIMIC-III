# Is every structured predictor observable at hour 48?

All 222 T5 variables audited against the prediction time. The design's whole
claim is about what a properly developed structured baseline can do *at hour
48*; any feature that is not knowable then makes the baseline artificially
strong, which shrinks the text increment, which inflates the collapse. That is
the direction that flatters our own conclusion, so it is the direction to be
strict about.

## Method

Each variable traced to its source block in `build_features_W1.py` /
`build_tiers_W1.py`, and each block checked for a `[admittime, admittime+48h)`
filter. Where a block had no filter, the exposure was measured directly against
the database rather than assumed.

*A caveat on the mechanical check:* grepping a function body for `admittime` or
`t48` gives a **false pass** on `step_context`, which mentions `c.admittime`
only inside an arithmetic expression and has no `WHERE` clause on `icustays` or
`services` at all. The blocks below were read, not grepped.

## Result

| group | n | source | windowed? | verdict |
|---|---|---|---|---|
| labs | 152 | `labevents` | yes | clean |
| vitals | 28 | `chartevents` | yes | clean |
| GCS | 7 | `chartevents` | yes | clean |
| interventions | 7 | `chartevents`, `procedureevents_mv`, `inputevents_*` | yes | clean |
| in/out | 3 | `outputevents` | yes | clean |
| body | 2 | `chartevents` | yes | clean |
| T1 physiology + demographics | 19 | `chartevents`, `labevents`, `admissions` | yes | clean |
| **T1 `elixhauser`** | **1** | **`diagnoses_icd`** | **no** | **LEAK — 100%** |
| context, admission attributes | 6 | `admissions` | n/a (known at admission) | clean |
| **context `n_icu_stays`** | **1** | **`icustays`, ungrouped count** | **no** | **LEAK — 6.3%** |
| **context `first_careunit`, `pre_icu_hours`** | **2** | **`icustays`, ARG_MIN over all stays** | **no** | **LEAK — 0.2%** |
| **context `first_service`, `surgical_service`** | **2** | **`services`, ARG_MIN over all transfers** | **no** | **LEAK — 0.05%** |

No T5 variable derives from `procedures_icd`, `diagnoses_icd` or `drgcodes`
other than `elixhauser`. `cardiac_arrest` and `mech_vent` use the time-windowed
chartevents definitions; the ICD-derived alternatives exist in the extractor as
`cardiac_arrest_icd_alt` and `vent_icd_alt` and are deliberately unused.

## Magnitudes

| variable | exposure | detail |
|---|---|---|
| `elixhauser` | **100%** | ICD-9 assigned at discharge; not knowable at any point during the stay |
| `n_icu_stays` | **6.3%** (1,613 / 25,692) | counts ICU stays across the whole admission; mean 1.07 vs 1.00 for the 48h-observable count |
| `first_careunit`, `pre_icu_hours` | **0.2%** (47) | first ICU stay begins after hour 48 |
| `first_service`, `surgical_service` | **0.05%** (14) | earliest service transfer is after hour 48 |

## Reading

**Elixhauser is the only one that matters quantitatively.** It affects every
admission; the others affect 0.05–6.3%, and `n_icu_stays` is nearly constant
(mean 1.07) so its predictive contribution is small to begin with. These are
specification defects rather than results defects, and the expectation is that
fixing them moves nothing.

**They should still be fixed, for three reasons.** The fix is trivial — add the
window to three SQL blocks. All four leaks sit in T5 and none in T1–T4, so they
act only on the *rich* baseline, which is the same direction as Elixhauser:
inflating the collapse. And "we audited all 222 and here is the table" is a much
better answer to a reviewer than "we checked the one we knew about."

## Proposed specification

1. Window `n_icu_stays`, `first_careunit`, `pre_icu_hours`, `first_service` and
   `surgical_service` to stays and transfers beginning before hour 48.
2. Replace discharge-coded `elixhauser` with `has_prior_admission` (binary) plus
   `prior_elixhauser` (score from the patient's *earlier* admissions, NaN for
   the 78.2% with none). See the coverage caveat below.
3. Keep the **replication arm** (F1 / Table 2) on the replication specificationification,
   discharge-coded Elixhauser included, explicitly labelled as reproducing
   Weissman. The rule applies from F3 onward.

**Coverage caveat on (2).** Only 21.8% of W1 admissions (5,610 / 25,692) have an
earlier admission in MIMIC-III, and MIMIC-III sees only BIDMC 2001–2012, so that
is a floor on true prior history. `prior_elixhauser` therefore does **not**
restore the comorbidity construct — for most of the cohort it is absent, and
what `has_prior_admission` mostly encodes is "this patient has been here
before." The observable baseline will be genuinely thinner on comorbidity than a
deployed model would be. That biases *against* the project's conclusion — thinner
baseline, larger text increment, smaller collapse — which is the safe direction,
and should be declared rather than left implicit.

## Open

`insurance`, `marital_status`, `language` and `religion` are admission-level
attributes in MIMIC-III and are treated here as known at admission. In some
hospital systems these fields are updated during or at the end of the stay. Not
checkable from the schema; flagged rather than resolved.
