# Do Clinical Notes Improve Early Prediction of Mortality or Prolonged ICU Stay?

Code for the study *"Do Clinical Notes Improve Early Prediction of Mortality or Prolonged ICU Stay?
Mostly Against a Thin Baseline"* (manuscript in preparation).

The study starts from Weissman et al. (2018), who found that adding clinical notes from the first
48 hours to 18 structured variables improved the prediction of in-hospital death or an ICU stay of
at least 7 days in MIMIC-III. We rebuilt that cohort and model. In our version the smallest baseline
(T1) has 21 predictors: admission type is coded as three indicators, and the comorbidity index, which
the original derived from diagnosis codes assigned at discharge, is replaced by two variables known
at 48 hours. We then enriched the structured baseline in five nested tiers (21 to 223 predictors) and
measured how much the notes still added at each tier, over ten random training/test divisions, two
learners (gradient boosting and logistic regression) and five text representations (bag-of-words,
LSA, two frozen transformer encoders, and a fine-tuned Bio_ClinicalBERT).

> Weissman GE, Hubbard RA, Ungar LH, et al. Inclusion of unstructured clinical text improves early
> prediction of death or prolonged ICU stay. *Crit Care Med* 2018;46(7):1125–1132.

## Data

MIMIC-III v1.4 requires credentialed access through PhysioNet. No patient data are in this
repository; everything under `out/` is generated locally and is git-ignored.

## Getting started

**Start with [`REPRODUCE.md`](REPRODUCE.md).** It lists the setup, the data pipeline in run order,
and, for every table and figure in the paper and supplement, the script, the command and the result
file.

## What is here

| Kind | Files |
|---|---|
| Database setup | `setup_mimic_duckdb.py` |
| Cohort, predictors, text | `build_*.py`, `tokenize_W1.py`, `encode_bert_*.py`, `align_text_W1.py`, `reselect_split_W1.py` |
| Main analyses | `run_repeated_reselect_W1.py` (ten-repetition curve), `splitsample_more_W1.py`, `finetune_full_W1.py`, `finetune_split_W1.py` |
| Checks and secondary analyses | `check_*.py`, `diag_*.py`, `tune_*.py`, `coverage_split_W1.py`, `fit_*.py`, `run_*.py` |
| Shared definition | `lr_tuned.py` (cross-validated logistic regression) |
| Figures | `make_figures_v43.py`, `make_cohort_figure.py`, `make_supp_figs.py` |
| Validation against the original paper | `validate_table1.py`, `run_models_W1.py` |

## Design documents

| File | Contents |
|---|---|
| `METHODS_weissman.md` | The original study's specification, quoted, and the places it leaves open |
| `SPEC_DECISION.md` | The rule for the observable predictor set, fixed before any result |
| `AUDIT_48h.md` | Check that every predictor is known at 48 hours |
| `FEATURES_W1.md` | How the structured predictors were built; deviations from the original |
| `TEXT_W1.md` | Note selection, cleaning and text features; deviations from the original |
| `MODELS_W1.md` | Replication of the original four models |

## Fidelity to the original cohort

The cohort reproduces the published flow (25,692 admissions vs. 25,947; composite outcome 21.0% vs.
21.2%), and the median notes per admission match exactly (6, IQR 4–11). Three points the original
paper leaves open were resolved against its own numbers:

1. **ICU length of stay** is summed over all ICU stays in the admission.
2. **Two note sets:** eligibility requires a clinical encounter note (excluding radiology, ECG and
   echo), while the text corpus excludes radiology only.
3. **Age** is fractional, with ages above 89 recoded to 90.

Details are in `METHODS_weissman.md` and `out/W1_table1_validation.csv` (from `validate_table1.py`).
