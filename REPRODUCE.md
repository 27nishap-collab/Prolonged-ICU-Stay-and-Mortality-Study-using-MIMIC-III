# Reproducibility map — "Do Clinical Notes Improve Early Prediction of Mortality or Prolonged ICU Stay?"

Paper draft v4.4 and SUPPLEMENT_v4.1 (October 2026). Every number in both documents comes from a
script in this folder. This file says which one, how it was run, and where its result lives.

## Setup

| what | where | override |
|---|---|---|
| MIMIC-III v1.4 as a DuckDB database (built by `setup_mimic_duckdb.py`) | `~/mnt/MIMIC_III_DB/mimic3.duckdb` (Cowork VM path; on the Mac `/Users/Shared/MIMIC_III/MIMIC_III_DB/mimic3.duckdb`) | `MIMIC3_DB=/path/to/mimic3.duckdb` |
| data and results folder | `out/` next to the scripts | `W1_DATA=/path/to/out` |
| figures | `../figs/` | `FIG_OUT=/path` |
| encoder weights | Hugging Face cache (`emilyalsentzer/Bio_ClinicalBERT`, `microsoft/BiomedNLP-BiomedBERT-base-uncased-abstract-fulltext`); tokenizers in `hf/` | `HF_HUB_OFFLINE=1` once cached |

- The database is built by `setup_mimic_duckdb.py` (step 0 below) from the 26 MIMIC-III v1.4 CSV files.
  The existing database matches it: notes stored as text, 2,083,180 note rows (the full v1.4 count).
  It also holds one extra table, `cohort`, which the loader did not create and no script here uses.
- `out/` is git-ignored: it holds patient-level data (PhysioNet credentialed access only).
- Defaults: `W1_SPEC=observable` (the paper's predictor set) and `W1_CORPUS=v2` (notes without
  discharge summaries). `W1_SPEC=replication` gives Weissman's original predictor list.
- Python 3.11 with numpy, pandas, scipy, scikit-learn, duckdb, pyarrow, matplotlib; the encoder
  scripts also need torch and transformers (Apple-silicon GPU via MPS for fine-tuning).
- Scripts that were first run in the cloud workspace store their archived results in `out/observable/`
  (observable specification) or `out/w1_<name>/` (original list). A rerun writes to
  `out/w1_<name>_observable/`, so the archived files are not overwritten.

## 1. Data pipeline (run once, in this order)

| step | script | command | output |
|---|---|---|---|
| 0. MIMIC-III CSVs → DuckDB | `setup_mimic_duckdb.py` | `MIMIC3_CSV=/path/to/mimic-iii-clinical-database-1.4 MIMIC3_DB=/path/to/mimic3.duckdb python3 setup_mimic_duckdb.py` (defaults: `/Users/Shared/MIMIC_III/...`) | `mimic3.duckdb` (26 tables) |
| cohort, funnel | `build_cohort_W1.py` | `python3 build_cohort_W1.py` | `out/cohort_W1.csv`, `out/W1_funnel_clinical.csv` |
| Weissman's 18 predictors | `build_features_W1.py` | `python3 build_features_W1.py` | `out/features_W1.parquet` |
| tiers T1–T5 | `build_tiers_W1.py` | `python3 build_tiers_W1.py all` | `out/tiers_W1.parquet`, `out/W1_tier_columns.json` |
| observable specification (S8) | `build_observable_W1.py` | `python3 build_observable_W1.py` | `out/tiers_W1_observable.parquet`, `out/W1_observable_manifest.json` |
| notes, 21 terms, word count, 500 phrases | `build_text_W1.py` | `python3 build_text_W1.py all --v2` | `out/model_tabular_W1_v2.parquet`, `out/text_open_W1_v2.npz` (25,320 rows: admissions with an in-window note) |
| phrases padded to all 25,692 admissions (zero rows for the 372 without an in-window note) | `align_text_W1.py` (first done as a one-off step on 2026-09-17; the script reproduces that file exactly) | `python3 align_text_W1.py` | `out/text_open_W1_v2_aligned.npz` |
| 4,687-phrase candidate pool | `reselect_split_W1.py` | `python3.11 reselect_split_W1.py dtm` | `out/text_full_W1_v2.npz` |
| LSA (single division) | `build_lsa_W1.py` | `python3 build_lsa_W1.py` | `out/lsa_wide_W1.npy`, `out/lsa_pool_W1.npy` |
| wordpiece windows | `tokenize_W1.py` | `W1_MAXWIN=all W1_TOKMODELS=bio_clinicalbert python3 tokenize_W1.py` (repeat until done), then `... merge`; same with `pubmedbert` | `out/tokall_<model>_W1.npz` (and `tok16_*` for the first-windows version) |
| frozen encoders, first 4 windows | `encode_bert_local.py` (Mac; `encode_bert_W1.py` is its cloud original) | `HF_HUB_OFFLINE=1 python3 encode_bert_local.py` | `out/bert_<model>_w0-4/` |
| max-pooled encoders | `encode_bert_pool_local.py` | `HF_HUB_OFFLINE=1 python3 encode_bert_pool_local.py` | `out/bertpool_<model>_w0-4/` |
| validation against Weissman Table 1 | `validate_table1.py` | `python3 validate_table1.py` | `out/W1_table1_validation.csv` |

Shared definitions: `lr_tuned.py` (the cross-validated logistic regression used by every newer script).

## 2. Main text

| item | script | command | result file |
|---|---|---|---|
| Figure 1 (cohort flow) | `make_cohort_figure.py` | `python3 make_cohort_figure.py` | `../figs/figure_cohort.*` |
| Table 2 (tiers) | `build_tiers_W1.py`, `build_observable_W1.py` | §1 | `out/W1_tier_columns_observable.json` |
| Table 3, Figure 2 (ten repetitions) | `run_repeated_reselect_W1.py`; figure `make_figures_v43.py` | `python3.11 run_repeated_reselect_W1.py`, then `... summary`; `python3 make_figures_v43.py` | `out/observable_v2/W1_repeated_reselect.csv` (arm `rechosen`); `../figs/figure2_saturation.*` |
| Table 4, bag-of-words / LSA / frozen encoders | `reselect_split_W1.py` (half-A phrases), `splitsample_more_W1.py` | `python3.11 reselect_split_W1.py select`; `splitsample_more_W1.py check / dtm_wide / lsa / encode / run / summary` (see its docstring) | `out/w1_ssmore_v2_observable/W1_ssmore_table.csv` |
| Table 4, fine-tuned encoder ([CLS] and mean) | `finetune_full_W1.py` | `caffeinate -dis env HF_HUB_OFFLINE=1 python3.11 finetune_full_W1.py run`, then `... summary` | `out/w1_ftfull_v2_observable/W1_ftfull.csv` |
| Table 5 (outcome components) | `run_outcome_split_W1.py` | `python3 run_outcome_split_W1.py` | archived `out/observable/W1_outcome_split.csv` |
| Prespecified clinical terms | `fit_attrib_W1.py` | `W1_SPEC=observable python3 fit_attrib_W1.py` | archived `out/observable/W1_attrib.csv` |
| Disposition vocabulary (Discussion) | `run_ablation_W1.py`, `run_reselect_W1.py`, `fit_reselect.py` | `python3 run_ablation_W1.py`; `python3 run_reselect_W1.py`; `python3 fit_reselect.py` | archived `out/observable/W1_ablation.csv`, `W1_reselect.csv`, `W1_ablation_with_reselect.csv` |
| 23.1% of test admissions share a patient with training (Limitations) | `run_sensitivity_W1.py` | `python3 run_sensitivity_W1.py` | printed; `out/w1_models/W1_sens_grouped_split.csv` |
| Replication of Weissman's four models | `run_models_W1.py` | `python3 run_models_W1.py` | `out/w1_models/W1_results.csv` |

## 3. Supplement

| section | script | result file |
|---|---|---|
| S1 Table S1, tuned row and chosen C | `run_repeated_reselect_W1.py` | `out/observable_v2/W1_repeated_reselect.csv`, `..._C.csv` |
| S1 Table S1, default row | `run_lr_default_reselect_W1.py` (`python3.11 run_lr_default_reselect_W1.py`, then `summary`) | `out/observable_v2/W1_repeated_reselect_lrdefault.csv` |
| S1 halving check under the tuned penalty | `check_lr_tuning_W1.py` | `out/w1_lrtune_v2_observable/W1_lrtune.csv` |
| S2 Table S2 (half-A vs full training set) | `reselect_split_W1.py` (`select`, `W1_SELECT_ON=train ... select`, `fit`) | `out/w1_ftsplit_v2_observable/W1_reselect_split.csv` |
| S3 Table S3.1 (contamination) | `finetune_bert_W1.py` (shared-patient fine-tuning), `diag_contamination_W1.py` | `out/w1_finetune_v2_observable/W1_contamination.csv` |
| S3 Table S3.2 (halving check) | `check_halfsize_W1.py` | `out/w1_splitsample_v2_observable/W1_halfsize.csv` |
| S3 split-sample fine-tuning, validation AUROC | `finetune_split_W1.py` (earlier version), `finetune_full_W1.py` | `out/w1_ftsplit_v2_observable/W1_ftsplit_val.csv`, `out/w1_ftfull_v2_observable/W1_ftfull_val.csv` |
| S4 readouts, both versions | `finetune_full_W1.py`, `finetune_split_W1.py` | `W1_ftfull.csv`, `W1_ftfull_textonly.csv`, `W1_ftsplit.csv`, `W1_ftsplit_textonly.csv` |
| S5 Table S5, Figure S1 (coverage) | `coverage_split_W1.py`, `finetune_full_W1.py`; figure `make_supp_figs.py` | `out/w1_ftsplit_v2_observable/W1_coverage_split.csv`, `out/w1_ftfull_v2_observable/W1_ftfull_coverage.csv`; `../figs/figureS1_coverage.*` |
| S6 Table S6 (GBM tuning) | `tune_gbm_split_W1.py` (`W1_SELECT=cv3 W1_DRAWS=0,1,2,3,4`) | `out/w1_ftsplit_v2_observable/W1_gbmtune_cv3.csv` |
| S7 Tables S7.1–S7.2 (representations, pooling) | `fit_bert_W1.py`, `fit_maxpool_W1.py`, `fit_lsa_W1.py` | archived `out/observable/W1_bert.csv`, `W1_maxpool.csv`, `W1_lsa.csv` |
| S8 observability audit | `build_observable_W1.py` (see also `AUDIT_48h.md`, `SPEC_DECISION.md`) | `out/W1_observable_manifest.json` |
| S9 Figure S2, Tables S9.1–S9.4 | `run_repeated_reselect_W1.py` (both specifications); figure `make_supp_figs.py` | `out/observable_v2/W1_repeated_reselect.csv`, `out/w1_repeated_v2/W1_repeated_reselect.csv`; `../figs/figureS2_partitions.*` |
| S10 replication table | `run_models_W1.py`, `validate_table1.py` | `out/w1_models/W1_results.csv`, `out/W1_table1_validation.csv` |
| S10 original predictor list, Figure S3 | `W1_SPEC=replication W1_CORPUS=v2 python3.11 run_repeated_reselect_W1.py`; figure `make_figures_v43.py` | `out/w1_repeated_v2/W1_repeated_reselect.csv`; `../figs/figureS3_weissman_list.*` |

## Scripts here that did not produce a reported number

`run_repeated_splits_W1.py` (ten-repetition curve with one reused phrase selection; superseded by
`run_repeated_reselect_W1.py`, whose `orig` arm supplies the reused-selection numbers quoted in S2;
its default-penalty rows are the reproduction check inside `run_lr_default_reselect_W1.py`),
`run_saturation_W1.py` (single-division curve, superseded), `run_splitnoise_W1.py` (earlier
split-noise check), `calibrate_finetune_local.py` (timing only), `encode_bert_W1.py` (cloud original
of `encode_bert_local.py`). `make_figure1.py` and `make_figure1_combined.py` draw figures that are not in
the paper and are not in git.
