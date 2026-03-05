# LIMUC Chapter 4 Reporting Toolkit

This folder contains Pass-1 reporting scripts that read persisted run folders under:

`Prototyping_reformat/DatasetAnalysis/LIMUC/**/results/*`

For final thesis reporting, use `11_chapter4_completion_audit.py` to generate the complete
publishable bundle in `4_reporting/out` from persisted results only.

## Scripts

1. `01_results_integrity_scanner.py`
- Scans all run folders.
- Flags missing artifacts.
- Detects smoke/subset runs (`test_rows != 1686`).
- Verifies split-hash consistency across full runs.
- Writes:
  - `results/tables/LIMUC_results_index.csv`
  - `results/tables/LIMUC_results_missing_artifacts.csv`
  - `results/tables/LIMUC_results_missing_expected_runs.csv`
  - `results/tables/LIMUC_results_summary_fullruns.csv`
  - `results/tables/chapter4_main_table_from_results.csv`

2. `02_build_chapter4_main_table.py`
- Builds chapter-ready comparison table from persisted results.
- By default includes only full runs (`n=1686`).
- Writes:
  - `results/tables/chapter4_main_comparison_table.csv`
  - `results/tables/chapter4_main_comparison_table.md`

3. `05_export_chapter4_figures.py`
- Exports/copies chapter figures from persisted artifacts.
- Writes:
  - `results/figures/class_distribution_by_split.png` (if metadata exists)
  - `results/figures/confusion_test_<run>.png` (best supervised + best generative)
  - `results/figures/pred_label_histogram_<run>.png` for generative full runs
  - `results/tables/pred_label_histogram_counts.csv`

4. `03_parser_audit.py`
- Creates parser-audit samples for a chosen generative run (defaults to `vlm_zero_shot_mayo`).
- Uses `pred_<split>_raw.csv` when available.
- Writes:
  - `results/tables/parser_audit_samples_<run>_<split>.csv`
  - `results/tables/parser_audit_summary_<run>_<split>.json`

5. `04_lora_load_proof.py`
- Produces LoRA load-proof artifacts for a selected LoRA run folder.
- Hashes adapter-like files and exports proof summary.
- Writes:
  - `results/tables/lora_adapter_file_hashes.csv`
  - `results/tables/lora_param_count.json`
  - `results/tables/lora_load_proof.txt`

6. `06_lora_ablation_table.py`
- Builds a LoRA ablation table from all run folders containing `lora` in name/model.
- Writes:
  - `results/tables/chapter4_lora_ablation_table.csv`
  - `results/tables/chapter4_lora_ablation_table.md`

7. `07_clinical_significance.py`
- Computes remission-slice metrics (0-1 vs 2-3) for full runs.
- Computes pairwise McNemar significance across full runs.
- Writes:
  - `results/tables/chapter4_remission_slice_from_results.csv`
  - `results/tables/chapter4_mcnemar_pairs_from_results.csv`

8. `08_qualitative_error_table.py`
- Builds qualitative comparison table for supervised vs generative predictions.
- Writes:
  - `results/tables/chapter4_qualitative_error_table.csv`
  - `results/tables/chapter4_qualitative_error_table_coverage.csv`

9. `11_chapter4_completion_audit.py`
- One-command final audit/report generator for Chapter 4 completion.
- Builds/rebuilds final artifacts in `4_reporting/out`:
  - `chapter4_audit_results_index.csv`
  - `chapter4_full_runs.csv`
  - `chapter4_missing_or_invalid_runs.md`
  - `chapter4_final_main_table.csv`
  - `chapter4_remission_slice_table.csv`
  - `chapter4_paired_significance.csv`
  - `chapter4_metric_ci_bootstrap.csv`
  - `generative_pred_distribution.csv/.png`
  - `figures/confusion_test_<best_*>.png`
  - `chapter4_completion_report.md`

10. `12_build_experiment_registry.py`
- Builds a unified Chapter 4 + Chapter 5 experiment registry.
- Covers:
  - Chapter 4 model runs (`*/results/*`) with metrics and split hash fields
  - Chapter 5 KB builds, wrapper runs, eval artifacts, completion audits
- Writes:
  - `out/pass1_chapter45_experiment_registry.csv`
  - `out/pass1_chapter4_full_runs_registry.csv`
  - `out/pass1_chapter45_experiment_registry_summary.json`

11. `13_data_quality_audit.py`
- Runs deep LIMUC data-quality checks for Pass 1:
  - split-file consistency and stable split-hash recomputation
  - patient-level leakage across split pairs
  - exact duplicate hash audit (`sha256`) and cross-split duplicate check
  - near-duplicate perceptual audit (`dHash` Hamming search)
  - image-quality diagnostics by split/class (brightness, contrast, sharpness, resolution outliers)
- Writes:
  - `out/pass1_data_quality_summary.json`
  - `out/pass1_image_audit_rows.csv`
  - `out/pass1_patient_leakage_pairs.csv`
  - `out/pass1_exact_duplicate_groups.csv`
  - `out/pass1_exact_duplicate_members.csv`
  - `out/pass1_near_duplicate_pairs.csv`
  - `out/pass1_image_quality_by_split_class.csv`
  - `out/pass1_image_quality_outliers.csv`
  - split consistency/distribution CSVs

12. `14_pass1_repro_data_quality_audit.py`
- One-command Pass 1 orchestrator:
  - runs script 12 then script 13
  - emits consolidated report artifacts
- Writes:
  - `out/pass1_repro_data_quality_report.json`
  - `out/pass1_repro_data_quality_report.md`

13. `15_chapter4_analytics_pass2.py`
- Runs Pass 2 Chapter 4 analytics upgrades:
  - calibration metrics (ECE/Brier/NLL) for probability-bearing full runs
  - temperature scaling on validation probabilities and before/after test comparison
  - reliability-bin exports and reliability plots
  - all-vs-all pairwise McNemar significance with exact p-values + FDR correction
  - paired bootstrap deltas for supervised vs generative pairs
  - ordinal-distance and boundary-confusion analysis
  - remission operating-point threshold sweeps (`P(Mayo 2/3)` thresholding)
- Writes:
  - `out/pass2_calibration_summary.csv`
  - `out/pass2_temperature_scaling_summary.csv`
  - `out/pass2_reliability_bins.csv`
  - `out/pass2_pairwise_mcnemar_all.csv`
  - `out/pass2_pairwise_bootstrap_deltas.csv`
  - `out/pass2_ordinal_error_profile.csv`
  - `out/pass2_ordinal_distance_distribution.csv`
  - `out/pass2_boundary_confusion_pairs.csv`
  - `out/pass2_remission_threshold_sweep.csv`
  - `out/pass2_remission_operating_points_summary.csv`
  - `out/pass2_chapter4_analytics_report.json`
  - `out/pass2_chapter4_analytics_report.md`
  - `out/figures/pass2_*.png`

14. `16_pass5_supervised_multiseed.py`
- Runs Pass 5 supervised multi-seed experiments (ResNet50) and aggregates results.
- Launches seed runs via `2_supervised_finetuning/train_resnet50_finetune.py` unless skipped.
- Produces mean/std/95% bootstrap CI summaries and per-class recall summaries.
- Writes:
  - `out/pass5_supervised_seed_runs.csv`
  - `out/pass5_supervised_metric_summary.csv`
  - `out/pass5_supervised_per_class_seed_rows.csv`
  - `out/pass5_supervised_per_class_recall_summary.csv`
  - `out/pass5_supervised_confusion_aggregate.npy`
  - `out/pass5_supervised_confusion_aggregate.png`
  - `out/figures/pass5_supervised_metric_ci.png`
  - `out/pass5_supervised_multiseed_report.json`
  - `out/pass5_supervised_multiseed_report.md`
  - `out/pass5_supervised_*_logs/seed_*.log`

15. `17_pass6_generative_multiseed.py`
- Runs Pass 6 generative multi-seed LoRA experiments plus controlled mode-2 ablations.
- Launches seed training via `3_vlm_severity/train_vlm_lora_mayo.py`.
- Runs controlled label-scoring eval via `3_vlm_severity/controlled_vlm_mayo_eval.py`.
- Aggregates:
  - LoRA mode1 (train-output) multi-seed metrics,
  - LoRA mode2 (controlled eval) multi-seed metrics,
  - per-class recall summaries with 95% bootstrap CI,
  - seed-wise mode2-minus-mode1 deltas,
  - McNemar exact tests (mode1 vs mode2),
  - zero-shot comparator rows and ablation table.
- Writes:
  - `out/pass6_generative_lora_mode1_seed_runs.csv`
  - `out/pass6_generative_lora_mode2_seed_runs.csv`
  - `out/pass6_generative_metric_summary.csv`
  - `out/pass6_generative_per_class_seed_rows.csv`
  - `out/pass6_generative_per_class_recall_summary.csv`
  - `out/pass6_generative_mode2_minus_mode1_by_seed.csv`
  - `out/pass6_generative_mcnemar_mode1_vs_mode2.csv`
  - `out/pass6_generative_zero_shot_baselines.csv`
  - `out/pass6_generative_ablation_table.csv`
  - `out/pass6_generative_confusion_aggregate_mode1.npy/.png`
  - `out/pass6_generative_confusion_aggregate_mode2.npy/.png`
  - `out/figures/pass6_generative_metric_ci.png`
  - `out/pass6_generative_multiseed_report.json`
  - `out/pass6_generative_multiseed_report.md`
  - `out/pass6_generative_*_logs/seed_*_{train,mode2}.log`

## Usage

From repo root:

```bash
python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/01_results_integrity_scanner.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/02_build_chapter4_main_table.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/05_export_chapter4_figures.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --best-supervised-run finetune_resnet50 \
  --best-generative-run vlm_zero_shot_mayo

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/03_parser_audit.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --run-name vlm_zero_shot_mayo \
  --split test

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/04_lora_load_proof.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --run-name vlm_lora_finetune_mayo

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/06_lora_ablation_table.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/07_clinical_significance.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/08_qualitative_error_table.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --supervised-run finetune_resnet50 \
  --generative-run vlm_zero_shot_mayo

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/11_chapter4_completion_audit.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out \
  --chapter-md Thesis/markdown/04_chapter_4_developing_the_proposed_approach.md

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/12_build_experiment_registry.py \
  --limuc-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --chapter5-root Prototyping_reformat/chapter5_pico_wrapper \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/13_data_quality_audit.py \
  --limuc-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out \
  --near-threshold 4 \
  --max-near-pairs 100000

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/14_pass1_repro_data_quality_audit.py \
  --limuc-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --chapter5-root Prototyping_reformat/chapter5_pico_wrapper \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out \
  --near-threshold 4 \
  --max-near-pairs 100000

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/15_chapter4_analytics_pass2.py \
  --dataset-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out \
  --bootstrap-iters 400 \
  --calibration-bins 15 \
  --remission-threshold-grid 501

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/16_pass5_supervised_multiseed.py \
  --limuc-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --seeds 11,23,42 \
  --amp \
  --tag pass5_supervised_latest \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out

python Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/17_pass6_generative_multiseed.py \
  --limuc-root Prototyping_reformat/DatasetAnalysis/LIMUC \
  --new-seeds 11,23 \
  --existing-runs vlm_lora_finetune_mayo_balanced_full_20260303 \
  --epochs 1 \
  --force-retrain \
  --tag pass6_generative_latest \
  --out-dir Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out \
  --eval-run-prefix vlm_lora_pass6_mode2_seed
```

## Notes

- These scripts treat `results/` as the source of truth.
- Historical `out/` folders are ignored in this pipeline.
- Smoke runs remain indexed but are excluded from the default chapter comparison table.
