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
```

## Notes

- These scripts treat `results/` as the source of truth.
- Historical `out/` folders are ignored in this pipeline.
- Smoke runs remain indexed but are excluded from the default chapter comparison table.
