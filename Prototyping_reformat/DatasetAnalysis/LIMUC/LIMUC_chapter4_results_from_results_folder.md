# LIMUC Chapter 4 Results Snapshot (From results/ Folders)

This file is generated from persisted artifacts under `Prototyping_reformat/DatasetAnalysis/LIMUC/**/results/*`.

## 1) Run Coverage

- Total run folders detected: **7**
- Full test-set runs (`n=1686`) detected: **5**
- Smoke/subset runs detected: **2**
- Unique split hash values: `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`

### 1.1 Artifact Presence by Run Folder

| run_folder | test_rows | metrics_test | pred_test | pred_val | run_meta | confusion_test.png |
|---|---:|---:|---:|---:|---:|---:|
| `resnet50_frozen_logreg` | 1686 | yes | yes | yes | yes | yes |
| `vit_frozen_logreg` | 1686 | yes | yes | yes | yes | yes |
| `finetune_resnet50` | 1686 | yes | yes | yes | yes | yes |
| `finetune_vit_or_swin` | 1686 | yes | yes | yes | yes | yes |
| `vlm_lora_finetune_mayo_smoke_20260222` | 16 | yes | yes | yes | yes | yes |
| `vlm_zero_shot_mayo` | 1686 | yes | yes | yes | yes | yes |
| `vlm_zero_shot_mayo_smoke_20260222` | 16 | yes | yes | yes | yes | yes |

## 2) Main Test-Set Metrics (Full Runs Only, n=1686)

| run_folder | model | accuracy | macro_f1 | balanced_acc | qwk | mae | rmse | parse_rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_frozen_logreg` | `resnet50_frozen_logreg` | 0.6198 | 0.5346 | 0.5420 | 0.6834 | 0.4324 | 0.7367 | NA |
| `vit_frozen_logreg` | `vit_frozen_logreg` | 0.6910 | 0.6192 | 0.6419 | 0.7620 | 0.3458 | 0.6503 | NA |
| `finetune_resnet50` | `resnet50_finetune` | 0.7527 | 0.6800 | 0.6858 | 0.8428 | 0.2533 | 0.5149 | NA |
| `finetune_vit_or_swin` | `vit_or_swin_finetune` | 0.7129 | 0.6675 | 0.6649 | 0.7642 | 0.3126 | 0.6137 | NA |
| `vlm_zero_shot_mayo` | `vlm_zero_shot` | 0.5486 | 0.1771 | 0.2500 | 0.0000 | 0.6987 | 1.1557 | 1.0000 |

## 3) Smoke/Subset Runs (Not Valid for Final Chapter-4 Main Comparison)

| run_folder | model | test_rows | accuracy | note |
|---|---|---:|---:|---|
| `vlm_lora_finetune_mayo_smoke_20260222` | `vlm_lora_finetune` | 16 | 0.7500 | subset/smoke run |
| `vlm_zero_shot_mayo_smoke_20260222` | `vlm_zero_shot` | 16 | 0.7500 | subset/smoke run |

## 4) Chapter-4 Relevance Check

- Full `clip_linear_baseline` run in `results/`: no
- Full LoRA run in `results/` (`n=1686`): no
- Structured Mayo+Evidence persisted run folder detected: no
- Best supervised (by full-run accuracy here): `finetune_resnet50`
- Best available generative full run here: `vlm_zero_shot_mayo`

## 5) Remaining Work to Close Chapter 4

1. Run and persist `clip_linear_baseline` into `1_frozen_encoders/results/clip_linear_baseline/` (missing in current results tree).
2. Run full LoRA (`n=1686`) and persist as a non-smoke folder under `3_vlm_severity/results/` (current LoRA is smoke only).
3. Persist structured Mayo+Evidence outputs to disk (`summary_metrics.json`, `pred_val.csv`, `pred_test.csv`) in a dedicated results folder.
4. Re-run final selected models with Pass-2 notebook updates so `run_meta.json` contains `run_id` and `timestamp_utc` for final thesis traceability.

## 6) Reporting Artifacts Now Available

Generated via `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting`:
- `results/tables/chapter4_main_comparison_table.csv`
- `results/tables/chapter4_remission_slice_from_results.csv`
- `results/tables/chapter4_mcnemar_pairs_from_results.csv`
- `results/tables/chapter4_qualitative_error_table.csv`
- `results/tables/chapter4_lora_ablation_table.csv`
- `results/tables/parser_audit_samples_vlm_zero_shot_mayo_test.csv`
- `results/tables/lora_load_proof.txt`
- `results/figures/confusion_test_finetune_resnet50.png`
- `results/figures/confusion_test_vlm_zero_shot_mayo.png`
- `results/figures/pred_label_histogram_vlm_zero_shot_mayo.png`

## 7) Notes

- The historical `LIMUC.md` report includes broader consolidated analysis; this snapshot is strictly from currently present `results/` artifacts.
- Some metrics in `LIMUC.md` (for example `clip_linear_baseline`) may come from earlier archived outputs when a matching current `results/` folder is absent.
- The class-distribution figure export depends on `0_dataset_prep/out/metadata/metadata_enriched.csv` being present.
