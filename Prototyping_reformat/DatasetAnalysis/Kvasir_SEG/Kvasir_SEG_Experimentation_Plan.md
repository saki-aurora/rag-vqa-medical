# Kvasir-SEG Comprehensive Experimentation Plan and Execution Report

_Updated: 2026-02-15_

## 1) Rerun Confirmation (Notebook Execution Status)

All implemented notebooks have produced persisted outputs in `out/` folders and are consistent with full split sizes.

| stage | notebook | status | evidence artifact |
|---|---|---|---|
| Data prep | `0_dataset_prep/01_build_metadata_images_and_masks.ipynb` | Completed | `0_dataset_prep/out/metadata/metadata_enriched.csv`, `split_hash.txt` |
| Data validation | `0_dataset_prep/02_validate_splits_and_integrity.ipynb` | Completed | `0_dataset_prep/out/metadata/integrity_report.json` |
| Morphology audit | `0_dataset_prep/03_dataset_morphology_and_visual_audit.ipynb` | Completed | `0_dataset_prep/out/metadata/morphology_summary.json`, `0_dataset_prep/out/visualizations/*` |
| Classic baseline 1 | `1_classic_seg_baselines/01_unet_resnet34_baseline.ipynb` | Completed | `1_classic_seg_baselines/out/unet_resnet34_baseline/metrics_test.json` |
| Classic baseline 2 | `1_classic_seg_baselines/02_deeplabv3plus_resnet50_baseline.ipynb` | Completed | `1_classic_seg_baselines/out/deeplabv3plus_resnet50_baseline/metrics_test.json` |
| Modern baseline | `2_modern_segmentation/01_segformer_b2_finetune.ipynb` | Completed | `2_modern_segmentation/out/segformer_b2_finetune/metrics_test.json` |
| Cross-dataset transfer | `3_generalization_and_ablation/01_cross_dataset_eval_kvasir_sessile.ipynb` | Completed | `3_generalization_and_ablation/out/cross_dataset_kvasir_sessile/status.json` |
| Ablations | `3_generalization_and_ablation/02_ablation_losses_augmentations_thresholds.ipynb` | Completed | `3_generalization_and_ablation/out/ablations/status.json` |
| Error analysis | `4_error_analysis/01_failure_mode_analysis.ipynb` | Completed | `4_error_analysis/out/model_summary.csv`, `hard_cases_top50.csv` |

## 2) Dataset QA and Split Integrity

### 2.1 Integrity Summary

Source: `0_dataset_prep/out/metadata/integrity_report.json`

| check | value |
|---|---|
| rows | 1000 |
| unique `img_id` | 1000 |
| missing images | 0 |
| missing masks | 0 |
| duplicate `img_id` | 0 |
| split leakage train-val | 0 |
| split leakage train-test | 0 |
| split leakage val-test | 0 |
| empty masks after threshold | 0 |

### 2.2 Split Definition

| split | count |
|---|---|
| train | 800 |
| val | 100 |
| test | 100 |

Split hash (frozen):

`66c42bf8279bf4b66d8603fbc4d56fb69bef7b09dc147e2266354a11acc8ca73`

### 2.3 Morphology Snapshot

Source: `0_dataset_prep/out/metadata/morphology_summary.json`

| metric | value |
|---|---|
| width mean | 625.292 |
| height mean | 545.228 |
| mask area ratio mean | 0.153910 |
| mask area ratio median | 0.114007 |
| component count mean | 1.207 |
| single-component share | 0.819 |
| multi-component share | 0.181 |

## 3) In-domain Segmentation Results (Kvasir-SEG Test)

Source artifacts:
- `1_classic_seg_baselines/out/*/metrics_test.json`
- `2_modern_segmentation/out/segformer_b2_finetune/metrics_test.json`

All three runs evaluated `n=100` test images.

| model | dice_mean | iou_mean | precision_mean | recall_mean | f1_mean | specificity_mean | loss |
|---|---|---|---|---|---|---|---|
| deeplabv3plus_resnet50_baseline | 0.7182 | 0.6111 | 0.8004 | 0.7247 | 0.7182 | 0.9794 | 0.2924 |
| unet_resnet34_baseline | 0.5707 | 0.4388 | 0.5748 | 0.7091 | 0.5707 | 0.9155 | 0.3994 |
| segformer_b2_finetune | 0.4749 | 0.3452 | 0.4220 | 0.7608 | 0.4749 | 0.8349 | 0.5025 |

Training history files confirm full planned epochs were executed:
- `unet_resnet34_baseline`: 8 epochs
- `deeplabv3plus_resnet50_baseline`: 6 epochs
- `segformer_b2_finetune`: 4 epochs

## 4) Cross-dataset Generalization (Kvasir-Sessile)

Source artifacts:
- `3_generalization_and_ablation/out/cross_dataset_kvasir_sessile/evaluation_summary.csv`
- `3_generalization_and_ablation/out/cross_dataset_kvasir_sessile/transfer_gap_summary.csv`
- `3_generalization_and_ablation/out/cross_dataset_kvasir_sessile/status.json`

Execution status:
- discovered runs: 3
- succeeded: 3
- failed: 0
- external set size: 196 images

### 4.1 External Metrics

| model | external_dice_mean | external_iou_mean | external_precision_mean | external_recall_mean |
|---|---|---|---|---|
| deeplabv3plus_resnet50_baseline | 0.4760 | 0.3608 | 0.4774 | 0.6274 |
| unet_resnet34_baseline | 0.3279 | 0.2198 | 0.3102 | 0.5734 |
| segformer_b2_finetune | 0.2868 | 0.1858 | 0.2129 | 0.7543 |

### 4.2 Domain-shift Drop (In-domain minus External)

| model | dice_drop | iou_drop |
|---|---|---|
| unet_resnet34_baseline | 0.2428 | 0.2190 |
| deeplabv3plus_resnet50_baseline | 0.2422 | 0.2503 |
| segformer_b2_finetune | 0.1880 | 0.1594 |

## 5) Ablation Outputs

Source artifacts:
- `3_generalization_and_ablation/out/ablations/threshold_sweep.csv`
- `3_generalization_and_ablation/out/ablations/threshold_best_per_run.csv`
- `3_generalization_and_ablation/out/ablations/status.json`

Status:
- threshold sweep rows: 15
- thresholds evaluated: `0.3, 0.4, 0.5, 0.6, 0.7`
- failures: 0
- train-time ablations executed in latest rerun: no (`run_train_ablations=false`)

### 5.1 Best Threshold per Model (By Dice)

| model | best_threshold | dice_mean | iou_mean |
|---|---|---|---|
| deeplabv3plus_resnet50_baseline | 0.4 | 0.7242 | 0.6160 |
| unet_resnet34_baseline | 0.4 | 0.5745 | 0.4413 |
| segformer_b2_finetune | 0.6 | 0.4810 | 0.3512 |

## 6) Error Analysis Summary

Source: `4_error_analysis/out/model_summary.csv`, `4_error_analysis/out/hard_cases_top50.csv`

| artifact | value |
|---|---|
| models summarized | 3 |
| hard-case rows exported | 50 |
| runs represented in hard cases | 3 |

Model ranking in `model_summary.csv` is consistent with in-domain test metrics:
1. `deeplabv3plus_resnet50_baseline`
2. `unet_resnet34_baseline`
3. `segformer_b2_finetune`

## 7) Deliverable Progress vs Original Plan

| deliverable | status | notes |
|---|---|---|
| Deterministic metadata + split artifacts | Completed | Split hash frozen and persisted |
| Integrity and morphology notebooks | Completed | Reports generated under `0_dataset_prep/out` |
| U-Net baseline | Completed | Metrics + per-image outputs saved |
| DeepLabV3+ baseline | Completed | Best in-domain metrics in current runs |
| SegFormer baseline | Completed | Metrics + per-image outputs saved |
| Kvasir-Sessile transfer evaluation | Completed | 3/3 runs succeeded |
| Threshold ablation | Completed | 15 evaluation rows, no failures |
| Loss/augmentation train-time ablations | Partially completed | Framework present; latest rerun kept disabled |
| Error analysis notebook | Completed | Summary + hard-cases exported |
| Optional MedSAM/SAM2 notebook | Not started | Still optional per compute budget |

## 8) Next Planned Actions (Thesis Integration)

1. Run multi-seed experiments (`>=3` seeds/model) for robust confidence intervals and paired tests.
2. Execute full train-time ablation matrix (`RUN_TRAIN_ABLATIONS=1`) and retain full-size outputs.
3. Integrate final Kvasir-SEG tables/figures into Chapter 3 comparative model section.
4. Use cross-dataset generalization findings as domain-shift evidence for Chapter 4/7 discussion.
