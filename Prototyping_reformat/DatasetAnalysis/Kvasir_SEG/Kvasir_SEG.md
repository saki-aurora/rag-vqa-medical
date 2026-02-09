# Kvasir_SEG Dataset Report

_Updated: 2026-02-09_

This report summarizes all persisted Kvasir-SEG ground-truth artifacts and computes dataset morphology statistics. No model-evaluation outputs are stored in this folder snapshot.

## 1) Dataset Snapshot

| set | count |
|---|---|
| images | 1000 |
| masks | 1000 |
| paired_images_masks | 1000 |
| images_without_mask | 0 |
| masks_without_image | 0 |

### 1.1 Image Resolution Summary

| metric | value |
|---|---|
| n_images | 1000.000000 |
| width_mean | 625.292000 |
| height_mean | 545.228000 |
| width_min | 332.000000 |
| width_max | 1920.000000 |
| height_min | 352.000000 |
| height_max | 1072.000000 |

### 1.2 Mask Foreground Coverage Summary

| metric | value |
|---|---|
| foreground_ratio_mean | 0.153910 |
| foreground_ratio_median | 0.114007 |
| foreground_ratio_p10 | 0.034108 |
| foreground_ratio_p90 | 0.319095 |
| foreground_ratio_min | 0.004739 |
| foreground_ratio_max | 0.811820 |

### 1.3 Mask Connected-component Summary

| metric | value |
|---|---|
| components_mean | 1.207000 |
| components_median | 1.000000 |
| single_component_share | 0.819000 |
| multi_component_share | 0.181000 |
| largest_component_ratio_mean | 0.152515 |
| largest_component_ratio_median | 0.110446 |

### 1.4 Bounding-box Summary

| metric | value |
|---|---|
| bbox_entries | 1000.000000 |
| bbox_count_mean | 1.071000 |
| bbox_count_max | 10.000000 |
| bbox_union_area_ratio_mean | 0.236413 |
| bbox_union_area_ratio_median | 0.176599 |
| bbox_vs_mask_bbox_iou_mean | 0.923643 |
| bbox_vs_mask_bbox_iou_median | 0.980719 |

## 2) Local Artifact Inventory

| track | path | key_files |
|---|---|---|
| Raw image/mask pairs | 0_dataset_prep/Kvasir-SEG/{images,masks} | 1000 JPEG images + 1000 mask JPEGs |
| Bounding-box annotations | 0_dataset_prep/Kvasir-SEG/kavsir_bboxes.json | Per-image width/height and polyp bbox coordinates |
| Prepared copy | 0_dataset_prep/out/Kvasir-SEG/kavsir_bboxes.json | Mirrored bbox JSON |

## 3) Local Evaluation Availability

| observation | impact |
|---|---|
| No persisted model-evaluation metrics JSON/CSV found in this folder | Performance comparison tables (Dice/IoU etc.) cannot be filled from local artifacts |

## 4) Metric Coverage vs Latest Kvasir-SEG Practice

| metric_family | report_status |
|---|---|
| Dice coefficient (DSC) | Recommended by Kvasir-SEG literature; not computable here (no model preds persisted) |
| IoU / mIoU | Recommended by Kvasir-SEG literature; not computable here |
| Precision / Recall / F1 | Common in recent segmentation papers; not computable here |
| HD95 / MAE boundary-distance | Used in recent transformer/CNN segmentation papers; not computable here |
| Dataset morphology stats (mask area, connected components, bbox-mask overlap) | Computed in this report from ground-truth artifacts |

### 4.1 Recent Source References

| source | metric_signals | link |
|---|---|---|
| Kvasir-SEG original benchmark paper (MICCAI MMMI 2020) | Dice and mIoU used as primary segmentation quality metrics | https://arxiv.org/abs/1911.07069 |
| Kvasir-SEG dataset page (Simula) | Benchmark context and links to baseline evaluations | https://datasets.simula.no/kvasir-seg/ |
| EffiSegNet (BMC Med Imaging, 2025) | Reports mDice and mIoU on Kvasir-SEG | https://pmc.ncbi.nlm.nih.gov/articles/PMC12470534/ |
| HSSAM polyp segmentation study (Sci Rep, 2025) | Reports Dice, mIoU, Precision, Recall, HD95, and MAE | https://www.nature.com/articles/s41598-025-12417-z |
