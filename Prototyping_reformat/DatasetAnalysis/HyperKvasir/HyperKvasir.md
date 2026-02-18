# HyperKvasir Dataset Report

_Updated: 2026-02-09_

This report consolidates all available HyperKvasir artifacts in this repository and adds derived metrics commonly used in recent HyperKvasir literature.

## 1) Dataset Snapshot

- Total images: **10662**
- Number of classes: **23**
- Test-set support range: **1 to 115** images/class (imbalance ratio **115.0x**)
- Rare classes (`support <= 5`): **7**
- Common classes (`support >= 90`): **7**

### 1.1 Split Distribution

| split | images | fraction_of_total |
|---|---|---|
| train | 8528 | 0.7998 |
| validation | 1069 | 0.1003 |
| test | 1065 | 0.0999 |

### 1.2 Per-class Counts (train/validation/test)

| label_id | label_name | train | validation | test | total |
|---|---|---|---|---|---|
| 3 | bbps-2-3 | 918 | 115 | 115 | 1148 |
| 12 | polyps | 822 | 103 | 103 | 1028 |
| 4 | cecum | 807 | 101 | 101 | 1009 |
| 5 | dyed-lifted-polyps | 802 | 100 | 100 | 1002 |
| 13 | pylorus | 799 | 100 | 100 | 999 |
| 6 | dyed-resection-margins | 791 | 99 | 99 | 989 |
| 22 | z-line | 746 | 93 | 93 | 932 |
| 15 | retroflex-stomach | 611 | 77 | 76 | 764 |
| 2 | bbps-0-1 | 517 | 64 | 65 | 646 |
| 19 | ulcerative-colitis-grade-2 | 354 | 45 | 44 | 443 |
| 7 | esophagitis-a | 322 | 41 | 40 | 403 |
| 14 | retroflex-rectum | 313 | 39 | 39 | 391 |
| 8 | esophagitis-b-d | 208 | 26 | 26 | 260 |
| 17 | ulcerative-colitis-grade-1 | 161 | 20 | 20 | 201 |
| 11 | impacted-stool | 105 | 13 | 13 | 131 |
| 21 | ulcerative-colitis-grade-3 | 106 | 14 | 13 | 133 |
| 1 | barretts-short-segment | 42 | 6 | 5 | 53 |
| 0 | barretts | 33 | 4 | 4 | 41 |
| 16 | ulcerative-colitis-grade-0-1 | 28 | 4 | 3 | 35 |
| 20 | ulcerative-colitis-grade-2-3 | 22 | 3 | 3 | 28 |
| 9 | hemorrhoids | 5 | 0 | 1 | 6 |
| 10 | ileum | 7 | 1 | 1 | 9 |
| 18 | ulcerative-colitis-grade-1-2 | 9 | 1 | 1 | 11 |

## 2) Local Artifact Inventory (used for this report)

| model | artifacts | splits |
|---|---|---|
| vit_frozen_logreg | 1_ViT/out/vit_frozen_logreg/metrics_val.json, metrics_test.json, predictions_val_test.csv | val + test |
| clip_linear | 3_feature_extraction/out/clip_linear/metrics_val.json, metrics_test.json, predictions_val_test.csv | val + test |
| vit_supervised | 2_supervised_finetuning/out/metrics_vit.json, history_vit.csv | test (overall + per-class); val history only |
| resnet50_supervised | 2_supervised_finetuning/out/metrics_resnet.json, history_resnet.csv | test (overall + per-class); val history only |
| blip2_zero_shot | 4_generative_vqa/out/blip2_zero_shot/predictions.csv, per_class*.csv | test |


> Note: `4_generative_vqa/Readme.md` states `blip2_zero_shot` is an old run and points to `blip2_zero_shot_second_run` as latest, but only old-run outputs are persisted under `out/`.
## 3) Overall Test Metrics (all saved model variants)

| rank_acc | model | n | accuracy | balanced_acc | precision_macro | recall_macro | f1_macro | f1_weighted | mcc | kappa |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | resnet50_supervised | 1065 | 0.8789 | 0.6266 | 0.6055 | 0.6266 | 0.5943 | 0.8786 | NA | NA |
| 2 | vit_supervised | 1065 | 0.8714 | 0.5391 | 0.5362 | 0.5391 | 0.5242 | 0.8559 | NA | NA |
| 3 | vit_frozen_logreg | 1065 | 0.8620 | 0.6130 | 0.6002 | 0.6130 | 0.6052 | 0.8641 | 0.8505 | 0.8504 |
| 4 | clip_linear | 1065 | 0.8620 | 0.5799 | 0.5663 | 0.5799 | 0.5721 | 0.8617 | 0.8503 | 0.8503 |
| 5 | blip2_zero_shot_clip | 1065 | 0.0638 | 0.0529 | 0.0500 | 0.0529 | 0.0254 | 0.0422 | 0.0386 | 0.0303 |

### 3.1 BLIP2 Variant-level Accuracy

| variant | accuracy | n |
|---|---|---|
| exact_match_raw_generation | 0.0000 | 1065 |
| exact_match_rule_mapped_label | 0.0000 | 1065 |
| clip_projected_label | 0.0638 | 1065 |

### 3.2 95% Wilson Confidence Intervals for Accuracy

| model | accuracy | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|
| resnet50_supervised | 0.8789 | 0.8579 | 0.8971 |
| vit_supervised | 0.8714 | 0.8499 | 0.8901 |
| vit_frozen_logreg | 0.8620 | 0.8400 | 0.8814 |
| clip_linear | 0.8620 | 0.8400 | 0.8814 |
| blip2_zero_shot_clip | 0.0638 | 0.0507 | 0.0802 |
| blip2_zero_shot_exact_match | 0.0000 | 0.0000 | 0.0036 |
| blip2_zero_shot_label_map_exact | 0.0000 | 0.0000 | 0.0036 |

## 4) Validation vs Test Drift (models with both splits)

| model | val_acc | test_acc | delta_test_minus_val_acc | val_f1_macro | test_f1_macro | delta_test_minus_val_f1_macro | val_mcc | test_mcc | delta_test_minus_val_mcc |
|---|---|---|---|---|---|---|---|---|---|
| vit_frozen_logreg | 0.8700 | 0.8620 | -0.0080 | 0.5829 | 0.6052 | 0.0223 | 0.8591 | 0.8505 | -0.0086 |
| clip_linear | 0.8279 | 0.8620 | 0.0341 | 0.5710 | 0.5721 | 0.0011 | 0.8138 | 0.8503 | 0.0365 |

## 5) Confusion-derived Error Metrics (test)

| model | specificity_macro | npv_macro | fpr_macro | fnr_macro |
|---|---|---|---|---|
| vit_frozen_logreg | 0.9937 | 0.9936 | 0.0063 | 0.3870 |
| clip_linear | 0.9937 | 0.9936 | 0.0063 | 0.4201 |
| blip2_zero_shot_clip | 0.9579 | 0.9576 | 0.0421 | 0.9471 |

## 6) Class-Imbalance Robustness Slices

### 6.1 Rare vs Common Class Recall

| model | rare_classes_mean_recall_(support<=5) | common_classes_mean_recall_(support>=90) | gap_common_minus_rare |
|---|---|---|---|
| resnet50_supervised | 0.1595 | 0.9375 | 0.7779 |
| vit_supervised | 0.0000 | 0.9432 | 0.9432 |
| vit_frozen_logreg | 0.1714 | 0.9120 | 0.7406 |
| clip_linear | 0.0476 | 0.9091 | 0.8615 |
| blip2_zero_shot_clip | 0.0000 | 0.0471 | 0.0471 |

### 6.2 Tail/Medium/Head Recall

| model | tail_mean_recall_(<=20) | medium_mean_recall_(21-89) | head_mean_recall_(>=90) |
|---|---|---|---|
| resnet50_supervised | 0.3490 | 0.7267 | 0.9375 |
| vit_supervised | 0.1100 | 0.7826 | 0.9432 |
| vit_frozen_logreg | 0.3058 | 0.7760 | 0.9120 |
| clip_linear | 0.2222 | 0.7922 | 0.9091 |
| blip2_zero_shot_clip | 0.0000 | 0.1477 | 0.0471 |

## 7) Per-class Comparison Tables

### 7.1 Best Model per Class by F1

| label_name | test_support | best_model_by_f1 | best_f1 | best_recall |
|---|---|---|---|---|
| bbps-2-3 | 115 | clip_linear | 0.9737 | 0.9652 |
| polyps | 103 | resnet50_supervised | 0.9852 | 0.9709 |
| cecum | 101 | resnet50_supervised | 0.9804 | 0.9901 |
| dyed-lifted-polyps | 100 | resnet50_supervised | 0.9561 | 0.9800 |
| pylorus | 100 | vit_frozen_logreg | 0.9950 | 0.9900 |
| dyed-resection-margins | 99 | resnet50_supervised | 0.9588 | 0.9394 |
| z-line | 93 | clip_linear | 0.8324 | 0.8280 |
| retroflex-stomach | 76 | resnet50_supervised | 0.9870 | 1.0000 |
| bbps-0-1 | 65 | resnet50_supervised | 0.9767 | 0.9692 |
| ulcerative-colitis-grade-2 | 44 | clip_linear | 0.6304 | 0.6591 |
| esophagitis-a | 40 | resnet50_supervised | 0.5586 | 0.7750 |
| retroflex-rectum | 39 | resnet50_supervised | 0.9620 | 0.9744 |
| esophagitis-b-d | 26 | vit_frozen_logreg | 0.6939 | 0.6538 |
| ulcerative-colitis-grade-1 | 20 | vit_frozen_logreg | 0.4783 | 0.5500 |
| impacted-stool | 13 | clip_linear | 0.8667 | 1.0000 |
| ulcerative-colitis-grade-3 | 13 | clip_linear | 0.5185 | 0.5385 |
| barretts-short-segment | 5 | resnet50_supervised | 0.1818 | 0.2000 |
| barretts | 4 | resnet50_supervised | 0.2857 | 0.2500 |
| ulcerative-colitis-grade-0-1 | 3 | resnet50_supervised | 0.4444 | 0.6667 |
| ulcerative-colitis-grade-2-3 | 3 | vit_frozen_logreg | 0.0000 | 0.0000 |
| hemorrhoids | 1 | vit_frozen_logreg | 0.0000 | 0.0000 |
| ileum | 1 | vit_frozen_logreg | 1.0000 | 1.0000 |
| ulcerative-colitis-grade-1-2 | 1 | vit_frozen_logreg | 0.0000 | 0.0000 |

### 7.2 Largest F1 Gains (ResNet50 supervised minus ViT supervised)

Top improvements:

| label_name | test_support | vit_sup_f1 | resnet_f1 | delta_f1_resnet_minus_vit | vit_sup_recall | resnet_recall | delta_recall_resnet_minus_vit |
|---|---|---|---|---|---|---|---|
| ulcerative-colitis-grade-3 | 13.0000 | 0.0000 | 0.4706 | 0.4706 | 0.0000 | 0.9231 | 0.9231 |
| ulcerative-colitis-grade-0-1 | 3.0000 | 0.0000 | 0.4444 | 0.4444 | 0.0000 | 0.6667 | 0.6667 |
| barretts | 4.0000 | 0.0000 | 0.2857 | 0.2857 | 0.0000 | 0.2500 | 0.2500 |
| ulcerative-colitis-grade-1 | 20.0000 | 0.1739 | 0.4186 | 0.2447 | 0.1000 | 0.4500 | 0.3500 |
| barretts-short-segment | 5.0000 | 0.0000 | 0.1818 | 0.1818 | 0.0000 | 0.2000 | 0.2000 |
| esophagitis-a | 40.0000 | 0.3889 | 0.5586 | 0.1697 | 0.3500 | 0.7750 | 0.4250 |
| retroflex-rectum | 39.0000 | 0.9091 | 0.9620 | 0.0529 | 0.8974 | 0.9744 | 0.0769 |
| dyed-lifted-polyps | 100.0000 | 0.9091 | 0.9561 | 0.0470 | 0.9500 | 0.9800 | 0.0300 |

Largest drops:

| label_name | test_support | vit_sup_f1 | resnet_f1 | delta_f1_resnet_minus_vit | vit_sup_recall | resnet_recall | delta_recall_resnet_minus_vit |
|---|---|---|---|---|---|---|---|
| ulcerative-colitis-grade-2 | 44.0000 | 0.6230 | 0.4262 | -0.1967 | 0.8636 | 0.2955 | -0.5682 |
| esophagitis-b-d | 26.0000 | 0.6531 | 0.4865 | -0.1666 | 0.6154 | 0.3462 | -0.2692 |
| impacted-stool | 13.0000 | 0.8387 | 0.8125 | -0.0262 | 1.0000 | 1.0000 | 0.0000 |
| bbps-2-3 | 115.0000 | 0.9689 | 0.9643 | -0.0046 | 0.9478 | 0.9391 | -0.0087 |
| ileum | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| hemorrhoids | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| ulcerative-colitis-grade-1-2 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| cecum | 101.0000 | 0.9804 | 0.9804 | 0.0000 | 0.9901 | 0.9901 | 0.0000 |

### 7.3 BLIP2 (CLIP-projected labels) Top per-class recall

| label_name | recall | support | hits |
|---|---|---|---|
| esophagitis-a | 0.8250 | 40 | 33 |
| dyed-lifted-polyps | 0.3000 | 100 | 30 |
| esophagitis-b-d | 0.0385 | 26 | 1 |
| ulcerative-colitis-grade-2 | 0.0227 | 44 | 1 |
| polyps | 0.0194 | 103 | 2 |
| dyed-resection-margins | 0.0101 | 99 | 1 |
| barretts | 0.0000 | 4 | 0 |
| bbps-2-3 | 0.0000 | 115 | 0 |
| cecum | 0.0000 | 101 | 0 |
| barretts-short-segment | 0.0000 | 5 | 0 |

- Classes with zero BLIP2 hits: **17 / 23**
## 8) Pairwise Significance Tests (McNemar, continuity-corrected)

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| vit_frozen_logreg vs clip_linear | 68 | 68 | 0.007353 | 0.931666 |
| vit_frozen_logreg vs blip2_zero_shot_clip | 21 | 871 | 808.072870 | 0.000000 |
| clip_linear vs blip2_zero_shot_clip | 19 | 869 | 811.712838 | 0.000000 |

## 9) Training Dynamics (supervised fine-tuning runs)

| model | epochs_trained | best_val_acc | best_epoch | final_train_loss | final_val_loss |
|---|---|---|---|---|---|
| vit_supervised | 3 | 0.8756 | 2 | 0.3349 | 0.4121 |
| resnet50_supervised | 3 | 0.8737 | 3 | 0.6261 | 0.4456 |

## 10) Metric Coverage vs Recent HyperKvasir Literature

| metric | recent_hyperkvasir_papers | local_report |
|---|---|---|
| Accuracy | Common | Computed for all models |
| Precision / Recall / F1 | Common | Computed (macro/weighted + per-class where available) |
| AUC / ROC | Common | Not computable from saved artifacts (no class probabilities/logits) |
| MCC | Reported in multiple recent papers | Computed where predictions are available |
| Cohen kappa | Used in HyperKvasir disease grading work | Computed where predictions are available |
| Specificity / FPR / FNR / NPV | Sensitivity/FNR seen; specificity often used in GI CAD | Computed one-vs-rest macro where predictions are available |
| Balanced accuracy | Used for class-imbalance settings | Computed for all classification models |
| Inference time | Reported in some recent methods | Not available in saved HyperKvasir artifacts |

### 10.1 Recent-paper metric conventions (primary sources)

| paper | metrics_reported | link |
|---|---|---|
| Curriculum self-supervised learning for HyperKvasir (Sci Rep, 2024) | Accuracy, Precision, Recall, F1-score | https://www.nature.com/articles/s41598-024-84591-7 |
| Reducing class overlap in ulcerative colitis severity (Sci Rep, 2025) | Accuracy, Precision, Recall, F1-score, AUC, MCC | https://www.nature.com/articles/s41598-025-98090-0 |
| HyperKvasir disease classification with NRMPO (Sci Rep, 2025) | Precision, Sensitivity, F1-score, FNR, execution time | https://www.nature.com/articles/s41598-025-01687-1 |
| GPT-4V for BBPS quality grading on HyperKvasir (BMJ Open Gastro, 2025) | Accuracy, Macro F1-score | https://pubmed.ncbi.nlm.nih.gov/40633642/ |
| Reflux esophagitis CAD on HyperKvasir subset (Ann Med, 2024) | AUROC, F1-score, MCC, Cohen kappa | https://pubmed.ncbi.nlm.nih.gov/38871614/ |
| Inflammatory bowel disease CAD on HyperKvasir subset (Healthcare, 2023) | Balanced accuracy, precision, F1-score (among others) | https://pubmed.ncbi.nlm.nih.gov/37892000/ |

## 11) Key Findings

- Best saved test accuracy is **0.8789** from **resnet50_supervised**; lowest is **0.0638** from **blip2_zero_shot_clip**.
- Supervised ResNet50 is strongest overall among persisted classification runs (accuracy, macro recall, and macro F1 all lead).
- Rare-class recall remains weak across all models, with very large head-vs-tail recall gaps driven by extreme class imbalance.
- BLIP2 raw exact-match is 0.0000; even CLIP-projected labels reach only 0.0638 accuracy on this 23-class task.
- AUC and inference-time comparisons cannot be reproduced from current saved artifacts because logits/probabilities and timing logs are not persisted.
## 12) Reproducibility Notes

- All metrics in this report are derived from files already present under `Prototyping_reformat/DatasetAnalysis/HyperKvasir/**/out/`.
- Balanced accuracy is computed as macro recall for single-label multiclass classification.
- Specificity/NPV/FPR/FNR are reported as one-vs-rest macro averages from confusion matrices.
- Confidence intervals use Wilson 95% intervals on observed correct counts.
