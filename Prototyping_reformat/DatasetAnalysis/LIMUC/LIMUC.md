# LIMUC Dataset Report

_Updated: 2026-02-09_

This report consolidates all persisted LIMUC severity-classification artifacts and provides additional paired and clinical-slice diagnostics.

## 1) Dataset Snapshot

- Total images/rows: **11276**
- Unique images: **11276**
- Unique patients: **564**
- Image resolution: **352x288** (all rows)

### 1.1 Split Distribution

| split | images | fraction |
|---|---|---|
| train | 8669 | 0.768801 |
| test | 1686 | 0.149521 |
| val | 921 | 0.081678 |

### 1.2 Class Distribution (Mayo 0-3)

| label_id | images | fraction |
|---|---|---|
| 0.000000 | 6105.000000 | 0.541415 |
| 1.000000 | 3052.000000 | 0.270663 |
| 2.000000 | 1254.000000 | 0.111210 |
| 3.000000 | 865.000000 | 0.076712 |

## 2) Local Artifact Inventory

| track | path | key_files |
|---|---|---|
| Dataset prep | 0_dataset_prep/out | metadata_enriched.csv, label_map.csv, split txt files |
| Frozen encoder baselines | 1_frozen_encoders/out/* | metrics_{train,val,test}.json, per_class_test.csv, pred_test.csv |
| Supervised fine-tuning | 2_supervised_finetuning/out/* | metrics_test.json, per_class_test.csv, pred_test.csv, training_history.csv |
| VLM severity | 3_vlm_severity/out/vlm_zero_shot_mayo | metrics_test.json, per_class_test.csv, pred_test.csv |

## 3) Overall Test Metrics (All Persisted Models)

| model | n | accuracy | balanced_acc | macro_f1 | weighted_f1 | qwk | mae | rmse | spearman | auroc_ovr | ece | parse_rate | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| finetune_resnet50 | 1686 | 0.753855 | 0.695008 | 0.682889 | 0.759663 | 0.835097 | 0.256821 | 0.528545 | 0.797353 | 0.929987 | 0.057975 | NA | 0.732732 | 0.773825 |
| finetune_vit_or_swin | 1686 | 0.727165 | 0.673848 | 0.672142 | 0.733037 | 0.806259 | 0.287070 | 0.564888 | 0.748538 | 0.918377 | 0.027893 | NA | 0.705405 | 0.747892 |
| vit_frozen_logreg | 1686 | 0.689798 | 0.641650 | 0.618454 | 0.698744 | 0.758806 | 0.348161 | 0.654848 | 0.724825 | 0.880440 | 0.149887 | NA | 0.667308 | 0.711426 |
| clip_linear_baseline | 1686 | 0.679122 | 0.635709 | 0.602016 | 0.689096 | 0.745502 | 0.367734 | 0.687112 | 0.722032 | 0.867623 | 0.102128 | NA | 0.656454 | 0.700976 |
| resnet50_frozen_logreg | 1686 | 0.619217 | 0.542258 | 0.533958 | 0.627385 | 0.679280 | 0.434757 | 0.742299 | 0.614094 | 0.827388 | 0.314162 | NA | 0.595793 | 0.642099 |
| vlm_zero_shot_mayo | 1686 | 0.548636 | 0.250000 | 0.177135 | 0.388731 | 0.000000 | 0.698695 | 1.155727 | NA | NA | NA | 1.000000 | 0.524799 | 0.572252 |

## 4) Paired Significance (McNemar, Test Set)

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| finetune_resnet50 vs vlm_zero_shot_mayo | 188 | 534 | 164.854571 | 0.000000 |
| finetune_vit_or_swin vs vlm_zero_shot_mayo | 206 | 507 | 126.227209 | 0.000000 |
| resnet50_frozen_logreg vs finetune_resnet50 | 370 | 143 | 99.563353 | 0.000000 |
| vit_frozen_logreg vs vlm_zero_shot_mayo | 215 | 453 | 84.085329 | 0.000000 |
| clip_linear_baseline vs vlm_zero_shot_mayo | 214 | 434 | 74.013889 | 0.000000 |
| resnet50_frozen_logreg vs finetune_vit_or_swin | 350 | 168 | 63.245174 | 0.000000 |
| clip_linear_baseline vs finetune_resnet50 | 275 | 149 | 36.851415 | 0.000000 |
| vit_frozen_logreg vs finetune_resnet50 | 256 | 148 | 28.339109 | 0.000000 |
| vit_frozen_logreg vs resnet50_frozen_logreg | 233 | 352 | 23.801709 | 0.000001 |
| resnet50_frozen_logreg vs vlm_zero_shot_mayo | 257 | 376 | 21.996840 | 0.000003 |
| resnet50_frozen_logreg vs clip_linear_baseline | 329 | 228 | 17.953321 | 0.000023 |
| clip_linear_baseline vs finetune_vit_or_swin | 261 | 180 | 14.512472 | 0.000139 |
| vit_frozen_logreg vs finetune_vit_or_swin | 249 | 186 | 8.836782 | 0.002952 |
| finetune_vit_or_swin vs finetune_resnet50 | 182 | 137 | 6.068966 | 0.013758 |
| vit_frozen_logreg vs clip_linear_baseline | 240 | 258 | 0.580321 | 0.446186 |

## 5) Per-class Comparison

### 5.1 Best Model per Class by F1

| class | support | model | precision | recall | f1 |
|---|---|---|---|---|---|
| 0.0 | 925.000000 | finetune_resnet50 | 0.916667 | 0.796757 | 0.852516 |
| 1.0 | 464.000000 | finetune_resnet50 | 0.614065 | 0.771552 | 0.683859 |
| 2.0 | 177.000000 | finetune_resnet50 | 0.568862 | 0.536723 | 0.552326 |
| 3.0 | 120.000000 | finetune_vit_or_swin | 0.692308 | 0.675000 | 0.683544 |

### 5.2 Hardest Classes per Model

| model | class | support | precision | recall | f1 |
|---|---|---|---|---|---|
| clip_linear_baseline | 2.0 | 177.000000 | 0.377682 | 0.497175 | 0.429268 |
| clip_linear_baseline | 1.0 | 464.000000 | 0.547368 | 0.560345 | 0.553781 |
| finetune_resnet50 | 2.0 | 177.000000 | 0.568862 | 0.536723 | 0.552326 |
| finetune_resnet50 | 3.0 | 120.000000 | 0.613636 | 0.675000 | 0.642857 |
| finetune_vit_or_swin | 2.0 | 177.000000 | 0.570552 | 0.525424 | 0.547059 |
| finetune_vit_or_swin | 1.0 | 464.000000 | 0.565365 | 0.717672 | 0.632479 |
| resnet50_frozen_logreg | 2.0 | 177.000000 | 0.328571 | 0.389831 | 0.356589 |
| resnet50_frozen_logreg | 1.0 | 464.000000 | 0.471845 | 0.523707 | 0.496425 |
| vit_frozen_logreg | 2.0 | 177.000000 | 0.427273 | 0.531073 | 0.473552 |
| vit_frozen_logreg | 1.0 | 464.000000 | 0.548134 | 0.601293 | 0.573484 |
| vlm_zero_shot_mayo | 1.0 | 464.000000 | 0.000000 | 0.000000 | 0.000000 |
| vlm_zero_shot_mayo | 2.0 | 177.000000 | 0.000000 | 0.000000 | 0.000000 |

## 6) Calibration and Probabilistic Diagnostics

| model | brier_multiclass | nll | ece |
|---|---|---|---|
| finetune_vit_or_swin | 0.357370 | 0.597607 | 0.027893 |
| finetune_resnet50 | 0.341132 | 0.563271 | 0.057975 |
| clip_linear_baseline | 0.455159 | 0.852288 | 0.102128 |
| vit_frozen_logreg | 0.454586 | 1.023172 | 0.149887 |
| resnet50_frozen_logreg | 0.673589 | 2.705077 | 0.314162 |

## 7) Clinical Remission Slice (Mayo 0-1 vs 2-3)

| model | n | remission_accuracy | remission_precision | remission_recall_sensitivity | remission_specificity | remission_f1 | remission_auroc |
|---|---|---|---|---|---|---|---|
| finetune_resnet50 | 1686 | 0.947805 | 0.968998 | 0.967603 | 0.855219 | 0.968300 | 0.983706 |
| finetune_vit_or_swin | 1686 | 0.937722 | 0.956615 | 0.968323 | 0.794613 | 0.962433 | 0.974058 |
| vit_frozen_logreg | 1686 | 0.902135 | 0.962236 | 0.917207 | 0.831650 | 0.939182 | 0.953558 |
| resnet50_frozen_logreg | 1686 | 0.886714 | 0.943047 | 0.917927 | 0.740741 | 0.930317 | 0.923865 |
| clip_linear_baseline | 1686 | 0.886714 | 0.964341 | 0.895608 | 0.845118 | 0.928705 | 0.941530 |
| vlm_zero_shot_mayo | 1686 | 0.823843 | 0.823843 | 1.000000 | 0.000000 | 0.903415 | NA |

## 8) Training Dynamics (Fine-tuned Classifiers)

| model | epochs | best_val_macro_f1 | best_epoch | final_train_loss | final_val_loss |
|---|---|---|---|---|---|
| finetune_vit_or_swin | 10 | 0.702719 | 9 | 0.574518 | 0.601686 |
| finetune_resnet50 | 15 | 0.724974 | 5 | 0.214657 | 0.702900 |

## 9) Metric Coverage vs Latest LIMUC/IBD Practice

| metric_family | report_status |
|---|---|
| Accuracy / Macro-F1 / Weighted-F1 | Computed for all persisted runs |
| Balanced accuracy | Computed for all persisted runs |
| Quadratic Weighted Kappa (QWK) | Computed for all persisted runs |
| MAE / RMSE / Spearman (ordinal severity) | Computed for all persisted runs |
| AUROC (OvR) | Computed for classifier runs with probabilities |
| Calibration (ECE, Brier, NLL) | ECE persisted; Brier/NLL derived here for probabilistic runs |
| Remission-focused metrics | Derived here from test predictions (sensitivity/specificity/F1/AUROC where available) |
| Paired significance tests | McNemar tests computed across model pairs |

### 9.1 Recent Source References

| source | metric_signals | link |
|---|---|---|
| LIMUC dataset repository (paper + protocol links) | Reports macro-F1 and QWK for multiclass severity and remission-oriented analyses | https://github.com/wanghaining/ulcerative_colitis |
| CLoE benchmark (arXiv 2025) | Uses severity accuracy and QWK as core benchmarking metrics | https://arxiv.org/abs/2506.08652 |
| Medical image understanding in IBD (MDPI, 2025) | Uses accuracy, F1, and QWK; reports AUROC and statistical significance tests | https://www.mdpi.com/2075-4418/15/13/1731 |
