# ImageCLEF MEDVQA-GI 2023 Dataset Report

_Updated: 2026-02-09_

This file is the dataset-level report for the next dataset after HyperKvasir, built from persisted artifacts under `ImageCLEF_MEDVQA_GI_2023/**/out` and `.../**/results`.

## 1) Dataset Snapshot

- Total QA rows: **36683**
- Unique images: **2000**
- Unique question types: **20**
- Average QA rows per image: **18.34**

### 1.1 Split summary

| split | rows | images | questions | fraction_rows |
|---|---|---|---|---|
| train | 29351 | 1600 | 20 | 0.8001 |
| validation | 7332 | 400 | 20 | 0.1999 |

### 1.2 Question profile

| question_id | family | train_n | val_n | train_unique_answers | val_unique_answers | is_binary_train |
|---|---|---|---|---|---|---|
| Are there any abnormalities in the image? | binary/boolean | 1600 | 400 | 4 | 5 | no |
| Are there any anatomical landmarks in the image? | binary/boolean | 1600 | 400 | 6 | 5 | no |
| Are there any instruments in the image? | binary/boolean | 1600 | 400 | 7 | 6 | no |
| Have all polyps been removed? | binary/boolean | 1600 | 400 | 3 | 3 | no |
| How many findings are present? | count | 1600 | 400 | 8 | 7 | no |
| How many instrumnets are in the image? | count | 1600 | 400 | 5 | 3 | no |
| How many polyps are in the image? | count | 1600 | 400 | 7 | 7 | no |
| Is there a green/black box artefact? | binary/boolean | 1600 | 400 | 2 | 2 | yes |
| Is there text? | binary/boolean | 1600 | 400 | 2 | 2 | yes |
| Is this finding easy to detect? | binary/boolean | 1600 | 400 | 3 | 3 | no |
| What color is the abnormality? | attribute | 1600 | 400 | 13 | 9 | no |
| What color is the anatomical landmark? | attribute | 1600 | 400 | 6 | 3 | no |
| What is the size of the polyp? | attribute | 1600 | 400 | 5 | 5 | no |
| What type of polyp is present? | attribute | 1600 | 400 | 4 | 4 | no |
| What type of procedure is the image taken from? | procedure | 1600 | 400 | 2 | 2 | yes |
| Where exactly in the image is the instrument located? | location | 140 | 43 | 1 | 1 | no |
| Where exactly in the image is the polyp located? | location | 411 | 89 | 1 | 1 | no |
| Where in the image is the abnormality? | location | 1600 | 400 | 10 | 10 | no |
| Where in the image is the anatomical landmark? | location | 1600 | 400 | 7 | 7 | no |
| Where in the image is the instrument? | location | 1600 | 400 | 10 | 9 | no |

## 2) Available Run Artifacts

| run | path | available |
|---|---|---|
| vilt_finetune | 2_vqa_models/results/01_vilt_finetune/validation | metrics_overall.json, metrics_per_question.csv, metrics_binary.csv, predictions.csv |
| qwen2_5_vl_zeroshot | 3_modern_vlm/results/05_qwen2_5_vl_zeroshot/{train,validation} | metrics_overall.json, metrics_per_question.csv, metrics_binary.csv, predictions.csv |
| qwen2_5_vl_lora_finetune | 3_modern_vlm/results/06_qwen2_5_vl_lora_finetune | train_summary.json, trainer_state.json, log_history.json (no validation predictions saved) |

## 3) Overall Metrics (label-id evaluation semantics used by local pipeline)

| model | split | n | accuracy | balanced_acc | precision_macro | recall_macro | f1_macro | f1_weighted | f1_micro | mcc | kappa | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| vilt_finetune | validation | 7332 | 0.9089 | 0.5853 | 0.5799 | 0.5853 | 0.5823 | 0.9069 | 0.9089 | 0.8876 | 0.8875 | saved predictions + label-id evaluation |
| qwen2_5_vl_zeroshot_raw | validation | 7332 | 0.0007 | 0.0433 | 0.0004 | 0.0433 | 0.0007 | 0.0002 | 0.0007 | -0.0696 | -0.0626 | saved predictions + label-id evaluation |
| qwen2_5_vl_zeroshot_projected | validation | 7332 | 0.0670 | 0.0899 | 0.0609 | 0.0899 | 0.0379 | 0.0879 | 0.0670 | -0.0296 | -0.0278 | derived lexical projection to known labels |
| qwen2_5_vl_zeroshot_raw | train | 29351 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | -0.0712 | -0.0641 | saved predictions + label-id evaluation |
| qwen2_5_vl_zeroshot_projected | train | 29351 | 0.0655 | 0.0486 | 0.0605 | 0.0486 | 0.0377 | 0.0854 | 0.0655 | -0.0301 | -0.0282 | derived lexical projection to known labels |

### 3.1 95% Wilson CIs for Accuracy

| model | split | accuracy | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|
| vilt_finetune | validation | 0.9089 | 0.9021 | 0.9153 |
| qwen2_5_vl_zeroshot_raw | validation | 0.0007 | 0.0003 | 0.0016 |
| qwen2_5_vl_zeroshot_projected | validation | 0.0670 | 0.0615 | 0.0729 |
| qwen2_5_vl_zeroshot_raw | train | 0.0000 | 0.0000 | 0.0001 |
| qwen2_5_vl_zeroshot_projected | train | 0.0655 | 0.0627 | 0.0684 |

### 3.2 Saved artifact `metrics_overall.json` values (consistency check)

| artifact | accuracy | macro_f1 | n |
|---|---|---|---|
| 2_vqa_models/results/01_vilt_finetune/validation/metrics_overall.json | 0.908893 | 0.581582 | 7332 |
| 3_modern_vlm/results/05_qwen2_5_vl_zeroshot/validation/metrics_overall.json | 0.000682 | 0.000682 | 7332 |
| 3_modern_vlm/results/05_qwen2_5_vl_zeroshot/train/metrics_overall.json | 0.000000 | 0.000000 | 29351 |

## 4) Per-question Model Comparison (validation)

| question_id | n | train_unique_answers | vilt_acc | qwen_raw_acc | qwen_proj_acc | delta_proj_minus_raw_acc | vilt_macro_f1 | qwen_raw_macro_f1 | qwen_proj_macro_f1 | delta_proj_minus_raw_macro_f1 | aligned_rows |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Are there any abnormalities in the image? | 400 | 4 | 0.9500 | 0.0025 | 0.0025 | 0.0000 | 0.7607 | 0.0010 | 0.0010 | 0.0000 | yes |
| Are there any anatomical landmarks in the image? | 400 | 6 | 0.9425 | 0.0000 | 0.0175 | 0.0175 | 0.6067 | 0.0000 | 0.0109 | 0.0109 | yes |
| Are there any instruments in the image? | 400 | 7 | 0.9025 | 0.0000 | 0.1800 | 0.1800 | 0.6171 | 0.0000 | 0.0657 | 0.0657 | yes |
| Have all polyps been removed? | 400 | 3 | 0.9525 | 0.0000 | 0.0200 | 0.0200 | 0.6296 | 0.0000 | 0.0267 | 0.0267 | yes |
| How many findings are present? | 400 | 8 | 0.8300 | 0.0025 | 0.0550 | 0.0525 | 0.3518 | 0.0007 | 0.0252 | 0.0245 | yes |
| How many instrumnets are in the image? | 400 | 5 | 0.9450 | 0.0000 | 0.0050 | 0.0050 | 0.6260 | 0.0000 | 0.0034 | 0.0034 | yes |
| How many polyps are in the image? | 400 | 7 | 0.9275 | 0.0025 | 0.0600 | 0.0575 | 0.2627 | 0.0007 | 0.0279 | 0.0272 | yes |
| Is there a green/black box artefact? | 400 | 2 | 0.9600 | 0.0000 | 0.5475 | 0.5475 | 0.9540 | 0.0000 | 0.4180 | 0.4180 | yes |
| Is there text? | 400 | 2 | 0.9725 | 0.0000 | 0.0375 | 0.0375 | 0.9622 | 0.0000 | 0.0729 | 0.0729 | yes |
| Is this finding easy to detect? | 400 | 3 | 0.8575 | 0.0000 | 0.0125 | 0.0125 | 0.5696 | 0.0000 | 0.0151 | 0.0151 | yes |
| What color is the abnormality? | 400 | 13 | 0.8450 | 0.0000 | 0.0800 | 0.0800 | 0.2323 | 0.0000 | 0.1417 | 0.1417 | yes |
| What color is the anatomical landmark? | 400 | 6 | 0.9675 | 0.0000 | 0.0100 | 0.0100 | 0.3278 | 0.0000 | 0.0191 | 0.0191 | yes |
| What is the size of the polyp? | 400 | 5 | 0.7500 | 0.0000 | 0.0000 | 0.0000 | 0.2636 | 0.0000 | 0.0000 | 0.0000 | yes |
| What type of polyp is present? | 400 | 4 | 0.8325 | 0.0000 | 0.0000 | 0.0000 | 0.6032 | 0.0000 | 0.0000 | 0.0000 | yes |
| What type of procedure is the image taken from? | 400 | 2 | 0.9975 | 0.0000 | 0.0000 | 0.0000 | 0.9969 | 0.0000 | 0.0000 | 0.0000 | yes |
| Where exactly in the image is the instrument located? | 43 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | yes |
| Where exactly in the image is the polyp located? | 89 | 1 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | 1.0000 | 0.0000 | 0.0000 | 0.0000 | yes |
| Where in the image is the abnormality? | 400 | 10 | 0.9175 | 0.0000 | 0.1400 | 0.1400 | 0.1878 | 0.0000 | 0.0282 | 0.0282 | yes |
| Where in the image is the anatomical landmark? | 400 | 7 | 0.9175 | 0.0050 | 0.0050 | 0.0000 | 0.2605 | 0.0014 | 0.0014 | 0.0000 | yes |
| Where in the image is the instrument? | 400 | 10 | 0.8625 | 0.0000 | 0.0550 | 0.0550 | 0.2827 | 0.0000 | 0.0299 | 0.0299 | yes |

### 4.1 Hardest questions for ViLT (by macro F1)

| question_id | n | train_unique_answers | vilt_acc | vilt_macro_f1 |
|---|---|---|---|---|
| Where in the image is the abnormality? | 400 | 10 | 0.9175 | 0.1878 |
| What color is the abnormality? | 400 | 13 | 0.8450 | 0.2323 |
| Where in the image is the anatomical landmark? | 400 | 7 | 0.9175 | 0.2605 |
| How many polyps are in the image? | 400 | 7 | 0.9275 | 0.2627 |
| What is the size of the polyp? | 400 | 5 | 0.7500 | 0.2636 |
| Where in the image is the instrument? | 400 | 10 | 0.8625 | 0.2827 |
| What color is the anatomical landmark? | 400 | 6 | 0.9675 | 0.3278 |
| How many findings are present? | 400 | 8 | 0.8300 | 0.3518 |

### 4.2 Hardest questions for Qwen projected (by macro F1)

| question_id | n | train_unique_answers | qwen_proj_acc | qwen_proj_macro_f1 |
|---|---|---|---|---|
| What is the size of the polyp? | 400 | 5 | 0.0000 | 0.0000 |
| What type of polyp is present? | 400 | 4 | 0.0000 | 0.0000 |
| What type of procedure is the image taken from? | 400 | 2 | 0.0000 | 0.0000 |
| Where exactly in the image is the instrument located? | 43 | 1 | 0.0000 | 0.0000 |
| Where exactly in the image is the polyp located? | 89 | 1 | 0.0000 | 0.0000 |
| Are there any abnormalities in the image? | 400 | 4 | 0.0025 | 0.0010 |
| Where in the image is the anatomical landmark? | 400 | 7 | 0.0050 | 0.0014 |
| How many instrumnets are in the image? | 400 | 5 | 0.0050 | 0.0034 |

### 4.3 Largest gains from lexical projection (Qwen projected minus raw)

| question_id | n | qwen_raw_acc | qwen_proj_acc | delta_proj_minus_raw_acc | qwen_raw_macro_f1 | qwen_proj_macro_f1 | delta_proj_minus_raw_macro_f1 |
|---|---|---|---|---|---|---|---|
| Is there a green/black box artefact? | 400 | 0.0000 | 0.5475 | 0.5475 | 0.0000 | 0.4180 | 0.4180 |
| Are there any instruments in the image? | 400 | 0.0000 | 0.1800 | 0.1800 | 0.0000 | 0.0657 | 0.0657 |
| Where in the image is the abnormality? | 400 | 0.0000 | 0.1400 | 0.1400 | 0.0000 | 0.0282 | 0.0282 |
| What color is the abnormality? | 400 | 0.0000 | 0.0800 | 0.0800 | 0.0000 | 0.1417 | 0.1417 |
| How many polyps are in the image? | 400 | 0.0025 | 0.0600 | 0.0575 | 0.0007 | 0.0279 | 0.0272 |
| Where in the image is the instrument? | 400 | 0.0000 | 0.0550 | 0.0550 | 0.0000 | 0.0299 | 0.0299 |
| How many findings are present? | 400 | 0.0025 | 0.0550 | 0.0525 | 0.0007 | 0.0252 | 0.0245 |
| Is there text? | 400 | 0.0000 | 0.0375 | 0.0375 | 0.0000 | 0.0729 | 0.0729 |
| Have all polyps been removed? | 400 | 0.0000 | 0.0200 | 0.0200 | 0.0000 | 0.0267 | 0.0267 |
| Are there any anatomical landmarks in the image? | 400 | 0.0000 | 0.0175 | 0.0175 | 0.0000 | 0.0109 | 0.0109 |

## 5) Family-level Aggregates (validation)

| family | n_validation_rows | vilt_acc | qwen_raw_acc | qwen_proj_acc | vilt_macro_f1 | qwen_raw_macro_f1 | qwen_proj_macro_f1 |
|---|---|---|---|---|---|---|---|
| attribute | 1600 | 0.8488 | 0.0000 | 0.0225 | 0.5153 | 0.0000 | 0.0340 |
| binary/boolean | 2800 | 0.9339 | 0.0004 | 0.1168 | 0.9222 | 0.0005 | 0.0906 |
| count | 1200 | 0.9008 | 0.0017 | 0.0400 | 0.3950 | 0.0011 | 0.0158 |
| location | 1332 | 0.9092 | 0.0015 | 0.0601 | 0.3115 | 0.0009 | 0.0205 |
| procedure | 400 | 0.9975 | 0.0000 | 0.0000 | 0.9969 | 0.0000 | 0.0000 |

## 6) Binary-question Detailed Metrics

| question_id | model | n | precision_macro | recall_macro | f1_macro | mcc | kappa |
|---|---|---|---|---|---|---|---|
| Is there a green/black box artefact? | qwen_projected | 400 | 0.5033 | 0.3578 | 0.4180 | 0.2698 | 0.2399 |
| Is there a green/black box artefact? | qwen_raw | 400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Is there a green/black box artefact? | vilt_finetune | 400 | 0.9690 | 0.9426 | 0.9540 | 0.9112 | 0.9082 |
| Is there text? | qwen_projected | 400 | 0.5167 | 0.0434 | 0.0729 | 0.0895 | 0.0185 |
| Is there text? | qwen_raw | 400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| Is there text? | vilt_finetune | 400 | 0.9573 | 0.9673 | 0.9622 | 0.9245 | 0.9244 |
| What type of procedure is the image taken from? | qwen_projected | 400 | 0.0000 | 0.0000 | 0.0000 | -0.0218 | -0.0014 |
| What type of procedure is the image taken from? | qwen_raw | 400 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| What type of procedure is the image taken from? | vilt_finetune | 400 | 0.9983 | 0.9955 | 0.9969 | 0.9937 | 0.9937 |

## 7) Paired Significance (McNemar, continuity corrected)

### 7.1 Overall paired comparisons

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| vilt vs qwen_raw | 5 | 6664 | 6647.01814365 | 0.00000000 |
| vilt vs qwen_projected | 31 | 6204 | 6109.63656776 | 0.00000000 |
| qwen_raw vs qwen_projected | 486 | 0 | 484.00205761 | 0.00000000 |

### 7.2 Most significant per-question gaps (ViLT vs Qwen projected)

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| vilt vs qwen_projected - What type of procedure is the image taken from? | 0 | 399 | 397.00250627 | 0.00000000 |
| vilt vs qwen_projected - Are there any abnormalities in the image? | 1 | 380 | 375.02362205 | 0.00000000 |
| vilt vs qwen_projected - How many instrumnets are in the image? | 0 | 376 | 374.00265957 | 0.00000000 |
| vilt vs qwen_projected - What color is the anatomical landmark? | 4 | 387 | 373.20716113 | 0.00000000 |
| vilt vs qwen_projected - Is there text? | 0 | 374 | 372.00267380 | 0.00000000 |
| vilt vs qwen_projected - Have all polyps been removed? | 1 | 374 | 369.02400000 | 0.00000000 |
| vilt vs qwen_projected - Are there any anatomical landmarks in the image? | 0 | 370 | 368.00270270 | 0.00000000 |
| vilt vs qwen_projected - Where in the image is the anatomical landmark? | 2 | 367 | 359.06775068 | 0.00000000 |
| vilt vs qwen_projected - How many polyps are in the image? | 1 | 348 | 343.02578797 | 0.00000000 |
| vilt vs qwen_projected - Is this finding easy to detect? | 0 | 338 | 336.00295858 | 0.00000000 |
| vilt vs qwen_projected - What type of polyp is present? | 0 | 333 | 331.00300300 | 0.00000000 |
| vilt vs qwen_projected - Where in the image is the instrument? | 5 | 328 | 311.36336336 | 0.00000000 |

## 8) Qwen2.5-VL LoRA Fine-tune Training Diagnostics

| model_name | run | num_train_samples | epochs | batch_size | learning_rate | use_4bit | compute_dtype | device_map |
|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-VL-7B-Instruct | 06_qwen2_5_vl_lora_finetune | 29351 | 1 | 1 | 0.0002 | 1 | bfloat16 | auto |

Training-curve summary from `trainer_state.json` log history:

| global_step | epochs | logged_points | loss_first | loss_last | loss_min | loss_median | grad_norm_max | lr_first | lr_last |
|---|---|---|---|---|---|---|---|---|---|
| 29351.000000 | 1.000000 | 1468.000000 | 8.228500 | 0.004900 | 0.000200 | 0.339200 | 19155.505859 | 0.000200 | 0.000000 |

## 9) Metric Coverage vs Latest MEDVQA-GI Practice

| metric | local_report_status |
|---|---|
| Accuracy | Computed (all available runs) |
| Macro F1 | Computed (all available runs) |
| Weighted/Micro F1 | Computed (derived) |
| Macro Precision/Recall | Computed (derived) |
| Balanced accuracy (macro recall) | Computed (derived) |
| MCC | Computed (derived) |
| Cohen kappa | Computed (derived) |
| 95% CI for accuracy | Computed (Wilson intervals) |
| McNemar significance tests | Computed (paired validation rows) |
| mIoU / Dice | Not computable from saved VQA prediction artifacts (subtask-3 masks not evaluated here) |
| BLEU / ROUGE | Not used in this closed-label MEDVQA-GI setup |

### 9.1 Recent/official source references

| source | reported_metrics | link |
|---|---|---|
| ImageCLEFmed MEDVQA-GI 2023 task page | F1, Accuracy, Recall, MCC, mIoU, Dice (across subtasks) | https://www.imageclef.org/2023/medical/vqa |
| ImageCLEFmedical 2023 MEDVQA overview chapter | Task 1 ranking by Accuracy; full table includes Accuracy and F1 | https://doi.org/10.1007/978-3-031-54605-1_4 |
| UIT-Saviors at MEDVQA-GI 2023 (participant paper) | Accuracy and F1-score (development + private test) | https://arxiv.org/abs/2307.02783 |
| ImageCLEFmed VQA 2024 task page | Accuracy, Precision, Recall, F1 (+ FID for synthesis setting) | https://www.imageclef.org/2024/medical/vqa |

## 10) Key Findings

- Best available validation run is **vilt_finetune** with accuracy **0.9089** and macro-F1 **0.5823**.
- Lowest validation run is **qwen2_5_vl_zeroshot_raw** with accuracy **0.0007** and macro-F1 **0.0007**.
- Qwen lexical projection increases validation accuracy from **0.0007** to **0.0670** (absolute **+0.0663**), but remains far below ViLT.
- Highest-difficulty questions are mostly high-cardinality attributes/locations (e.g., abnormality color and location questions).
- Binary questions are near-saturated for ViLT and near-zero for raw Qwen outputs; projection partially recovers binary performance only in limited cases.
- LoRA run has training logs saved, but no persisted validation prediction file/metrics in this repository snapshot, so no fair model-to-model comparison is possible yet.
## 11) Reproducibility Notes

- `label_id` metrics follow the same per-question label-map strategy used in local notebooks (`common.py` + `add_label_ids`).
- Qwen "projected" metrics are additional diagnostics from this report, created by lexical mapping of generated text to known answer labels per question.
- McNemar tests use paired validation rows (confirmed aligned by `(image_id, question_id, answer_norm, split)`).
- mIoU/Dice are not included because this report covers persisted VQA prediction artifacts, not segmentation-task run outputs.
