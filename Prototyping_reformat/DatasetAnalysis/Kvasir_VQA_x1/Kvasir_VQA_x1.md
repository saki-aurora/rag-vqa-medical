# Kvasir_VQA_x1 Dataset Report

_Updated: 2026-02-09_

This report consolidates all persisted Kvasir-VQA-X1 artifacts and adds derived comparisons across generative and closed-set evaluation tracks.

## 1) Dataset Snapshot

- Total QA rows: **159549**
- Unique images: **6449**
- Avg QA rows per image: **24.74**
- Complexity levels available: **3**
- Atomic question classes (expanded): **18**

### 1.1 Split Distribution

| split | rows | fraction_rows |
|---|---|---|
| train | 143594 | 0.899999 |
| test | 15955 | 0.100001 |

### 1.2 Complexity Distribution

| complexity | rows | fraction_rows |
|---|---|---|
| 1.000000 | 54856.000000 | 0.343819 |
| 2.000000 | 52349.000000 | 0.328106 |
| 3.000000 | 52344.000000 | 0.328075 |

### 1.3 Top Atomic Question Classes

| question_class | rows | fraction_rows |
|---|---|---|
| procedure_type | 23667 | 0.148337 |
| text_presence | 23600 | 0.147917 |
| polyp_removal_status | 23599 | 0.147911 |
| box_artifact_presence | 23476 | 0.147140 |
| instrument_count | 21272 | 0.133326 |
| finding_count | 20571 | 0.128932 |
| polyp_type | 20008 | 0.125403 |
| polyp_size | 19642 | 0.123110 |
| instrument_location | 18915 | 0.118553 |
| instrument_presence | 18877 | 0.118315 |
| abnormality_location | 18809 | 0.117889 |
| polyp_count | 18809 | 0.117889 |
| abnormality_color | 18607 | 0.116622 |
| abnormality_presence | 18535 | 0.116171 |
| landmark_location | 13134 | 0.082320 |
| landmark_presence | 12031 | 0.075406 |
| finding_presence | 2500 | 0.015669 |
| landmark_color | 534 | 0.003347 |

### 1.4 Composite-question Cardinality

| atomic_classes_per_row | rows | fraction_rows |
|---|---|---|
| 1.000000 | 54856.000000 | 0.343819 |
| 2.000000 | 52349.000000 | 0.328106 |
| 3.000000 | 52344.000000 | 0.328075 |

### 1.5 Most Frequent Question-class Combinations

| question_class_combo | rows | fraction_rows |
|---|---|---|
| ['polyp_removal_status'] | 3945 | 0.024726 |
| ['text_presence'] | 3941 | 0.024701 |
| ['box_artifact_presence'] | 3940 | 0.024695 |
| ['procedure_type'] | 3939 | 0.024688 |
| ['instrument_count'] | 3555 | 0.022282 |
| ['finding_count'] | 3429 | 0.021492 |
| ['polyp_type'] | 3331 | 0.020878 |
| ['polyp_size'] | 3258 | 0.020420 |
| ['instrument_location'] | 3154 | 0.019768 |
| ['instrument_presence'] | 3148 | 0.019731 |
| ['polyp_count'] | 3134 | 0.019643 |
| ['abnormality_color'] | 3125 | 0.019586 |
| ['abnormality_location'] | 3112 | 0.019505 |
| ['abnormality_presence'] | 3099 | 0.019423 |
| ['finding_presence'] | 2500 | 0.015669 |
| ['landmark_location'] | 2158 | 0.013526 |
| ['landmark_presence'] | 2000 | 0.012535 |
| ['box_artifact_presence' 'procedure_type'] | 506 | 0.003171 |
| ['text_presence' 'procedure_type'] | 487 | 0.003052 |
| ['polyp_removal_status' 'text_presence'] | 474 | 0.002971 |

### 1.6 Answer Length Statistics

| metric | value |
|---|---|
| mean_answer_tokens | 10.384991 |
| median_answer_tokens | 9.000000 |
| p90_answer_tokens | 18.000000 |
| p99_answer_tokens | 28.000000 |
| max_answer_tokens | 48.000000 |

## 2) Local Artifact Inventory

| track | path | key_files |
|---|---|---|
| Unified eval reporting | 2_modeling/12_eval_reporting/results/tables | leaderboard.csv, breakdown_by_class.csv, breakdown_by_complexity.csv, original_vs_transformed.csv |
| Modern zero-shot VLM | 2_modeling/10_modern_vlm/results/* | qwen2_5_vl_zeroshot/metrics.json, medgemma_zeroshot/metrics.json |
| LoRA fine-tune (MedGemma) | 2_modeling/11_lora_finetune/results/medgemma_lora_original | metrics.json, predictions.jsonl, run_config.json, train_progress.json |
| BLIP/BLIP2 | 2_modeling/04_blip_finetune + 06_blip2_finetune | metrics.json, metrics_test.json, predictions.jsonl |
| Classical baselines | 2_modeling/{01_text_only,02_image_only,03_fusion,05_resnet50_image_only} | metrics_test.json + prediction CSVs |
| RAG eval | 2_modeling/09_rag_blip2_eval/out | metrics.json, predictions.csv |

## 3) Generative VQA Leaderboard (Persisted Artifacts)

| model | em | token_f1 | anls | bleu | rouge_l | meteor | count |
|---|---|---|---|---|---|---|---|
| medgemma_lora_original | 0.000000 | 0.508473 | 0.340755 | NA | NA | NA | 15955.000000 |
| medgemma_zeroshot | 0.000063 | 0.213080 | 0.017498 | 0.033341 | 0.158501 | 0.141180 | 15955.000000 |
| llava_zeroshot | 0.000000 | 0.212437 | 0.007032 | 0.025760 | 0.163942 | 0.150097 | 15955.000000 |
| qwen2_5_vl_zeroshot | 0.000000 | 0.172788 | 0.000000 | 0.017084 | 0.123496 | 0.187288 | 15955.000000 |
| rag_blip2_eval | 0.000000 | NA | NA | NA | NA | NA | 200.000000 |
| blip_vqa_finetune | NA | NA | NA | 0.273109 | 0.612978 | NA | 15955.000000 |
| blip2_finetune | NA | NA | NA | 0.000000 | 0.015831 | NA | NA |

### 3.1 Exact-match 95% Wilson Confidence Intervals

| model | em | n | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|
| medgemma_zeroshot | 0.000063 | 15955.000000 | 0.000011 | 0.000355 |
| qwen2_5_vl_zeroshot | 0.000000 | 15955.000000 | 0.000000 | 0.000241 |
| llava_zeroshot | 0.000000 | 15955.000000 | 0.000000 | 0.000241 |
| rag_blip2_eval | 0.000000 | 200.000000 | 0.000000 | 0.018845 |
| medgemma_lora_original | 0.000000 | 15955.000000 | 0.000000 | 0.000241 |

### 3.2 Deltas vs `qwen2_5_vl_zeroshot`

| model | delta_token_f1_vs_qwen | delta_anls_vs_qwen | delta_em_vs_qwen |
|---|---|---|---|
| medgemma_lora_original | 0.335685 | 0.340755 | 0.000000 |
| medgemma_zeroshot | 0.040292 | 0.017498 | 0.000063 |
| llava_zeroshot | 0.039649 | 0.007032 | 0.000000 |
| qwen2_5_vl_zeroshot | 0.000000 | 0.000000 | 0.000000 |
| blip_vqa_finetune | NA | NA | NA |
| blip2_finetune | NA | NA | NA |
| rag_blip2_eval | NA | NA | 0.000000 |

## 4) Closed-set / Classification-style Metrics

| model | n | accuracy | balanced_acc | macro_f1 | weighted_f1 | top_3_accuracy | top_5_accuracy | notes |
|---|---|---|---|---|---|---|---|---|
| fusion_tfidf_vit_logreg | 5893 | 0.814865 | NA | 0.150749 | NA | NA | NA |  |
| text_yesno_tfidf_logreg | 1540 | 0.777922 | NA | 0.777291 | NA | NA | NA |  |
| vlm_zeroshot_label_mapped | 15955 | 0.561642 | NA | 0.005817 | NA | NA | NA | oov_rate=0.9730 |
| text_topk_tfidf_logreg | 4252 | 0.422389 | 0.235698 | 0.204103 | 0.328639 | 0.740828 | 0.866416 |  |
| image_resnet50_logreg | 5952 | 0.233535 | NA | 0.008556 | NA | NA | NA |  |
| text_bert_classifier | 9148 | 0.158942 | 0.006654 | 0.002327 | 0.076046 | 0.293288 | 0.356362 |  |
| image_vit_logreg | 4252 | 0.020461 | NA | 0.006352 | NA | NA | NA |  |

## 5) Breakdown and Robustness Tables (Generative Track)

### 5.1 Best Model per Question Class (by token F1)

| question_class | model | count | em | token_f1 | anls |
|---|---|---|---|---|---|
| text_presence | medgemma_zeroshot | 2454 | 0.000000 | 0.323969 | 0.029318 |
| instrument_count | medgemma_zeroshot | 2208 | 0.000000 | 0.300177 | 0.052119 |
| box_artifact_presence | medgemma_zeroshot | 2319 | 0.000000 | 0.294114 | 0.033141 |
| instrument_location | medgemma_zeroshot | 1913 | 0.000523 | 0.278065 | 0.014572 |
| landmark_color | llava_zeroshot | 55 | 0.000000 | 0.277837 | 0.000000 |
| instrument_presence | medgemma_zeroshot | 1837 | 0.000544 | 0.263131 | 0.015082 |
| landmark_location | llava_zeroshot | 1224 | 0.000000 | 0.250122 | 0.008013 |
| procedure_type | medgemma_zeroshot | 2303 | 0.000000 | 0.249978 | 0.024203 |
| polyp_count | medgemma_zeroshot | 1900 | 0.000000 | 0.247796 | 0.012816 |
| polyp_size | llava_zeroshot | 1936 | 0.000000 | 0.221228 | 0.003951 |
| abnormality_color | llava_zeroshot | 1862 | 0.000000 | 0.221034 | 0.000855 |
| finding_count | medgemma_zeroshot | 2115 | 0.000000 | 0.215266 | 0.014261 |
| landmark_presence | llava_zeroshot | 1264 | 0.000000 | 0.209348 | 0.006945 |
| polyp_type | medgemma_zeroshot | 1959 | 0.000510 | 0.208531 | 0.011259 |
| polyp_removal_status | llava_zeroshot | 2312 | 0.000000 | 0.207454 | 0.001629 |
| abnormality_location | llava_zeroshot | 1901 | 0.000000 | 0.206951 | 0.001131 |
| abnormality_presence | medgemma_zeroshot | 1823 | 0.000000 | 0.168470 | 0.002446 |
| finding_presence | medgemma_zeroshot | 237 | 0.000000 | 0.056308 | 0.000000 |

### 5.2 Hardest Class for Each Question Class (lowest token F1)

| question_class | model | count | em | token_f1 | anls |
|---|---|---|---|---|---|
| finding_presence | qwen2_5_vl_zeroshot | 237 | 0.000000 | 0.001603 | 0.000000 |
| abnormality_presence | llava_zeroshot | 1823 | 0.000000 | 0.149770 | 0.000851 |
| finding_count | qwen2_5_vl_zeroshot | 2115 | 0.000000 | 0.160905 | 0.000000 |
| abnormality_color | qwen2_5_vl_zeroshot | 1862 | 0.000000 | 0.161059 | 0.000000 |
| landmark_color | qwen2_5_vl_zeroshot | 55 | 0.000000 | 0.182130 | 0.000000 |
| landmark_presence | medgemma_zeroshot | 1264 | 0.000000 | 0.183505 | 0.002955 |
| polyp_count | qwen2_5_vl_zeroshot | 1900 | 0.000000 | 0.184266 | 0.000000 |
| polyp_type | qwen2_5_vl_zeroshot | 1959 | 0.000000 | 0.185374 | 0.000000 |
| polyp_size | qwen2_5_vl_zeroshot | 1936 | 0.000000 | 0.186602 | 0.000000 |
| abnormality_location | medgemma_zeroshot | 1901 | 0.000000 | 0.191448 | 0.003360 |
| polyp_removal_status | medgemma_zeroshot | 2312 | 0.000000 | 0.191859 | 0.008967 |
| landmark_location | medgemma_zeroshot | 1224 | 0.000000 | 0.193467 | 0.009036 |
| procedure_type | qwen2_5_vl_zeroshot | 2303 | 0.000000 | 0.204954 | 0.000000 |
| instrument_presence | qwen2_5_vl_zeroshot | 1837 | 0.000000 | 0.222355 | 0.000000 |
| instrument_count | qwen2_5_vl_zeroshot | 2208 | 0.000000 | 0.231791 | 0.000000 |
| instrument_location | qwen2_5_vl_zeroshot | 1913 | 0.000000 | 0.232128 | 0.000000 |
| box_artifact_presence | llava_zeroshot | 2319 | 0.000000 | 0.259598 | 0.020115 |
| text_presence | qwen2_5_vl_zeroshot | 2454 | 0.000000 | 0.261770 | 0.000000 |

### 5.3 Token F1 by Complexity

| complexity | llava_zeroshot | medgemma_zeroshot | qwen2_5_vl_zeroshot |
|---|---|---|---|
| 1.000000 | 0.151298 | 0.145365 | 0.079875 |
| 2.000000 | 0.217874 | 0.216663 | 0.171684 |
| 3.000000 | 0.271474 | 0.280927 | 0.271952 |

### 5.4 Original vs Transformed Slice

| em | token_f1 | anls | count | model | is_transformed |
|---|---|---|---|---|---|
| 0.000000 | 0.172788 | 0.000000 | 15955 | qwen2_5_vl_zeroshot | False |
| 0.000063 | 0.213080 | 0.017498 | 15955 | medgemma_zeroshot | False |
| 0.000000 | 0.212437 | 0.007032 | 15955 | llava_zeroshot | False |

## 6) Paired Significance (Approximate Local EM Reconstruction)

Local McNemar tests below are computed from normalized exact-match reconstruction on saved prediction files.

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| qwen2_5_vl_zeroshot vs medgemma_zeroshot | 1 | 0 | 0.000000 | 1.000000 |
| qwen2_5_vl_zeroshot vs medgemma_lora_original | 0 | 0 | 0.000000 | 1.000000 |
| medgemma_zeroshot vs medgemma_lora_original | 0 | 1 | 0.000000 | 1.000000 |

## 7) LoRA Training Diagnostics

| model | variant | train_rows | eval_rows | epochs | batch_size | grad_accum | learning_rate | device | epoch_loss_first | epoch_loss_last | completed_epochs |
|---|---|---|---|---|---|---|---|---|---|---|---|
| google/medgemma-4b-it | original | 143594 | 15955 | 1 | 4 | 4 | 0.000200 | NVIDIA H100 80GB HBM3 | 0.555220 | 0.555220 | 1 |

## 8) Metric Coverage vs Latest X1/Medico Practice

| metric_family | report_status |
|---|---|
| Exact Match (EM) | Computed for modern VLMs/LoRA and RAG runs |
| Token-level F1 | Computed for modern VLMs/LoRA (leaderboard + per-class/per-complexity) |
| ANLS | Computed for modern VLMs/LoRA |
| BLEU / ROUGE-L / METEOR | Computed for zero-shot leaderboard models; BLIP/BLIP2 include BLEU+ROUGE-L only |
| CIDEr | Not persisted in local artifacts |
| Label-mapped Accuracy / Macro-F1 | Computed in 03_vlm_modern_baseline_zeroshot and classical baselines |
| Robustness: original vs transformed | Table present but current rows are only non-transformed in persisted eval snapshot |

### 8.1 Recent Source References

| source | metric_signals | link |
|---|---|---|
| MediaEval 2025 Medico task page | Generative VQA evaluation with NLP-style metrics for answer quality | https://multimediaeval.github.io/editions/2025/tasks/medico/ |
| MediaEval-Medico-2025 official repository | Uses BLEU, ROUGE-L, METEOR, plus medical correctness/relevance and other diagnostics | https://github.com/simula/MediaEval-Medico-2025 |
| Kvasir-VQA-X1 dataset paper (2025) | Introduces transformed-complexity VQA setting and benchmark context for modern VLM evaluation | https://arxiv.org/abs/2506.09958 |
