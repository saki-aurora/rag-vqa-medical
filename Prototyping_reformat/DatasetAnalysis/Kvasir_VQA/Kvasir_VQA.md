# Kvasir_VQA Dataset Report

_Updated: 2026-02-09_

This report consolidates all persisted Kvasir_VQA artifacts and computes additional classification diagnostics for available benchmark subsets.

## 1) Dataset Snapshot

- Total QA rows: **58849**
- Unique images: **6500**
- Avg QA rows per image: **9.05**
- Unique normalized question templates: **20**

### 1.1 Source Distribution and Annotation Density

| source | qa_rows | unique_images | qa_per_image |
|---|---|---|---|
| Ulcerative Colitis | 16890 | 1000 | 16.890000 |
| Esophagitis | 16723 | 1000 | 16.723000 |
| Polyps | 13539 | 1000 | 13.539000 |
| Instrument | 9197 | 1000 | 9.197000 |
| Normal | 2500 | 2500 | 1.000000 |

### 1.2 Question Type Distribution

| question_type | rows | fraction_rows |
|---|---|---|
| Yes/No | 26515 | 0.450560 |
| Entity | 10528 | 0.178899 |
| Counting | 10118 | 0.171932 |
| Location | 8424 | 0.143146 |
| Color | 3213 | 0.054597 |
| Other | 51 | 0.000867 |

### 1.3 Answer Type Distribution

| answer_type | rows | fraction_rows |
|---|---|---|
| Yes/No | 15243 | 0.259019 |
| Token | 15137 | 0.257218 |
| None/NA | 10520 | 0.178763 |
| Numeric | 10156 | 0.172577 |
| List | 7793 | 0.132424 |

### 1.4 Most Frequent Question Templates

| question_norm | rows | fraction_rows |
|---|---|---|
| have all polyps been removed | 3945 | 0.067036 |
| is this finding easy to detect | 3941 | 0.066968 |
| is there text | 3941 | 0.066968 |
| is there a green black box artefact | 3940 | 0.066951 |
| what type of procedure is the image taken from | 3939 | 0.066934 |
| how many instrumnets are in the image | 3555 | 0.060409 |
| how many findings are present | 3429 | 0.058268 |
| what type of polyp is present | 3331 | 0.056602 |
| what is the size of the polyp | 3258 | 0.055362 |
| where in the image is the instrument | 3154 | 0.053595 |
| are there any instruments in the image check all that are present | 3148 | 0.053493 |
| how many polyps are in the image | 3134 | 0.053255 |
| what color is the abnormality if more than one separate with | 3125 | 0.053102 |
| where in the image is the abnormality | 3112 | 0.052881 |
| are there any abnormalities in the image check all that are present | 3100 | 0.052677 |
| does this image contain any finding | 2500 | 0.042482 |
| where in the image is the anatomical landmark | 2158 | 0.036670 |
| are there any anatomical landmarks in the image check all that are present | 2000 | 0.033985 |
| what color is the anatomical landmark if more than one separate with | 88 | 0.001495 |
| none | 51 | 0.000867 |

## 2) Local Artifact Inventory

| track | path | key_files |
|---|---|---|
| Dataset prep | 0_dataset_prep/out | metadata_enriched.csv + visualizations/*.csv |
| Text baseline (UC answer-type) | 1_text_baselines_models/out/uc_answer_type_tfidf_dt | predictions_with_rows.csv, classification_report.txt, run_meta.json |
| BLIP transformer yes/no | 4_VQA_transformers/out | blip_vqa_base_yesno/metrics.json, blip_yesno_forced_choice/metrics.json |
| Custom fusion benchmarks | 5_custom_fusion_VQA_models/out/reports | main_table.csv + m1/m2/blip2 prediction CSVs |
| Phase-3 comparison | evaluation_comparison/phase3_results | summary_uc_phase3.csv + empty prediction placeholders |

## 3) UC Answer-type Text Baseline (`tfidf + decision_tree`)

| model | n | accuracy | balanced_acc | precision_macro | recall_macro | f1_macro | f1_weighted | mcc | kappa | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| tfidf_decision_tree_uc_answer_type | 3378 | 0.996152 | 0.995360 | 0.996000 | 0.995360 | 0.995642 | 0.996155 | 0.995050 | 0.995034 | 0.993426 | 0.997750 |

### 3.1 Per-class Performance (UC answer types)

| class | support | precision | recall | f1 |
|---|---|---|---|---|
| None/NA | 1129.000000 | 1.000000 | 1.000000 | 1.000000 |
| Token | 637.000000 | 0.980000 | 1.000000 | 0.989899 |
| Yes/No | 597.000000 | 1.000000 | 0.981575 | 0.990702 |
| Numeric | 596.000000 | 1.000000 | 1.000000 | 1.000000 |
| List | 419.000000 | 1.000000 | 0.995227 | 0.997608 |

## 4) Yes/No Benchmarks Across Available Runs

| model | n | accuracy | balanced_acc | precision_macro | recall_macro | f1_macro | f1_weighted | mcc | kappa | coverage_non_unknown | unknown_rate | notes | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet_gru_m1_yesno | 443 | 0.986456 | 0.973673 | 0.956646 | 0.973673 | 0.964953 | 0.986580 | 0.930163 | 0.929909 | 1.000000 | 0.000000 | custom fusion benchmark subset | 0.970770 | 0.993778 |
| vit_bertlite_m2_yesno | 443 | 0.950339 | 0.906593 | 0.854616 | 0.906593 | 0.878126 | 0.952033 | 0.759432 | 0.756447 | 1.000000 | 0.000000 | custom fusion benchmark subset | 0.925956 | 0.966979 |
| blip2_zeroshot_yesno | 443 | 0.893905 | 0.509376 | 0.697846 | 0.509376 | 0.492332 | 0.848040 | 0.086138 | 0.032436 | 1.000000 | 0.000000 | custom fusion benchmark subset | 0.861765 | 0.919273 |
| blip_vqa_base_yesno_forced_choice | 12267 | 0.518301 | 0.514587 | 0.516357 | 0.514587 | 0.502888 | 0.504865 | 0.030894 | 0.029379 | 1.000000 | 0.000000 |  | 0.509455 | 0.527136 |
| blip_vqa_base_yesno_freegen | 500 | 0.000000 | NA | NA | NA | 0.000000 | NA | NA | NA | 0.000000 | 1.000000 | all outputs unknown in persisted run | 0.000000 | 0.007624 |

### 4.1 Pairwise Significance (McNemar, aligned subsets)

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| m1_yesno vs m2_yesno | 1 | 17 | 12.500000 | 0.000407 |

## 5) Attribute Subset Benchmarks (Custom Fusion)

| model | n | accuracy | balanced_acc | precision_macro | recall_macro | f1_macro | f1_weighted | mcc | kappa | 95%_wilson_low | 95%_wilson_high |
|---|---|---|---|---|---|---|---|---|---|---|---|
| resnet_gru_m1_attribute | 352 | 0.670455 | 0.376936 | 0.370063 | 0.376936 | 0.367341 | 0.652101 | 0.535752 | 0.529238 | 0.619741 | 0.717488 |
| vit_bertlite_m2_attribute | 352 | 0.656250 | 0.374696 | 0.409337 | 0.374696 | 0.355586 | 0.626094 | 0.541947 | 0.511492 | 0.605186 | 0.703941 |

### 5.1 Pairwise Significance (McNemar)

| pair | n01_(A_wrong_B_right) | n10_(A_right_B_wrong) | chi2_cc | p_value |
|---|---|---|---|---|
| m1_attr vs m2_attr | 51 | 55 | 0.084906 | 0.770756 |

## 6) Custom Fusion Main Table (Persisted)

| Model | Type | Accuracy | Macro-F1 |
|---|---|---|---|
| ResNet+GRU (M1) | Yes/No | 0.986456 | 0.964953 |
| ViT+BERT-lite (M2) | Yes/No | 0.950339 | 0.878126 |
| BLIP-2 zero-shot | Yes/No | 0.893905 | 0.492332 |
| ResNet+GRU (M1) | Attribute | 0.670455 | 0.367341 |
| ViT+BERT-lite (M2) | Attribute | 0.656250 | 0.355586 |

## 7) Phase-3 Comparison Status (BLIP/ViLT/GIT)

| model | n_examples | BLEU_avg | ROUGE_L | time_sec | time_per_example_sec |
|---|---|---|---|---|---|
| blip | 0 | NA | NA | 0.001179 | 0.001179 |
| vilt | 0 | NA | NA | 0.000778 | 0.000778 |
| git | 0 | NA | NA | 0.000772 | 0.000772 |
| blip2 | 0 | NA | NA | 0.000685 | 0.000685 |

### 7.1 Prediction Artifact Availability

| file | size_bytes | has_rows |
|---|---|---|
| vilt_uc_predictions.csv | 1 | 0 |
| blip_uc_predictions.csv | 1 | 0 |
| blip2_uc_predictions.csv | 1 | 0 |
| git_uc_predictions.csv | 1 | 0 |

## 8) Metric Coverage vs Latest Kvasir-VQA Practice

| metric_family | report_status |
|---|---|
| Accuracy / Macro-F1 (classification) | Computed for UC baseline, BLIP forced-choice, and custom fusion subsets |
| Balanced accuracy, MCC, Cohen kappa | Derived in this report for all classification subsets with labels |
| Precision/Recall (macro + class-wise) | Computed for all classification subsets |
| BLEU / ROUGE-L / METEOR / CIDEr | Literature-recommended for generative Kvasir-VQA; not available from current persisted phase-3 files (n_examples=0) |
| Runtime-per-example | Available in summary_uc_phase3.csv (but no evaluated examples) |

### 8.1 Recent Source References

| source | metric_signals | link |
|---|---|---|
| Kvasir-VQA benchmark paper (arXiv 2024) | Reports BLEU, ROUGE-L, METEOR, and CIDEr for text-image VQA tasks | https://arxiv.org/html/2409.04556v2 |
| Kvasir-VQA repository | Provides benchmark splits/tasks and evaluation context for medical GI VQA | https://github.com/lxtGH/Kvasir-VQA |
