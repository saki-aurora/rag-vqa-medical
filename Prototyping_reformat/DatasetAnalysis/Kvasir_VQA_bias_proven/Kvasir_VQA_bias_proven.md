# Kvasir_VQA_bias_proven Dataset Report

_Updated: 2026-02-09_

This folder contains dataset-bias analysis artifacts for Kvasir-VQA. It does not include persisted model-evaluation runs.

## 1) Dataset Snapshot

- Total QA rows: **58849**
- Unique images: **6500**
- Avg QA rows per image: **9.05**

### 1.1 Source-level Annotation Density

| source | qa_rows | unique_images | qa_per_image |
|---|---|---|---|
| Ulcerative Colitis | 16890 | 1000 | 16.890000 |
| Esophagitis | 16723 | 1000 | 16.723000 |
| Polyps | 13539 | 1000 | 13.539000 |
| Instrument | 9197 | 1000 | 9.197000 |
| Normal | 2500 | 2500 | 1.000000 |

### 1.2 Source x Question-type Crosstab

| source | Color | Counting | Entity | Location | Other | Yes/No |
|---|---|---|---|---|---|---|
| Esophagitis | 1077 | 2967 | 2967 | 2964 | 11 | 6737 |
| Instrument | 164 | 1595 | 1265 | 1129 | 26 | 5018 |
| Normal | 0 | 0 | 0 | 0 | 0 | 2500 |
| Polyps | 978 | 2576 | 3316 | 1354 | 8 | 5307 |
| Ulcerative Colitis | 994 | 2980 | 2980 | 2977 | 6 | 6953 |

### 1.3 Top Question Templates

| question_norm | count | fraction_rows |
|---|---|---|
| have all polyps been removed | 3945 | 0.067036 |
| is this finding easy to detect | 3941 | 0.066968 |
| is there text | 3941 | 0.066968 |
| is there a green/black box artefact | 3940 | 0.066951 |
| what type of procedure is the image taken from | 3939 | 0.066934 |
| how many instrumnets are in the image | 3555 | 0.060409 |
| how many findings are present | 3429 | 0.058268 |
| what type of polyp is present | 3331 | 0.056602 |
| what is the size of the polyp | 3258 | 0.055362 |
| where in the image is the instrument | 3154 | 0.053595 |
| are there any instruments in the image? check all that are present | 3148 | 0.053493 |
| how many polyps are in the image | 3134 | 0.053255 |
| what color is the abnormality? if more than one separate with ; | 3125 | 0.053102 |
| where in the image is the abnormality | 3112 | 0.052881 |
| are there any abnormalities in the image? check all that are present | 3100 | 0.052677 |
| does this image contain any finding | 2500 | 0.042482 |
| where in the image is the anatomical landmark | 2158 | 0.036670 |
| are there any anatomical landmarks in the image? check all that are present | 2000 | 0.033985 |
| what color is the anatomical landmark? if more than one separate with ; | 88 | 0.001495 |
| none | 51 | 0.000867 |

### 1.4 Top Answers

| answer | rows | fraction_rows |
|---|---|---|
| none | 10520 | 0.178763 |
| no | 8382 | 0.142432 |
| yes | 6861 | 0.116587 |
| 0 | 4538 | 0.077113 |
| 1 | 4380 | 0.074428 |
| not relevant | 3024 | 0.051386 |
| colonoscopy | 2944 | 0.050026 |
| center center left center right lower center lower left lower right upper center upper left upper right | 1794 | 0.030485 |
| pink red white | 1271 | 0.021598 |
| polyp | 1137 | 0.019321 |
| 2 | 1116 | 0.018964 |
| gastroscopy | 993 | 0.016874 |
| ulcerative colitis | 992 | 0.016857 |
| oesophagitis | 922 | 0.015667 |
| z line | 833 | 0.014155 |
| tube | 824 | 0.014002 |
| paris ip | 568 | 0.009652 |
| paris is | 464 | 0.007885 |
| pink red | 431 | 0.007324 |
| 11 20mm | 406 | 0.006899 |

## 2) Bias Indicators

| indicator | value |
|---|---|
| max/min QA-per-image across sources | 16.890000 |
| top-1 template share | 0.067036 |
| top-5 template cumulative share | 0.334857 |
| yes/no-style question prefix share (proxy) | 0.450560 |
| none/no/not-relevant answer share | 0.372581 |
| question-template entropy | 2.885450 |
| answer entropy | 3.114473 |

## 3) Local Artifact Inventory

| track | path | key_files |
|---|---|---|
| Bias analysis narrative | 2_dataset_analysis/Results.md | Interpretive dataset health-check summary |
| Raw metadata sample | generated_outputs/metadata.csv | source/question/answer/img_id |
| Bias visualization tables | generated_outputs/2/visualizations | crosstab_source_qtype.csv, top_question_templates.csv |

## 4) Model-evaluation Availability

| observation | impact |
|---|---|
| No model prediction/evaluation artifacts are persisted in this folder | Only dataset-bias metrics (not model-performance metrics) can be reported |

## 5) Recommended Bias-aware Metric Suite for Future Runs

| metric_family | why_it_matters |
|---|---|
| Group-wise accuracy / macro-F1 | Measures performance parity across sources/question-types under imbalance |
| Worst-group accuracy | Captures failure on minority groups hidden by global averages |
| Calibration (ECE/Brier) by group | Detects overconfident bias patterns |
| Template-overlap robustness | Checks generalization beyond repeated question templates |
| Counterfactual consistency (yes/no inversions, paraphrases) | Tests shortcut reliance and lexical bias |

### 5.1 Source References

| source | metric_signals | link |
|---|---|---|
| Kvasir_VQA_bias_proven local analysis outputs | Provides source-vs-question-type crosstabs and template frequency concentration used for bias quantification | generated_outputs/2/visualizations/ |
| Kvasir-VQA benchmark context | Defines underlying QA structure used by this bias-focused derivative workspace | https://github.com/lxtGH/Kvasir-VQA |
