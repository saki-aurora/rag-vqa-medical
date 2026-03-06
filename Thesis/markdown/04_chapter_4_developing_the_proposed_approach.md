# Chapter 4: Developing the Proposed Approach

## 4.1 Chapter Overview and Goals

Chapter 3 established a reproducible reliability baseline across GI MedVQA settings and showed a consistent gap between supervised severity models and naive zero-shot severity prompting on LIMUC.  
Chapter 4 operationalizes that gap into a concrete method: a controlled severity-oriented pipeline for Ulcerative Colitis (UC) that preserves visual reliability while introducing generative capabilities under explicit output constraints.

This chapter has four goals:

1. Define a reproducible UC severity task formulation for Mayo 0-3 scoring.
2. Build strong supervised and zero-shot baselines in the same protocol.
3. Implement parameter-efficient VLM adaptation (LoRA) for controlled generative severity output.
4. Compare methods with ordinal and clinically aligned metrics, then analyze failure patterns.

## 4.2 Datasets and Scope

### 4.2.1 Primary Dataset Required

The only dataset strictly required to execute Chapter 4 in this repository is:

- **LIMUC** (UC severity, Mayo 0-3), expected by:
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb`

Expected local structure (as referenced by the LIMUC data-prep notebook):

- `Datasets/LIMUC/train_and_validation_sets`
- `Datasets/LIMUC/test_set`
- optional: `Datasets/LIMUC/patient_based_classified_images` for patient-aware split workflows

Frozen LIMUC metadata snapshot used in this chapter (`metadata_enriched.csv`) contains `11,276` frames:

| Split | N |
|---|---:|
| Train | 8669 |
| Val | 921 |
| Test | 1686 |

| Mayo class | N (all splits) |
|---|---:|
| 0 | 6105 |
| 1 | 3052 |
| 2 | 1254 |
| 3 | 865 |

### 4.2.2 Optional External Robustness Dataset

If cross-dataset robustness is included in Chapter 4, add:

- **HyperKvasir UC subset** (or a clinically filtered UC subset) as external-only evaluation.

This is optional and should be framed as domain-shift testing, not primary training evidence.

### 4.2.3 Datasets Not Required for Core Chapter 4 Severity Claims

The following are useful for broader thesis context but are not mandatory for Chapter 4 core UC severity execution:

- Kvasir-VQA
- Kvasir-VQA-x1
- ImageCLEF MEDVQA-GI

## 4.3 Proposed Pipeline

Chapter 4 uses the pipeline:

`Dataset curation and split freeze -> Supervised/VLM fine-tuning -> Controlled generation or label scoring -> Optional retrieval support -> Evaluation and statistical validation`

### 4.3.1 Data Preparation Layer

Primary notebook:

- `Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb`

Core outputs used downstream:

- metadata table with split assignment
- label mapping (Mayo classes)
- split hash for reproducibility traceability

### 4.3.2 Strong Supervised Baseline Layer

Frozen baselines:

- `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/resnet50_frozen_logreg.ipynb`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/vit_frozen_logreg.ipynb`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/clip_linear_baseline.ipynb`

Fine-tuned baselines:

- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_resnet50.ipynb`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_vit_or_swin.ipynb`

These models define the reliability anchor for all generative comparisons.

### 4.3.3 Zero-Shot Generative Severity Layer

Primary notebook:

- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_zero_shot_mayo.ipynb`

Task framing:

- fixed severity question prompt
- constrained expected format (`SCORE: X`)
- score parser with invalid handling

### 4.3.4 Parameter-Efficient Generative Adaptation Layer

Primary notebook:

- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_lora_finetune_mayo.ipynb`

Method:

- LoRA adaptation on vision-language generation stack
- supervised target format with explicit score token

Dependency note:

- `peft` is required for this stage and can be installed into the active environment.

### 4.3.5 Optional Retrieval-Support Layer

If Chapter 4 includes a retrieval-backed extension (lightweight RAG-style prompt context), the local reference pattern is:

- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/2_modeling/09_rag_blip2_eval/01_rag_blip2_eval.ipynb`

This should be reported as an extension path, not the primary severity model.

## 4.4 Experimental Design

### 4.4.1 Task Definition and Evaluation Lanes

Input:

- colonoscopy frame
- fixed severity question prompt

Output:

- Mayo score in `{0,1,2,3}`

Optional output extension:

- short evidence phrase constrained to visual findings only

Lane definitions used in Chapter 4:

- `mode1` (`lora_mode1_train`): free generation constrained by prompt format, then strict parser extraction of `SCORE: <0|1|2|3>`.
- `mode2` (`lora_mode2_eval`): score is chosen by candidate-label likelihood after the `SCORE:` prefix (`sequence_logprob` strategy), without relying on free-text parsing.

`mode1` is the primary generative lane; `mode2` is retained as a controlled ablation.

### 4.4.2 Metric Bundle

Primary metrics:

- Accuracy
- Macro-F1
- Balanced Accuracy
- Quadratic Weighted Kappa (QWK)
- MAE / RMSE (ordinal distance)

Supplementary metrics:

- Per-class precision/recall/F1
- Confusion matrix
- Parse/unknown rate for generative outputs
- Remission-oriented slice (`0-1` vs `2-3`): sensitivity, specificity, F1

### 4.4.3 Statistical and Reliability Checks

- Paired significance testing (McNemar where paired predictions are available)
- Confidence interval reporting for key metrics
- Error decomposition by class and clinical threshold behavior

### 4.4.4 Frozen Training Configuration Summary

Configuration values are frozen from:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_multiseed_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_multiseed_report.json`

| Lane | Core setup |
|---|---|
| Pass 5 supervised | ResNet50 fine-tune, seeds `11/23/42`, epochs `15`, batch size `16`, LR `3e-4`, weight decay `1e-4`. |
| Pass 6 generative | BLIP2-Flan-T5-XL + LoRA, seeds `11/23/77`, epochs `2`, batch size `2`, grad accumulation `4`, LR `5e-5`, LoRA `r=8`, `alpha=16`, dropout `0.1`, balanced sampling, label-token-only objective. |

## 4.5 Frozen Results and Claim Boundary (LIMUC)

This section is synchronized to the Chapter 4 freeze package and is restricted to frozen Pass 5/6/7 artifacts:
- `CH4_PART1_SCOPE_FREEZE_20260306.md`
- `CH4_PART2_REPRO_FREEZE_20260306.md`
- `CH4_PART3_WRITING_PACK_20260306.md`

### 4.5.1 Primary KPI and Reporting Policy

The primary optimization and reporting KPI is internal LIMUC `mode1/test` QWK. Accuracy, macro-F1, balanced accuracy, and parse rate are treated as secondary companion metrics. The chapter objective is reproducible comparative evaluation rather than hitting a pre-specified numeric cutoff. External HyperKvasir UC proxy scores are reported as stress-test evidence only and are not used as the model-selection target.

### 4.5.2 Internal Multi-Seed Results (Pass 5 vs Pass 6)

Source artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_multiseed_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_multiseed_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_mode1_qc.csv`

| Lane | Seeds | Accuracy | Macro-F1 | Balanced accuracy | QWK | 95% CI (QWK) | Parse rate |
|---|---|---:|---:|---:|---:|---|---:|
| Pass 5 supervised | 11/23/42 | 0.737643 | 0.667330 | 0.670907 | 0.818649 | [0.807920, 0.830582] | -- |
| Pass 6 `lora_mode1_train` | 11/23/77 | 0.781930 | 0.727920 | 0.736292 | 0.863656 | [0.862382, 0.865836] | 1.000000 |
| Pass 6 `lora_mode2_eval` | 11/23/77 | 0.548636 | 0.177135 | 0.250000 | 0.000000 | [0.000000, 0.000000] | 1.000000 |

Compared with the official Pass 5 baseline, the retained mode1 lane improves internal QWK by `+0.045007` and macro-F1 by `+0.060590`. Mode2 remains a controlled negative-result lane. The mode1 QC table confirms stable non-degenerate convergence (`3/3` seeds pass).

![Frozen internal KPI comparison (Pass5 vs Pass6) (F02)](figures/ch4_representations/F02_ch4_core_metric_comparison.png)

*Frozen internal KPI comparison using Pass 5 supervised and Pass 6 generative lane aggregates.*

![Frozen metric means with 95% confidence intervals (F03)](figures/ch4_representations/F03_ch4_radar_profile.png)

*Frozen Chapter 4 metric means with 95% confidence intervals from Pass 5 and Pass 6 reports.*

![Mode1 seed-level QC heatmap (Pass6) (F05)](figures/ch4_representations/F05_ch4_mcnemar_significance_heatmap.png)

*Mode1 seed-level QC for Pass 6 generative training (legacy figure ID/file name retained for freeze compatibility).*

### 4.5.3 External Stress-Test (Pass 7 HyperKvasir UC Proxy)

Source artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_drop_table.csv`

External protocol note (from `metadata_hyperkvasir_uc_proxy_mayo_floor.csv`):
- external set size: `851` frames (test-only)
- class distribution: `0:35`, `1:212`, `2:471`, `3:133`
- mapping policy: `floor_for_interval_labels` for interval findings (`0-1 -> 0`, `1-2 -> 1`, `2-3 -> 2`)

| Lane | Internal QWK | External QWK | Delta (external-internal) | Internal parse rate | External parse rate |
|---|---:|---:|---:|---:|---:|
| `resnet50_supervised` | 0.828762 | 0.359597 | -0.469165 | -- | -- |
| `vlm_lora_mode1` | 0.862752 | 0.000000 | -0.862752 | 1.0 | 0.0 |
| `vlm_lora_mode2` | 0.000000 | 0.000000 | 0.000000 | 1.0 | 1.0 |

External performance on the current proxy mapping is poor for both families, with especially severe degradation for mode1. This is interpreted as domain-shift and label-compatibility limitation evidence, not as contradiction of the internal LIMUC result.

![External stress-test drops (Pass7) (F04)](figures/ch4_representations/F04_ch4_remission_slice_comparison.png)

*External stress-test drops (external minus internal) from Pass 7 HyperKvasir UC proxy evaluation.*

### 4.5.4 Frozen Artifact Set Used in Chapter 4 Tables and Figures

Primary citation artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_mode1_qc.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_drop_table.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`

Supporting diagnostic figures:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/pass5_supervised_metric_ci.png`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/pass6_generative_metric_ci.png`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_confusion_aggregate.png`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_confusion_aggregate_mode1.png`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_confusion_aggregate_mode2.png`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/generative_pred_distribution.png`

![Aggregate confusion panel (Pass5/Pass6) (F06)](figures/ch4_representations/F06_ch4_confusion_panel.png)

*Aggregate confusion panel for Pass 5 supervised and Pass 6 mode1/mode2 lanes.*

### 4.5.5 Exploratory Optimization (Appendix-Only)

| Sweep | Source artifact | Best observed test QWK | Best run | Reporting role |
|---|---|---:|---|---|
| Pass 8 internal fusion | `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_internal_fusion_20260306T085817Z/pass8_internal_fusion_candidates.csv` | 0.866368 | `baseline_vlm_vote3` | Appendix only |
| Pass 8 supervised scout (5090) | `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_scout_r2_20260306T091539Z/pass8_supervised_all_runs.csv` | 0.870416 | `swin_t_ce_m_e8_seed011` | Appendix only |
| Pass 8 supervised focus (5090) | `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_focus_20260306T092204Z/pass8_supervised_all_runs.csv` | 0.861687 | `swin_t_ce_s_e16_seed011` | Appendix only |

These exploratory sweeps are informative but remain outside headline claims; all reported exploratory configurations stayed below `QWK = 0.90`, and are therefore retained as future-improvement context only.

## 4.6 Chapter-4 Completion Status

### 4.6.1 Freeze and Reporting Completion

- Scope and claim boundary lock: `CH4_PART1_SCOPE_FREEZE_20260306.md`
- Repo/code/data reproducibility freeze: `CH4_PART2_REPRO_FREEZE_20260306.md`
- Writing asset map and citation pack: `CH4_PART3_WRITING_PACK_20260306.md`
- Chapter text and figure synchronization: `CH4_PART4_CHAPTER_TEXT_SYNC_20260306.md`, `CH4_PART5_FIGURE_SYNC_20260306.md`, `CH4_PART6_CHAPTER_FIGURE_INSERT_SYNC_20260306.md`
- Final readiness gate: `CH4_PART7_DISSERTATION_READINESS_GATE_20260306.md`

### 4.6.2 Residual Technical Risk (Explicit)

- Structured `Mayo + Evidence` output remains deferred and is excluded from primary Chapter 4 claims.
- External proxy labels are not fully Mayo-native; external scores remain limitation evidence.

## 4.7 Final Writing Notes

1. Use frozen Pass 5/6/7 artifacts for all Chapter 4 headline numeric claims.
2. Keep external dataset discussion explicitly under robustness limitations.
3. Keep exploratory Pass 8 results in appendix/future-work framing only.
4. Preserve run provenance references whenever final numbers are cited.

## 4.8 Limitations to Explicitly State

1. Single-frame severity grading is not equivalent to full-procedure interpretation.
2. Class-boundary ambiguity (`0<->1`, `1<->2`, `2<->3`) remains an error source.
3. Mode2 controlled scoring fails under this setup and is not used as primary generative lane.
4. On current HyperKvasir proxy labels, external claims are not deployment-ready.
5. The best frozen internal QWK remains below `0.90`, indicating remaining headroom for future data-cleanup-driven improvement.

## 4.9 Chapter Summary and Transition

Chapter 4 now has a frozen, reproducible severity pipeline with clear claim boundaries. On internal LIMUC, the multi-seed generative mode1 lane outperforms the frozen supervised multi-seed baseline in QWK and macro-F1, while controlled mode2 collapses and is retained as a negative-result ablation. External proxy evaluation shows large domain-shift degradation and is reported as limitation evidence rather than generalization proof.  
These frozen findings provide the handoff to Chapter 5, where evidence-grounded and safety-constrained clinical synthesis is developed around a fixed upstream severity component.

