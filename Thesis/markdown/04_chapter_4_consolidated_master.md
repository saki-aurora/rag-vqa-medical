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

`Dataset -> Model -> Controlled Generation -> Optional Retrieval Support -> Fine-Tuning -> Evaluation`

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

### 4.4.1 Task Definition

Input:

- colonoscopy frame
- fixed severity question prompt

Output:

- Mayo score in `{0,1,2,3}`

Optional output extension:

- short evidence phrase constrained to visual findings only

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

## 4.5 Frozen Results and Claim Boundary (LIMUC)

This section is synchronized to the Chapter 4 freeze package and is restricted to frozen Pass 5/6/7 artifacts:
- `CH4_PART1_SCOPE_FREEZE_20260306.md`
- `CH4_PART2_REPRO_FREEZE_20260306.md`
- `CH4_PART3_WRITING_PACK_20260306.md`

### 4.5.1 Primary KPI and Reporting Policy

The primary optimization and reporting KPI is internal LIMUC `mode1/test` QWK. Accuracy, macro-F1, balanced accuracy, and parse rate are treated as secondary companion metrics. External HyperKvasir UC proxy scores are reported as stress-test evidence only and are not used as the model-selection target.

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

![Mode1 convergence QC across seeds (Pass6) (F05)](figures/ch4_representations/F05_ch4_mcnemar_significance_heatmap.png)

*Mode1 convergence QC by seed for Pass 6 generative training.*

### 4.5.3 External Stress-Test (Pass 7 HyperKvasir UC Proxy)

Source artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_drop_table.csv`

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

These exploratory sweeps are informative but remain outside headline claims because none crossed `QWK >= 0.90`.

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
5. Internal `QWK >= 0.90` was not achieved in the frozen chapter evidence.

## 4.9 Chapter Summary and Transition

Chapter 4 now has a frozen, reproducible severity pipeline with clear claim boundaries. On internal LIMUC, the multi-seed generative mode1 lane outperforms the frozen supervised multi-seed baseline in QWK and macro-F1, while controlled mode2 collapses and is retained as a negative-result ablation. External proxy evaluation shows large domain-shift degradation and is reported as limitation evidence rather than generalization proof.  
These frozen findings provide the handoff to Chapter 5, where evidence-grounded and safety-constrained clinical synthesis is developed around a fixed upstream severity component.

## 4.10 Consolidated Delivery Record (Parts 1-8)

This section consolidates all Chapter 4 completion work into this single dissertation source file.
Use this file as the Chapter 4 single source of truth for writing and final edits.

### 4.10.1 Part-by-Part Completion Summary

| Part | Date | Output Artifact | Status | Dissertation-Relevant Outcome |
|---|---|---|---|---|
| Part 1 | 2026-03-06 | `CH4_PART1_SCOPE_FREEZE_20260306.md` | Complete | Locked KPI boundary: internal LIMUC `mode1/test` QWK as primary claim target. |
| Part 2 | 2026-03-06 | `CH4_PART2_REPRO_FREEZE_20260306.md` | Complete | Repo/code/data freeze documented with commit, split hash, and rerun templates. |
| Part 3 | 2026-03-06 | `CH4_PART3_WRITING_PACK_20260306.md`, `CH4_PART3_ASSET_MANIFEST_20260306.csv` | Complete | Writing pack and citation-ready asset map prepared with frozen numbers. |
| Part 4 | 2026-03-06 | `CH4_PART4_CHAPTER_TEXT_SYNC_20260306.md` | Complete | Chapter 4 narrative synchronized to Pass 5/6/7 claim boundary. |
| Part 5 | 2026-03-06 | `CH4_PART5_FIGURE_SYNC_20260306.md` | Complete | Regenerated/staged Chapter 4 figures `F02` to `F06` from frozen artifacts. |
| Part 6 | 2026-03-06 | `CH4_PART6_CHAPTER_FIGURE_INSERT_SYNC_20260306.md` | Complete | Inserted `F02` to `F06` into Chapter 4 markdown text. |
| Part 7 | 2026-03-06 | `CH4_PART7_DISSERTATION_READINESS_GATE_20260306.md` | Complete | Readiness gate passed: value/path/figure validation complete. |
| Part 8 | 2026-03-06 | `CH4_PART8_FINAL_WRITING_PASS_20260306.md` | Complete | Final dissertation-style Chapter 4 writing pass delivered. |

### 4.10.2 Frozen Headline Results (Official)

Internal official baseline (Pass 5 supervised, seeds 11/23/42):
- Accuracy: `0.737643`
- Macro-F1: `0.667330`
- Balanced accuracy: `0.670907`
- QWK: `0.818649`
- 95% CI QWK: `[0.807920, 0.830582]`

Internal official generative primary (Pass 6 mode1, seeds 11/23/77):
- Accuracy: `0.781930`
- Macro-F1: `0.727920`
- Balanced accuracy: `0.736292`
- QWK: `0.863656`
- Parse rate: `1.000000`
- 95% CI QWK: `[0.862382, 0.865836]`

Controlled ablation (Pass 6 mode2):
- Accuracy: `0.548636`
- Macro-F1: `0.177135`
- Balanced accuracy: `0.250000`
- QWK: `0.000000`
- Parse rate: `1.000000`

External stress-test (Pass 7 HyperKvasir UC proxy):
- `resnet50_supervised` QWK: `0.828762 -> 0.359597` (delta `-0.469165`)
- `vlm_lora_mode1` QWK: `0.862752 -> 0.000000` (delta `-0.862752`)
- `vlm_lora_mode1` parse rate: `1.0 -> 0.0`

### 4.10.3 Claim Guardrail (Locked)

Allowed headline claims:
1. On internal LIMUC, Pass 6 mode1 multi-seed exceeds the frozen Pass 5 supervised multi-seed baseline on QWK and macro-F1.
2. Mode2 is retained as a controlled negative-result ablation.
3. External HyperKvasir proxy degradation is limitation evidence (domain shift and label mismatch), not generalization proof.

Disallowed headline claims:
1. Do not claim `QWK >= 0.90` achieved on frozen Chapter 4 evidence.
2. Do not claim external deployment readiness from current proxy labels.
3. Do not mix exploratory Pass 8 results into frozen headline tables.

### 4.10.4 Reproducibility Freeze Snapshot

Frozen snapshot from Part 2:
- Freeze timestamp (UTC): `2026-03-06T14:30:03Z`
- Branch at freeze: `LIMUC`
- Commit at freeze: `24dff5f5542a7ac9fdef31795c64e143d1e68ee6`
- LIMUC split hash: `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`
- Canonical repo path: `/mnt/hf/thesis/rag-vqa-medical`

### 4.10.5 Final Artifact Bundle for Dissertation Citations

Primary metric/report artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_mode1_qc.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_drop_table.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`

Staged Chapter 4 figures used in this markdown:
- `Thesis/markdown/figures/ch4_representations/F02_ch4_core_metric_comparison.png`
- `Thesis/markdown/figures/ch4_representations/F03_ch4_radar_profile.png`
- `Thesis/markdown/figures/ch4_representations/F04_ch4_remission_slice_comparison.png`
- `Thesis/markdown/figures/ch4_representations/F05_ch4_mcnemar_significance_heatmap.png`
- `Thesis/markdown/figures/ch4_representations/F06_ch4_confusion_panel.png`

Exploratory appendix-only artifacts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_internal_fusion_20260306T085817Z/pass8_internal_fusion_candidates.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_scout_r2_20260306T091539Z/pass8_supervised_all_runs.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_focus_20260306T092204Z/pass8_supervised_all_runs.csv`

### 4.10.6 Final Validation Status

Final checks completed for this chapter package:
- Numeric consistency check against frozen sources: `26/26` pass.
- Embedded figure link resolution (`F02` to `F06`): `5/5` pass.
- Cited artifact path resolution in chapter text: pass.
- Readiness decision: `PASS` for dissertation drafting under frozen Pass 5/6/7 boundary.

### 4.10.7 Deferred Future Work (Not in Headline Claims)

- Data cleanup and relabel pass focused on class-boundary ambiguity (`0<->1`, `1<->2`).
- Re-run top supervised and generative pipelines on cleaned metadata.
- Evaluate on a Mayo-compatible external dataset with native `0/1/2/3` labels.

## 4.11 Chapter 4 Representation Pack (What, Where, Aim)

All Chapter 4 representations are now consolidated under:
- `Thesis/markdown/figures/ch4_representations/`

### 4.11.1 Main-Text Representations

| Rep ID | File | Where to place/use | Aim of representation |
|---|---|---|---|
| R4.1 | `F02_ch4_core_metric_comparison.png` | Section `4.5.2` (internal results) | Show headline internal KPI comparison across Pass 5 baseline, Pass 6 mode1, and Pass 6 mode2. |
| R4.2 | `F03_ch4_radar_profile.png` | Section `4.5.2` (internal results) | Show mean metrics with uncertainty (95% CI) to support stability claims, not only point estimates. |
| R4.3 | `F05_ch4_mcnemar_significance_heatmap.png` | Section `4.5.2` (mode1 stability) | Show seed-level QC convergence for retained mode1 lane. |
| R4.4 | `F04_ch4_remission_slice_comparison.png` | Section `4.5.3` (external stress test) | Show internal-to-external degradation pattern and domain-shift limitation. |
| R4.5 | `F06_ch4_confusion_panel.png` | Section `4.5.4` (diagnostics) | Show class-wise error structure across supervised, mode1, and mode2 lanes. |

### 4.11.2 Optional Diagnostic/Appendix Representations

| Rep ID | File | Where to place/use | Aim of representation |
|---|---|---|---|
| R4.6 | `pass5_supervised_metric_ci.png` | Appendix or extended `4.5.2` | Detailed CI plot for Pass 5 supervised aggregate metrics. |
| R4.7 | `pass6_generative_metric_ci.png` | Appendix or extended `4.5.2` | Detailed CI plot for Pass 6 mode1/mode2 aggregate metrics. |
| R4.8 | `pass5_supervised_confusion_aggregate.png` | Appendix diagnostics | Show supervised-only confusion pattern at class level. |
| R4.9 | `pass6_generative_confusion_aggregate_mode1.png` | Appendix diagnostics | Show retained mode1 confusion pattern at class level. |
| R4.10 | `pass6_generative_confusion_aggregate_mode2.png` | Appendix diagnostics | Show controlled mode2 collapse pattern as negative-result evidence. |
| R4.11 | `generative_pred_distribution.png` | Appendix diagnostics | Show predicted-class distribution sanity for generative outputs. |

### 4.11.3 Table/Data Representations (Source Tables)

| Rep ID | File | Where to place/use | Aim of representation |
|---|---|---|---|
| R4.12 | `pass5_supervised_metric_summary.csv` | Table source for `4.5.2` | Canonical Pass 5 frozen metrics and CIs. |
| R4.13 | `pass6_generative_metric_summary.csv` | Table source for `4.5.2` | Canonical Pass 6 mode1/mode2 frozen metrics and CIs. |
| R4.14 | `pass6_generative_mode1_qc.csv` | Table source for mode1 QC statements | Seed-level QC evidence (`3/3` pass). |
| R4.15 | `pass7_external_drop_table.csv` | Table source for `4.5.3` | Canonical internal-to-external drop evidence. |
| R4.16 | `pass7_external_validation_report.json` | Supporting citation for `4.5.3` | Full external evaluation context and metric provenance. |
| R4.17 | `pass6_generative_mcnemar_mode1_vs_mode2.csv` | Appendix statistical test table | Paired significance support for mode1 vs mode2 behavior gap. |
| R4.18 | `pass5_supervised_per_class_recall_summary.csv` | Appendix per-class table | Supervised class-wise recall profile. |
| R4.19 | `pass6_generative_per_class_recall_summary.csv` | Appendix per-class table | Generative class-wise recall profile by lane. |
| R4.20 | `ch4_frozen_internal_metrics.csv` | Supplemental table source | Compact thesis-figure-ready internal metric export. |
| R4.21 | `ch4_pass7_drop_subset.csv` | Supplemental table source | Compact thesis-figure-ready external-drop export. |
| R4.22 | `ch4_pass6_mode1_qc.csv` | Supplemental table source | Compact thesis-figure-ready mode1 QC export. |
