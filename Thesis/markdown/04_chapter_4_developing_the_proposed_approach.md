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

## 4.5 Persisted Results Snapshot (LIMUC)

This section summarizes persisted outputs under `Prototyping_reformat/DatasetAnalysis/LIMUC/**/results/*` plus reporting tables generated in `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables`.

### 4.5.1 Dataset Reproducibility Summary

- Total images: `11,276`
- Patients: `564`
- Split counts: train `8,669`, val `921`, test `1,686`
- Label counts (Mayo 0/1/2/3): `6105 / 3052 / 1254 / 865`
- Split hash: `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`

### 4.5.2 Main Test-Set Comparison (Persisted Full Runs, `n=1686`)

Source table:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_main_comparison_table.csv`

| Model (run folder) | Accuracy | Macro-F1 | Balanced Acc | QWK | MAE | RMSE | Parse Rate |
|---|---:|---:|---:|---:|---:|---:|---:|
| `resnet50_frozen_logreg` | 0.6198 | 0.5346 | 0.5420 | 0.6834 | 0.4324 | 0.7367 | n/a |
| `vit_frozen_logreg` | 0.6910 | 0.6192 | 0.6419 | 0.7620 | 0.3458 | 0.6503 | n/a |
| `finetune_resnet50` | **0.7527** | **0.6800** | **0.6858** | **0.8428** | **0.2533** | **0.5149** | n/a |
| `finetune_vit_or_swin` | 0.7129 | 0.6675 | 0.6649 | 0.7642 | 0.3126 | 0.6137 | n/a |
| `vlm_zero_shot_mayo` | 0.5486 | 0.1771 | 0.2500 | 0.0000 | 0.6987 | 1.1557 | 1.0000 |

Interpretation:
- The best reliability anchor remains **`finetune_resnet50`**.
- The best currently persisted full generative run is **`vlm_zero_shot_mayo`**, which still collapses to class-0 behavior.
- A full persisted LoRA run is not yet available in `results/` (only smoke run exists).

### 4.5.3 LoRA Status in Persisted Results

Persisted LoRA artifacts currently available:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo_smoke_20260222`

Current persisted LoRA status:
- test rows: `16` (smoke/subset only)
- accuracy: `0.7500`
- balanced accuracy: `0.3333`
- QWK: `0.0`

Interpretation:
- This smoke run is useful for pipeline sanity checks but is not valid for the main Chapter-4 full test-set comparison.

### 4.5.4 Remission Slice and Paired Significance (From `results/`)

Source tables:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_remission_slice_from_results.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_mcnemar_pairs_from_results.csv`

Key remission slice results (`0-1` vs `2-3`):
- `finetune_resnet50`: remission accuracy `0.9484`, sensitivity `0.9762`, specificity `0.8182`, remission F1 `0.9689`
- `vlm_zero_shot_mayo`: remission accuracy `0.8238`, sensitivity `1.0000`, specificity `0.0000`, remission F1 `0.9034`

Key McNemar gap:
- `finetune_resnet50` vs `vlm_zero_shot_mayo`: `n01=168`, `n10=512`, `chi2_cc=173.0132`, `p=1.63e-39`

### 4.5.5 Qualitative Error and Ablation Artifacts

Generated artifacts:
- qualitative error table: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_qualitative_error_table.csv`
- qualitative coverage: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_qualitative_error_table_coverage.csv`
- LoRA ablation table: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/tables/chapter4_lora_ablation_table.csv`

Current qualitative coverage snapshot:
- `correct_both`: available `757`, sampled `4`
- `supervised_correct_generative_wrong`: available `512`, sampled `4`
- `generative_correct_supervised_wrong`: available `168`, sampled `4`

Current ablation status:
- only one LoRA run is currently persisted and it is smoke (`n=16`); full-run ablation remains pending.

### 4.5.6 Figure Assets and Captions (Ready for Thesis Import)

- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/figures/confusion_test_finetune_resnet50.png`
  - caption: Confusion matrix for best supervised UC severity model on LIMUC test set.
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/figures/confusion_test_vlm_zero_shot_mayo.png`
  - caption: Confusion matrix for zero-shot generative Mayo scoring on LIMUC test set.
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/results/figures/pred_label_histogram_vlm_zero_shot_mayo.png`
  - caption: Predicted-label distribution for zero-shot generative Mayo scoring (class-collapse diagnostic).

### 4.5.7 Structured Generative Output (Mayo + Evidence)

Structured notebook test summary (final output cell):
- Parse rate: `0.0563`
- Full-set accuracy: `0.0314`
- Answered-only accuracy: `0.5579`
- Answered macro-F1: `0.1791`
- Answered QWK: `0.0`
- Evidence present rate: `0.0`

Interpretation:
- The structured prompt is not yet producing reliable two-field outputs; most predictions fail strict parsing and evidence extraction.
- Persisted structured `results/` artifacts are still pending for this pathway.

### 4.5.8 Key Chapter-4 Findings from Current Evidence

1. The supervised pipeline clearly outperforms generative baselines on ordinal reliability.
2. Zero-shot VLM scoring is reproducibly weak for Mayo grading under the current prompt/parser protocol.
3. The current LoRA run does not yet improve over zero-shot and needs targeted retraining/tuning.
4. A controlled generative format (`Mayo + Evidence`) is feasible in design but not yet operationally robust.

## 4.6 What Remains to Complete Chapter 4

### 4.6.1 Must-Complete Technical Tasks

- [ ] Re-run and persist full artifacts for:
  - `1_frozen_encoders/clip_linear_baseline.ipynb`
  - `3_vlm_severity/vlm_lora_finetune_mayo.ipynb`
- [ ] Save structured evaluation outputs to disk (`summary_metrics.json`, `pred_val.csv`, `pred_test.csv`) for strict reproducibility.
- [x] Generate a unified comparison table CSV and import it into the thesis chapter.
- [x] Export confusion matrices for:
  - best supervised model (`finetune_resnet50`)
  - best generative model (current `vlm_zero_shot_mayo` or improved LoRA run)
- [x] Add remission-oriented slice (`0-1` vs `2-3`) from `pred_test.csv` files.
- [x] Add a paired significance subsection (McNemar and/or bootstrap CI) for supervised vs generative gap.

### 4.6.2 Writing Tasks to Finalize This Chapter

- [ ] Replace the phrase "current runs" with finalized run IDs and timestamps.
- [x] Add figure references and captions for confusion matrices and class distribution.
- [x] Add one qualitative error table with representative false negatives and false positives.
- [x] Add a short ablation table (prompt format, LoRA rank/LR/epochs, parsing strictness).

## 4.7 Implementation Notes for Final Pass

1. Keep LIMUC as primary evidence and clearly mark any external dataset as robustness-only.
2. Freeze prompt/parsing rules before final metrics extraction to avoid protocol drift.
3. Log environment and model IDs in run metadata for each final artifact.
4. Report split hash in the chapter whenever final numbers are cited.

## 4.8 Limitations to Explicitly State

1. Single-frame severity grading is not equivalent to full-procedure interpretation.
2. Class imbalance continues to affect moderate/severe reliability.
3. Generative format constraints reduce but do not eliminate clinically unsafe outputs.
4. External deployment claims remain out of scope.

## 4.9 Chapter Summary and Transition

Chapter 4 now has a reproducible supervised reference and a baseline generative severity pathway on LIMUC. The evidence currently supports a strong supervised advantage and highlights concrete engineering gaps to close for generative adaptation (LoRA and structured evidence output).  
These results form the handoff to Chapter 5, where evidence-grounded and scenario-driven QA can be expanded with retrieval support under tighter reliability controls.
