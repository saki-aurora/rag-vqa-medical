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

## 4.5 Result Compilation Plan

This chapter will compile values directly from notebook output cells and saved run summaries in the notebook workflows.

### 4.5.1 Mandatory Comparison Table

Table 4.x (main model comparison) should include:

- best supervised classifier
- zero-shot VLM severity run
- LoRA-adapted VLM severity run

Columns:

- accuracy, macro-F1, balanced accuracy, QWK, MAE, RMSE, parse/unknown rate

### 4.5.2 Mandatory Clinical Slice Table

Table 4.x (remission slice):

- remission sensitivity/recall
- remission specificity
- remission F1
- remission accuracy

### 4.5.3 Mandatory Figures

- confusion matrix (best supervised model)
- confusion matrix (best generative model)
- class support distribution
- ordinal error profile or predicted-vs-true severity relation

## 4.6 Implementation Notes

1. Core Chapter 4 evidence should remain tied to LIMUC first, then optional external robustness.
2. Notebook prompt and parser logic must be fixed before final comparison to avoid protocol drift.
3. Dependency upgrades should be minimal and logged (for example, adding `peft`).
4. If run artifacts are regenerated, report split hash and run metadata alongside final tables.

## 4.7 Limitations to Explicitly State

1. Single-frame severity grading is not equivalent to full-procedure interpretation.
2. Class imbalance still affects moderate/severe reliability.
3. Generative format control reduces but does not eliminate reasoning errors.
4. External clinical deployment claims remain out of scope.

## 4.8 Chapter Summary and Transition

Chapter 4 delivers a controlled generative UC severity pipeline anchored to strong supervised reliability and ordinally aligned evaluation.  
This provides the technical base for the next chapter, where scenario-driven, evidence-aware question answering is expanded toward PICO-oriented decision support.

