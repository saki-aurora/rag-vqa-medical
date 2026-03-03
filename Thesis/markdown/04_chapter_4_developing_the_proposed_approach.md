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

## 4.5 Generative AI Technique Results (LIMUC)

This section reports the implemented Chapter-4 generative technique as persisted artifacts under `Prototyping_reformat/DatasetAnalysis/LIMUC/**/results/*`, with final reporting outputs centralized in `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/`.

### 4.5.1 Generative Severity Scoring Formulation

The generative severity task is defined as follows: input is a single colonoscopy frame, output is a Mayo endoscopic subscore in `{0,1,2,3}`, and evaluation uses ordinal and clinically aligned metrics.

Mode 1 (baseline generative decoding) uses a fixed prompt and strict parser:

`You are a medical imaging assistant. Rate ulcerative colitis severity using the Mayo endoscopic subscore. Output EXACTLY in this format: SCORE: <0|1|2|3>`

Parsing rule is deterministic and constrained: extract the first valid digit `0-3` after the literal token `SCORE:`; otherwise mark invalid (`parse_ok = false`) and include parse rate in final metrics.

### 4.5.2 Controlled Decoding via Label Scoring

Mode 2 implements controlled generative decoding by scoring candidate label tokens `{0,1,2,3}` at the next-token position immediately following the prefix `SCORE:` and selecting argmax. The per-class confidence vector `(p0,p1,p2,p3)` is computed with softmax over those candidate logits.

This method keeps the model in generative-token space while removing free-form output variance and parser brittleness. In persisted outputs, Mode 2 is therefore always parsable (`parse_rate = 1.0`) and yields deterministic label outputs with explicit confidence columns.

Persisted Mode-2 full runs:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mode2_label_scoring_full_20260301`
- run id: `vlm_zero_shot_mode2_label_scoring_full_20260301_20260301T020047Z`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_zero_shot_mode2_label_sampling_full_20260302`
- run id: `vlm_zero_shot_mode2_label_sampling_full_20260302_20260302T065858Z`

### 4.5.3 LoRA Adaptation

The LoRA adaptation lane is persisted as:
- legacy run: `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo`
- run id: `vlm_lora_finetune_mayo_20260227T214308Z`
- finalized balanced full run: `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_finetune_mayo_balanced_full_20260303`
- run id: `vlm_lora_finetune_mayo_balanced_full_20260303_20260303T080754Z`
- base model: `Salesforce/blip2-flan-t5-xl`
- training config recorded in run metadata: `epochs=3`, `batch_size=2`, `grad_accum=4`, `lr=5e-5`, `balanced_sampling=true`

Observed full-test results for the finalized balanced LoRA run are:
- accuracy `0.7200`
- macro-F1 `0.6816`
- balanced accuracy `0.6985`
- QWK `0.8231`
- parse rate `1.0000`

Compared with persisted zero-shot (`vlm_zero_shot_mayo`, QWK `0.0000`) and controlled zero-shot Mode-2 sampling (`vlm_zero_shot_mode2_label_sampling_full_20260302`, QWK `0.0061`), balanced LoRA improves all key ordinal metrics and becomes the best generative run in Chapter 4.

### 4.5.4 Results and Analytics

Final results-only master table:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_final_main_table.csv`

Required run coverage audit:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_audit_results_index.csv`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_missing_or_invalid_runs.md`

All required full runs (`n=1686`) are present in `results/`: `finetune_resnet50`, `finetune_vit_or_swin`, `resnet50_frozen_logreg`, `vit_frozen_logreg`, `clip_linear_baseline`, `vlm_zero_shot_mayo`, `vlm_lora_finetune_mayo`; plus finalized controlled and adapted generative runs including `vlm_zero_shot_mode2_label_sampling_full_20260302` and `vlm_lora_finetune_mayo_balanced_full_20260303`.

Clinical remission slice (`0-1` vs `2-3`) table:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_remission_slice_table.csv`

Paired significance:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_paired_significance.csv`
- best supervised vs best generative pair (current table): `finetune_resnet50` vs `vlm_lora_finetune_mayo_balanced_full_20260303`
- McNemar p-value: `2.813208e-04`

Bootstrap confidence intervals (results-only, from persisted `pred_test.csv`):
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_metric_ci_bootstrap.csv`
- includes 95% CI for accuracy, macro-F1, and QWK for each full run.

Figure references (final path):
- class distribution: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/class_distribution_by_split.png`
- supervised confusion: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/confusion_test_finetune_resnet50.png`
- best generative confusion: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/confusion_test_vlm_lora_finetune_mayo_balanced_full_20260303.png`
- generative label histograms:
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/figures/pred_label_histogram_vlm_lora_finetune_mayo_balanced_full_20260303.png`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/generative_pred_distribution.png`

### 4.5.5 Failure Analysis

Qualitative error artifact:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/chapter4_qualitative_error_table.csv`

Coverage summary:
- `correct_both`: sampled representative cases where both systems are correct.
- `supervised_correct_generative_wrong`: dominant failure mode, concentrated around ordinal confusion and moderate/severe boundaries.
- `generative_correct_supervised_wrong`: minority slice, retained for balanced error interpretation.

Error pattern remains primarily ordinal-boundary confusion (especially `0<->1` and `2<->3`) rather than degenerate single-class collapse. This aligns with the improved balanced accuracy and QWK in the finalized LoRA run while preserving room for boundary-focused calibration in future work.

### 4.5.6 Scope Decision on Evidence Generation

Structured two-field generation (`Mayo + Evidence`) remains out of scope for Chapter 4 final claims unless parsing is robust at scale. Current evidence-format behavior remains weak and is therefore treated as a negative result path. The clinically grounded evidence/citation wrapper is deferred to Chapter 5 under the PICO-oriented GenAI wrapper design, where retrieval grounding is explicit.

## 4.6 Chapter-4 Completion Status

### 4.6.1 Completed Items

- [x] Results-only run audit and index (`chapter4_audit_results_index.csv`) under `4_reporting/out/`
- [x] Final Chapter-4 master comparison table from full runs (`chapter4_final_main_table.csv`)
- [x] Required full baseline run persisted for `clip_linear_baseline`
- [x] Controlled generative Mode 1 and Mode 2 code paths implemented and persisted
- [x] Controlled non-degenerate generative full run persisted (`vlm_zero_shot_mode2_label_sampling_full_20260302`)
- [x] Balanced LoRA full generative run persisted and selected as best generative (`vlm_lora_finetune_mayo_balanced_full_20260303`)
- [x] Remission slice and paired significance exports (`chapter4_remission_slice_table.csv`, `chapter4_paired_significance.csv`)
- [x] Figure exports and qualitative error table in final `4_reporting/out/` location
- [x] Chapter text updated with final run IDs/timestamps and artifact references

### 4.6.2 Residual Technical Risk (Explicit)

- [x] LoRA adapter proof exported and validated under `4_reporting/out/lora_load_proof.txt` (`Status: PASS`), with hashed adapter payload and run metadata linkage. Training provenance is persisted via both `training_history.csv`/`training_summary.json` and run metadata (`epochs`, `lr`, LoRA config, adapter path).
- [x] Structured `Mayo + Evidence` is intentionally retained as a deferred/negative-result lane for Chapter 4 and excluded from primary severity claims.

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

Chapter 4 now has a reproducible supervised reference and a defensible generative severity pathway on LIMUC. The finalized evidence shows that balanced LoRA improves substantially over zero-shot and becomes the best generative run, while the supervised anchor (`finetune_resnet50`) remains statistically stronger in paired comparison.  
These results form the handoff to Chapter 5, where evidence-grounded and scenario-driven QA can be expanded with retrieval support under tighter reliability controls.
