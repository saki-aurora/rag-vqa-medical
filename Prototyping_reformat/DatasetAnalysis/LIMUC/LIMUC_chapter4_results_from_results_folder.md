# LIMUC Chapter 4 Results Snapshot (From `results/` Folders)

This file summarizes persisted artifacts under:
`Prototyping_reformat/DatasetAnalysis/LIMUC/**/results/*`

Ground-truth reporting exports for Chapter 4 are generated in:
`Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/`

## 1) Run Coverage

Source:
- `4_reporting/out/results_index.csv`

Current scan summary:
- Run folders scanned: **17**
- Full runs (`test_rows == 1686`): **11**
- Smoke/subset runs: **2**
- Canonical split hash (full runs): `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`

## 2) Required Full-Run Check

Source:
- `4_reporting/out/chapter4_missing_runs.md`

Required Chapter-4 full runs:
- `finetune_resnet50` ✅
- `finetune_vit_or_swin` ✅
- `resnet50_frozen_logreg` ✅
- `vit_frozen_logreg` ✅
- `clip_linear_baseline` ✅
- `vlm_zero_shot_mayo` ✅
- `vlm_lora_finetune_mayo` ✅

Missing required runs:
- **None**

## 3) Main Results Table (Full Runs Only)

Source:
- `4_reporting/out/chapter4_final_main_table.csv`

This table includes supervised baselines plus persisted generative runs (zero-shot, LoRA run, and controlled-mode outputs) with `test_rows == 1686`.

## 4) Controlled Generative Outputs Persisted

### 4.1 Mode 1 (Free generation + strict parser)

- Zero-shot Mode-1 artifact:
  - `3_vlm_severity/results/vlm_zero_shot_mode1_freegen_from_results_20260301/`
- LoRA Mode-1 artifact:
  - `3_vlm_severity/results/vlm_lora_mode1_freegen_from_results_20260301/`

Each includes:
- `pred_test.csv`
- `metrics_test.json`
- `confusion_test.png`
- `pred_label_histogram.png`
- `parser_audit_samples.csv`
- `run_meta.json`

### 4.2 Mode 2 (Controlled label scoring)

- Zero-shot Mode-2 full run:
  - `3_vlm_severity/results/vlm_zero_shot_mode2_label_scoring_full_20260301/`
  - Includes `p0,p1,p2,p3` confidence columns in `pred_test.csv`.
- LoRA Mode-2 persisted proxy output:
  - `3_vlm_severity/results/vlm_lora_mode2_label_scoring_from_results_20260301/`
  - Run metadata explicitly notes this is derived from persisted LoRA predictions because adapter weight files are not currently available in this workspace.

## 5) Clinical and Statistical Tables

Sources:
- Remission slice: `4_reporting/out/chapter4_remission_slice_table.csv`
- McNemar pairs: `4_reporting/out/chapter4_mcnemar_pairs_from_results.csv`
- Best supervised vs best generative pair: `4_reporting/out/chapter4_paired_significance.csv`

## 6) Figures for Chapter 4

Final figures are in:
- `4_reporting/out/figures/`

Key files:
- `class_distribution_by_split.png`
- `confusion_test_finetune_resnet50.png`
- `confusion_test_vlm_zero_shot_mode2_label_scoring_full_20260301.png`
- `pred_label_histogram_vlm_zero_shot_mode2_label_scoring_full_20260301.png`
- `pred_label_histogram_vlm_lora_mode2_label_scoring_from_results_20260301.png`

## 7) Residual Risk

- `4_reporting/out/lora_load_proof.txt` currently reports LoRA adapter evidence as `INCOMPLETE` for `vlm_lora_finetune_mayo` because adapter weight files are absent in the persisted folder.
- Structured `Mayo + Evidence` remains a deferred/negative-result lane for Chapter 4 main claims.
