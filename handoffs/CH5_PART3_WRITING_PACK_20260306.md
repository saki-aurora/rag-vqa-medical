# Chapter 5 Writing Pack (Part 3)

Date: 2026-03-06
Input freezes consumed:
- `CH5_PART1_SCOPE_FREEZE_20260306.md`
- `CH5_PART2_REPRO_FREEZE_20260306.md`

## 1) Purpose

This pack converts frozen Chapter 5 evidence into a writing-ready asset map and citation-ready metric blocks.

Machine-readable manifest:
- `CH5_PART3_ASSET_MANIFEST_20260306.csv`

## 2) Frozen Numbers to Cite (Primary Claims)

### 2.1 PICO extraction (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`

- Required-field macro-F1: `0.7572418125609615`
- All-field macro-F1: `0.6751992097736779`
- `n_queries=20`

Required-field F1 values:
- Population: `1.0000`
- Intervention: `0.7692`
- Comparator: `0.9362`
- Outcomes: `0.4444`
- Severity anchors: `0.6364`

### 2.2 Retrieval quality (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`

- @1: precision `0.2000`, recall `0.1000`, hit `0.2000`
- @3: precision `0.1667`, recall `0.2500`, hit `0.3000`
- @5: precision `0.1600`, recall `0.4500`, hit `0.6000`
- `n_queries=10`

### 2.3 Answer grounding quality (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`

- Outputs: `50`
- Claims extracted: `142`
- Claims evaluated: `138`
- Refusal count: `4` (rate `0.08`)
- Citation coverage: `1.0000`
- Strict claim support: `0.8695652173913043`
- Hallucination proxy: `0.0000`
- Contradiction proxy: `0.0000`

### 2.4 Completion gate
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306/chapter5_completion_report.json`

- Audit status: `PASS`
- Checklist pass: `6/6`
- KB: `3` docs / `12` chunks

## 3) Table Order for Chapter 5 Draft

Recommended insertion order (all paths already in manifest):

1. `T5_01` PICO aggregate metrics (pass4)
2. `T5_02` retrieval @k metrics + CI (pass4)
3. `T5_03` answer grounding/safety metrics (pass4)
4. `T5_04` wrapper configuration snapshot (pass4)
5. `T5_05` completion audit status
6. `T5_06` retrieval ablation summary (supporting)

## 4) Figure Order for Chapter 5 Draft

Use these frozen staged figures:

1. `F5_01` PICO field precision/recall/F1: `Thesis/markdown/figures/generated/F07_ch5_pico_field_precision_recall_f1.png`
2. `F5_02` retrieval @k + 95% CI: `Thesis/markdown/figures/generated/F08_ch5_retrieval_at_k_curve_with_ci.png`
3. `F5_03` retrieval ablation comparison: `Thesis/markdown/figures/generated/F09_ch5_retrieval_ablation_comparison.png`
4. `F5_04` answer quality and grounding KPI panel: `Thesis/markdown/figures/generated/F10_ch5_answer_quality_grounding_kpi_panel.png`

## 5) Ready-to-Paste Claim Blocks

### Block A: Core wrapper result

"In the frozen pass4 Chapter 5 run, the PICO wrapper achieved required-field macro-F1 0.7572, with top-5 retrieval hit rate 0.60 and recall 0.45, while maintaining citation-linked output structure across all evaluated responses."

### Block B: Safety-grounding statement

"The answer layer preserved citation coverage at 1.0 with hallucination and contradiction proxies at 0.0 in the frozen lexical evaluation protocol, while refusal handling remained active for policy-sensitive prompts (4/50 outputs)."

### Block C: Boundary statement

"These Chapter 5 results establish reproducible wrapper behavior on an internal benchmark and do not imply external clinical deployment readiness without broader evidence curation and clinician semantic adjudication."

## 6) What Is Ready vs Deferred

Ready for dissertation writing now:
- Scope freeze (Part 1)
- Repro/code freeze (Part 2)
- Writing asset map and metric blocks (Part 3, this file)

Deferred (future work):
- Larger retrieval gold set and multi-rater adjudication.
- Expanded external guideline/trial KB.
- Prospective clinician-in-the-loop UI study.

## 7) Verification

Quick check command:

```bash
cd /mnt/hf/thesis/rag-vqa-medical
python - <<'PY'
import pandas as pd
m=pd.read_csv('CH5_PART3_ASSET_MANIFEST_20260306.csv')
print(m[['asset_id','asset_type','status','exists']].to_string(index=False))
PY
```
