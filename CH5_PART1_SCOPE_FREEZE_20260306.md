# Chapter 5 Scope Freeze (Part 1)

Date: 2026-03-06
Scope owner: dissertation drafting freeze for Chapter 5

## 1) Official KPI Lock

Primary KPI bundle (optimization + claim focus):
- PICO required-field macro-F1 (`P/I/C/O + severity anchors`)
- Retrieval Hit@5 and Recall@5
- Citation-grounding safety bundle:
  - citation coverage
  - strict claim support
  - hallucination proxy
  - contradiction proxy
  - refusal rate

Secondary diagnostics:
- PICO all-field macro-F1
- Precision@k and Recall@k at `k=1,3,5`
- Retrieval bootstrap confidence intervals
- completion-audit checklist pass/fail state

Policy:
- Chapter 5 is a wrapper/evidence-layer chapter, not a replacement for Chapter 4 severity validation.
- Chapter 5 headline claims use frozen pass4 artifacts only.

## 2) Official Results to Report (Headline)

These are the frozen, official Chapter 5 numbers for dissertation claims.

### 2.1 PICO extraction (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`

- `n_queries=20`
- Required-field macro-F1: `0.7572418125609615`
- All-field macro-F1: `0.6751992097736779`

Required-field component F1 values:
- Population: `1.0000`
- Intervention: `0.7692`
- Comparator: `0.9362`
- Outcomes: `0.4444`
- Severity anchors: `0.6364`

### 2.2 Retrieval quality (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`

- `n_queries=10`
- @1: precision `0.2000`, recall `0.1000`, hit rate `0.2000`
- @3: precision `0.1667`, recall `0.2500`, hit rate `0.3000`
- @5: precision `0.1600`, recall `0.4500`, hit rate `0.6000`

Bootstrap CI highlights (@5):
- Precision@5 CI: `[0.08, 0.26]`
- Recall@5 CI: `[0.20, 0.70]`
- Hit@5 CI: `[0.30, 0.90]`

### 2.3 Answer quality and grounding (pass4)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`

- Outputs: `50`
- Claims extracted: `142`
- Claims evaluated: `138`
- Policy claims excluded: `4`
- Refusal count: `4` (rate `0.08`)
- Citation coverage: `1.0000`
- Citation correctness (heuristic): `1.0000`
- Claim support (strict): `0.8695652173913043`
- Hallucination proxy: `0.0000`
- Contradiction proxy: `0.0000`
- Citation link integrity: `1.0000`

### 2.4 Completion audit gate (chapter freeze)
Source: `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306/chapter5_completion_report.json`

- Status: `PASS`
- Checklist: `6/6` pass
- Wrapper outputs counted: `50`
- KB chunks/docs: `12 / 3`

## 3) Final Claim Boundary (for writing)

Headline claims allowed:
1. Chapter 5 provides a reproducible, citation-aware PICO wrapper over frozen Chapter 4 boundaries.
2. Required-field PICO extraction and top-5 retrieval show usable internal baseline behavior.
3. Citation-linked synthesis with refusal handling is operational and auditable in frozen pass4 artifacts.

Claims to avoid in headline:
1. Do not claim deployment readiness from current internal KB/gold subset sizes.
2. Do not claim clinician-level semantic correctness from lexical heuristics alone.
3. Do not treat Chapter 5 retrieval/synthesis metrics as a substitute for Chapter 4 model reliability evidence.

## 4) Future-Run Bucket (Deferred)

Deferred for post-freeze work:
- Expand KB beyond sample internal documents.
- Increase retrieval gold size and clinician adjudication.
- Add manual semantic review protocol for claim grounding.
- Add UI-driven prospective evaluation with clinician raters.

## 5) Frozen Artifact Paths

- `Prototyping_reformat/chapter5_pico_wrapper/results/pipeline_pass4_latest/pipeline_summary.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_pass4_latest/kb_manifest.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/run_config.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass3_ablation/retrieval_ablation_summary.tsv`
- `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306/chapter5_completion_report.json`
- `Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`
