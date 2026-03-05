# Chapter 5 PICO Wrapper Workspace

This directory contains the Chapter 5 implementation workspace for a
PICO-driven GenAI wrapper on top of Chapter 4 severity outputs.

## Scope of Part 1 (Scaffold + Contracts)

Implemented in this step:

- Package scaffold under `pico_wrapper/`
- Strict data contracts in `pico_wrapper/schemas.py`
- Safety policy helpers in `pico_wrapper/safety.py`
- Run-id and JSON I/O utilities in `pico_wrapper/utils_io.py`
- Placeholders for upcoming modules (`pico_extract`, `kb_ingest`,
  `retriever`, `synthesis`, `wrapper`)
- Standard-library tests in `tests/`

## Scope of Part 2 (KB Ingestion + Retrieval Core)

Implemented in this step:

- `pico_wrapper/kb_ingest.py`:
  - scans `.md`/`.txt` KB docs
  - section-aware chunking with offsets
  - persisted `chunks.jsonl`, `kb_manifest.json`
  - index backend:
    - TF-IDF (if `scikit-learn` is available)
    - keyword overlap fallback (pure Python)
- `pico_wrapper/retriever.py`:
  - PICO-driven query composition
  - backend-aware retrieval over persisted index
- `scripts/build_kb.py` CLI
- Added sample KB docs in `data/kb/sample_docs/`
- Added tests: `test_kb_ingest.py`, `test_retriever.py`

Build KB command:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/build_kb.py \
  --kb_dir Prototyping_reformat/chapter5_pico_wrapper/data/kb \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest
```

## Scope of Part 3 (Wrapper Baseline Pipeline)

Implemented in this step:

- `pico_wrapper/pico_extract.py`:
  - rule-based PICO extraction
  - severity-anchor extraction (MES/Mayo/UCEIS terms)
  - optional LLM-mode hook with safe fallback
- `pico_wrapper/synthesis.py`:
  - deterministic structured synthesis
  - citation-linked claims
  - severity summary integration
  - uncertainty + limitations + standard disclaimer
- `pico_wrapper/wrapper.py`:
  - orchestrates extract -> retrieve -> synthesize
  - dosing refusal logic via safety rules
  - explicit LLM-mode fallback tracking
- `scripts/run_wrapper.py` CLI

Run wrapper command:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_wrapper.py \
  --query "Does biologic therapy improve remission in adults with UC over 12 weeks?" \
  --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_latest
```

Alternative (using `--kb_index_dir` as requested by acceptance criteria):

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_wrapper.py \
  --query "What evidence supports remission improvement with biologics?" \
  --kb_index_dir Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_index \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_latest
```

Local browser UI (no extra dependencies):

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_ui.py \
  --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json \
  --port 8502
```

Then open `http://127.0.0.1:8502`.

## Scope of Part 4 (Evaluation Suite + Querysets)

Implemented in this step:

- `scripts/make_queryset.py`:
  - creates synthetic physician queries (`queries.jsonl`)
  - creates PICO gold subset (`pico_gold.jsonl`)
  - creates retrieval gold subset (`retrieval_gold.jsonl`)
- `scripts/eval_pico.py`:
  - computes field-level precision/recall/F1 for P/I/C/O + severity anchors
- `scripts/eval_retrieval.py`:
  - computes precision@k / recall@k / hit-rate@k on labeled retrieval subset
- `scripts/eval_answers.py`:
  - citation coverage
  - citation correctness heuristic
  - hallucination proxy (unsupported claims)
  - manual rubric template generation for human scoring

Part 4 run commands:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/make_queryset.py \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/data/queries \
  --n_queries 50 \
  --n_pico_gold 20

python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_wrapper.py \
  --query_file Prototyping_reformat/chapter5_pico_wrapper/data/queries/queries.jsonl \
  --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json \
  --retrieval_k 5 \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest

python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_pico.py \
  --pico_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/pico_gold.jsonl \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest

python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_retrieval.py \
  --retrieval_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/retrieval_gold.jsonl \
  --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json \
  --k_values 1,3,5 \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest

python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_answers.py \
  --wrapper_outputs Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest/wrapper_outputs.jsonl \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest
```

## Scope of Part 5 (Audit + Thesis Write-Up Sync)

Implemented in this step:

- `scripts/chapter5_completion_audit.py`:
  - validates KB manifest/index status
  - validates wrapper run coverage (minimum query count)
  - validates eval artifact presence/readability
  - validates Chapter 5 markdown references to Chapter 5 artifacts + frozen Chapter 4 run id
  - writes PASS/FAIL report under `results/<audit_run_id>/`
- Chapter 5 markdown file:
  - `Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`
  - includes architecture, setup, results tables, limitations, and artifact pointers

Audit command:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/chapter5_completion_audit.py \
  --kb_manifest Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json \
  --wrapper_outputs Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest/wrapper_outputs.jsonl \
  --pico_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/pico_eval.json \
  --retrieval_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/retrieval_eval.json \
  --answer_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest/answer_eval.json \
  --chapter_md Thesis/markdown/05_chapter_5_genai_wrapper_pico.md \
  --min_queries 20
```

## Scope of Pass 3 (Retrieval/Safety/Eval Upgrades)

Implemented:

- hybrid retrieval backend (keyword + TF-IDF + semantic-LSA fusion),
- optional reranker with configurable pool/alpha,
- low-evidence abstention and escalation-aware safety behavior,
- stricter answer checks (strict support + contradiction proxy),
- retrieval bootstrap confidence intervals.

Current tuned default:

- `rerank_alpha=0.20` (selected from pass3 ablation).

Pass 3 refreshed artifacts:

- `results/kb_build_pass3_latest/`
- `results/wrapper_eval_pass3_latest/`
- `results/eval_pass3_latest/`

## Scope of Pass 4 (UI/Productization + Automation)

Implemented:

- `scripts/run_ui.py` upgraded with:
  - image upload path for severity context,
  - severity predictor integration:
    - lookup mode from Chapter 4 prediction CSV (default),
    - command hook mode (`--severity_predict_cmd`) for external predictors,
  - safety alert banner for refusal/abstention/caution cases,
  - side-by-side output panes (PICO, claims, evidence, full JSON),
  - session history panel backed by `ui_sessions.jsonl`,
  - export/report support (`.json` + `.md`) with persisted report files.
- `scripts/run_full_pipeline.py`:
  - one-command reproducible Chapter 5 pipeline:
    `make_queryset -> build_kb -> run_wrapper -> eval_pico -> eval_retrieval -> eval_answers -> completion_audit`,
  - writes deterministic `<tag>` artifact directories and a pipeline summary JSON.
- new helper module:
  - `pico_wrapper/ui_support.py`
- new tests:
  - `tests/test_ui_support.py`

Pass 4 pipeline command:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_full_pipeline.py \
  --tag pass4_latest
```

Pass 4 artifacts:

- `results/kb_build_pass4_latest/`
- `results/wrapper_eval_pass4_latest/`
- `results/eval_pass4_latest/`
- `results/chapter5_completion_audit_pass4_latest/`
- `results/pipeline_pass4_latest/pipeline_summary.json`
- UI run/report outputs:
  - `results/ui_pass4_latest/`

## Directory Layout

- `pico_wrapper/`: core package
- `scripts/`: CLI scripts (build/run/eval)
- `data/`: local KB and query assets
- `results/`: reproducible run outputs
- `tests/`: schema and utility tests

## Notes

- Chapter 4 is treated as frozen evidence.
- The baseline Chapter 5 pipeline runs without external API keys.
- Current status includes Parts 1-5 (scaffold, retrieval, wrapper, evaluation, audit + chapter sync).
