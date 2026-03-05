# Chapter 5 Scripts

## Core scripts

- `build_kb.py`: Build chunked KB and retrieval indices.
- `run_wrapper.py`: Run wrapper pipeline over one query or query file.
- `run_ui.py`: Local browser UI with image upload, severity integration, session history, and export.
- `make_queryset.py`: Generate synthetic query/eval sets.
- `eval_pico.py`: PICO extraction metrics.
- `eval_retrieval.py`: Retrieval metrics + bootstrap CI.
- `eval_answers.py`: Citation/claim support checks + strict support metrics.
- `chapter5_completion_audit.py`: PASS/FAIL completion audit.
- `run_full_pipeline.py`: One-command reproducible pipeline for Chapter 5.

## One-command pipeline

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_full_pipeline.py \
  --tag pass4_latest
```

This writes:

- `results/kb_build_<tag>/`
- `results/wrapper_eval_<tag>/`
- `results/eval_<tag>/`
- `results/chapter5_completion_audit_<tag>/`
- `results/pipeline_<tag>/pipeline_summary.json`

## UI launch

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_ui.py \
  --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_pass4_latest/kb_manifest.json \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/ui_pass4_latest \
  --port 8502
```

Then open `http://127.0.0.1:8502`.
