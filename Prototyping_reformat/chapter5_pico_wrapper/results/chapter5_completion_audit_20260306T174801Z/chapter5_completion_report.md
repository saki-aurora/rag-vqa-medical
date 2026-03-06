# FAIL

## Chapter 5 Completion Checklist
- ✅ KB index built and manifest valid
- ✅ Wrapper ran on at least N=20 queries
- ✅ PICO evaluation file exists and is readable
- ✅ Retrieval evaluation file exists and is readable
- ✅ Answer evaluation file exists and is readable
- ❌ Chapter 5 markdown exists and references frozen Chapter 5 artifacts + frozen Chapter 4 boundary

## Key Paths
- KB manifest: `/mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_pass4_latest/kb_manifest.json`
- Wrapper outputs: `/mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl`
- PICO eval: `/mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`
- Retrieval eval: `/mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`
- Answer eval: `/mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`
- Chapter markdown: `/mnt/hf/thesis/rag-vqa-medical/Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`

## Summary
- wrapper outputs counted: `50`
- kb chunks: `12`
- kb docs: `3`

## Missing / Fixes Required
- Chapter markdown missing reference token: Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl

## Minimal Punch List
1. Build KB index:
   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/build_kb.py --kb_dir Prototyping_reformat/chapter5_pico_wrapper/data/kb --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest`
2. Run wrapper on query set:
   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_wrapper.py --query_file Prototyping_reformat/chapter5_pico_wrapper/data/queries/queries.jsonl --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json --retrieval_k 5 --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest`
3. Generate evaluations:
   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_pico.py --pico_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/pico_gold.jsonl --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`
   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_retrieval.py --retrieval_gold Prototyping_reformat/chapter5_pico_wrapper/data/queries/retrieval_gold.jsonl --manifest_path Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_latest/kb_manifest.json --k_values 1,3,5 --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`
   `python Prototyping_reformat/chapter5_pico_wrapper/scripts/eval_answers.py --wrapper_outputs Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_latest/wrapper_outputs.jsonl --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/eval_latest`
4. Update Chapter 5 markdown artifact references and rerun audit.

- frozen Chapter 5 artifact tokens:
  - `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`
  - `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`
  - `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`
  - `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl`
- frozen Chapter 4 boundary tokens:
  - `CH4_PART1_SCOPE_FREEZE_20260306.md`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
