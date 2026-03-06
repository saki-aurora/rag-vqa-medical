# Chapter 5 Reproducibility Freeze (Part 2)

Date: 2026-03-06
Freeze timestamp (UTC): 2026-03-06T17:54:37Z

## 1) Repo Freeze Stamp

- Branch: `LIMUC`
- Commit (HEAD): `29ea484bb90fb43dd24c4ac1c0dc7b3bda436d21`
- Canonical workspace path:
  - `/home/arcturus/Desktop/thesis/rag-vqa-medical` resolves to `/mnt/hf/thesis/rag-vqa-medical`

Working tree notes at freeze:
- tracked modifications:
  - `Prototyping_reformat/chapter5_pico_wrapper/scripts/chapter5_completion_audit.py`
  - `Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`
- untracked chapter outputs:
  - `Thesis/markdown/figures/ch5_representations/`
  - `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306/`

## 2) Code Manifest (Local Hashes)

Core package modules:
- `pico_wrapper/schemas.py`: `cf70cc195c63ec2100b9eda74b3d2749f454962d`
- `pico_wrapper/pico_extract.py`: `49bb3aecb9a688d29da25d62b5c738e137bf6e4e`
- `pico_wrapper/kb_ingest.py`: `eb35919eda491c9b0ddb57dea78f2ff6c26803e1`
- `pico_wrapper/retriever.py`: `bb4f71c65295c24fabc0a12126acc3c40df4c3b4`
- `pico_wrapper/synthesis.py`: `a0f153c49dac6fd7d80d54831d5ab9e16b2aa0c8`
- `pico_wrapper/safety.py`: `5a37b2e5f501d35c187dec632cf46c919a49da99`
- `pico_wrapper/wrapper.py`: `e647796fb6e90bb411c5ebc18915d5c07c7295e6`
- `pico_wrapper/ui_support.py`: `181768dd5d35f45b966995f52a6814283b270592`

Core scripts:
- `scripts/make_queryset.py`: `79a4cccc356a009f5a6286c74ee9585a947eed72`
- `scripts/build_kb.py`: `78f1f56c5d14c5c11f498740d6a498614ce940d9`
- `scripts/run_wrapper.py`: `76fd4a77ace4efc09e38c9299d8a2276acf156c2`
- `scripts/run_ui.py`: `a71c355171d67424575242131d18689f0db33131`
- `scripts/eval_pico.py`: `5215d8ca8ae16f8a1ec9b56700c3f1e4528256a1`
- `scripts/eval_retrieval.py`: `3e3dcbb4dd1d256d4c4378f2380cbd17b2ec8f9e`
- `scripts/eval_answers.py`: `cd032b505bdd25b535cf8278e886d39b35acec12`
- `scripts/run_full_pipeline.py`: `5bf5a8ee4a4eab420b35391acd040d66cf1af3f9`
- `scripts/chapter5_completion_audit.py` (local modified): `e2a685c443694bec074dd815e63bcc515198e6a3`

Chapter file (local modified):
- `Thesis/markdown/05_chapter_5_genai_wrapper_pico.md`: `0f59631e93922692f5bfca6f4152fa87f282d82f`

## 3) Data / Queryset Freeze

Query assets:
- `Prototyping_reformat/chapter5_pico_wrapper/data/queries/queries.jsonl` (`n=50`)
- `Prototyping_reformat/chapter5_pico_wrapper/data/queries/pico_gold.jsonl` (`n=20`)
- `Prototyping_reformat/chapter5_pico_wrapper/data/queries/retrieval_gold.jsonl` (`n=10`)

SHA256:
- `queries.jsonl`: `35c538281774cefae5893006f68045bcaff47579a67baa52f8d6ecc6a3e79fc8`
- `pico_gold.jsonl`: `48588d874e41f3d6bd875460e46e1cf9a11fb615a8e170c8b43dca303cd3b4ae`
- `retrieval_gold.jsonl`: `5dbb279a2f2abb227843980d853163a6cc61ec5047e4c0a5323c8830f7e756f4`

Wrapper outputs:
- `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl` (`n=50`)

## 4) Official Artifact Lock (For Dissertation)

Primary frozen artifacts:
- `Prototyping_reformat/chapter5_pico_wrapper/results/pipeline_pass4_latest/pipeline_summary.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_pass4_latest/kb_manifest.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/run_config.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json`
- `Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306/chapter5_completion_report.json`

Supporting exploratory/ablation artifact:
- `Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass3_ablation/retrieval_ablation_summary.tsv`

## 5) Command Provenance (Frozen Pass4)

Command chain source:
- `Prototyping_reformat/chapter5_pico_wrapper/results/pipeline_pass4_latest/pipeline_summary.json`

Executed sequence (all returncode `0`):
1. `make_queryset`
2. `build_kb`
3. `run_wrapper` (backend=`hybrid`, rerank alpha=`0.2`)
4. `eval_pico`
5. `eval_retrieval` (bootstrap iters=`2000`, seed=`42`)
6. `eval_answers` (strict overlap threshold enabled)
7. `completion_audit`

## 6) Deterministic Rerun Template (Chapter 5)

```bash
export REPO=/mnt/hf/thesis/rag-vqa-medical
cd "$REPO"

python Prototyping_reformat/chapter5_pico_wrapper/scripts/run_full_pipeline.py \
  --tag pass4_latest
```

To rebuild only the freeze gate report:

```bash
python Prototyping_reformat/chapter5_pico_wrapper/scripts/chapter5_completion_audit.py \
  --kb_manifest Prototyping_reformat/chapter5_pico_wrapper/results/kb_build_pass4_latest/kb_manifest.json \
  --wrapper_outputs Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl \
  --run_config Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/run_config.json \
  --pico_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/pico_eval.json \
  --retrieval_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/retrieval_eval.json \
  --answer_eval Prototyping_reformat/chapter5_pico_wrapper/results/eval_pass4_latest/answer_eval.json \
  --chapter_md Thesis/markdown/05_chapter_5_genai_wrapper_pico.md \
  --min_queries 20 \
  --out_dir Prototyping_reformat/chapter5_pico_wrapper/results/chapter5_completion_audit_ch5_freeze_20260306 \
  --audit_run_id chapter5_completion_audit_ch5_freeze_20260306
```

## 7) Quick Verification Commands

```bash
cd /mnt/hf/thesis/rag-vqa-medical

git branch --show-current
git rev-parse HEAD
git status --short

wc -l Prototyping_reformat/chapter5_pico_wrapper/data/queries/queries.jsonl \
      Prototyping_reformat/chapter5_pico_wrapper/data/queries/pico_gold.jsonl \
      Prototyping_reformat/chapter5_pico_wrapper/data/queries/retrieval_gold.jsonl \
      Prototyping_reformat/chapter5_pico_wrapper/results/wrapper_eval_pass4_latest/wrapper_outputs.jsonl
```
