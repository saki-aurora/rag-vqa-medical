# AGENT HANDOFF - NEXT CHAT (2026-03-06)

## TL;DR
- Branch: `LIMUC`
- Current HEAD: `38f9a2ad80b846ffef9bcf68d141924d1141df78`
- No training/eval process is currently running.
- Pass 6 is complete and strong on internal LIMUC.
- Pass 7 (external HyperKvasir UC proxy) is complete and shows major domain-shift drop.
- Pass 7 scripts are present but still untracked (not committed yet).

---

## Current Git State
- `git status --short` currently shows untracked files:
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/eval_resnet50_checkpoint.py`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/18_pass7_external_validation.py`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/__pycache__/eval_resnet50_checkpoint.cpython-312.pyc`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/__pycache__/18_pass7_external_validation.cpython-312.pyc`
- Recommendation before commit: remove `__pycache__` artifacts, commit the two new `.py` scripts.

---

## Pass 6 (Internal LIMUC) - Final Status
Source: `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_multiseed_report.json`

- Mode1 QC: `3 pass / 0 fail`
  - `vlm_lora_objfix_b200_seed011`
  - `vlm_lora_objfix_b200_seed023`
  - `vlm_lora_objfix_b200_seed077`
- Mode1 aggregate (mean):
  - accuracy: `0.7819296164491893`
  - macro_f1: `0.7279199995261996`
  - qwk: `0.8636564897888555`
  - parse_rate: `1.0`
- Mode2 aggregate (mean):
  - accuracy: `0.5486358244365361`
  - macro_f1: `0.17713519724243584`
  - qwk: `0.0`
  - parse_rate: `1.0`

Pass 6 run-level mode1 table source:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_lora_mode1_seed_runs.csv`

---

## External Dataset Prepared for Pass 7
Zip available:
- `Datasets/hyperkvasir/hyper-kvasir-labeled-images.zip` (~3.7GB)

Prepared external proxy metadata:
- `Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor.csv`
- Summary file:
  - `Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor_summary.txt`

Summary values:
- `n_rows=851`
- `class_counts=0:35, 1:212, 2:471, 3:133`
- `missing_image_ids=0`

Mapping policy used:
- HyperKvasir UC findings -> Mayo-like 0/1/2/3
- In-between labels are floor-mapped:
  - `0-1 -> 0`
  - `1-2 -> 1`
  - `2-3 -> 2`

Important: this is a proxy external set and label mapping is imperfect relative to native Mayo labels.

---

## Pass 7 Script Additions (Uncommitted)
Added scripts:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/eval_resnet50_checkpoint.py`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/18_pass7_external_validation.py`

Notes:
- Syntax was previously validated via `py_compile`.
- `18_pass7_external_validation.py` orchestrates:
  - supervised external eval (ResNet checkpoint)
  - generative external eval (VLM mode1/mode2)
  - internal-vs-external drop table/report

---

## Pass 7 Execution (Completed)
Successful run tag:
- `pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1`

Run root:
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/`

Key outputs:
- `.../pass7_external_drop_table.csv`
- `.../pass7_external_validation_report.json`
- `.../pass7_external_validation_report.md`
- `.../resnet50_external_eval/metrics_test.json`
- `.../vlm_external_eval/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1_vlm_test/mode1_free_generation/metrics_test.json`
- `.../vlm_external_eval/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1_vlm_test/mode2_label_scoring/metrics_test.json`

Executed with interpreter:
- `/home/arcturus/miniforge3/envs/vlm-lora-gpu/bin/python`

Reason:
- `conda run -n <env>` resolved to pyenv python in this machine; explicit env python path was required for compatible `peft/transformers`.

---

## Pass 7 Results (Internal vs External)
Source: `.../pass7_external_validation_report.json`

Generated UTC:
- `2026-03-06T07:50:59.997725+00:00`

Internal references (from LIMUC artifacts):
- `resnet50_supervised`:
  - accuracy `0.7633451957295374`, macro_f1 `0.6694020252436621`, qwk `0.8287620300398653`
- `vlm_lora_mode1`:
  - accuracy `0.7811387900355872`, macro_f1 `0.7201470808012675`, qwk `0.8627519281519653`, parse_rate `1.0`
- `vlm_lora_mode2`:
  - accuracy `0.5486358244365361`, macro_f1 `0.1771351972424358`, qwk `0.0`, parse_rate `1.0`

External (HyperKvasir UC proxy):
- `resnet50_supervised`:
  - accuracy `0.4336075205640423`, macro_f1 `0.3874229862099845`, qwk `0.35959664371448996`
- `vlm_lora_mode1`:
  - accuracy `0.041128084606345476`, macro_f1 `0.019751693002257337`, qwk `0.0`, parse_rate `0.0`
- `vlm_lora_mode2`:
  - accuracy `0.041128084606345476`, macro_f1 `0.019751693002257337`, qwk `0.0`, parse_rate `1.0`

Interpretation:
- Internal performance remains strong (Pass 6).
- External generalization on this proxy set is poor, especially VLM.
- This is expected under heavy domain shift + non-native Mayo label mapping.

---

## Known Operational Notes
- Prior detached `nohup` launch from tool context did not persist reliably; interactive session worked.
- For reliable launches, use direct command and keep session attached (or run directly in user shell).
- No active pass processes now:
  - `18_pass7_external_validation.py`
  - `controlled_vlm_mayo_eval.py`
  - `17_pass6_generative_multiseed.py`
  - `train_vlm_lora_mayo.py`

---

## Suggested Next Steps (for next Codex chat)
1. Clean and commit Pass 7 code additions:
   - remove pycache files
   - commit two new scripts on `LIMUC`
2. Decide objective explicitly:
   - If target is near `0.9`, optimize internal LIMUC only (realistic with further sweeps/ensembles).
   - If target is external robustness, acquire a true Mayo-compatible external dataset (native 0/1/2/3 labels).
3. If internal 0.9 push starts now:
   - supervised backbone upgrade + sweep + seed ensemble + calibration
   - keep current Pass 6 mode1 as generative baseline anchor

---

## Quick Verify Commands
Check no jobs running:
```bash
pgrep -fa "18_pass7_external_validation.py|controlled_vlm_mayo_eval.py|train_vlm_lora_mayo.py|17_pass6_generative_multiseed.py" | grep -v "bash -c" || true
```

Check Pass 7 final report:
```bash
cat Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json
```

Check repo state:
```bash
git branch --show-current
git rev-parse HEAD
git status --short
```
