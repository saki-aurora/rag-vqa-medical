# Chapter 4 Reproducibility Freeze (Part 2)

Date: 2026-03-06
Freeze timestamp (UTC): 2026-03-06T14:30:03Z

## 1) Repo Freeze Stamp

- Branch: `LIMUC`
- Commit (HEAD): `24dff5f5542a7ac9fdef31795c64e143d1e68ee6`
- Working tree status at freeze:
  - untracked: `CH4_PART1_SCOPE_FREEZE_20260306.md`
  - no tracked file modifications
- Canonical workspace path:
  - `/home/arcturus/Desktop/thesis/rag-vqa-medical` resolves to `/mnt/hf/thesis/rag-vqa-medical`

## 2) Code Manifest (Tracked Scripts + Git Blob IDs)

These are the scripts that define official Chapter 4 pass execution and reporting.

- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/eval_resnet50_checkpoint.py`  
  blob: `3ba0ba1c40b5069395cc9072ac2bc601367e27da`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/train_resnet50_finetune.py`  
  blob: `c84bc95fb77c6999e2884903dd825cf20ca01ad9`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/train_supervised_backbone.py`  
  blob: `7a1b385cbbd2ca3368a934f787e70e21b291cfda`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/train_vlm_lora_mayo.py`  
  blob: `eda3ea5a9d1826270eed72e9ea4c79dc2afd0d76`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/controlled_vlm_mayo_eval.py`  
  blob: `300ade3b3f60f6f84bec2d132828a69a55fac6d5`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/16_pass5_supervised_multiseed.py`  
  blob: `88f5674f7c4c8e14086dab9205343711af2d0ca3`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/17_pass6_generative_multiseed.py`  
  blob: `6b5e3e81c2c118217b8cddcc704c0990f71d70f7`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/18_pass7_external_validation.py`  
  blob: `111b71b2e941065e60171f4e38700a7f253934b5`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/19_pass8_internal_fusion.py`  
  blob: `7c170622403f6bfde7c67682961c378551fa3c53`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/20_pass8_supervised_push.py`  
  blob: `825fda31eef42f2afc1d6cd31543e7bcdd0d8cad`

## 3) Data/Split Freeze

- Internal metadata (LIMUC):
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/out/metadata/metadata_enriched.csv`
- Split hash used by official runs:
  - `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`
- External proxy metadata (Pass 7):
  - `Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor.csv`
  - summary: `Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor_summary.txt`
  - rows/classes: `n_rows=851`, `class_counts=0:35,1:212,2:471,3:133`

## 4) Official Artifact Lock (For Dissertation)

Primary frozen artifacts:

- Pass 5 supervised (official baseline):
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_metric_summary.csv`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_multiseed_report.json`
  - key QWK mean: `0.8186494575`
- Pass 6 generative (official primary):
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_metric_summary.csv`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_multiseed_report.json`
  - key mode1 QWK mean: `0.8636564898`
- Pass 7 external stress test:
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json`
  - `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_drop_table.csv`
  - key external QWK: ResNet `0.3595966437`, VLM mode1 `0.0`

Exploratory (not headline claims):

- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_internal_fusion_20260306T085817Z/`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_scout_r2_20260306T091539Z/`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass8_supervised_5090_focus_20260306T092204Z/`

## 5) Command Provenance (Executed)

### 5.1 Pass 5 seed training command (from `seed_011.log` first line)

```bash
/home/arcturus/.pyenv/versions/3.12.2/bin/python3.12 \
  /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/train_resnet50_finetune.py \
  --limuc-root /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC \
  --out-dir /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_resnet50_pass5_seed011 \
  --seed 11 --epochs 15 --batch-size 16 --num-workers 4 --lr 0.0003 --weight-decay 0.0001 \
  --device auto --log-every 50 --amp
```

### 5.2 Pass 6 seed training command (from `seed_077_train.log` first line)

```bash
/root/work/venv-vqa/bin/python \
  /root/work/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/train_vlm_lora_mayo.py \
  --data-root /root/work/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC \
  --run-name vlm_lora_objfix_b200_seed077 --seed 77 --model-name Salesforce/blip2-flan-t5-xl \
  --epochs 2 --batch-size 2 --grad-accum 4 --lr 5e-05 --weight-decay 0.0 --max-new-tokens 8 \
  --num-workers 8 --logging-steps 25 --save-steps 400 --save-total-limit 1 \
  --lora-r 8 --lora-alpha 16 --lora-dropout 0.1 --balanced-sampling --force-cuda \
  --label-token-only --class-token-loss-weight 1.0 --template-token-loss-weight 0.0
```

### 5.3 Pass 7 executed commands (from `pass7_external_validation_report.json`)

```bash
/home/arcturus/miniforge3/envs/vlm-lora-gpu/bin/python \
  /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/eval_resnet50_checkpoint.py \
  --meta-csv /mnt/hf/thesis/rag-vqa-medical/Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor.csv \
  --split test \
  --checkpoint /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/finetune_resnet50/best_resnet50.pt \
  --out-dir /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/resnet50_external_eval \
  --limuc-root /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC \
  --batch-size 32 --num-workers 8 --seed 42 --run-name resnet50_external_eval --device cuda

/home/arcturus/miniforge3/envs/vlm-lora-gpu/bin/python \
  /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/controlled_vlm_mayo_eval.py \
  --meta-csv /mnt/hf/thesis/rag-vqa-medical/Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor.csv \
  --split test --model-name Salesforce/blip2-flan-t5-xl \
  --adapter-dir /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_objfix_b200_seed077/lora_adapter \
  --mode both --mode2-strategy sequence_logprob --seed 42 \
  --run-name pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1_vlm_test \
  --out-dir /mnt/hf/thesis/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/vlm_external_eval \
  --log-every 50 --force-cuda
```

### 5.4 Pass 8 provenance (exploratory only)

- Internal fusion output includes `best_candidate_pred_test.csv`, so it was run with `--save-predictions`.
- Supervised 5090 exploratory runs are fully reconstructable from first lines in:
  - `.../pass8_supervised_5090_scout_r2_20260306T091539Z/logs/*.log`
  - `.../pass8_supervised_5090_focus_20260306T092204Z/logs/*.log`
- Example (best scout run):

```bash
/root/work/venv-vqa/bin/python \
  /root/work/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/train_supervised_backbone.py \
  --limuc-root /root/work/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC \
  --out-dir /root/work/rag-vqa-medical/Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/results/swin_t_ce_m_e8_seed011 \
  --run-id swin_t_ce_m_e8_seed011 --seed 11 --epochs 8 --batch-size 16 --num-workers 8 \
  --lr 0.00015 --weight-decay 0.0002 --min-lr 1e-06 --device cuda --log-every 80 \
  --backbone swin_t --loss ce --aug-strength medium --image-size 224 --resize-size 256 \
  --class-weighting balanced --scheduler cosine --label-smoothing 0.0 --early-stop-patience 0 --amp
```

## 6) Deterministic Rerun Templates (Chapter 4)

Set root once:

```bash
export REPO=/mnt/hf/thesis/rag-vqa-medical
cd "$REPO"
```

### 6.1 Rebuild official Pass 5 report

```bash
/home/arcturus/.pyenv/versions/3.12.2/bin/python3.12 \
  "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/16_pass5_supervised_multiseed.py" \
  --limuc-root "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC" \
  --python /home/arcturus/.pyenv/versions/3.12.2/bin/python3.12 \
  --seeds 11,23,42 --epochs 15 --batch-size 16 --num-workers 4 \
  --lr 0.0003 --weight-decay 0.0001 --amp --device auto --tag pass5_supervised
```

### 6.2 Rebuild official Pass 6 report (obj-fix run family)

```bash
/root/work/venv-vqa/bin/python \
  "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/17_pass6_generative_multiseed.py" \
  --limuc-root "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC" \
  --python /root/work/venv-vqa/bin/python \
  --run-prefix vlm_lora_objfix_b200_seed \
  --new-seeds 77 \
  --existing-runs vlm_lora_objfix_b200_seed011,vlm_lora_objfix_b200_seed023 \
  --force-retrain --epochs 2 --batch-size 2 --grad-accum 4 \
  --lr 5e-05 --weight-decay 0.0 --num-workers 8 --save-steps 400 --save-total-limit 1 \
  --balanced-sampling --force-cuda --label-token-only \
  --class-token-loss-weight 1.0 --template-token-loss-weight 0.0 \
  --eval-run-prefix vlm_lora_pass6_mode2_seed --eval-mode2-strategy sequence_logprob \
  --exclude-nonconverged-mode1 --mode1-min-qwk 0.2 --mode1-min-pred-classes 2 --mode1-max-train-loss 3.0 \
  --tag pass6_generative
```

### 6.3 Rebuild official Pass 7 report

```bash
/home/arcturus/miniforge3/envs/vlm-lora-gpu/bin/python \
  "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/18_pass7_external_validation.py" \
  --limuc-root "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC" \
  --python /home/arcturus/miniforge3/envs/vlm-lora-gpu/bin/python \
  --meta-csv "$REPO/Datasets/hyperkvasir/pass7_uc_proxy/metadata/metadata_hyperkvasir_uc_proxy_mayo_floor.csv" \
  --split test --tag pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1 \
  --resnet-batch-size 32 --num-workers 8 --eval-log-every 50 --eval-mode2-strategy sequence_logprob \
  --vlm-run-dir "$REPO/Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/vlm_lora_objfix_b200_seed077" \
  --force-cuda
```

## 7) Quick Verification Commands

```bash
cd /mnt/hf/thesis/rag-vqa-medical

git branch --show-current
git rev-parse HEAD
git status --short

jq '.generated_utc, .task' Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass5_supervised_multiseed_report.json
jq '.generated_utc, .task' Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass6_generative_multiseed_report.json
jq '.generated_utc, .task' Prototyping_reformat/DatasetAnalysis/LIMUC/4_reporting/out/pass7_external_hyperkvasir_uc_proxy_floor_20260306_r1/pass7_external_validation_report.json
```

## 8) Chapter 4 Claim Guardrail (Aligned With Part 1)

- Dissertation headline claims should use frozen Pass 5/6/7 artifacts only.
- Pass 8 stays explicitly exploratory unless promoted via a future freeze update.
- Internal primary KPI remains LIMUC `mode1/test` QWK; external HyperKvasir proxy remains limitation/stress-test evidence.
