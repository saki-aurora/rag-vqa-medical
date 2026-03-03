# LIMUC Chapter 4 Notebook Run Order

This is the full notebook order for Chapter 4 (baselines + generative).

## Environment and config checks (completed in this workspace)

- Kernel to use: `vqa-rag`
- Python: `3.11.13`
- GPU: `NVIDIA GeForce RTX 3090` with CUDA available
- Required packages present in `vqa-rag`: `torch`, `transformers`, `accelerate`, `peft`, `datasets`, `evaluate`, `bitsandbytes`, `sentencepiece`
- LIMUC data present:
  - `Datasets/LIMUC/train_and_validation_sets`
  - `Datasets/LIMUC/test_set`
  - `Datasets/LIMUC/patient_based_classified_images`
- Metadata already prepared:
  - `0_dataset_prep/out/metadata/metadata_enriched.csv` (`11276` rows)
  - split hash: `d71d3864f86c77641c029b050ab26b74e62f8425940c4131fd708a964b78008b`
- Notebook config alignment:
  - `vlm_lora_finetune_mayo.ipynb` kernel metadata set to `vqa-rag`
  - `vlm_structured_mayo_evidence_eval.ipynb` now includes `TRANSFORMERS_NO_TF` and `USE_TF=0`

## One-time shell setup before runs

Run from repo root:

```bash
cd /home/aristotle/Desktop/rag-vqa-medical
conda activate vqa-rag

# Full runs:
export MAX_SAMPLES=0
export FORCE_CUDA=1

# Optional speed/memory knobs:
export BATCH_SIZE=1
export RUN_ID=chapter4_final_$(date -u +%Y%m%dT%H%M%SZ)
# export VLM_MODEL=Salesforce/blip2-flan-t5-xl
```

Use this execution pattern for each notebook:

```bash
jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=vqa-rag \
  <NOTEBOOK_PATH>
```

Persisted outputs are written under `*/results/<run_name>/` and include `run_meta.json` with `run_id` and `timestamp_utc`.

## Preferred LoRA path (CLI, reproducible)

Use the standalone trainer instead of relying on notebook state:

```bash
cd /home/arcturus/Desktop/thesis/rag-vqa-medical
/home/arcturus/miniforge3/envs/vqa-rag/bin/python \
  Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/train_vlm_lora_mayo.py \
  --run-name vlm_lora_finetune_mayo_balanced_full_$(date -u +%Y%m%d) \
  --epochs 3 \
  --batch-size 2 \
  --grad-accum 4 \
  --lr 5e-5 \
  --balanced-sampling \
  --force-cuda
```

Monitor progress:

```bash
tail -f Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/results/<run_name>/train.log
```

This writes full Chapter-4-compatible artifacts:
- `pred_test.csv`, `pred_val.csv`
- `metrics_test.json`, `metrics_val.json`
- `confusion_test.png`, `pred_label_histogram.png`
- `parser_audit_samples.csv`
- `training_history.csv`, `training_summary.json`
- `lora_config.json`, `lora_param_count.json`, `run_meta.json`

## Required notebook order for Chapter 4

1. `Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb`
2. `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/resnet50_frozen_logreg.ipynb`
3. `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/vit_frozen_logreg.ipynb`
4. `Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/clip_linear_baseline.ipynb`
5. `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_resnet50.ipynb`
6. `Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_vit_or_swin.ipynb`
7. `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_zero_shot_mayo.ipynb`
8. `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_lora_finetune_mayo.ipynb`

## Optional notebook (recommended for "generative AI" framing)

9. `Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_structured_mayo_evidence_eval.ipynb`

This adds structured output:
- `Mayo: <0|1|2|3>`
- `Evidence: <short visual phrase>`

## Run all in one command block

```bash
cd /home/aristotle/Desktop/rag-vqa-medical
conda activate vqa-rag
export MAX_SAMPLES=0
export FORCE_CUDA=1
export BATCH_SIZE=1
export RUN_ID=chapter4_batch_$(date -u +%Y%m%dT%H%M%SZ)

for nb in \
  Prototyping_reformat/DatasetAnalysis/LIMUC/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/resnet50_frozen_logreg.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/vit_frozen_logreg.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/1_frozen_encoders/clip_linear_baseline.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_resnet50.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/2_supervised_finetuning/finetune_vit_or_swin.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_zero_shot_mayo.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_lora_finetune_mayo.ipynb \
  Prototyping_reformat/DatasetAnalysis/LIMUC/3_vlm_severity/vlm_structured_mayo_evidence_eval.ipynb
do
  echo "Running $nb"
  jupyter nbconvert --to notebook --execute --inplace \
    --ExecutePreprocessor.kernel_name=vqa-rag \
    "$nb" || break
done
```
