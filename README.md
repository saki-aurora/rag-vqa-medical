# RAG + VQA for GI Endoscopy (Thesis Workspace)

Author: Sarthak Kaushik  
Supervisors: Dr. Sabah Mohammad, Dr. Jinan Fiadhi  
Affiliation: Lakehead University

## What this repository is
This is a research workspace for GI-endoscopy AI experiments, focused on:

- medical VQA (visual question answering),
- RAG-style VQA evaluation on GI data,
- segmentation and severity grading support experiments,
- thesis writing artifacts and references.

It is notebook-first (not a packaged Python library). Most experiments are organized by dataset and stage under `Prototyping_reformat/DatasetAnalysis/`.

## Start here (short version)
If you are new to this repo, do this first:

1. Create the environment from `environment.yml`.
2. Open Jupyter Lab.
3. Work in `Prototyping_reformat/DatasetAnalysis/` (this is the active track).
4. Run `0_dataset_prep` notebooks before model notebooks.
5. Use the per-dataset report files (`*.md`) to see current persisted results.

## Active vs legacy content

- Active experimentation track: `Prototyping_reformat/`
- Older experimentation track: `Prototyping/`
- Archived snapshots: `legacy/`
- Thesis draft content: `Thesis/` and `Thesis/markdown/`

If you only want current work, prioritize `Prototyping_reformat/`.

## Repository map

| Path | Purpose |
|---|---|
| `Prototyping_reformat/DatasetAnalysis/` | Main experiment pipelines, grouped by dataset |
| `Prototyping_reformat/cloud_setup_scripts/` | GPU-specific setup scripts (A100/H100/H200/GB200/5090) |
| `Prototyping/` | Earlier prototype notebooks and outputs |
| `Thesis/markdown/` | Chapter drafts, references, edits tracking |
| `Papers/` | Collected papers used in the thesis |
| `Datasets/` | Dataset pointers/download notes |
| `Timeline/` | Milestone/timeline assets |

## Environment setup

### Option 1: Conda (recommended for local reproducibility)

```bash
conda env create -f environment.yml
conda activate vqa-rag
jupyter lab
```

Notes:

- `environment.yml` is a GPU-oriented environment (PyTorch + Transformers + RAPIDS stack included).
- Python in this environment is 3.11.

### Option 2: GPU cloud bootstrap scripts

For cloud machines, use the script matching your GPU:

- `Prototyping_reformat/cloud_setup_scripts/setup_vqa_rag_a100.sh`
- `Prototyping_reformat/cloud_setup_scripts/setup_vqa_rag_h100.sh`
- `Prototyping_reformat/cloud_setup_scripts/setup_vqa_rag_h200.sh`
- `Prototyping_reformat/cloud_setup_scripts/setup_vqa_rag_gb200.sh`
- `Prototyping_reformat/cloud_setup_scripts/setup_vqa_rag_blackwell_5090.sh`

These scripts:

- create a virtual environment,
- install PyTorch and dependencies,
- register a Jupyter kernel (`vqa-rag`),
- write a `vqa-rag.env` file with cache and dataset env vars.

## Data setup by dataset track

All active tracks are under `Prototyping_reformat/DatasetAnalysis/`.

| Dataset track | How data is sourced | Where data/artifacts are expected | First notebook |
|---|---|---|---|
| `HyperKvasir` | Hugging Face (`sahilur/hyper-kvasir-labeled-images`) | `HyperKvasir/0_dataset_prep/out/` | `0_dataset_prep/01_build_metadata_images_and_manifests.ipynb` |
| `Kvasir_VQA` | Hugging Face (`SimulaMet-HOST/Kvasir-VQA`) | `Kvasir_VQA/0_dataset_prep/out/` | `0_dataset_prep/01_build_metadata_images_and_manifests.ipynb` |
| `Kvasir_VQA_x1` | Hugging Face (`SimulaMet/Kvasir-VQA-x1`) + host images | `Kvasir_VQA_x1/0_dataset_prep/out/` | `0_dataset_prep/01_build_metadata_images_and_manifests.ipynb` |
| `ImageCLEF_MEDVQA_GI_2023` | Manual zip download | Put zips in `ImageCLEF_MEDVQA_GI_2023/0_dataset_prep/dataset_download_zip/` | `0_dataset_prep/00_unpack_dataset_zips.ipynb` |
| `Kvasir_SEG` | Local extracted data or zip (`kvasir-seg.zip`), optional download path in notebook helpers | `Kvasir_SEG/0_dataset_prep/` and `Kvasir_SEG/0_dataset_prep/out/` | `0_dataset_prep/01_build_metadata_images_and_masks.ipynb` |
| `LIMUC` | Local dataset folders (train/val/test structure) | `Datasets/LIMUC/` by default | `LIMUC/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb` |

Additional dataset links are listed in `Datasets/Readme.md`.

## Notebook execution flow

General rule for all dataset tracks:

1. Run `0_dataset_prep/*` first.
2. Run model notebooks in numbered order (`1_*`, `2_*`, ...).
3. Run analysis/report notebooks last.

### Example: Kvasir_VQA_x1 (main modern VQA/RAG track)

Recommended order:

1. `Kvasir_VQA_x1/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb`
2. `Kvasir_VQA_x1/0_dataset_prep/02_validate_splits_and_integrity.ipynb`
3. `Kvasir_VQA_x1/1_dataset_analysis/01_explore_dataset.ipynb`
4. `Kvasir_VQA_x1/1_dataset_analysis/02_metrics_and_answer_normalization.ipynb`
5. `Kvasir_VQA_x1/2_modeling/...` (choose model family)
6. `Kvasir_VQA_x1/2_modeling/12_eval_reporting/01_unified_eval_tables_figures.ipynb`

Most Kvasir_VQA_x1 modeling notebooks expect:

```bash
export KVASIR_VQA_X1_ROOT=/absolute/path/to/Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1
```

### Non-interactive notebook execution

```bash
jupyter nbconvert --to notebook --execute \
  Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb
```

## Where to find current results quickly

Each active dataset track has a summary report markdown file:

- `Prototyping_reformat/DatasetAnalysis/HyperKvasir/HyperKvasir.md`
- `Prototyping_reformat/DatasetAnalysis/ImageCLEF_MEDVQA_GI_2023/ImageCLEF_MEDVQA_GI_2023.md`
- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA/Kvasir_VQA.md`
- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1/Kvasir_VQA_x1.md`
- `Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_bias_proven/Kvasir_VQA_bias_proven.md`
- `Prototyping_reformat/DatasetAnalysis/Kvasir_SEG/Kvasir_SEG.md`
- `Prototyping_reformat/DatasetAnalysis/LIMUC/LIMUC.md`

These summarize persisted artifacts, metrics, and reproducibility notes.

## Output and versioning conventions

- Generated outputs usually go to `out/` directories inside each dataset track.
- Lightweight summaries often go to `results/` directories.
- `out/` is generally git-ignored to keep the repository manageable.

Check `.gitignore` and `.gitattributes`:

- large artifacts (`*.pt`, `*.npz`, etc.) are ignored or tracked with Git LFS rules,
- `Datasets/` is treated as re-creatable local data.

If you need large tracked artifacts, run:

```bash
git lfs install
git lfs pull
```

## Thesis writing artifacts

Main writing workspace:

- `Thesis/markdown/00_abstract.md`
- `Thesis/markdown/01_chapter_1_introduction.md`
- `Thesis/markdown/02_chapter_2_survey_of_vqa_techniques.md`
- `Thesis/markdown/03_chapter_3_investigating_existing_vqa_techniques_across_gi_endoscopy_datasets_v2.md`
- `Thesis/markdown/refs.md`
- `Thesis/markdown/TOC.md`

## Troubleshooting

- CUDA/PyTorch mismatch: use the GPU-specific setup script for your hardware.
- `KVASIR_VQA_X1_ROOT` error: set the env var before running x1 modeling notebooks.
- ImageCLEF zip not found: place both official zips in `ImageCLEF_MEDVQA_GI_2023/0_dataset_prep/dataset_download_zip/`.
- Hugging Face throttling/timeouts: retry with local snapshot options used in dataset prep notebooks.
- Missing data in `LIMUC`: verify expected subfolders under `Datasets/LIMUC/` (`train_and_validation_sets`, `test_set`, `patient_based_classified_images`).

## Scope and safety

This repository is for research and educational use.  
It is not a clinical product and must not be used for direct medical decision-making.
