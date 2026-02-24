# Chapter 4 Work Plan — Generative Vision-Language Modeling for Ulcerative Colitis Severity

This plan is written as an **execution checklist** (coding + experiments + analysis artifacts) that an AI coding agent (e.g., Codex) can follow.

## Chapter 4 objective

Build and evaluate **generative AI** techniques to **rate Ulcerative Colitis (UC) endoscopic severity** (Mayo 0–3) using VQA-style methods, and compare them against strong discriminative baselines.

**Primary deliverables for the thesis chapter**
- A reproducible dataset + split protocol for UC severity.
- Baselines: supervised classifier(s) + zero-shot VLM.
- Proposed method(s): **parameter-efficient fine-tuning (LoRA/QLoRA)** of a VLM for **severity VQA**, plus optional multi-task “rubric” prompting.
- Full evaluation bundle: **accuracy, macro-F1, balanced accuracy, QWK, MAE/RMSE**, per-class metrics, confusion matrices, and clinical slices (remission vs active disease).
- Error analysis and ablation studies (prompting, decoding constraints, class imbalance handling).
- Figures + tables ready to drop into the thesis.

---

## 0) Repo structure to create (recommended)

Create a clean `chapter4_uc_severity/` module to avoid mixing with earlier experiments:

```text
chapter4_uc_severity/
  README.md
  configs/
    limuc.yaml
    hyperkvasir_uc.yaml
    models.yaml
    prompts.yaml
  data/
    raw/                  # NOT committed
    processed/            # NOT committed
    splits/               # committed (csv with IDs only)
  scripts/
    00_download_prepare_data.py
    01_make_splits.py
    02_train_classifier.py
    03_eval_classifier.py
    04_eval_vlm_zeroshot.py
    05_finetune_vlm_lora.py
    06_eval_vlm_finetuned.py
    07_calibration.py
    08_error_analysis.py
    09_make_figures_tables.py
  src/
    datasets.py
    prompts.py
    metrics.py
    modeling_classifier.py
    modeling_vlm.py
    calibration.py
    utils.py
  outputs/
    runs/
    tables/
    figures/
```

**Versioning rule:** commit *code + configs + split manifests + tables/figures*, but **never commit raw images**.

---

## 1) Data strategy (most important decision)

### 1.1 Use the best public dataset for UC severity

**Primary dataset (recommended): LIMUC** (Mayo 0–3, large enough to train models properly).
- Use **patient-level split** if patient IDs are provided (avoid leakage).
- Keep an external evaluation option (HyperKvasir UC subset) for domain shift.

**Secondary dataset (optional): HyperKvasir UC subset**
- Useful as a *cross-dataset generalization* test.
- HyperKvasir UC grade labels can be ambiguous; treat it as **out-of-domain robustness**, not as the main training label source.

### 1.2 Data preparation tasks

**Script: `00_download_prepare_data.py`**
- Download/unpack dataset(s) into `data/raw/`.
- Build a single canonical dataframe:
  - `image_id`, `image_path`, `label_mayo` (0–3), `patient_id` (if available), `source_dataset`.
- Validate label counts and missing labels.
- Export to `data/processed/uc_severity_master.csv`.

**Script: `01_make_splits.py`**
- Create splits with a fixed seed and log it.
- If `patient_id` exists: split by patient (GroupShuffleSplit).
- Save:
  - `data/splits/limuc_train.csv`, `limuc_val.csv`, `limuc_test.csv`
  - optionally `hyperkvasir_test.csv`
- Save class distribution report per split.

### 1.3 Class imbalance handling (must-do)

Implement **at least 2** of these and compare:
- Class-weighted cross entropy.
- Focal loss (gamma sweep).
- WeightedRandomSampler / oversampling in dataloader.
- Mild augmentation policies (careful with color; endoscopy is color-sensitive).

Store each choice as a config flag and report results.

---

## 2) Define the UC severity “VQA task”

The key Chapter 4 requirement is **generative VQA-style scoring**, not only a classifier.

### 2.1 Canonical VQA question(s)

Create 1–3 templates (keep them stable for fair comparison):

- Q1: `What is the Mayo endoscopic subscore (0,1,2,3) for this colonoscopy image? Answer with a single digit.`
- Q2: `Classify ulcerative colitis severity using the Mayo endoscopic score (0–3). Respond only with 0, 1, 2, or 3.`
- Q3 (optional for robustness): `Which Mayo score best describes the endoscopic severity? (0=normal,1=mild,2=moderate,3=severe). Output only the number.`

### 2.2 Output normalization (critical)

**Implement strict parsing** so generative models are comparable:

`parse_mayo(answer_text) -> {0,1,2,3, None}`

Rules:
- If a digit 0–3 exists → take first valid.
- Else map common words:
  - `normal/remission` → 0
  - `mild` → 1
  - `moderate` → 2
  - `severe` → 3
- Else return None → count as **unknown / abstain**.

Report:
- unknown rate
- accuracy on *answered* subset and on *full* subset (treat unknown as wrong)

**File:** `src/metrics.py` and `src/prompts.py`

---

## 3) Baseline 1 — Strong discriminative classifier(s)

Even though Chapter 4 focuses on generative VLMs, you need a **strong supervised baseline** as the reliability anchor.

### 3.1 Models to run (minimum)

**Run at least 2:**
- ResNet-50 (ImageNet init, full fine-tune)
- ViT-B/16 or Swin-T (ImageNet init, full fine-tune)

Optional (stronger, if compute allows):
- ConvNeXt-T/S
- EfficientNetV2

### 3.2 Ordinal-aware variants (recommended)

Because Mayo is ordinal, implement at least one ordinal technique:
- **Ordinal regression head** (cumulative logits) OR
- **CORAL** loss OR
- Train classifier as regression (0–3) + round, then compare.

### 3.3 Training protocol (must be reproducible)

**Script: `02_train_classifier.py`**
- Deterministic seed
- Mixed precision
- Early stopping on validation **macro-F1** or **QWK**
- Save:
  - best checkpoint
  - per-epoch metrics CSV
  - confusion matrix on test

**Script: `03_eval_classifier.py`**
- Evaluate on test split.
- Output:
  - `outputs/tables/classifier_main_metrics.csv`
  - `outputs/figures/confmat_resnet50.png`, etc.

---

## 4) Baseline 2 — Zero-shot VLM severity VQA

Goal: show that naive zero-shot is not sufficient (consistent with your Chapter 3 narrative), unless a model is unusually strong.

### 4.1 Models to evaluate (choose based on what you can run)

Pick **2–4** from your existing toolchain to avoid pipeline churn:
- BLIP-2 (if already integrated)
- Qwen2.5-VL (if already integrated)
- MedGemma (if already integrated)
- LLaVA-Med (if you have it available)

### 4.2 Evaluation design

**Script: `04_eval_vlm_zeroshot.py`**
- For each image:
  - run model with fixed prompt Q1
  - store raw output text
  - parse → predicted label
- Save:
  - `outputs/runs/vlm_zeroshot_<model>.jsonl` (image_id, prompt, raw_answer, parsed_label)
  - `outputs/tables/vlm_zeroshot_metrics.csv`

Metrics:
- accuracy, macro-F1, balanced accuracy
- QWK, MAE/RMSE
- unknown rate

---

## 5) Proposed method — LoRA/QLoRA fine-tuning of a VLM for Mayo scoring

This is the “core contribution” section for Chapter 4.

### 5.1 Choose the finetuning target model

Selection criteria:
- open weights (so it’s reproducible)
- stable HF ecosystem (Transformers + PEFT)
- works with endoscopy images reasonably

**Recommended practical path:**
- Start with **MedGemma LoRA** if you already used it successfully on Kvasir-VQA-x1.
- Alternatively use Qwen2.5-VL LoRA if that stack is already installed.

### 5.2 Training dataset format

Create a supervised instruction-style dataset:
- Input: (image, prompt)
- Target: `"0"` / `"1"` / `"2"` / `"3"`

Optional: add a second target field for explanation (see 6.2), but keep the *primary* run label-only for stable metrics.

### 5.3 Finetuning tasks

**Script: `05_finetune_vlm_lora.py`**
- Implement LoRA adapters over:
  - language layers (q_proj, v_proj, k_proj, o_proj)
  - optionally cross-attention layers (if model has)
- Use QLoRA (4-bit) if GPU RAM is tight.
- Key hyperparameters to sweep (small grid):
  - LoRA rank: {8, 16, 32}
  - lr: {1e-4, 2e-4, 5e-4}
  - epochs: {1, 3, 5} (LIMUC is not tiny; 3–5 is realistic)
  - label smoothing on/off
- Log GPU, batch size, effective batch size, and wall-clock.

**Output:**
- adapter checkpoint
- train/val curves
- best config in `outputs/runs/vlm_lora_best.yaml`

### 5.4 Evaluation of finetuned VLM

**Script: `06_eval_vlm_finetuned.py`**
- Same pipeline as zero-shot eval, but using finetuned adapter.
- Compare directly to:
  - classifier best
  - VLM zero-shot

---

## 6) Make it “Generative AI” (beyond just emitting a digit)

Chapter 4 will read stronger if you demonstrate a *controlled* generative capability while preserving safety.

### 6.1 Two-output structured response (recommended)

Have the finetuned model output:

```text
Mayo: <0-3>
Evidence: <short phrase; max 15 words; only visual findings>
```

Example:
- `Mayo: 2 | Evidence: marked erythema, absent vascular pattern, erosions`

Important: keep “Evidence” constrained to **visual descriptors only** (avoid treatment recommendations here).

### 6.2 How to create evidence labels (3 options)

Pick one option and document it clearly:

1) **Weak rubric supervision (fast, acceptable for thesis if disclosed):**
   - Derive evidence keywords from Mayo grade definition.
   - Example mapping:
     - 0 → normal mucosa, visible vascular pattern
     - 1 → mild erythema, decreased vascular pattern
     - 2 → marked erythema, friability, erosions
     - 3 → spontaneous bleeding, ulceration
   - Limitation: assumes grade→feature mapping always holds.

2) **Human annotation on a small subset (best quality, small scale):**
   - Manually annotate ~200 images with evidence tags.
   - Use them for evaluation only (not training), or for light finetune.

3) **Model-generated explanations with a verifier (advanced):**
   - Use a second model / rule-checker to reject explanations that contain forbidden content.

### 6.3 Explanation evaluation (minimum viable)

Even without new labels, you can run automated checks:
- “forbidden keyword” rate (e.g., medication names, dosing, prognosis terms)
- length compliance
- evidence keyword consistency with predicted grade (rubric-based consistency)

Save examples for the thesis:
- 3 correct with good evidence
- 3 incorrect with analysis

---

## 7) Analysis bundle (what you must produce for the chapter)

### 7.1 Primary tables

**Script: `09_make_figures_tables.py`** should export CSVs and thesis-ready images.

Table A: Main model comparison (test set)
- classifier(s)
- VLM zero-shot
- VLM LoRA (label-only)
- VLM LoRA (label + evidence) (if done)

Metrics:
- Accuracy, Macro-F1, Balanced Acc
- QWK
- MAE, RMSE
- Unknown rate

Table B: Per-class precision/recall/F1 + support.

Table C: Remission slice (0–1 vs 2–3)
- sensitivity, specificity, F1

### 7.2 Figures (minimum)

- Confusion matrices: best classifier and best VLM.
- Class distribution (train/val/test).
- Reliability/correlation plot for ordinal error (pred vs true).
- Optional: calibration curve if you implement calibration (below).

### 7.3 Statistical tests (recommended)

- McNemar test for paired accuracy differences (classifier vs VLM LoRA).
- Bootstrap CI for macro-F1 and QWK.

---

## 8) Calibration + abstention (nice-to-have but very thesis-friendly)

Severity grading is a high-stakes ordinal task; add a “don’t know” gate.

### 8.1 Classifier calibration

**Script: `07_calibration.py`**
- Temperature scaling on validation.
- Report ECE and calibrated confidence distribution.
- Define abstention rule:
  - if max prob < threshold → abstain

### 8.2 Generative confidence proxy

If model exposes token logprobs:
- take probability of the emitted digit token
- abstain if below threshold

Report performance vs abstention rate curve.

---

## 9) Optional advanced extension (only if time/compute)

### 9.1 Metric learning / triplet loss retrieval (inspired by the Crohn’s “thick data” framework)

Goal: learn a severity-aware embedding space and use it for:
- nearest-neighbor retrieval of similar cases
- hard-negative mining
- few-shot robustness

Deliverable:
- a retrieval demo: for a test image, show top-3 nearest neighbors and their labels.

This can become a short subsection in Chapter 4 or a bridge to Chapter 5.

---

## 10) Writing integration checklist (to finish Chapter 4 cleanly)

Once experiments are done, update the Chapter 4 text:

- Replace placeholder metrics with **exact numbers** from `outputs/tables/`.
- Insert confusion matrices and at least 1 representative qualitative figure.
- Add a subsection explicitly defining:
  - dataset(s)
  - patient-level split (if used)
  - prompt(s)
  - parsing rules
  - what “unknown” means
- Add an “Ablations” subsection:
  - prompt template comparison
  - LoRA rank / LR impact (small table)
  - imbalance strategy impact (small table)
- Add “Limitations”:
  - dataset bias, single-frame grading, label noise, external validity.

---

## Definition of done (DoD)

Chapter 4 work is “done” when all items below exist:

- [ ] `data/splits/*.csv` committed (IDs only) + split stats documented
- [ ] Classifier baseline trained + evaluated with full metric bundle
- [ ] Zero-shot VLM evaluated with parsing + unknown rate
- [ ] LoRA/QLoRA finetuned VLM evaluated
- [ ] `outputs/tables/` contains thesis-ready CSV tables
- [ ] `outputs/figures/` contains thesis-ready PNG/PDF figures
- [ ] Chapter 4 text updated to reference the produced artifacts
