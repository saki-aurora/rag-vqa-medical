#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$WORKDIR/vqa-rag}"
KERNEL_NAME="${KERNEL_NAME:-vqa-rag}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (vqa-rag)}"
REPO_DIR="${REPO_DIR:-$WORKDIR/rag-vqa-medical}"

REQ_FILE="${REQ_FILE:-$WORKDIR/requirements-vqa-rag.txt}"

# Kvasir-VQA x1 prep (enabled by default for notebook readiness)
KVASIR_ROOT="${KVASIR_ROOT:-$REPO_DIR/Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1}"
RUN_KVASIR_PREP="${RUN_KVASIR_PREP:-1}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# GB200/Blackwell: use PyTorch nightly cu128
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"
echo "[GB200] Using PyTorch NIGHTLY cu128 index: ${TORCH_INDEX_URL}"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "⚠️  Repo not found at $REPO_DIR. Clone it first, or set REPO_DIR."
else
  echo "✅ Repo found at $REPO_DIR"
fi

cat > "$REQ_FILE" <<'REQEOF'
transformers==4.53.2
datasets==3.6.0
accelerate==1.12.0
evaluate==0.4.6
timm==1.0.22
safetensors==0.5.3
tokenizers==0.21.4
peft==0.13.2
sentencepiece==0.2.0

scikit-learn==1.7.0
numpy==2.4.2
pandas==2.3.1
scipy==1.16.0
pillow==11.3.0
tqdm==4.67.1
regex==2024.11.6
requests==2.32.4
matplotlib==3.10.7
seaborn==0.13.2
scikit-image==0.26.0
statsmodels==0.14.6
nltk==3.9.1
sacrebleu==2.5.1
rouge-score==0.1.2
tabulate==0.9.0

jupyterlab
ipykernel
ipywidgets
nbconvert
REQEOF

if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

PIP_CMD="python -m pip"
if command -v uv >/dev/null 2>&1; then
  PIP_CMD="uv pip"
fi

# Install PyTorch nightly cu128 for GB200
$PIP_CMD install --upgrade --pre torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
$PIP_CMD install -r "$REQ_FILE"

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

# Caches + env vars
mkdir -p "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"
cat > "$WORKDIR/vqa-rag.env" <<EOFENV
export HF_HOME=$WORKDIR/.cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/.cache/huggingface/hub
export HF_DATASETS_CACHE=$WORKDIR/.cache/huggingface/datasets
export TORCH_HOME=$WORKDIR/.cache/torch
export KVASIR_VQA_X1_ROOT=$KVASIR_ROOT
EOFENV

# Prepare Kvasir-VQA x1 data + manifest (runs once; set RUN_KVASIR_PREP=0 to skip)
if [[ -d "$REPO_DIR" && "$RUN_KVASIR_PREP" == "1" ]]; then
  echo "\n🧪 Preparing Kvasir-VQA x1 dataset (this can take a while)..."
  if [[ -d "$KVASIR_ROOT" ]]; then
    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
      "$KVASIR_ROOT/0_dataset_prep/01_build_metadata_images_and_manifests.ipynb"

    jupyter nbconvert --to notebook --execute --inplace \
      --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
      "$KVASIR_ROOT/0_dataset_prep/02_validate_splits_and_integrity.ipynb"
  else
    echo "⚠️  KVASIR_ROOT not found at $KVASIR_ROOT; skipping dataset prep."
  fi
fi

# Verify CUDA works (this catches "no kernel image" early)
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    x = torch.randn(4, device="cuda")
    y = x @ x.T
    print("CUDA matmul ok:", y.shape)
PY

echo "✅ Done. Source env vars with: source $WORKDIR/vqa-rag.env"
