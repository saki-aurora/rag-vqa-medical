#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$WORKDIR/vqa-rag}"
KERNEL_NAME="${KERNEL_NAME:-vqa-rag}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (vqa-rag)}"

REQ_FILE="${REQ_FILE:-$WORKDIR/requirements-vqa-rag.txt}"

echo "[Blackwell] Using PyTorch NIGHTLY cu128 (CUDA 12.8) index"
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

cat > "$REQ_FILE" <<'EOF'
transformers==4.53.2
datasets==3.6.0
accelerate==1.12.0
evaluate==0.4.6
timm==1.0.22
safetensors==0.5.3
tokenizers==0.21.4

scikit-learn==1.7.0
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
EOF

if [[ ! -d "$VENV_DIR" ]]; then
  python -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1090
source "$VENV_DIR/bin/activate"

python -m pip install -U pip wheel setuptools

# Use uv if available (much faster resolver); otherwise pip
PIP_CMD="python -m pip"
if command -v uv >/dev/null 2>&1; then
  PIP_CMD="uv pip"
fi

# Install PyTorch nightly cu128 for Blackwell
$PIP_CMD install --upgrade --pre torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"

# Install the rest
$PIP_CMD install -r "$REQ_FILE"

# Register kernel
python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

# Caches
mkdir -p "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"
cat > "$WORKDIR/vqa-rag.env" <<EOF
export HF_HOME=$WORKDIR/.cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/.cache/huggingface/hub
export HF_DATASETS_CACHE=$WORKDIR/.cache/huggingface/datasets
export TORCH_HOME=$WORKDIR/.cache/torch
EOF

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
