#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$WORKDIR/vqa-rag}"
KERNEL_NAME="${KERNEL_NAME:-vqa-rag}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (vqa-rag)}"
REPO_DIR="${REPO_DIR:-$WORKDIR/rag-vqa-medical}"

REQ_FILE="${REQ_FILE:-$WORKDIR/requirements-vqa-rag.txt}"
KVASIR_ROOT="${KVASIR_ROOT:-$REPO_DIR/Prototyping_reformat/DatasetAnalysis/Kvasir_VQA_x1}"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

# Detect driver-reported CUDA capability to choose wheel channel
CUDA_VER="$(nvidia-smi 2>/dev/null | awk -F 'CUDA Version: ' 'NF>1{print $2}' | awk '{print $1}' || true)"
TORCH_CUDA="cu121"
if [[ -n "${CUDA_VER}" ]]; then
  # crude version compare: 12.8+ => cu128
  MAJ="${CUDA_VER%%.*}"; MIN="${CUDA_VER#*.}"; MIN="${MIN%%.*}"
  if [[ "$MAJ" -gt 12 || ( "$MAJ" -eq 12 && "$MIN" -ge 8 ) ]]; then
    TORCH_CUDA="cu128"
  fi
fi

TORCH_INDEX_URL="https://download.pytorch.org/whl/${TORCH_CUDA}"
echo "[H100] CUDA Version (nvidia-smi): ${CUDA_VER:-unknown} => Using ${TORCH_CUDA} wheels (${TORCH_INDEX_URL})"

cat > "$REQ_FILE" <<'EOF'
transformers==4.53.2
datasets==3.6.0
accelerate==1.12.0
evaluate==0.4.6
timm==1.0.22
safetensors==0.5.3
tokenizers==0.21.4
peft==0.13.2
sentencepiece==0.2.0
python-dotenv==1.0.1

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

PIP_CMD="python -m pip"
if command -v uv >/dev/null 2>&1; then
  PIP_CMD="uv pip"
fi

# Stable PyTorch for H100
$PIP_CMD install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX_URL"
$PIP_CMD install -r "$REQ_FILE"

python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "$KERNEL_DISPLAY"

mkdir -p "$WORKDIR/.cache/huggingface" "$WORKDIR/.cache/torch"
cat > "$WORKDIR/vqa-rag.env" <<EOF
export HF_HOME=$WORKDIR/.cache/huggingface
export TRANSFORMERS_CACHE=$WORKDIR/.cache/huggingface/hub
export HF_DATASETS_CACHE=$WORKDIR/.cache/huggingface/datasets
export TORCH_HOME=$WORKDIR/.cache/torch
export KVASIR_VQA_X1_ROOT=$KVASIR_ROOT
EOF

python - <<'PY'
import torch

def parse_version(v: str):
    core = str(v).split("+", 1)[0]
    parts = []
    for token in core.split("."):
        digits = "".join(ch for ch in token if ch.isdigit())
        parts.append(int(digits) if digits else 0)
    while len(parts) < 3:
        parts.append(0)
    return tuple(parts[:3])

print("Torch:", torch.__version__)
if parse_version(torch.__version__) < (2, 6, 0):
    raise SystemExit(
        "ERROR: torch>=2.6 is required for Gemma3/MedGemma multimodal masking. "
        "Try nightly wheels if stable is older on your image: "
        "pip install --upgrade --pre torch torchvision torchaudio --index-url "
        "https://download.pytorch.org/whl/nightly/<cu121|cu128>"
    )
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

echo "✅ Done. Source env vars with: source $WORKDIR/vqa-rag.env"
