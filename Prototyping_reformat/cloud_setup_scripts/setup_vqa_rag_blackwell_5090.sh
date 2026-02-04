#!/usr/bin/env bash
set -euo pipefail

WORKDIR="${WORKDIR:-/workspace}"
VENV_DIR="${VENV_DIR:-$WORKDIR/vqa-rag}"
KERNEL_NAME="${KERNEL_NAME:-vqa-rag}"
KERNEL_DISPLAY="${KERNEL_DISPLAY:-Python (vqa-rag)}"
REPO_DIR="${REPO_DIR:-$WORKDIR/rag-vqa-medical}"

REQ_FILE="${REQ_FILE:-$WORKDIR/requirements-vqa-rag.txt}"
DATASET_URL="${DATASET_URL:-http://216.211.50.9:9090/5827695.zip}"
DATASET_ZIP="${DATASET_ZIP:-$WORKDIR/5827695.zip}"
DATASET_DIR="${DATASET_DIR:-$REPO_DIR/Datasets/LIMUC}"

echo "[Blackwell] Using PyTorch NIGHTLY cu128 (CUDA 12.8) index"
TORCH_INDEX_URL="https://download.pytorch.org/whl/nightly/cu128"

mkdir -p "$WORKDIR"
cd "$WORKDIR"

if [[ ! -d "$REPO_DIR" ]]; then
  echo "⚠️  Repo not found at $REPO_DIR. Clone it first, or set REPO_DIR."
else
  echo "✅ Repo found at $REPO_DIR"
fi

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

# Download + extract dataset (if repo exists)
if [[ -d "$REPO_DIR" ]]; then
  mkdir -p "$DATASET_DIR"
  if [[ ! -f "$DATASET_ZIP" ]]; then
    echo "⬇️  Downloading dataset from $DATASET_URL"
    if command -v curl >/dev/null 2>&1; then
      curl -L "$DATASET_URL" -o "$DATASET_ZIP"
    elif command -v wget >/dev/null 2>&1; then
      wget -O "$DATASET_ZIP" "$DATASET_URL"
    else
      echo "❌ Neither curl nor wget is available. Install one and re-run."
      exit 1
    fi
  else
    echo "✅ Dataset zip already exists at $DATASET_ZIP"
  fi

  echo "📦 Extracting dataset to $DATASET_DIR"
  python - <<'PY'
import os
import shutil
import zipfile
from pathlib import Path

zip_path = Path(os.environ["DATASET_ZIP"])
out_dir = Path(os.environ["DATASET_DIR"])
out_dir.mkdir(parents=True, exist_ok=True)

with zipfile.ZipFile(zip_path, "r") as z:
    names = [n for n in z.namelist() if not n.endswith("/")]
    top_levels = {n.split("/")[0] for n in names if n}
    top = next(iter(top_levels)) if len(top_levels) == 1 else None

    if top and top not in {"0_dataset_prep"}:
        tmp = out_dir.parent / (out_dir.name + "_tmp_extract")
        if tmp.exists():
            shutil.rmtree(tmp)
        z.extractall(tmp)
        src = tmp / top
        if src.exists():
            for item in src.iterdir():
                dest = out_dir / item.name
                if dest.exists():
                    if dest.is_dir():
                        shutil.rmtree(dest)
                    else:
                        dest.unlink()
                shutil.move(str(item), str(dest))
        else:
            z.extractall(out_dir)
        shutil.rmtree(tmp, ignore_errors=True)
    else:
        z.extractall(out_dir)

expected_dirs = [
    out_dir / "train_and_validation_sets",
    out_dir / "test_set",
    out_dir / "patient_based_classified_images",
]
if any(p.exists() for p in expected_dirs):
    print("✅ Found expected dataset folders in", out_dir)
else:
    # If the zip had a nested LIMUC/..., move it up
    nested = out_dir / "LIMUC"
    if nested.exists() and nested.is_dir():
        for item in nested.iterdir():
            dest = out_dir / item.name
            if dest.exists():
                if dest.is_dir():
                    shutil.rmtree(dest)
                else:
                    dest.unlink()
            shutil.move(str(item), str(dest))
        try:
            nested.rmdir()
        except Exception:
            pass
        print("✅ Moved nested LIMUC contents into", out_dir)
    else:
        print("⚠️  Expected dataset folders not found. Check the zip structure.")
PY
fi

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
