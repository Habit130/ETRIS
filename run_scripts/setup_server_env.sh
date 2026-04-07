#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-etris-plantseg}"
HF_VIT_URL="https://huggingface.co/jinaai/clip-models/resolve/main/ViT-B-16.pt?download=true"
HF_VIT_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
TMP_CONDARC="$(mktemp)"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"

cleanup() {
  rm -f "${TMP_CONDARC}"
}
trap cleanup EXIT

cat > "${TMP_CONDARC}" <<'EOF'
channels:
  - conda-forge
default_channels: []
channel_priority: strict
show_channel_urls: true
EOF

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  CONDARC="${TMP_CONDARC}" conda env update -n "${ENV_NAME}" --file "${ROOT_DIR}/environment.server.yml" --prune
else
  CONDARC="${TMP_CONDARC}" conda env create -n "${ENV_NAME}" --file "${ROOT_DIR}/environment.server.yml"
fi

conda activate "${ENV_NAME}"
python -m pip uninstall -y opencv-python opencv-contrib-python >/dev/null 2>&1 || true
python -m pip install --no-deps --upgrade "opencv-python-headless==4.10.0.84"
python -m pip show opencv-python-headless
mkdir -p "${ROOT_DIR}/pretrain"

python "${ROOT_DIR}/tools/download_hf_asset.py" \
  --url "${HF_VIT_URL}" \
  --output "${ROOT_DIR}/pretrain/ViT-B-16.pt" \
  --sha256 "${HF_VIT_SHA256}"

python - <<'PY'
import importlib
import traceback

modules = [
    ("cv2", "cv2"),
    ("ftfy", "ftfy"),
    ("lmdb", "lmdb"),
    ("pyarrow", "pyarrow"),
    ("regex", "regex"),
    ("torch", "torch"),
    ("wandb", "wandb"),
    ("yaml", "yaml"),
    ("PIL.Image", "PIL.Image"),
    ("pycocotools.mask", "pycocotools.mask"),
]

loaded = {}
for label, module_name in modules:
    try:
        loaded[label] = importlib.import_module(module_name)
        print(f"import-ok: {label}")
    except Exception:
        print(f"import-failed: {label}")
        traceback.print_exc()
        raise

print("environment-ok", loaded["torch"].__version__)
PY
