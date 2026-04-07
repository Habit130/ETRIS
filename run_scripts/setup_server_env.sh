#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_NAME="${1:-etris-plantseg}"
HF_VIT_URL="https://huggingface.co/jinaai/clip-models/resolve/main/ViT-B-16.pt?download=true"
HF_VIT_SHA256="5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f"
CHANNEL_ARGS=(--override-channels -c conda-forge -c pytorch -c nvidia)

if ! command -v conda >/dev/null 2>&1; then
  echo "conda is required but was not found in PATH." >&2
  exit 1
fi

eval "$(conda shell.bash hook)"
conda config --set channel_priority strict >/dev/null 2>&1 || true

if conda env list | awk '{print $1}' | grep -qx "${ENV_NAME}"; then
  conda env update "${CHANNEL_ARGS[@]}" -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.server.yml" --prune
else
  conda env create "${CHANNEL_ARGS[@]}" -n "${ENV_NAME}" -f "${ROOT_DIR}/environment.server.yml"
fi

conda activate "${ENV_NAME}"
mkdir -p "${ROOT_DIR}/pretrain"

python "${ROOT_DIR}/tools/download_hf_asset.py" \
  --url "${HF_VIT_URL}" \
  --output "${ROOT_DIR}/pretrain/ViT-B-16.pt" \
  --sha256 "${HF_VIT_SHA256}"

python -c "import cv2, ftfy, lmdb, pyarrow, regex, torch, wandb, yaml; from PIL import Image; import pycocotools.mask; print('environment-ok', torch.__version__)"
