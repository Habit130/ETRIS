#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/config/plantseg/bridge_v16.yaml}"
shift $(( $# > 0 ? 1 : 0 ))
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

python test.py \
  --config "${CONFIG_PATH}" \
  "$@"
