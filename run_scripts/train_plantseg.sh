#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG_PATH="${1:-${ROOT_DIR}/config/plantseg/bridge_v16.yaml}"
shift $(( $# > 0 ? 1 : 0 ))
cd "${ROOT_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-4}"
export MASTER_PORT="${MASTER_PORT:-29999}"

torchrun --nproc_per_node=1 --master_port="${MASTER_PORT}" \
  train.py \
  --config "${CONFIG_PATH}" \
  "$@"
