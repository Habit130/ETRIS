#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${1:-etris}"
ENV_FILE="${2:-environment.linux.yml}"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found. Please install Miniconda or Anaconda first."
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

conda env create -n "${ENV_NAME}" -f "${ENV_FILE}"

echo
echo "Environment created."
echo "Activate it with:"
echo "  conda activate ${ENV_NAME}"
