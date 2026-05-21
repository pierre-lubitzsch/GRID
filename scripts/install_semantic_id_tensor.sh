#!/usr/bin/env bash
# Copy a merged_predictions_tensor.pt into embeddings/<dataset>/ for training scripts.
#
# Usage:
#   bash scripts/install_semantic_id_tensor.sh rsc15 logs/inference/runs/.../pickle/merged_predictions_tensor.pt

set -euo pipefail

DATASET="${1:?dataset name required}"
SRC="${2:?source .pt path required}"

GRID_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
resolve_grid_dataset "${DATASET}"

DEST="${GRID_SEMANTIC_ID_PATH}"
mkdir -p "$(dirname "${DEST}")"
cp -f "${SRC}" "${DEST}"
echo "Installed ${SRC} -> ${DEST}"
