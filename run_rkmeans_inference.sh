#!/usr/bin/env bash
#SBATCH --job-name=rkmeans_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

# Step 3b: run RKMeans inference using a trained checkpoint from Step 3a.
#
# Usage:
#   sbatch run_rkmeans_inference.sh <ckpt_path> [dataset] [embedding_path]
# Example:
#   sbatch run_rkmeans_inference.sh checkpoints/last.ckpt beauty

CKPT_PATH="${1:-}"
DATASET="${2:-beauty}"
EMBEDDING_PATH="${3:-}"

if [ -z "${CKPT_PATH}" ]; then
  echo "Missing ckpt_path."
  echo "Usage: sbatch run_rkmeans_inference.sh <ckpt_path> [dataset] [embedding_path]"
  exit 1
fi

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ ! -f "${EMBEDDING_PATH}" ]; then
  echo "Embedding file not found: ${EMBEDDING_PATH}"
  echo "Pass an explicit embedding_path as third arg if needed."
  exit 1
fi

if [ -z "${EMBEDDING_PATH}" ]; then
  EMBEDDING_PATH="$(ls -t logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -n1 || true)"
fi

echo "[$(date -Is)] Starting rkmeans inference on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR}"

python -u -m src.inference \
  experiment=rkmeans_inference_flat \
  data_dir="${GRID_DATA_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies=3 \
  codebook_width=256 \
  "ckpt_path='${CKPT_PATH}'" \
  callbacks.pickle_writer.should_merge_files_on_main=false

echo "[$(date -Is)] rkmeans inference finished, merging pickle shards..."

LATEST_RUN_DIR="$(ls -dt logs/inference/runs/*/* 2>/dev/null | head -n 1)"
PICKLE_DIR="${LATEST_RUN_DIR}/pickle"
export PICKLE_DIR

if [ -z "${LATEST_RUN_DIR}" ] || [ ! -d "${PICKLE_DIR}" ]; then
  echo "Could not find latest run pickle directory under logs/inference/runs."
  exit 1
fi

python - <<'PY'
import os
import pickle
import torch
from src.utils.tensor_utils import (
    deduplicate_rows_in_tensor,
    merge_list_of_keyed_tensors_to_single_tensor,
    transpose_tensor_from_file,
)

pickle_dir = os.environ["PICKLE_DIR"]
files = sorted(
    [f for f in os.listdir(pickle_dir) if f.startswith("predictions_") and f.endswith(".pkl")]
)
if not files:
    raise RuntimeError(f"No pickle shard files found in {pickle_dir}")

merged = []
for name in files:
    with open(os.path.join(pickle_dir, name), "rb") as fh:
        merged.extend(pickle.load(fh))

with open(os.path.join(pickle_dir, "merged_predictions.pkl"), "wb") as fh:
    pickle.dump(merged, fh)

tensor = merge_list_of_keyed_tensors_to_single_tensor(
    data=merged,
    index_key="item_id",
    value_key="cluster_ids",
)
pt_path = os.path.join(pickle_dir, "merged_predictions_tensor.pt")
torch.save(tensor.cpu(), pt_path)

# Match rkmeans_inference_flat post-processing behavior.
deduplicate_rows_in_tensor(file_path=pt_path)
transpose_tensor_from_file(file_path=pt_path)

print(f"Merged {len(merged)} rows into {pt_path}")
PY

echo "[$(date -Is)] Merge complete: ${PICKLE_DIR}/merged_predictions_tensor.pt"
bash "${GRID_DIR}/scripts/install_semantic_id_tensor.sh" "${DATASET}" "${PICKLE_DIR}/merged_predictions_tensor.pt"
