#!/usr/bin/env bash
#SBATCH --job-name=rkmeans_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_a100_80gb_pcie:1

set -euo pipefail

# Step 3a: train RKMeans codebooks from embeddings.
#
# Usage:
#   sbatch run_rkmeans_train.sh [dataset] [embedding_path]
# Example:
#   sbatch run_rkmeans_train.sh beauty logs/inference/runs/<step2_run>/pickle/merged_predictions_tensor.pt

DATASET="${1:-beauty}"
EMBEDDING_PATH="${2:-}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ -z "${EMBEDDING_PATH}" ]; then
  EMBEDDING_PATH="$(ls -t logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -n1 || true)"
fi
if [ ! -f "${EMBEDDING_PATH}" ]; then
  echo "Embedding file not found: ${EMBEDDING_PATH:-<empty>}"
  exit 1
fi

echo "[$(date -Is)] Starting rkmeans train on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR}"
echo "Using embedding_path=${EMBEDDING_PATH}"

# Save checkpoints to the compute node's local disk first — torch.save() can
# hang on the scratch parallel filesystem when called from a compute node.
LOCAL_CKPT_DIR="${HOME}/tmp/rkmeans_ckpts_${SLURM_JOB_ID:-$$}"
mkdir -p "${LOCAL_CKPT_DIR}"

python -u -m src.train \
  experiment=rkmeans_train_flat \
  data_dir="${GRID_DATA_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies=3 \
  codebook_width=256 \
  "callbacks.model_checkpoint.dirpath=${LOCAL_CKPT_DIR}" \
  "${@:3}"

# Copy checkpoints from local disk to the most recent Hydra run dir on scratch.
LATEST_RUN_DIR="$(ls -dt "${GRID_DIR}/logs/train/runs"/*/* 2>/dev/null | head -1 || true)"
if [ -n "${LATEST_RUN_DIR}" ] && ls "${LOCAL_CKPT_DIR}"/*.ckpt &>/dev/null; then
  mkdir -p "${LATEST_RUN_DIR}/checkpoints"
  cp "${LOCAL_CKPT_DIR}"/*.ckpt "${LATEST_RUN_DIR}/checkpoints/"
  echo "[$(date -Is)] Checkpoint(s) copied to ${LATEST_RUN_DIR}/checkpoints/"
  ls "${LATEST_RUN_DIR}/checkpoints/"
else
  echo "[$(date -Is)] WARNING: no checkpoint found in ${LOCAL_CKPT_DIR}"
fi

echo "[$(date -Is)] rkmeans train finished"
