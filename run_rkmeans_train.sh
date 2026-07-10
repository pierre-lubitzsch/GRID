#!/usr/bin/env bash
#SBATCH --job-name=rkmeans_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:2

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

# Number of RKMeans codebooks and codebook size. Defaults reproduce the standard
# TIGER setup (3 codebooks -> +1 dedup digit at inference -> 4 hierarchies).
# "Longer IDs for finer-grained neighborhoods" (Jul 3): for final ID length
# L in {8, 16} with codebook size FIXED at 256, set RKMEANS_HIER=L-1 (7 or 15)
# and keep CODEBOOK_WIDTH=256. run_rkmeans_inference.sh MUST use the same values.
RKMEANS_HIER="${RKMEANS_HIER:-3}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"

echo "[$(date -Is)] Starting rkmeans train on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR}"
echo "Using embedding_path=${EMBEDDING_PATH}"
echo "Codebooks: num_hierarchies=${RKMEANS_HIER} codebook_width=${CODEBOOK_WIDTH} (final ID length = ${RKMEANS_HIER}+1 after dedup)"

LOCAL_CKPT_DIR="${TMPDIR:-/tmp}/rkmeans_ckpts_${SLURM_JOB_ID:-$$}"
mkdir -p "${LOCAL_CKPT_DIR}"

python -u -m src.train \
  experiment=rkmeans_train_flat \
  data_dir="${GRID_DATA_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies="${RKMEANS_HIER}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  "callbacks.model_checkpoint.dirpath=${LOCAL_CKPT_DIR}" \
  "${@:3}"

LATEST_RUN_DIR="$(ls -d "${GRID_DIR}/logs/train/runs"/*/* 2>/dev/null | sort | tail -1 || true)"
if [ -n "${LATEST_RUN_DIR}" ] && ls "${LOCAL_CKPT_DIR}"/*.ckpt &>/dev/null; then
  mkdir -p "${LATEST_RUN_DIR}/checkpoints"
  cp "${LOCAL_CKPT_DIR}"/*.ckpt "${LATEST_RUN_DIR}/checkpoints/"
  echo "[$(date -Is)] Checkpoint copied to ${LATEST_RUN_DIR}/checkpoints/"
else
  echo "[$(date -Is)] ERROR: no checkpoint found in ${LOCAL_CKPT_DIR}"
  exit 1
fi

echo "[$(date -Is)] rkmeans train finished"
