#!/usr/bin/env bash
#SBATCH --job-name=tiger_inference_after_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=2-00:00:00
#SBATCH --partition=gpu

set -euo pipefail

# After tiger_train_flat: pick latest *.ckpt under logs/train/runs/*/*/checkpoints/, then run
# tiger_inference_flat (same as README “4. Generate Recommendations”).
#
# Usage:
#   sbatch run_tiger_inference_after_train.sh [dataset] [semantic_id_path]
# Example:
#   sbatch run_tiger_inference_after_train.sh beauty logs/inference/runs/<step3>/pickle/merged_predictions_tensor.pt
#
# Passes through to run_tiger_inference.sh (see there for TIGER_DATA_DIR, Hydra quoting).

DATASET="${1:-beauty}"
SEMANTIC_ID_PATH="${2:-}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

CKPT_PATH="$(ls -t logs/train/runs/*/*/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -z "${CKPT_PATH}" ]; then
  echo "No checkpoint found under logs/train/runs/*/*/checkpoints/*.ckpt"
  exit 1
fi

echo "Using ckpt_path=${CKPT_PATH}"
bash run_tiger_inference.sh "${CKPT_PATH}" "${DATASET}" "${SEMANTIC_ID_PATH}"
