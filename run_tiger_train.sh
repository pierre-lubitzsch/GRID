#!/usr/bin/env bash
#SBATCH --job-name=tiger_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --partition=pgpu
#SBATCH --time=2-00:00:00

# Resource notes:
# * --gres=gpu:nvidia_h200:2 + --partition=pgpu: requests 2 H200s (s-sc-pgpu[11-16]).
#   Override at submit time with `sbatch --gres=gpu:nvidia_h200:N ...`.

set -euo pipefail

# -----------------------------------------------------------------------------
# 4. Train Generative Recommendation Model with Semantic IDs (README)
#
#   python -m src.train experiment=tiger_train_flat \
#       data_dir=data/amazon_data/beauty \
#       semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
#       num_hierarchies=4
#
# num_hierarchies=4: add 1 vs RKMeans (step 3) because the previous step appends one
# additional digit to de-duplicate semantic IDs (3 codebooks -> 4 hierarchies here).
# -----------------------------------------------------------------------------
#
# Wrapper usage:
#   sbatch run_tiger_train.sh [dataset] [clean|poison] [semantic_id_path]
#
# data_dir: README uses data/amazon_data/<dataset>. This checkout often has data under
# src/data/amazon_data/<dataset> — override with:  TIGER_DATA_DIR=src/data/amazon_data/beauty
# (default below follows the repo layout.)
#
# Progress bar: off in configs/experiment/tiger_train_flat.yaml. Local bar:
#   add trainer.enable_progress_bar=true

DATASET="${1:-beauty}"
ARG2="${2:-}"
ARG3="${3:-}"
POISONING_RATIO="${POISONING_RATIO:-${4:-0.01}}"
N_TARGET_ITEMS="${N_TARGET_ITEMS:-${5:-10}}"
VARIANT="clean"
SEMANTIC_ID_PATH=""

case "${ARG2}" in
  clean|poison)
    VARIANT="${ARG2}"
    SEMANTIC_ID_PATH="${ARG3}"
    ;;
  "")
    ;;
  *)
    if [[ "${ARG2}" == *.pt ]]; then
      SEMANTIC_ID_PATH="${ARG2}"
    else
      echo "Unknown arg '${ARG2}'. Use clean|poison or a .pt semantic_id_path."
      exit 1
    fi
    ;;
esac

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ -n "${TIGER_DATA_DIR:-}" ]; then
  DATA_DIR="${TIGER_DATA_DIR}"
elif [ "${VARIANT}" = "poison" ]; then
  DATA_DIR="${GRID_POISON_DATA_DIR}"
else
  DATA_DIR="${GRID_DATA_DIR}"
fi

AUTO_SEMANTIC_ID_PATH="$(ls -t \
  logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt \
  embeddings/*/merged_predictions_tensor.pt \
  embeddings/merged_predictions_tensor.pt \
  2>/dev/null | head -n1 || true)"

if [ -n "${SEMANTIC_ID_PATH}" ] && [ ! -f "${SEMANTIC_ID_PATH}" ]; then
  echo "Provided semantic_id_path not found: ${SEMANTIC_ID_PATH}"
  if [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
    echo "Falling back to latest discovered tensor: ${AUTO_SEMANTIC_ID_PATH}"
    SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
  fi
fi

if [ -z "${SEMANTIC_ID_PATH}" ] && [ -f "${GRID_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${GRID_SEMANTIC_ID_PATH}"
fi
if [ -z "${SEMANTIC_ID_PATH}" ] && [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
fi

if [ ! -f "${SEMANTIC_ID_PATH}" ]; then
  echo "Semantic ID tensor not found: ${SEMANTIC_ID_PATH:-<empty>}"
  echo "Step 3 must produce pickle/merged_predictions_tensor.pt first."
  echo "Expected under logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt"
  exit 1
fi

# Build a unique, informative run directory: date/time_jobID_dataset_variant[_pctX_nY]
JOB_ID="${SLURM_JOB_ID:-local$$}"
TS="$(date +%Y-%m-%d/%H-%M-%S)"
RUN_LABEL="${DATASET}_${VARIANT}"
if [ "${VARIANT}" = "poison" ]; then
  PCT_LABEL="$(python3 -c "r=${POISONING_RATIO}; print(f'pct{int(round(r*100))}')")"
  RUN_LABEL="${RUN_LABEL}_${PCT_LABEL}_n${N_TARGET_ITEMS}"
fi
HYDRA_RUN_DIR="logs/train/runs/${TS}_job${JOB_ID}_${RUN_LABEL}"

echo "[$(date -Is)] Starting tiger train (tiger_train_flat) dataset=${DATASET} variant=${VARIANT}"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Run dir: ${HYDRA_RUN_DIR}"

python -u -m src.train \
  experiment=tiger_train_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  num_hierarchies=4 \
  hydra.run.dir="${HYDRA_RUN_DIR}"

LATEST_CKPT="$(ls -t logs/train/runs/*/*/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -n "${LATEST_CKPT}" ]; then
  echo "[$(date -Is)] tiger train finished"
  echo "Latest checkpoint (for tiger inference): ${LATEST_CKPT}"
else
  echo "[$(date -Is)] tiger train finished (no .ckpt under logs/train/runs/*/*/checkpoints/)"
fi
