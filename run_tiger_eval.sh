#!/usr/bin/env bash
#SBATCH --job-name=tiger_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# Evaluate a TIGER checkpoint on a (clean) test set.
#
# Reproducibility: EVAL_SEED defaults to 2 (matches unlearning wrappers if
# you use the same value for UNLEARN_SEED).
#
# Wraps `python -m scripts.eval_ckpt_on_test` so the experiment defaults
# (DDP, multi-GPU, large batch) are forced down to a single-device test loop
# that matches the unlearning workflow's eval semantics.
#
# Wrapper usage:
#   sbatch run_tiger_eval.sh <ckpt_path> <data_dir> [semantic_id_path] \
#                            [extra hydra overrides...]
#
# Example (evaluate the unlearned ckpt on clean Beauty):
#   sbatch run_tiger_eval.sh \
#       logs/unlearn/runs/2026-05-07/16-51-18/checkpoints/unlearned.ckpt \
#       src/data/amazon_data/beauty \
#       embeddings/beauty/merged_predictions_tensor.pt
# -----------------------------------------------------------------------------

CKPT_PATH="${1:-}"
DATA_DIR="${2:-}"
SEMANTIC_ID_PATH="${3:-}"
shift $(( $# < 3 ? $# : 3 ))
EXTRA_OVERRIDES=("$@")

if [ -z "${CKPT_PATH}" ] || [ -z "${DATA_DIR}" ]; then
  echo "Missing required argument(s)."
  echo "Usage: sbatch run_tiger_eval.sh <ckpt_path> <data_dir> [semantic_id_path] [extra hydra overrides...]"
  exit 1
fi

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

EVAL_SEED="${EVAL_SEED:-2}"
export PYTHONHASHSEED="${EVAL_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"

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
if [ -z "${SEMANTIC_ID_PATH}" ] && [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
fi
if [ ! -f "${SEMANTIC_ID_PATH}" ]; then
  echo "Semantic ID tensor not found: ${SEMANTIC_ID_PATH:-<empty>}"
  exit 1
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "Checkpoint not found: ${CKPT_PATH}"
  exit 1
fi
if [ ! -d "${DATA_DIR}/testing" ]; then
  echo "Expected ${DATA_DIR}/testing/ but it doesn't exist."
  exit 1
fi

echo "[$(date -Is)] Evaluating ${CKPT_PATH}"
echo "Using seed=${EVAL_SEED} (set EVAL_SEED to override)"
echo "Using data_dir=${DATA_DIR}    (testing shards under ${DATA_DIR}/testing/)"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]}"
fi

# Hydra: quote values that contain '=' (Lightning checkpoint filenames).
# trainer.* overrides force a single-device test loop (mirrors the unlearning
# experiment's accelerator config so DDP/sync-batchnorm don't kick in).
python -u -m scripts.eval_ckpt_on_test \
  experiment=tiger_train_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "ckpt_path='${CKPT_PATH}'" \
  num_hierarchies=4 \
  seed="${EVAL_SEED}" \
  train=False test=True \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.sync_batchnorm=false \
  trainer.num_nodes=1 \
  trainer.deterministic=true \
  callbacks.model_checkpoint=null \
  callbacks.early_stopping=null \
  "${EXTRA_OVERRIDES[@]}"

echo "[$(date -Is)] Evaluation finished"
