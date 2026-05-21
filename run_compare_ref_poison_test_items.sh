#!/usr/bin/env bash
#SBATCH --job-name=ref_poi_diff
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=04:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# Compare clean-reference vs poisoned checkpoints on the test set.
# Saves semantic IDs of items predicted correctly by ref but not by poisoned,
# and counts overlap with items in bandwagon spam training sessions.
#
# Usage:
#   sbatch run_compare_ref_poison_test_items.sh <reference_ckpt> <poisoned_ckpt> \
#       [semantic_id_path] [test_data_dir] [poison_data_dir]
#
# Example:
#   sbatch run_compare_ref_poison_test_items.sh \
#       logs/train/runs/2026-05-13/13-01-47/checkpoints/checkpoint_epoch=000_step=004400.ckpt \
#       logs/train/runs/2026-05-06/16-02-49/checkpoints/checkpoint_best.ckpt \
#       embeddings/beauty/merged_predictions_tensor.pt \
#       src/data/amazon_data/beauty \
#       src/data/amazon_data/beauty_spam_seed2_pct1_n10
# -----------------------------------------------------------------------------

REF_CKPT="${1:-}"
POISON_CKPT="${2:-}"
SEMANTIC_ID_PATH="${3:-}"
TEST_DATA_DIR="${4:-src/data/amazon_data/beauty}"
POISON_DATA_DIR="${5:-src/data/amazon_data/beauty_spam_seed2_pct1_n10}"

if [ -z "${REF_CKPT}" ] || [ -z "${POISON_CKPT}" ]; then
  echo "Usage: sbatch $0 <reference_ckpt> <poisoned_ckpt> [semantic_id_path] [test_data_dir] [poison_data_dir]"
  exit 1
fi

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

EVAL_SEED="${EVAL_SEED:-2}"
export PYTHONHASHSEED="${EVAL_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"

AUTO_SEMANTIC_ID_PATH="$(ls -t \
  embeddings/*/merged_predictions_tensor.pt \
  logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt \
  2>/dev/null | head -n1 || true)"
if [ -z "${SEMANTIC_ID_PATH}" ] && [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
fi
for p in "${REF_CKPT}" "${POISON_CKPT}" "${SEMANTIC_ID_PATH}"; do
  if [ ! -f "${p}" ]; then
    echo "File not found: ${p}"
    exit 1
  fi
done
if [ ! -d "${TEST_DATA_DIR}/testing" ]; then
  echo "Missing ${TEST_DATA_DIR}/testing/"
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="${GRID_DIR}/logs/eval/ref_poison_diff/${STAMP}"
mkdir -p "${OUT_DIR}"

echo "[$(date -Is)] ref vs poison test-item diff"
echo "  reference: ${REF_CKPT}"
echo "  poisoned:  ${POISON_CKPT}"
echo "  test data: ${TEST_DATA_DIR}"
echo "  poison sessions: ${POISON_DATA_DIR}/training/data_spam_*.tfrecord.gz"
echo "  output:    ${OUT_DIR}"

python -u -m scripts.compare_ref_poison_test_items \
  experiment=tiger_train_flat \
  data_dir="${TEST_DATA_DIR}" \
  poison_data_dir="${POISON_DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "reference_ckpt_path='${REF_CKPT}'" \
  "poisoned_ckpt_path='${POISON_CKPT}'" \
  num_hierarchies=4 \
  seed="${EVAL_SEED}" \
  train=False \
  test=True \
  trainer.devices=1 \
  trainer.accelerator=gpu \
  hydra.run.dir="${OUT_DIR}"

echo "[$(date -Is)] Finished"
echo "  summary: ${OUT_DIR}/summary.json"
echo "  items:   ${OUT_DIR}/ref_right_poison_wrong.json"
