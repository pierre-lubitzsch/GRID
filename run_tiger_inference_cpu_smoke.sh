#!/usr/bin/env bash
# Smoke-test tiger inference on CPU (no GPU queue). Catches config/Hydra/runtime errors quickly.
#
# Usage (from repo root):
#   bash run_tiger_inference_cpu_smoke.sh <ckpt_path> [dataset] [semantic_id_path]
#
# Optional env:
#   LIMIT_PRED_BATCHES=3   # Passed as +trainer.limit_predict_batches (Hydra struct: must use +)

set -euo pipefail

CKPT_PATH="${1:-}"
DATASET="${2:-beauty}"
SEMANTIC_ID_PATH="${3:-}"
LIMIT_PRED_BATCHES="${LIMIT_PRED_BATCHES:-3}"

if [ -z "${CKPT_PATH}" ]; then
  echo "Missing ckpt_path."
  echo "Usage: bash run_tiger_inference_cpu_smoke.sh <ckpt_path> [beauty|sports|toys] [semantic_id_path]"
  exit 1
fi

case "${DATASET}" in
  beauty|sports|toys) ;;
  *)
    echo "Invalid dataset: '${DATASET}'"
    exit 1
    ;;
esac

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

DATA_DIR="${TIGER_DATA_DIR:-src/data/amazon_data/${DATASET}}"

AUTO_SEMANTIC_ID_PATH="$(ls -t \
  logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt \
  logs/inference/runs/*/*/pickle/*_merged_predictions_tensor_latest.pt \
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

echo "[$(date -Is)] CPU smoke: tiger inference dataset=${DATASET} limit_predict_batches=${LIMIT_PRED_BATCHES}"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Using ckpt_path=${CKPT_PATH}"

export CUDA_VISIBLE_DEVICES=""

# DDP + GPU defaults in yaml are overridden for single-process CPU.
python -u -m src.inference \
  experiment=tiger_inference_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "ckpt_path='${CKPT_PATH}'" \
  num_hierarchies=4 \
  trainer.accelerator=cpu \
  trainer.devices=1 \
  trainer.strategy=auto \
  trainer.sync_batchnorm=false \
  "+trainer.limit_predict_batches=${LIMIT_PRED_BATCHES}" \
  trainer.enable_progress_bar=true

echo "[$(date -Is)] CPU smoke finished"
