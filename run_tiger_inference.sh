#!/usr/bin/env bash
#SBATCH --job-name=tiger_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=2-00:00:00

# --cpus-per-task=8 matches num_workers=8 in tiger_inference_flat.yaml.
# --gpus-per-node=1 + --partition=gpu: --partition=gpu alone does NOT
# allocate a GPU on this cluster; always request one explicitly.

set -euo pipefail

# -----------------------------------------------------------------------------
# 4. Generate Recommendations (TIGER inference) — README
#
#   python -m src.inference experiment=tiger_inference_flat \
#       data_dir=data/amazon_data/beauty \
#       semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
#       ckpt_path=<checkpoint_from_tiger_train_above> \
#       num_hierarchies=4
#
# num_hierarchies=4: same as training (+1 vs RKMeans step 3 for the de-dup digit).
# ckpt_path: under logs/train/runs/.../checkpoints/ from tiger_train_flat.
# Quote ckpt_path if the filename contains '=' (e.g. checkpoint_epoch=..._step=....ckpt).
# -----------------------------------------------------------------------------
#
# Wrapper usage:
#   sbatch run_tiger_inference.sh <ckpt_path> [dataset] [semantic_id_path]
#
# data_dir: see run_tiger_train.sh — TIGER_DATA_DIR overrides (default src/data/...).

CKPT_PATH="${1:-}"
DATASET="${2:-beauty}"
SEMANTIC_ID_PATH="${3:-}"

if [ -z "${CKPT_PATH}" ]; then
  echo "Missing ckpt_path (checkpoint from tiger_train_flat)."
  echo "Usage: sbatch run_tiger_inference.sh <ckpt_path> [beauty|sports|toys] [semantic_id_path]"
  exit 1
fi

case "${DATASET}" in
  beauty|sports|toys) ;;
  *)
    echo "Invalid dataset: '${DATASET}'"
    echo "Usage: sbatch run_tiger_inference.sh <ckpt_path> [beauty|sports|toys] [semantic_id_path]"
    exit 1
    ;;
esac

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

DATA_DIR="${TIGER_DATA_DIR:-src/data/amazon_data/${DATASET}}"

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
  echo "Expected step 3 pickle: .../pickle/merged_predictions_tensor.pt"
  exit 1
fi

echo "[$(date -Is)] Starting tiger inference (tiger_inference_flat) dataset=${DATASET}"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Using ckpt_path=${CKPT_PATH}"

# Hydra: quote values that contain '=' (Lightning checkpoint filenames).
python -u -m src.inference \
  experiment=tiger_inference_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "ckpt_path='${CKPT_PATH}'" \
  num_hierarchies=4

echo "[$(date -Is)] tiger inference finished"
