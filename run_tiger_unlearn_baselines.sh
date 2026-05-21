#!/usr/bin/env bash
#SBATCH --job-name=tiger_unlearn_baselines
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=1-00:00:00

set -euo pipefail

# Wrapper for baseline unlearning algorithms (finetune, neg_train, filter, unified).
#
# Usage:
#   sbatch run_tiger_unlearn_baselines.sh <algorithm> <ckpt_path> <data_dir> \
#       [semantic_id_path] [extra hydra overrides...]
#
# Algorithms: finetune | neg_train | filter | unified | scif
#
# Retrain upper bound (external):
#   sbatch run_tiger_train.sh <dataset> clean

ALGO="${1:-}"
CKPT_PATH="${2:-}"
DATA_DIR="${3:-}"
SEMANTIC_ID_PATH="${4:-}"
shift $(( $# < 4 ? $# : 4 ))
EXTRA_OVERRIDES=("$@")

if [ -z "${ALGO}" ] || [ -z "${CKPT_PATH}" ] || [ -z "${DATA_DIR}" ]; then
  echo "Usage: sbatch run_tiger_unlearn_baselines.sh <algorithm> <ckpt_path> <data_dir> [semantic_id_path] [hydra overrides...]"
  exit 1
fi

case "${ALGO}" in
  finetune) EXPERIMENT="tiger_unlearn_finetune_flat" ;;
  neg_train) EXPERIMENT="tiger_unlearn_neg_train_flat" ;;
  filter) EXPERIMENT="tiger_unlearn_filter_flat" ;;
  unified) EXPERIMENT="tiger_unlearn_unified_flat" ;;
  scif) EXPERIMENT="tiger_unlearn_scif_flat" ;;
  retrain)
    echo "Retrain baseline: run_tiger_train.sh on cleaned/retain data (no unlearn entry)."
    exit 0
    ;;
  *)
    echo "Unknown algorithm=${ALGO}; choose finetune|neg_train|filter|unified|scif|retrain"
    exit 1
    ;;
esac

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/unlearn_run_dir.sh
source "${GRID_DIR}/scripts/unlearn_run_dir.sh"
UNLEARN_OUTPUT_DIR="$(unlearn_build_output_dir "${GRID_DIR}")"
unlearn_allocate_output_dir "${UNLEARN_OUTPUT_DIR}"

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
RESOLVED="$(resolve_grid_dataset "${DATA_DIR}" "${SEMANTIC_ID_PATH}")"
DATA_DIR="$(echo "${RESOLVED}" | awk '{print $1}')"
if [ -z "${SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="$(echo "${RESOLVED}" | awk '{print $2}')"
fi

export PYTHONHASHSEED="${UNLEARN_SEED:-2}"
export CUBLAS_WORKSPACE_CONFIG=:4096:8

NUM_HIERARCHIES="${NUM_HIERARCHIES:-4}"
HYDRA_OVERRIDES=(
  "experiment=${EXPERIMENT}"
  "data_dir=${DATA_DIR}"
  "semantic_id_path=${SEMANTIC_ID_PATH}"
  "ckpt_path=${CKPT_PATH}"
  "num_hierarchies=${NUM_HIERARCHIES}"
  "paths.output_dir=${UNLEARN_OUTPUT_DIR}"
  "unlearning_run_tag=${UNLEARN_RUN_TAG:-baseline_${ALGO}}"
  "${EXTRA_OVERRIDES[@]}"
)

echo "[unlearn-baseline] algo=${ALGO} experiment=${EXPERIMENT} output=${UNLEARN_OUTPUT_DIR}"
python -m src.unlearn "${HYDRA_OVERRIDES[@]}"
