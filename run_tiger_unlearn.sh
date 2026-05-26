#!/usr/bin/env bash
#SBATCH --job-name=tiger_unlearn
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --partition=pgpu
#SBATCH --time=1-00:00:00

# --cpus-per-task=8: SCIF iterates over forget/retain dataloaders multiple
# times (one HVP pass per CG iteration), so even with num_workers=0 in
# tiger_unlearn_scif_flat.yaml we want some parallelism for the
# materialised retain subset filter (neighborhood_sampler).
# --gres=gpu:nvidia_h200:1 + --partition=pgpu: SCIF runs single-device autograd
# (devices=1, strategy=auto in the experiment config) on one H200.

set -euo pipefail

# -----------------------------------------------------------------------------
# TIGER unlearning (SCIF) — runs `python -m src.unlearn`
#
# Reproducibility: export UNLEARN_SEED=2 (default) before sbatch to fix
# PYTHONHASHSEED / Hydra seed; optional CUBLAS_WORKSPACE_CONFIG is set below.
#
# Pre-requisites:
#   1. A poisoned dataset directory produced by
#        python -m src.data.poisoning.bandwagon
#      and a forget/retain split produced by
#        python -m src.data.unlearning.split_forget_retain
#      under the same directory (training_forget/, training_retain/,
#      forget_manifest.json, items/, evaluation/, testing/).
#   2. A TIGER checkpoint trained on that poisoned dataset (the model we are
#      unlearning *from*).
#   3. The semantic-id tensor used for the train run.
#
# Wrapper usage:
#   sbatch run_tiger_unlearn.sh <ckpt_path> <data_dir> [semantic_id_path] \
#                               [neighborhood_aware] [extra hydra overrides...]
#
# Example:
#   sbatch run_tiger_unlearn.sh \
#       logs/train/runs/2026-05-06/13-00-00/checkpoints/checkpoint_epoch=003.ckpt \
#       src/data/amazon_data/beauty_spam_seed42_pct1_n10 \
#       embeddings/beauty/merged_predictions_tensor.pt \
#       true \
#       unlearning.neighbor_aware_factor=8 unlearning.target_params=all
# -----------------------------------------------------------------------------

CKPT_PATH="${1:-}"
DATA_DIR="${2:-}"
SEMANTIC_ID_PATH="${3:-}"
NEIGHBORHOOD_AWARE="${4:-false}"
shift $(( $# < 4 ? $# : 4 ))
EXTRA_OVERRIDES=("$@")

if [ -z "${CKPT_PATH}" ] || [ -z "${DATA_DIR}" ]; then
  echo "Missing required argument(s)."
  echo "Usage: sbatch run_tiger_unlearn.sh <ckpt_path> <data_dir> [semantic_id_path] [neighborhood_aware:true|false] [extra hydra overrides...]"
  exit 1
fi

case "${NEIGHBORHOOD_AWARE}" in
  true|false) ;;
  *)
    echo "neighborhood_aware must be 'true' or 'false', got '${NEIGHBORHOOD_AWARE}'"
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

# Reproducibility (override with UNLEARN_SEED=... before sbatch).
UNLEARN_SEED="${UNLEARN_SEED:-2}"
export PYTHONHASHSEED="${UNLEARN_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"

# Auto-discover semantic_id_path if not provided. Mirrors run_tiger_inference.sh.
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
  echo "Expected RKMeans output: .../pickle/merged_predictions_tensor.pt"
  exit 1
fi

if [ ! -d "${DATA_DIR}/training_forget" ] || [ ! -d "${DATA_DIR}/training_retain" ]; then
  echo "data_dir must contain training_forget/ and training_retain/ subdirs."
  echo "Run 'python -m src.data.unlearning.split_forget_retain --data_dir ${DATA_DIR} --forget_manifest ${DATA_DIR}/forget_manifest.json' first."
  exit 1
fi

echo "[$(date -Is)] Starting TIGER unlearning (SCIF, neighborhood_aware=${NEIGHBORHOOD_AWARE})"
echo "Using output_dir=${UNLEARN_OUTPUT_DIR}"
echo "Using seed=${UNLEARN_SEED} (set UNLEARN_SEED to override)"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Using ckpt_path=${CKPT_PATH}"
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]}"
fi

# Hydra: quote values that contain '=' (Lightning checkpoint filenames).
python -u -m src.unlearn \
  experiment=tiger_unlearn_scif_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "ckpt_path='${CKPT_PATH}'" \
  num_hierarchies=4 \
  seed="${UNLEARN_SEED}" \
  unlearning.neighborhood_aware=${NEIGHBORHOOD_AWARE} \
  hydra.run.dir="${UNLEARN_OUTPUT_DIR}" \
  unlearning_run_tag="${UNLEARN_RUN_TAG:-}" \
  "${EXTRA_OVERRIDES[@]}"

echo "[$(date -Is)] TIGER unlearning finished"
echo "Output: ${UNLEARN_OUTPUT_DIR}"
echo "Unlearned ckpt: ${UNLEARN_OUTPUT_DIR}/checkpoints/unlearned.ckpt"
