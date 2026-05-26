#!/usr/bin/env bash
#SBATCH --job-name=tiger_unlearn_seq
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --partition=pgpu
#SBATCH --time=1-00:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# Sequential TIGER unlearning (SCIF) -- runs `python -m src.unlearn_sequential`
#
# Reproducibility: UNLEARN_SEED (default 2) — see run_tiger_unlearn.sh.
#
# Pre-requisites: same as run_tiger_unlearn.sh (poisoned dataset, train_forget /
# train_retain split, TIGER checkpoint, semantic-id tensor).
#
# Wrapper usage:
#   sbatch run_tiger_unlearn_sequential.sh <ckpt_path> <dataset|data_dir> \
#       [semantic_id_path] [neighborhood_aware:true|false] [request_batch_size:int] \
#       [neighborhood_aware_sample_rate:float] [extra hydra overrides...]
#
# <dataset> is a short name (beauty, rsc15, ...) resolved via
# scripts/resolve_unlearn_dataset.sh, or a path containing '/' used as data_dir.
#
# Poison dataset selection (env vars, defaults shown):
#   POISONING_RATIO=0.01  N_TARGET_ITEMS=10  POISON_SEED=2
#
# Example (dataset name):
#   sbatch run_tiger_unlearn_sequential.sh \
#       logs/train/.../checkpoint_epoch=003.ckpt \
#       beauty \
#       embeddings/beauty/merged_predictions_tensor.pt \
#       true 8 0.5 \
#       unlearning.max_requests=4 unlearning.target_params=all
#
# After unlearning, optionally evaluates the final unlearned.ckpt on the clean
# test split (default on). Set UNLEARN_RUN_POST_EVAL=false to skip.
# Clean eval dir comes from the dataset registry (override with UNLEARN_EVAL_DATA_DIR).
# -----------------------------------------------------------------------------

CKPT_PATH="${1:-}"
DATASET_OR_DIR="${2:-}"
SEMANTIC_ID_PATH="${3:-}"
NEIGHBORHOOD_AWARE="${4:-false}"
REQUEST_BATCH_SIZE="${5:-8}"

# Arg 6 is sample_rate when numeric; otherwise treat it as the first Hydra override.
NEIGHBORHOOD_AWARE_SAMPLE_RATE="${UNLEARN_NEIGHBORHOOD_AWARE_SAMPLE_RATE:-1.0}"
if [ "$#" -ge 6 ] && python3 - <<PY
import sys
try:
    r = float("${6:-}")
    sys.exit(0 if 0.0 <= r <= 1.0 else 1)
except ValueError:
    sys.exit(1)
PY
then
  NEIGHBORHOOD_AWARE_SAMPLE_RATE="${6}"
  shift 6
else
  shift $(( $# < 5 ? $# : 5 ))
fi
EXTRA_OVERRIDES=("$@")

if [ -z "${CKPT_PATH}" ] || [ -z "${DATASET_OR_DIR}" ]; then
  echo "Missing required argument(s)."
  echo "Usage: sbatch run_tiger_unlearn_sequential.sh <ckpt_path> <dataset|data_dir> \\"
  echo "  [semantic_id_path] [neighborhood_aware:true|false] [request_batch_size:int] \\"
  echo "  [neighborhood_aware_sample_rate:float] [extra hydra overrides...]"
  echo "Known datasets: beauty, sports, toys, rsc15, rsc15_smoke (see scripts/resolve_unlearn_dataset.sh)"
  exit 1
fi

POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET_ITEMS="${N_TARGET_ITEMS:-10}"
POISON_SEED="${POISON_SEED:-2}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
# shellcheck source=scripts/resolve_unlearn_dataset.sh
source "${GRID_DIR}/scripts/resolve_unlearn_dataset.sh"

if [[ "${DATASET_OR_DIR}" == */* ]]; then
  DATA_DIR="${DATASET_OR_DIR}"
else
  DATASET="${DATASET_OR_DIR}"
  if ! resolve_unlearn_dataset "${DATASET}"; then
    exit 1
  fi
  DATA_DIR="${UNLEARN_DATA_DIR}"
  if [ -z "${SEMANTIC_ID_PATH}" ]; then
    SEMANTIC_ID_PATH="${UNLEARN_DEFAULT_SEMANTIC_ID_PATH}"
  fi
fi

case "${NEIGHBORHOOD_AWARE}" in
  true|false) ;;
  *)
    echo "neighborhood_aware must be 'true' or 'false', got '${NEIGHBORHOOD_AWARE}'"
    exit 1
    ;;
esac

if ! [[ "${REQUEST_BATCH_SIZE}" =~ ^[0-9]+$ ]] || [ "${REQUEST_BATCH_SIZE}" -le 0 ]; then
  echo "request_batch_size must be a positive integer, got '${REQUEST_BATCH_SIZE}'"
  exit 1
fi

if ! python3 - <<PY
import sys
r = float("${NEIGHBORHOOD_AWARE_SAMPLE_RATE}")
sys.exit(0 if 0.0 <= r <= 1.0 else 1)
PY
then
  echo "neighborhood_aware_sample_rate must be in [0, 1], got '${NEIGHBORHOOD_AWARE_SAMPLE_RATE}'"
  exit 1
fi

cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/unlearn_run_dir.sh
source "${GRID_DIR}/scripts/unlearn_run_dir.sh"
UNLEARN_OUTPUT_DIR="$(unlearn_build_output_dir "${GRID_DIR}" "${REQUEST_BATCH_SIZE}")"
unlearn_allocate_output_dir "${UNLEARN_OUTPUT_DIR}"

UNLEARN_SEED="${UNLEARN_SEED:-2}"
export PYTHONHASHSEED="${UNLEARN_SEED}"
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
  echo "Expected RKMeans output: .../pickle/merged_predictions_tensor.pt"
  exit 1
fi

if [ ! -d "${DATA_DIR}/training_forget" ] || [ ! -d "${DATA_DIR}/training_retain" ]; then
  echo "data_dir must contain training_forget/ and training_retain/ subdirs."
  echo "Run 'python -m src.data.unlearning.split_forget_retain --data_dir ${DATA_DIR} --forget_manifest ${DATA_DIR}/forget_manifest.json' first."
  exit 1
fi

echo "[$(date -Is)] Starting sequential TIGER unlearning (SCIF)"
echo "Using output_dir=${UNLEARN_OUTPUT_DIR}"
echo "Using seed=${UNLEARN_SEED} (set UNLEARN_SEED to override)"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Using ckpt_path=${CKPT_PATH}"
echo "request_batch_size=${REQUEST_BATCH_SIZE} | neighborhood_aware=${NEIGHBORHOOD_AWARE} | neighborhood_aware_sample_rate=${NEIGHBORHOOD_AWARE_SAMPLE_RATE}"
if [ -n "${DATASET:-}" ]; then
  echo "dataset=${DATASET} | eval_data_dir=${UNLEARN_EVAL_DATA_DIR:-<unset>}"
fi
if [ -f "${DATA_DIR}/forget_manifest.json" ]; then
  EXPECTED_BATCHES="$(python3 - <<PY
import json, math
with open("${DATA_DIR}/forget_manifest.json") as f:
    n = len(json.load(f).get("spam_user_ids") or [])
bs = int("${REQUEST_BATCH_SIZE}")
print(math.ceil(n / bs) if n and bs else "")
PY
)"
  if [ -n "${EXPECTED_BATCHES}" ]; then
    echo "Expected unlearning batches: ${EXPECTED_BATCHES} (from forget_manifest; less if unlearning.max_requests is set)"
  fi
fi
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ]; then
  echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]}"
fi

# Hydra: quote values that contain '=' (Lightning checkpoint filenames).
python -u -m src.unlearn_sequential \
  experiment=tiger_unlearn_scif_sequential \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  "ckpt_path='${CKPT_PATH}'" \
  num_hierarchies=4 \
  seed="${UNLEARN_SEED}" \
  unlearning.neighborhood_aware=${NEIGHBORHOOD_AWARE} \
  "unlearning.neighborhood_aware_sample_rate=${NEIGHBORHOOD_AWARE_SAMPLE_RATE}" \
  unlearning.request_batch_size=${REQUEST_BATCH_SIZE} \
  hydra.run.dir="${UNLEARN_OUTPUT_DIR}" \
  unlearning_run_tag="${UNLEARN_RUN_TAG:-bs${REQUEST_BATCH_SIZE}}" \
  "${EXTRA_OVERRIDES[@]}"

echo "[$(date -Is)] Sequential TIGER unlearning finished"
echo "Output: ${UNLEARN_OUTPUT_DIR}"
FINAL_CKPT="${UNLEARN_OUTPUT_DIR}/checkpoints/unlearned.ckpt"
echo "Final ckpt: ${FINAL_CKPT}"

UNLEARN_RUN_POST_EVAL="${UNLEARN_RUN_POST_EVAL:-true}"
# Set by resolve_unlearn_dataset for short names; else default or env override.
UNLEARN_EVAL_DATA_DIR="${UNLEARN_EVAL_DATA_DIR:-src/data/amazon_data/beauty}"

if [ "${UNLEARN_RUN_POST_EVAL}" = "true" ]; then
  if [ ! -f "${FINAL_CKPT}" ]; then
    echo "WARNING: skipping post-unlearn eval — final ckpt not found: ${FINAL_CKPT}"
  elif [ ! -d "${UNLEARN_EVAL_DATA_DIR}/testing" ]; then
    echo "WARNING: skipping post-unlearn eval — missing ${UNLEARN_EVAL_DATA_DIR}/testing/"
  else
    EVAL_OUT="${UNLEARN_OUTPUT_DIR}/eval"
    mkdir -p "${EVAL_OUT}"
    echo "[$(date -Is)] Post-unlearn eval on clean test: ${UNLEARN_EVAL_DATA_DIR}"
    python -u -m scripts.eval_ckpt_on_test \
      experiment=tiger_train_flat \
      data_dir="${UNLEARN_EVAL_DATA_DIR}" \
      "semantic_id_path='${SEMANTIC_ID_PATH}'" \
      "ckpt_path='${FINAL_CKPT}'" \
      num_hierarchies=4 \
      seed="${UNLEARN_SEED}" \
      train=False test=True \
      trainer.devices=1 \
      trainer.strategy=auto \
      trainer.sync_batchnorm=false \
      trainer.num_nodes=1 \
      trainer.deterministic=true \
      callbacks.model_checkpoint=null \
      callbacks.early_stopping=null \
      hydra.run.dir="${EVAL_OUT}"
    echo "[$(date -Is)] Post-unlearn eval finished"
    echo "Metrics: ${EVAL_OUT}/csv/version_0/metrics.csv"
  fi
else
  echo "Post-unlearn eval skipped (UNLEARN_RUN_POST_EVAL=${UNLEARN_RUN_POST_EVAL})"
fi
