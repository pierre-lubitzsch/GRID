#!/usr/bin/env bash
#SBATCH --job-name=tiger_eval_3way
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=08:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# Evaluate three TIGER checkpoints on the same (clean) test split, then run
# compute_relative_utility (clean reference vs sequential-final vs poisoned).
#
# Usage:
#   sbatch run_tiger_eval_three_way.sh <sequential_final_ckpt> <clean_ref_ckpt> \
#       <poisoned_ckpt> [semantic_id_path] [data_dir]
#
# Defaults: semantic_id and data_dir follow run_tiger_eval.sh. Eval seed defaults
# to 2 (override with EVAL_SEED=...).
#
# Example:
#   sbatch run_tiger_eval_three_way.sh \
#       logs/unlearn/runs/2026-05-08/.../checkpoints/unlearned.ckpt \
#       logs/train/runs/2026-04-01/19-22-14/checkpoints/checkpoint_epoch=000_step=004600.ckpt \
#       logs/train/runs/2026-05-06/16-02-49/checkpoints/checkpoint_best.ckpt \
#       embeddings/beauty/merged_predictions_tensor.pt \
#       src/data/amazon_data/beauty
# -----------------------------------------------------------------------------

SEQ_CKPT="${1:-}"
CLEAN_CKPT="${2:-}"
POISON_CKPT="${3:-}"
SEMANTIC_ID_PATH="${4:-}"
DATA_DIR="${5:-src/data/amazon_data/beauty}"

if [ -z "${SEQ_CKPT}" ] || [ -z "${CLEAN_CKPT}" ] || [ -z "${POISON_CKPT}" ]; then
  echo "Usage: sbatch $0 <sequential_final_ckpt> <clean_ref_ckpt> <poisoned_ckpt> [semantic_id_path] [data_dir]"
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

for p in "${SEQ_CKPT}" "${CLEAN_CKPT}" "${POISON_CKPT}"; do
  if [ ! -f "${p}" ]; then
    echo "Checkpoint not found: ${p}"
    exit 1
  fi
done
if [ ! -d "${DATA_DIR}/testing" ]; then
  echo "Expected ${DATA_DIR}/testing/"
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BASE="${GRID_DIR}/logs/eval/three_way/${STAMP}"
mkdir -p "${BASE}"

echo "[$(date -Is)] Three-way eval  seed=${EVAL_SEED}  out_base=${BASE}"
echo "  sequential_final: ${SEQ_CKPT}"
echo "  clean_reference:  ${CLEAN_CKPT}"
echo "  poisoned:         ${POISON_CKPT}"

run_one() {
  local label="$1"
  local ckpt_path="$2"
  local out_dir="${BASE}/${label}"
  mkdir -p "${out_dir}"
  echo "[$(date -Is)] Evaluating [${label}] -> ${out_dir}"
  python -u -m scripts.eval_ckpt_on_test \
    experiment=tiger_train_flat \
    data_dir="${DATA_DIR}" \
    "semantic_id_path='${SEMANTIC_ID_PATH}'" \
    "ckpt_path='${ckpt_path}'" \
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
    hydra.run.dir="${out_dir}"
}

run_one "clean_ref" "${CLEAN_CKPT}"
run_one "poisoned" "${POISON_CKPT}"
run_one "sequential_final" "${SEQ_CKPT}"

REF_CSV="${BASE}/clean_ref/csv/version_0/metrics.csv"
SEQ_CSV="${BASE}/sequential_final/csv/version_0/metrics.csv"
POI_CSV="${BASE}/poisoned/csv/version_0/metrics.csv"

for f in "${REF_CSV}" "${SEQ_CSV}" "${POI_CSV}"; do
  if [ ! -f "${f}" ]; then
    echo "Missing metrics.csv: ${f}"
    exit 1
  fi
done

echo "[$(date -Is)] Relative utility (reference = clean train-on-clean ckpt)"
python -u -m scripts.compute_relative_utility \
  --reference "${REF_CSV}" \
  --unlearned "${SEQ_CSV}" \
  --label_unlearned sequential_final \
  --extra "${POI_CSV}" \
  --label_extra poisoned \
  --out_json "${BASE}/relative_utility.json"

echo "[$(date -Is)] Done. Artifacts under ${BASE}"
