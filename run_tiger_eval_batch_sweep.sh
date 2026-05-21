#!/usr/bin/env bash
#SBATCH --job-name=tiger_eval_sweep
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --time=12:00:00

set -euo pipefail

# -----------------------------------------------------------------------------
# Evaluate clean + poisoned + many sequential-unlearned checkpoints (one per
# request batch size) on the same clean test split, then print one combined
# relative-utility table.
#
# Usage:
#   sbatch run_tiger_eval_batch_sweep.sh <clean_ckpt> <poisoned_ckpt> \
#       [semantic_id_path] [data_dir] [batch_size ...]
#
# If no batch sizes are given, discovers every logs/unlearn/runs/job*_bs<N> dir
# that already contains checkpoints/unlearned.ckpt (latest dir per N).
#
# Example (explicit batch sizes):
#   sbatch run_tiger_eval_batch_sweep.sh \
#       logs/train/runs/2026-05-13/13-53-31/checkpoints/checkpoint_epoch=000_step=003300.ckpt \
#       logs/train/runs/2026-05-13/13-01-47/checkpoints/checkpoint_epoch=000_step=004400.ckpt \
#       embeddings/beauty/merged_predictions_tensor.pt \
#       src/data/amazon_data/beauty \
#       1 2 4 8 16 32 64 128 256
# -----------------------------------------------------------------------------

CLEAN_CKPT="${1:-}"
POISON_CKPT="${2:-}"
SEMANTIC_ID_PATH="${3:-}"
DATA_DIR="${4:-src/data/amazon_data/beauty}"
shift $(( $# < 4 ? $# : 4 )) || true
BATCH_SIZES=("$@")

if [ -z "${CLEAN_CKPT}" ] || [ -z "${POISON_CKPT}" ]; then
  echo "Usage: sbatch $0 <clean_ckpt> <poisoned_ckpt> [semantic_id_path] [data_dir] [batch_size ...]"
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
  if [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
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

for p in "${CLEAN_CKPT}" "${POISON_CKPT}"; do
  if [ ! -f "${p}" ]; then
    echo "Checkpoint not found: ${p}"
    exit 1
  fi
done
if [ ! -d "${DATA_DIR}/testing" ]; then
  echo "Expected ${DATA_DIR}/testing/"
  exit 1
fi

resolve_seq_ckpt_for_bs() {
  local bs="$1"
  local run_dir ckpt
  run_dir="$(ls -1dt "${GRID_DIR}"/logs/unlearn/runs/job*_bs"${bs}" 2>/dev/null | head -1 || true)"
  if [ -z "${run_dir}" ]; then
    return 1
  fi
  ckpt="${run_dir}/checkpoints/unlearned.ckpt"
  if [ ! -f "${ckpt}" ] && [ ! -L "${ckpt}" ]; then
    return 1
  fi
  printf '%s\n' "${ckpt}"
}

if [ "${#BATCH_SIZES[@]}" -eq 0 ]; then
  mapfile -t BATCH_SIZES < <(
    ls -1d "${GRID_DIR}"/logs/unlearn/runs/job*_bs* 2>/dev/null \
      | sed -n 's/.*_bs\([0-9][0-9]*\)$/\1/p' \
      | sort -n -u
  )
  if [ "${#BATCH_SIZES[@]}" -eq 0 ]; then
    echo "No batch sizes given and none discovered under logs/unlearn/runs/job*_bs*"
    exit 1
  fi
  echo "Auto-discovered batch sizes: ${BATCH_SIZES[*]}"
fi

declare -a SEQ_BS=()
declare -a SEQ_CKPTS=()
for bs in "${BATCH_SIZES[@]}"; do
  if ! [[ "${bs}" =~ ^[0-9]+$ ]]; then
    echo "Invalid batch size (expected integer): ${bs}"
    exit 1
  fi
  ckpt="$(resolve_seq_ckpt_for_bs "${bs}" || true)"
  if [ -z "${ckpt}" ]; then
    echo "WARNING: skipping bs=${bs} — no finished unlearned.ckpt under logs/unlearn/runs/job*_bs${bs}"
    continue
  fi
  SEQ_BS+=("${bs}")
  SEQ_CKPTS+=("${ckpt}")
  echo "  bs=${bs} -> ${ckpt}"
done

if [ "${#SEQ_CKPTS[@]}" -eq 0 ]; then
  echo "No sequential checkpoints ready to evaluate."
  exit 1
fi

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
BASE="${GRID_DIR}/logs/eval/batch_sweep/${STAMP}"
mkdir -p "${BASE}"

echo "[$(date -Is)] Batch-sweep eval  seed=${EVAL_SEED}  out_base=${BASE}"
echo "  clean_reference: ${CLEAN_CKPT}"
echo "  poisoned:        ${POISON_CKPT}"

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
for i in "${!SEQ_BS[@]}"; do
  run_one "seq_bs${SEQ_BS[$i]}" "${SEQ_CKPTS[$i]}"
done

REF_CSV="${BASE}/clean_ref/csv/version_0/metrics.csv"
POI_CSV="${BASE}/poisoned/csv/version_0/metrics.csv"
for f in "${REF_CSV}" "${POI_CSV}"; do
  if [ ! -f "${f}" ]; then
    echo "Missing metrics.csv: ${f}"
    exit 1
  fi
done

RUN_ARGS=(--reference "${REF_CSV}" --run "poisoned:${POI_CSV}")
for i in "${!SEQ_BS[@]}"; do
  seq_csv="${BASE}/seq_bs${SEQ_BS[$i]}/csv/version_0/metrics.csv"
  if [ ! -f "${seq_csv}" ]; then
    echo "Missing metrics.csv: ${seq_csv}"
    exit 1
  fi
  RUN_ARGS+=(--run "seq_bs${SEQ_BS[$i]}:${seq_csv}")
done

echo "[$(date -Is)] Combined relative utility (reference = clean)"
python -u -m scripts.compute_relative_utility \
  "${RUN_ARGS[@]}" \
  --out_json "${BASE}/relative_utility.json"

{
  echo "batch_sizes=${SEQ_BS[*]}"
  echo "clean_ckpt=${CLEAN_CKPT}"
  echo "poisoned_ckpt=${POISON_CKPT}"
  for i in "${!SEQ_BS[@]}"; do
    echo "seq_bs${SEQ_BS[$i]}_ckpt=${SEQ_CKPTS[$i]}"
  done
} > "${BASE}/manifest.txt"

echo "[$(date -Is)] Done. Artifacts under ${BASE}"
