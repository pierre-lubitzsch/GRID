#!/usr/bin/env bash
#SBATCH --job-name=rsc15_poison
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --partition=cpu

set -euo pipefail

# ERASE-style bandwagon spam on converted RSC15 (TFRecord layout), then forget/retain split.
#
# Matches beauty_spam_seed2_pct1_n10 / ERASE --spam defaults:
#   seed=2, poisoning_ratio=0.01, n_target_items=10, sprinkled placement,
#   p_two_targets=0.119 (rsc15 fraud session target-click distribution).
#
# Usage:
#   ./run_rsc15_poison.sh                    # default: src/data/erase_data/rsc15
#   ./run_rsc15_poison.sh rsc15              # short name (resolve_grid_dataset.sh)
#   ./run_rsc15_poison.sh rsc15_smoke
#   ./run_rsc15_poison.sh src/data/erase_data/rsc15   # explicit path
#
# Fast path (default): stats from src/data/rsc15.inter (one pandas pass, like ERASE
# create_fraud_sessions_sbr.py) instead of scanning all training TFRecords.
# Forget/retain split copies shards by name (no second 8M-row scan).
#
# Env overrides:
#   POISON_SEED=2  POISONING_RATIO=0.01  N_TARGET_ITEMS=10  OVERWRITE=1
#   STATS_INTER=src/data/rsc15.inter  N_CLEAN_USERS=7990324
#   USE_TFRECORD_SCAN=1  # slow legacy path (two full TFRecord passes)

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

POISON_SEED="${POISON_SEED:-2}"
POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET_ITEMS="${N_TARGET_ITEMS:-10}"
ROWS_PER_SHARD="${ROWS_PER_SHARD:-5000}"
OVERWRITE="${OVERWRITE:-}"

ARG="${1:-rsc15}"
# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"

if [[ "${ARG}" == */* ]]; then
  CLEAN_DIR="${ARG}"
  BASE="$(basename "${CLEAN_DIR}")"
  PARENT="$(dirname "${CLEAN_DIR}")"
  PCT="$(python3 -c "print(int(round(${POISONING_RATIO} * 100)))")"
  OUT_DIR="${PARENT}/${BASE}_spam_seed${POISON_SEED}_pct${PCT}_n${N_TARGET_ITEMS}"
else
  if ! resolve_grid_dataset "${ARG}"; then
    exit 1
  fi
  CLEAN_DIR="${GRID_DATA_DIR}"
  OUT_DIR="${GRID_POISON_DATA_DIR}"
fi

if [ ! -d "${CLEAN_DIR}/training" ]; then
  echo "Clean dataset not found: ${CLEAN_DIR}/training"
  echo "Pass a path with '/' or a known short name: rsc15, rsc15_smoke, beauty, ..."
  exit 1
fi

STATS_INTER="${STATS_INTER:-src/data/rsc15.inter}"
N_CLEAN_USERS="${N_CLEAN_USERS:-}"
if [ -z "${N_CLEAN_USERS}" ] && [ -f "${CLEAN_DIR}/dataset_meta.json" ]; then
  N_CLEAN_USERS="$(python3 -c "import json; print(json.load(open('${CLEAN_DIR}/dataset_meta.json'))['splits']['training'])")"
fi

BW_ARGS=(
  --data_dir "${CLEAN_DIR}"
  --out_dir "${OUT_DIR}"
  --attack bandwagon
  --target_strategy unpopular
  --poisoning_ratio "${POISONING_RATIO}"
  --n_target_items "${N_TARGET_ITEMS}"
  --placement sprinkled
  --p_two_targets 0.119
  --seed "${POISON_SEED}"
  --rows_per_shard "${ROWS_PER_SHARD}"
)
if [ -z "${USE_TFRECORD_SCAN:-}" ] && [ -f "${STATS_INTER}" ]; then
  echo "[$(date -Is)] ERASE-fast stats from ${STATS_INTER} (n_clean_users=${N_CLEAN_USERS:-<from .inter>})"
  BW_ARGS+=(--stats-inter "${STATS_INTER}")
  if [ -n "${N_CLEAN_USERS}" ]; then
    BW_ARGS+=(--n-clean-users "${N_CLEAN_USERS}")
  fi
else
  echo "[$(date -Is)] Legacy path: scanning all training TFRecords for stats"
fi
if [ -n "${OVERWRITE}" ]; then
  BW_ARGS+=(--overwrite)
fi

echo "[$(date -Is)] Bandwagon poison: ${CLEAN_DIR} -> ${OUT_DIR}"
python -u -m src.data.poisoning.bandwagon "${BW_ARGS[@]}"

echo "[$(date -Is)] Splitting training_forget / training_retain"
SPLIT_ARGS=(
  --data_dir "${OUT_DIR}"
  --forget_manifest "${OUT_DIR}/forget_manifest.json"
)
if [ -z "${USE_TFRECORD_SCAN:-}" ]; then
  SPLIT_ARGS+=(--segregated-shards)
fi
if [ -n "${OVERWRITE}" ]; then
  SPLIT_ARGS+=(--overwrite)
fi
python -u -m src.data.unlearning.split_forget_retain "${SPLIT_ARGS[@]}"

echo "[$(date -Is)] Done."
echo "  Poisoned dataset: ${OUT_DIR}"
echo "  Next: embeddings + RKMeans on clean items, then:"
echo "    sbatch run_tiger_train.sh rsc15 poison <semantic_id_path>"
echo "    sbatch run_tiger_train.sh rsc15 clean <semantic_id_path>"
echo "    sbatch run_tiger_unlearn_sequential.sh <poison_ckpt> rsc15 <semantic_id_path> true 8 1.0"
