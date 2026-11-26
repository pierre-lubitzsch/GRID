#!/usr/bin/env bash
#SBATCH --job-name=fig1_js
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
legacy_args=()
if [[ "${LEGACY_BEAUTY2:-0}" == "1" ]]; then legacy_args+=(--legacy-beauty2); fi
if [[ "${ALL_UNRELATED:-0}" == "1" ]]; then legacy_args+=(--all-unrelated); fi
if [[ -n "${OUT_STEM:-}" ]]; then legacy_args+=(--out-stem "${OUT_STEM}"); fi
if [[ -n "${NEIGHBOR_DEF:-}" ]]; then legacy_args+=(--neighbor-def "${NEIGHBOR_DEF}"); fi
python -u scripts/run_fig1_js_eval.py --dataset "${DATASET:?}" --seed "${SEED:?}" \
  --stratum "${STRATUM:-mid}" \
  --n-queries "${N_QUERIES:-16}" --item-chunk "${ITEM_CHUNK:-128}" \
  --metric "${METRIC:-js}" --max-neighbors "${MAX_NEIGHBORS:-256}" \
  --scope-batch-size "${SCOPE_BATCH_SIZE:-16}" "${legacy_args[@]}"
