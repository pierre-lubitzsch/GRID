#!/usr/bin/env bash
# Submit sequential TIGER unlearning jobs with neighborhood-aware retain sampling.
# Sweeps request_batch_size; all neighborhood knobs are set in the block below.
#
# Usage (from repo root):
#   bash submit_tiger_unlearn_neighborhood_bs.sh
#
# Optional env before launch:
#   UNLEARN_SEED=2          # global Hydra seed (retain shuffle + SCIF)
#   UNLEARN_RUN_POST_EVAL=false

set -euo pipefail

# -----------------------------------------------------------------------------
# Paths (required)
# -----------------------------------------------------------------------------
CKPT="logs/train/runs/2026-05-13/13-01-47/checkpoints/checkpoint_epoch=000_step=004400.ckpt"
DATASET="beauty"               # beauty | sports | toys | rsc15 | rsc15_smoke (see resolve_unlearn_dataset.sh)
SID=""                         # empty = default for DATASET from registry

# -----------------------------------------------------------------------------
# Sequential unlearning
# -----------------------------------------------------------------------------
REQUEST_BATCH_SIZES=(1 2 4 8 16 32 64 128 256)
# REQUEST_BATCH_SIZES=(8)  # single job for a quick smoke test

# -----------------------------------------------------------------------------
# Neighborhood-aware retain sampler (see neighborhood_sampler.py)
# -----------------------------------------------------------------------------
RETAIN_SAMPLE_SIZE=16          # budget multiplier: retain rows cap = this * |D_f| per request
RETAIN_MAX_ROWS=""             # optional hard cap after shuffle; empty = no extra cap
PROGRESSIVE_SID_PREFIX=true    # true: prefix lengths k-1 .. 1; false: fixed sid_prefix_length
SID_PREFIX_LENGTH=2            # only used when PROGRESSIVE_SID_PREFIX=false
NEIGHBORHOOD_AWARE_SAMPLE_RATE=1.0  # 1=nbh only, 0=uniform only, 0.5=mixed

# -----------------------------------------------------------------------------
# SCIF / other unlearning knobs (passed as Hydra overrides)
# -----------------------------------------------------------------------------
TARGET_PARAMS=all              # all | sid_embeddings | encoder_only
UPDATE_MAX_NORM=1.0
BATCH_SIZE_PER_DEVICE=256

# Extra Hydra overrides (optional), e.g.:
# EXTRA_OVERRIDES=(unlearning.max_requests=4 unlearning.cg_max_iter=100)
EXTRA_OVERRIDES=()

# -----------------------------------------------------------------------------
# SLURM
# -----------------------------------------------------------------------------
SLURM_PARTITION=gpu
SLURM_GPUS=1
SLURM_CPUS=8

# -----------------------------------------------------------------------------

NEIGHBORHOOD_OVERRIDES=(
  "unlearning.retain_sample_size=${RETAIN_SAMPLE_SIZE}"
  "unlearning.progressive_sid_prefix=${PROGRESSIVE_SID_PREFIX}"
  "unlearning.sid_prefix_length=${SID_PREFIX_LENGTH}"
  "unlearning.target_params=${TARGET_PARAMS}"
  "unlearning.update_max_norm=${UPDATE_MAX_NORM}"
  "unlearning.batch_size_per_device=${BATCH_SIZE_PER_DEVICE}"
)

if [ -n "${RETAIN_MAX_ROWS}" ]; then
  NEIGHBORHOOD_OVERRIDES+=("unlearning.retain_max_rows=${RETAIN_MAX_ROWS}")
fi

for bs in "${REQUEST_BATCH_SIZES[@]}"; do
  TAG="${DATASET}_nbh_bs${bs}_rs${RETAIN_SAMPLE_SIZE}_rate${NEIGHBORHOOD_AWARE_SAMPLE_RATE}"
  if [ -n "${RETAIN_MAX_ROWS}" ]; then
    TAG="${TAG}_cap${RETAIN_MAX_ROWS}"
  fi
  if [ "${PROGRESSIVE_SID_PREFIX}" = "false" ]; then
    TAG="${TAG}_pfx${SID_PREFIX_LENGTH}"
  fi

  echo "Submitting dataset=${DATASET} bs=${bs} rate=${NEIGHBORHOOD_AWARE_SAMPLE_RATE} (tag=${TAG})"
  sbatch -p "${SLURM_PARTITION}" --gpus-per-node="${SLURM_GPUS}" --cpus-per-task="${SLURM_CPUS}" \
    --job-name="unlearn_${TAG}" \
    run_tiger_unlearn_sequential.sh \
      "${CKPT}" "${DATASET}" "${SID}" \
      true \
      "${bs}" \
      "${NEIGHBORHOOD_AWARE_SAMPLE_RATE}" \
      "unlearning_run_tag=${TAG}" \
      "${NEIGHBORHOOD_OVERRIDES[@]}" \
      "${EXTRA_OVERRIDES[@]}"
done
