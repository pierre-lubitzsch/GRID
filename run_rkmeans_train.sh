#!/usr/bin/env bash
#SBATCH --job-name=rkmeans_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:2

set -euo pipefail

# Step 3a: train RKMeans codebooks from embeddings.
#
# Usage:
#   sbatch run_rkmeans_train.sh [dataset] [embedding_path]
# Example:
#   sbatch run_rkmeans_train.sh beauty logs/inference/runs/<step2_run>/pickle/merged_predictions_tensor.pt

DATASET="${1:-beauty}"
EMBEDDING_PATH="${2:-}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ -z "${EMBEDDING_PATH}" ]; then
  # Resolve the DENSE CONTENT embeddings for THIS dataset. Two traps here, both
  # hit in practice (job 10451266 trained "beauty" on food's tensor):
  #  * no dataset filter -> `ls -t` returns whatever ran most recently, which is
  #    another dataset's embeddings whenever two builds overlap;
  #  * `logs/inference/runs/*/*` also matches `sid/<ds>_.../`, which holds the
  #    OUTPUT semantic ids ([L, N] ints), not the [N, D] input embeddings. Feeding
  #    those in dies deep inside the dataloader with an opaque IndexError.
  # Prefer the canonical per-dataset tensor, then this dataset's embeddings runs.
  EMBEDDING_PATH="embeddings/${DATASET}_merged_predictions_tensor_latest.pt"
  if [ ! -f "${EMBEDDING_PATH}" ]; then
    EMBEDDING_PATH="$(ls -t logs/inference/runs/embeddings/${DATASET}_*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -n1 || true)"
  fi
fi
if [ ! -f "${EMBEDDING_PATH}" ]; then
  echo "Embedding file not found: ${EMBEDDING_PATH:-<empty>}"
  exit 1
fi

# Number of RKMeans codebooks and codebook size. Defaults reproduce the standard
# TIGER setup (3 codebooks -> +1 dedup digit at inference -> 4 hierarchies).
# "Longer IDs for finer-grained neighborhoods" (Jul 3): for final ID length
# L in {8, 16} with codebook size FIXED at 256, set RKMEANS_HIER=L-1 (7 or 15)
# and keep CODEBOOK_WIDTH=256. run_rkmeans_inference.sh MUST use the same values.
RKMEANS_HIER="${RKMEANS_HIER:-3}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"

echo "[$(date -Is)] Starting rkmeans train on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR}"
echo "Using embedding_path=${EMBEDDING_PATH}"
# Shape guard: content embeddings are [N_items, D] with D in the hundreds/thousands.
# A semantic-id tensor is [L, N] with L ~ 3-4. Catch the mix-up HERE, with a
# readable message, instead of as an IndexError inside a dataloader worker.
python3 - "${EMBEDDING_PATH}" "${DATASET}" <<'PYEOF' || exit 1
import sys, torch
t = torch.load(sys.argv[1], map_location="cpu")
if isinstance(t, dict):
    t = next(iter(t.values()))
t = torch.as_tensor(t)
if t.dim() != 2 or min(t.shape) < 16:
    raise SystemExit(
        f"embedding_path {sys.argv[1]} has shape {tuple(t.shape)}, which is not a "
        f"[N_items, D] content-embedding tensor for '{sys.argv[2]}'. A [L, N] "
        "semantic-id tensor from logs/inference/runs/sid/ is the usual mix-up."
    )
print(f"  embedding tensor OK: {tuple(t.shape)} ({t.dtype})")
PYEOF
echo "Codebooks: num_hierarchies=${RKMEANS_HIER} codebook_width=${CODEBOOK_WIDTH} (final ID length = ${RKMEANS_HIER}+1 after dedup)"

LOCAL_CKPT_DIR="${TMPDIR:-/tmp}/rkmeans_ckpts_${SLURM_JOB_ID:-$$}"
mkdir -p "${LOCAL_CKPT_DIR}"

# Quantizer choice. rkmeans (DEFAULT) reproduces every existing semantic ID;
# rqvae / rvq select the autoencoder + VectorQuantization stacks instead. The
# default is unchanged, so existing invocations behave exactly as before.
# TRAIN and INFERENCE must use the SAME value -- a checkpoint trained with one
# quantizer cannot be loaded by another (different module tree).
QUANTIZER="${QUANTIZER:-rkmeans}"
case "${QUANTIZER}" in
  rkmeans|rqvae|rvq) ;;
  *) echo "QUANTIZER must be rkmeans|rqvae|rvq, got '${QUANTIZER}'" >&2; exit 1 ;;
esac
# Fail fast with a useful message: not every quantizer has both a train and
# an inference config (rvq currently has no train counterpart), and a missing
# experiment otherwise surfaces as an opaque Hydra composition error.
_EXP_CFG="configs/experiment/${QUANTIZER}_train_flat.yaml"
if [ ! -f "${GRID_DIR:-$PWD}/${_EXP_CFG}" ] && [ ! -f "${_EXP_CFG}" ]; then
  echo "No train config for QUANTIZER=${QUANTIZER} (expected ${_EXP_CFG})." >&2
  echo "Available: $(ls configs/experiment/*_train_flat.yaml 2>/dev/null | xargs -n1 basename | sed 's/_train_flat.yaml//' | tr '\n' ' ')" >&2
  exit 1
fi
echo "Quantizer: ${QUANTIZER}"

# Pin the Hydra run dir instead of guessing where the checkpoint should land.
# The old code did
#     LATEST_RUN_DIR="$(ls -d logs/train/runs/*/* | sort | tail -1)"
# with NO dataset filter, so two concurrent codebook trainings (e.g. toys and
# sports) could copy one dataset's checkpoint into the other's run dir -- and
# every semantic ID built from it would then be silently wrong. It also picks a
# different dir per DDP rank. (Jobs 10385160/61 happened to land correctly only
# because their copy steps did not overlap; verified after the fact against
# each run's .hydra/config.yaml embedding_path.)
RUN_TAG="${DATASET}_${QUANTIZER}_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
LATEST_RUN_DIR="${GRID_DIR}/logs/train/runs/codebook/${RUN_TAG}"
mkdir -p "${LATEST_RUN_DIR}"
echo "Run dir: ${LATEST_RUN_DIR}"

python -u -m src.train \
  experiment=${QUANTIZER}_train_flat \
  data_dir="${GRID_DATA_DIR}" \
  hydra.run.dir="${LATEST_RUN_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies="${RKMEANS_HIER}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  "callbacks.model_checkpoint.dirpath=${LOCAL_CKPT_DIR}" \
  "${@:3}"

if [ -n "${LATEST_RUN_DIR}" ] && ls "${LOCAL_CKPT_DIR}"/*.ckpt &>/dev/null; then
  mkdir -p "${LATEST_RUN_DIR}/checkpoints"
  cp "${LOCAL_CKPT_DIR}"/*.ckpt "${LATEST_RUN_DIR}/checkpoints/"
  echo "[$(date -Is)] Checkpoint copied to ${LATEST_RUN_DIR}/checkpoints/"
else
  echo "[$(date -Is)] ERROR: no checkpoint found in ${LOCAL_CKPT_DIR}"
  exit 1
fi

echo "[$(date -Is)] rkmeans train finished"
