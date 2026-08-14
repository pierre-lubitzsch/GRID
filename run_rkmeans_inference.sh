#!/usr/bin/env bash
#SBATCH --job-name=rkmeans_inference
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --time=2-00:00:00
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:2

set -euo pipefail

# Step 3b: run RKMeans inference using a trained checkpoint from Step 3a.
#
# Usage:
#   sbatch run_rkmeans_inference.sh <ckpt_path> [dataset] [embedding_path]
# Example:
#   sbatch run_rkmeans_inference.sh checkpoints/last.ckpt beauty

CKPT_PATH="${1:-}"
DATASET="${2:-beauty}"
EMBEDDING_PATH="${3:-}"

if [ -z "${CKPT_PATH}" ]; then
  echo "Missing ckpt_path."
  echo "Usage: sbatch run_rkmeans_inference.sh <ckpt_path> [dataset] [embedding_path]"
  exit 1
fi

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ ! -f "${EMBEDDING_PATH}" ]; then
  echo "Embedding file not found: ${EMBEDDING_PATH}"
  echo "Pass an explicit embedding_path as third arg if needed."
  exit 1
fi

if [ -z "${EMBEDDING_PATH}" ]; then
  EMBEDDING_PATH="$(ls -t logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -n1 || true)"
fi

# Must match run_rkmeans_train.sh (the checkpoint's codebooks have this shape).
# For final ID length L in {8, 16} with fixed codebook size 256, set
# RKMEANS_HIER=L-1 (7 or 15); the dedup step below appends the +1 digit.
RKMEANS_HIER="${RKMEANS_HIER:-3}"
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"

echo "[$(date -Is)] Starting rkmeans inference on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR}"
echo "Codebooks: num_hierarchies=${RKMEANS_HIER} codebook_width=${CODEBOOK_WIDTH} (final ID length = ${RKMEANS_HIER}+1 after dedup)"

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
# an inference config (rvq currently has no inference counterpart), and a missing
# experiment otherwise surfaces as an opaque Hydra composition error.
_EXP_CFG="configs/experiment/${QUANTIZER}_inference_flat.yaml"
if [ ! -f "${GRID_DIR:-$PWD}/${_EXP_CFG}" ] && [ ! -f "${_EXP_CFG}" ]; then
  echo "No inference config for QUANTIZER=${QUANTIZER} (expected ${_EXP_CFG})." >&2
  echo "Available: $(ls configs/experiment/*_inference_flat.yaml 2>/dev/null | xargs -n1 basename | sed 's/_inference_flat.yaml//' | tr '\n' ' ')" >&2
  exit 1
fi
echo "Quantizer: ${QUANTIZER}"

# Pin the Hydra run dir to a dataset+job-scoped path rather than guessing it
# afterwards. The old code did
#     LATEST_RUN_DIR="$(ls -d logs/inference/runs/*/* | sort | tail -1)"
# which has NO dataset filter, so two concurrent runs (e.g. toys and sports)
# would both merge whichever dir sorted last -- silently assigning one dataset's
# semantic IDs from the other's predictions. That glob also matches the
# `logs/inference/runs/embeddings/<ds>_<jobid>` dirs written by
# generate_embeddings.sh, so it could even merge raw embedding shards.
RUN_TAG="${DATASET}_${QUANTIZER}_${SLURM_JOB_ID:-$(date +%Y%m%d-%H%M%S)}"
LATEST_RUN_DIR="logs/inference/runs/sid/${RUN_TAG}"
PICKLE_DIR="${LATEST_RUN_DIR}/pickle"
export PICKLE_DIR
mkdir -p "${PICKLE_DIR}"
echo "Run dir: ${LATEST_RUN_DIR}"

python -u -m src.inference \
  experiment=${QUANTIZER}_inference_flat \
  data_dir="${GRID_DATA_DIR}" \
  hydra.run.dir="${LATEST_RUN_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies="${RKMEANS_HIER}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  "ckpt_path='${CKPT_PATH}'" \
  callbacks.pickle_writer.should_merge_files_on_main=false

echo "[$(date -Is)] rkmeans inference finished, merging pickle shards..."

if [ ! -d "${PICKLE_DIR}" ]; then
  echo "Pickle directory missing: ${PICKLE_DIR}"
  exit 1
fi
if [ -z "$(ls -A "${PICKLE_DIR}"/*.pkl 2>/dev/null)" ]; then
  echo "No pickle shards in ${PICKLE_DIR} -- inference did not write to the pinned run dir."
  exit 1
fi

python - <<'PY'
import os
import pickle
import torch
from src.utils.tensor_utils import (
    deduplicate_rows_in_tensor,
    merge_list_of_keyed_tensors_to_single_tensor,
    transpose_tensor_from_file,
)

pickle_dir = os.environ["PICKLE_DIR"]
files = sorted(
    [f for f in os.listdir(pickle_dir) if f.startswith("predictions_") and f.endswith(".pkl")]
)
if not files:
    raise RuntimeError(f"No pickle shard files found in {pickle_dir}")

merged = []
for name in files:
    with open(os.path.join(pickle_dir, name), "rb") as fh:
        merged.extend(pickle.load(fh))

with open(os.path.join(pickle_dir, "merged_predictions.pkl"), "wb") as fh:
    pickle.dump(merged, fh)

tensor = merge_list_of_keyed_tensors_to_single_tensor(
    data=merged,
    index_key="item_id",
    value_key="cluster_ids",
)
pt_path = os.path.join(pickle_dir, "merged_predictions_tensor.pt")
torch.save(tensor.cpu(), pt_path)

# Match rkmeans_inference_flat post-processing behavior.
deduplicate_rows_in_tensor(file_path=pt_path)
transpose_tensor_from_file(file_path=pt_path)

print(f"Merged {len(merged)} rows into {pt_path}")
PY

echo "[$(date -Is)] Merge complete: ${PICKLE_DIR}/merged_predictions_tensor.pt"
# Install guard for longer IDs: the canonical embeddings/<dataset>/ path is the
# L=4 tensor (RKMEANS_HIER=3 + dedup digit) that all standard runs read. A
# non-default RKMEANS_HIER would silently OVERWRITE it with a different-L tensor,
# so install those to the per-L dir embeddings/<dataset>_L<L>/ instead (same
# layout run_generate_sid.sh uses).
if [ "${RKMEANS_HIER}" = "3" ]; then
  bash "${GRID_DIR}/scripts/install_semantic_id_tensor.sh" "${DATASET}" "${PICKLE_DIR}/merged_predictions_tensor.pt"
else
  L=$(( RKMEANS_HIER + 1 ))
  DEST_DIR="embeddings/${DATASET}_L${L}"
  mkdir -p "${DEST_DIR}"
  cp -f "${PICKLE_DIR}/merged_predictions_tensor.pt" "${DEST_DIR}/merged_predictions_tensor.pt"
  echo "Installed L=${L} tensor -> ${DEST_DIR}/merged_predictions_tensor.pt (canonical embeddings/${DATASET}/ left untouched)"
fi
