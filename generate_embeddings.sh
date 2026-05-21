#!/usr/bin/env bash
#SBATCH --job-name=generate_embeddings
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
# Cluster default for gpu/pgpu is 2h if you omit --time; we request the partition max (2 days).
#SBATCH --time=2-00:00:00
# Default: pgpu, 2× H200 (experiment uses DDP + devices=-1). Override if you need another layout.
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:2

set -euo pipefail

# Optional arg:
#   1) dataset: beauty (default), sports, toys, rsc15, rsc15_smoke
#
# Walltime: #SBATCH --time=2-00:00:00 (2 days). Override with: sbatch --time=08:00:00 ...
#
# Single-GPU (gpu partition), e.g. A100 80GB PCIe:
#   sbatch --partition=gpu --gres=gpu:nvidia_a100_80gb_pcie:1 generate_embeddings.sh beauty
#
# More H200s on pgpu:
#   sbatch --partition=pgpu --gres=gpu:nvidia_h200:4 generate_embeddings.sh beauty
#
# Multi-CPU: keep predict num_workers=0 (TensorFlow TFRecords + forked DataLoader workers break).
DATASET="${1:-beauty}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

echo "[$(date -Is)] Starting inference on dataset=${DATASET}"
echo "Using data_dir=${GRID_DATA_DIR} (items/)"

python -u -m src.inference \
  experiment=sem_embeds_inference_flat \
  data_dir="${GRID_DATA_DIR}" \
  data_loading.datamodule.predict_dataloader_config.num_workers=0 \
  data_loading.datamodule.predict_dataloader_config.timeout=0 \
  data_loading.datamodule.predict_dataloader_config.persistent_workers=false \
  callbacks.pickle_writer.should_merge_files_on_main=false

echo "[$(date -Is)] Inference finished, merging pickle shards..."

LATEST_RUN_DIR="$(ls -dt logs/inference/runs/*/* 2>/dev/null | head -n 1)"
PICKLE_DIR="${LATEST_RUN_DIR}/pickle"
export PICKLE_DIR

if [ -z "${LATEST_RUN_DIR}" ] || [ ! -d "${PICKLE_DIR}" ]; then
  echo "Could not find latest run pickle directory under logs/inference/runs."
  exit 1
fi

python - <<'PY'
import os
import pickle
import torch

pickle_dir = os.environ["PICKLE_DIR"]
files = sorted([f for f in os.listdir(pickle_dir) if f.endswith(".pkl")])
if not files:
    raise RuntimeError(f"No pickle shard files found in {pickle_dir}")

merged = []
for name in files:
    with open(os.path.join(pickle_dir, name), "rb") as fh:
        merged.extend(pickle.load(fh))

with open(os.path.join(pickle_dir, "merged_predictions.pkl"), "wb") as fh:
    pickle.dump(merged, fh)

# Sort by item_id so rows are deterministic; build a sequential tensor so the
# file works regardless of whether item IDs are dense-from-0 (Amazon) or
# sparse large integers (rsc15).  Always save as a dict with both "embeddings"
# and "item_ids" so load_dense_embeddings can reconstruct the raw-ID mapping.
merged_sorted = sorted(merged, key=lambda r: int(r["item_id"]))
item_ids = torch.tensor([int(r["item_id"]) for r in merged_sorted], dtype=torch.int64)
emb_list = [torch.tensor(r["embedding"]) for r in merged_sorted]
embeddings = torch.stack(emb_list).float()

payload = {"embeddings": embeddings.cpu(), "item_ids": item_ids.cpu()}
torch.save(payload, os.path.join(pickle_dir, "merged_predictions_tensor.pt"))
print(
    f"Merged {len(merged_sorted)} rows into {pickle_dir}/merged_predictions_tensor.pt "
    f"(item_id range {int(item_ids.min())}..{int(item_ids.max())})"
)
PY

echo "[$(date -Is)] Merge complete: ${PICKLE_DIR}/merged_predictions_tensor.pt"
echo "Install for training: bash scripts/install_semantic_id_tensor.sh ${DATASET} ${PICKLE_DIR}/merged_predictions_tensor.pt"
