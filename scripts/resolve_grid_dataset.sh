#!/usr/bin/env bash
# Resolve GRID data paths from a short dataset name.
#
# Usage:
#   source scripts/resolve_grid_dataset.sh
#   resolve_grid_dataset beauty
#
# Exports:
#   GRID_DATA_DIR          clean dataset (train / eval / test)
#   GRID_POISON_DATA_DIR   bandwagon-poisoned dataset (for unlearning)
#   GRID_SEMANTIC_ID_PATH  default semantic-id tensor for training / unlearning
#
# Override before calling:
#   GRID_DATA_DIR, GRID_POISON_DATA_DIR, GRID_SEMANTIC_ID_PATH
#
# Poison dataset naming is derived from env vars (all have defaults):
#   POISONING_RATIO=0.01  POISON_SEED=2  N_TARGET_ITEMS=10

resolve_grid_dataset() {
  local name="${1:-${DATASET:-}}"
  if [ -z "${name}" ]; then
    echo "resolve_grid_dataset: dataset name required" >&2
    return 1
  fi

  local clean="" poison="" sid=""

  # Compute poison suffix once from env vars.
  local _pct _seed _n _sfx
  _pct="$(python3 -c "print(int(round(${POISONING_RATIO:-0.01} * 100)))")"
  _seed="${POISON_SEED:-2}"
  _n="${N_TARGET_ITEMS:-10}"
  _sfx="_spam_seed${_seed}_pct${_pct}_n${_n}"

  case "${name}" in
    beauty)
      clean="src/data/amazon_data/beauty"
      poison="${clean}${_sfx}"
      sid="embeddings/beauty/merged_predictions_tensor.pt"
      ;;
    sports)
      clean="src/data/amazon_data/sports"
      poison="${clean}${_sfx}"
      sid="embeddings/sports/merged_predictions_tensor.pt"
      ;;
    toys)
      clean="src/data/amazon_data/toys"
      poison="${clean}${_sfx}"
      sid="embeddings/toys/merged_predictions_tensor.pt"
      ;;
    rsc15)
      clean="src/data/erase_data/rsc15"
      poison="${clean}${_sfx}"
      sid="embeddings/rsc15/merged_predictions_tensor.pt"
      ;;
    rsc15_smoke)
      clean="src/data/erase_data/rsc15_smoke"
      poison="${clean}${_sfx}"
      sid="embeddings/rsc15_smoke/merged_predictions_tensor.pt"
      ;;
    *)
      # Dynamic fallback: resolve from directory structure.
      # Supports any dataset created under src/data/erase_data/<name>.
      local data_dir="src/data/erase_data/${name}"
      if [ ! -d "${data_dir}" ]; then
        echo "Unknown dataset '${name}'. Known: beauty, sports, toys, rsc15, rsc15_smoke" >&2
        echo "  Also tried '${data_dir}' — directory not found." >&2
        return 1
      fi
      clean="${data_dir}"
      poison="${data_dir}${_sfx}"
      # Prefer a dataset-specific SID tensor.  If absent, read
      # sid_tensor_compatible_with from dataset_meta.json (written by
      # subsample_rsc15.py --from-grid-dir) to reuse the source dataset's tensor.
      sid="embeddings/${name}/merged_predictions_tensor.pt"
      local meta="${data_dir}/dataset_meta.json"
      if [ ! -f "${sid}" ] && [ -f "${meta}" ]; then
        local compat_dir
        compat_dir=$(grep -o '"sid_tensor_compatible_with": *"[^"]*"' "${meta}" \
                     | sed 's/.*": *"\([^"]*\)"/\1/')
        if [ -n "${compat_dir}" ] && [ "${compat_dir}" != "null" ]; then
          local compat_name
          compat_name=$(basename "${compat_dir}")
          sid="embeddings/${compat_name}/merged_predictions_tensor.pt"
        fi
      fi
      ;;
  esac

  export GRID_DATA_DIR="${GRID_DATA_DIR:-${clean}}"
  export GRID_POISON_DATA_DIR="${GRID_POISON_DATA_DIR:-${poison}}"
  export GRID_SEMANTIC_ID_PATH="${GRID_SEMANTIC_ID_PATH:-${sid}}"
}
