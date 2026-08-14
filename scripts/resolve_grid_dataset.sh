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
#   POISON_METHOD=bandwagon   # bandwagon|segment|clone_append
# The method token is empty for 'bandwagon' so existing datasets keep their
# names; other methods insert '_<method>' (e.g. <base>_spam_segment_seed...).

resolve_grid_dataset() {
  local name="${1:-${DATASET:-}}"
  if [ -z "${name}" ]; then
    echo "resolve_grid_dataset: dataset name required" >&2
    return 1
  fi

  local clean="" poison="" sid=""

  # Compute poison suffix once from env vars.
  local _pct _seed _n _sfx _method _mtok
  _pct="$(python3 -c "print(int(round(${POISONING_RATIO:-0.01} * 100)))")"
  _seed="${POISON_SEED:-2}"
  _n="${N_TARGET_ITEMS:-10}"
  _method="${POISON_METHOD:-bandwagon}"
  if [ "${_method}" = "bandwagon" ]; then _mtok=""; else _mtok="_${_method}"; fi
  # clone_inject with >1 injection per session gets a distinct token
  # (_clone_injectx<count>) so different injection counts don't share a dir;
  # count=1 (default) keeps the plain _clone_inject name.
  if [ "${_method}" = "clone_inject" ] && [ "${CLONE_INJECT_COUNT:-1}" != "1" ]; then
    _mtok="${_mtok}x${CLONE_INJECT_COUNT}"
  fi
  # Target-selection strategy token: empty for 'unpopular' (default, backward
  # compatible); 'mid'/'popular'/'random' insert _tgt<strategy> so the three
  # popularity variants of one (method,n,seed,pct) live in separate dirs.
  local _strat _stok
  _strat="${TARGET_STRATEGY:-unpopular}"
  if [ "${_strat}" = "unpopular" ]; then _stok=""; else _stok="_tgt${_strat}"; fi
  _sfx="_spam${_mtok}${_stok}_seed${_seed}_pct${_pct}_n${_n}"

  case "${name}" in
    beauty)
      clean="src/data/amazon_data/beauty"
      poison="${clean}${_sfx}"
      sid="embeddings/beauty/merged_predictions_tensor.pt"
      ;;
    beauty_v2)
      # Same Beauty DATA as `beauty`, but the 2026-08-12 width-256 RQ-KMeans
      # codebook (train job 10397484) instead of the May-2026 one.
      #
      # Why a separate namespace: RQ-KMeans is not reproducible across runs here
      # -- retraining with the same config/embeddings agrees with the published
      # codes at 0.18-0.37%, i.e. chance (1/256 = 0.39%). So the new codebook is
      # a DIFFERENT clustering and its semantic IDs are incompatible with every
      # model trained on the old ones. Installing over embeddings/beauty/ would
      # silently invalidate ~1272 recorded unlearning runs; the legacy tensor is
      # kept at merged_predictions_tensor.LEGACY_may29.pt.
      #
      # Adopted so all runs share one codebook AND so faithful TRACER is possible
      # (its centroids survive for this one, unlike the May-2026 codebook).
      # Poison dirs deliberately resolve to the SHARED beauty_spam_* datasets:
      # poisoning acts on user sequences, not semantic IDs, so they are reusable.
      clean="src/data/amazon_data/beauty"
      poison="${clean}${_sfx}"
      sid="embeddings/beauty_v2/merged_predictions_tensor.pt"
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
    food)
      # CANONICAL food build (Amazon Reviews 2023 "Grocery and Gourmet Food",
      # renamed from grocery 2026-08-13). Built by
      # src/data/amazon_data/convert_amazon_2023.py with items filtered at
      # 3-core but users at >=4 interactions.
      #
      # Why the threshold is split: alcohol is only 0.039% of interactions and
      # iterative k-core cascades away most of it, while leave-one-out needs
      # users >= 4 (train = seq[:-2] must hold >= 2 items) or they vanish from
      # training while staying in eval/test. Measured alcohol pool per variant:
      #   food_k5        20 items / 123 interactions   (1e-4 needs 313 -> no)
      #   food_k4        54 items / 269 interactions   (1e-4 needs 381 -> no)
      #   food (this)   145 items / 458 interactions   (1e-4 needs 405 -> yes)
      # Splits are symmetric (168/168/168). Carries sensitive_items_alcohol.json
      # and forget_manifest_alcohol_1e-5_seed2.json.
      clean="src/data/amazon_data/food"
      poison="${clean}${_sfx}"
      sid="embeddings/food/merged_predictions_tensor.pt"
      ;;
    food_k5|food_k4|food_k3_broken)
      # Superseded builds, kept for reference. food_k5 (5-core) cannot satisfy
      # ERASE's 1e-4 forget ratio; food_k3_broken has a BROKEN split (39% of
      # users in eval/test but not training, built before the --min-seq-len fix).
      clean="src/data/amazon_data/${name}"
      poison="${clean}${_sfx}"
      sid="embeddings/${name}/merged_predictions_tensor.pt"
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
