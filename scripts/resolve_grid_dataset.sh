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

resolve_grid_dataset() {
  local name="${1:-${DATASET:-}}"
  if [ -z "${name}" ]; then
    echo "resolve_grid_dataset: dataset name required" >&2
    return 1
  fi

  local clean="" poison="" sid=""

  case "${name}" in
    beauty)
      clean="src/data/amazon_data/beauty"
      poison="src/data/amazon_data/beauty_spam_seed2_pct1_n10"
      sid="embeddings/beauty/merged_predictions_tensor.pt"
      ;;
    sports)
      clean="src/data/amazon_data/sports"
      poison="src/data/amazon_data/sports_spam_seed2_pct1_n10"
      sid="embeddings/sports/merged_predictions_tensor.pt"
      ;;
    toys)
      clean="src/data/amazon_data/toys"
      poison="src/data/amazon_data/toys_spam_seed2_pct1_n10"
      sid="embeddings/toys/merged_predictions_tensor.pt"
      ;;
    rsc15)
      clean="src/data/erase_data/rsc15"
      poison="src/data/erase_data/rsc15_spam_seed2_pct1_n10"
      sid="embeddings/rsc15/merged_predictions_tensor.pt"
      ;;
    rsc15_smoke)
      clean="src/data/erase_data/rsc15_smoke"
      poison="src/data/erase_data/rsc15_smoke_spam_seed2_pct1_n10"
      sid="embeddings/rsc15_smoke/merged_predictions_tensor.pt"
      ;;
    *)
      echo "Unknown dataset '${name}'. Known: beauty, sports, toys, rsc15, rsc15_smoke" >&2
      return 1
      ;;
  esac

  export GRID_DATA_DIR="${GRID_DATA_DIR:-${clean}}"
  export GRID_POISON_DATA_DIR="${GRID_POISON_DATA_DIR:-${poison}}"
  export GRID_SEMANTIC_ID_PATH="${GRID_SEMANTIC_ID_PATH:-${sid}}"
}
