#!/usr/bin/env bash
# Resolve GRID unlearning paths from a short dataset name.
# Thin wrapper around resolve_grid_dataset.sh for backward compatibility.

_resolve_unlearn_grid_dir() {
  local script_dir
  script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  # shellcheck source=scripts/resolve_grid_dataset.sh
  source "${script_dir}/resolve_grid_dataset.sh"
}

resolve_unlearn_dataset() {
  _resolve_unlearn_grid_dir
  if ! resolve_grid_dataset "${1:-${DATASET:-}}"; then
    return 1
  fi
  export UNLEARN_DATA_DIR="${UNLEARN_POISON_DATA_DIR:-${GRID_POISON_DATA_DIR}}"
  export UNLEARN_EVAL_DATA_DIR="${UNLEARN_EVAL_DATA_DIR:-${GRID_DATA_DIR}}"
  export UNLEARN_DEFAULT_SEMANTIC_ID_PATH="${UNLEARN_SEMANTIC_ID_PATH:-${GRID_SEMANTIC_ID_PATH}}"
}
