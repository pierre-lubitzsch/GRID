#!/usr/bin/env bash
# resolve_item_embeddings <dataset> -> GRID_ITEM_EMBEDDINGS
#
# Pre-quantization item embeddings, needed by
#   * unlearning.coherence_neighbor_method=embedding (the default for the
#     sensitive setting, where the neighbourhood IS the concept), and
#   * algorithm=tracer, which projects onto the codebook.
#
# They live in TWO layouts, which is what made the naive
# `embeddings/<ds>_merged_predictions_tensor_latest.pt` default a beauty-only
# path: beauty's payload was installed there by hand, while every dataset
# generated since is still in the inference run dir generate_embeddings.sh wrote.
# Passing a nonexistent path is not harmless -- the unlearning module raises, so
# the job dies after queueing and its run dir then blocks its own resubmission.
#
# ITEM_EMB_OVERRIDE always wins. Sets GRID_ITEM_EMBEDDINGS to "" and returns 1
# when nothing is found, so callers can fail fast with their own message.
resolve_item_embeddings() {
  local ds="$1"
  GRID_ITEM_EMBEDDINGS=""
  if [ -n "${ITEM_EMB_OVERRIDE:-}" ]; then
    GRID_ITEM_EMBEDDINGS="${ITEM_EMB_OVERRIDE}"
    [ -f "${GRID_ITEM_EMBEDDINGS}" ] || {
      echo "ITEM_EMB_OVERRIDE not found: ${GRID_ITEM_EMBEDDINGS}" >&2; return 1; }
    return 0
  fi
  local direct="embeddings/${ds}_merged_predictions_tensor_latest.pt"
  if [ -f "${direct}" ]; then
    GRID_ITEM_EMBEDDINGS="${direct}"
    return 0
  fi
  local hit
  hit="$(ls -1t logs/inference/runs/embeddings/${ds}_*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -1 || true)"
  if [ -n "${hit}" ]; then
    GRID_ITEM_EMBEDDINGS="${hit}"
    return 0
  fi
  echo "No pre-quantization item embeddings for '${ds}'. Looked for:" >&2
  echo "  ${direct}" >&2
  echo "  logs/inference/runs/embeddings/${ds}_*/pickle/merged_predictions_tensor.pt" >&2
  echo "Generate them with ./generate_embeddings.sh ${ds}, or pass ITEM_EMB_OVERRIDE=<path>." >&2
  return 1
}
