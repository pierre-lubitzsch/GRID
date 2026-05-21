#!/usr/bin/env bash
# Collect post-unlearn test metrics from logs/unlearn/runs (no GPU re-eval).
#
# Usage (from repo root):
#   bash run_collect_unlearn_eval_table.sh
#
# Or with explicit paths:
#   bash run_collect_unlearn_eval_table.sh \
#     logs/eval/batch_sweep/2026-05-20_16-44-14 \
#     logs/unlearn/runs

set -euo pipefail

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"

# -----------------------------------------------------------------------------
# Defaults — edit if your batch-sweep stamp differs
# -----------------------------------------------------------------------------
BATCH_SWEEP_DIR="${1:-logs/eval/batch_sweep/2026-05-20_16-44-14}"
RUNS_ROOT="${2:-logs/unlearn/runs}"

REF_CSV="${BATCH_SWEEP_DIR}/clean_ref/csv/version_0/metrics.csv"
POI_CSV="${BATCH_SWEEP_DIR}/poisoned/csv/version_0/metrics.csv"

STAMP="$(date +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="${GRID_DIR}/logs/eval/collected/${STAMP}"

if [ ! -f "${REF_CSV}" ]; then
  echo "Missing reference metrics: ${REF_CSV}"
  echo "Pass your batch-sweep directory as the first argument."
  exit 1
fi

echo "[$(date -Is)] Collecting unlearn eval metrics (no re-run)"
echo "  reference:     ${REF_CSV}"
echo "  poisoned:      ${POI_CSV}"
echo "  runs_root:     ${RUNS_ROOT}"
echo "  sweep_fallback:${BATCH_SWEEP_DIR}"
echo "  output:        ${OUT_DIR}"

POISON_ARG=()
if [ -f "${POI_CSV}" ]; then
  POISON_ARG=(--poisoned "${POI_CSV}")
else
  echo "WARNING: poisoned metrics not found at ${POI_CSV}; table will omit poisoned row."
fi

python -u -m scripts.collect_unlearn_eval_table \
  --reference "${REF_CSV}" \
  "${POISON_ARG[@]}" \
  --runs-root "${RUNS_ROOT}" \
  --batch-sweep-dir "${BATCH_SWEEP_DIR}" \
  --out-dir "${OUT_DIR}"

echo "[$(date -Is)] Done. Artifacts: ${OUT_DIR}"
