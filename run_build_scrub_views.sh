#!/usr/bin/env bash
#SBATCH --job-name=build_scrub_views
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --time=04:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# -----------------------------------------------------------------------------
# Finish the scrubbed val/test views (scripts/build_scrubbed_eval_view.py) and
# re-verify the ones already on disk. CPU only. Views are write-once: the builder
# refuses to touch a view that exists, so only missing (dataset, seed) cells are
# built. Already built on 2026-11-06: beauty s2,3,5,7,11 (built + verified) and
# sports s2,3,5 (built, verification pending).
#
#   sbatch run_build_scrub_views.sh
# Override: BUILD="sports:7,11 toys:2,3,5,7,11"  VERIFY="sports:2,3,5"
# -----------------------------------------------------------------------------
BUILD="${BUILD:-sports:7,11 toys:2,3,5,7,11}"
VERIFY="${VERIFY:-beauty:2,3,5,7,11 sports:2,3,5}"
LOG=logs/eval/sensitive_scrub/build_views_job${SLURM_JOB_ID:-local}.log
mkdir -p logs/eval/sensitive_scrub
export TF_CPP_MIN_LOG_LEVEL=2
{
for spec in ${VERIFY}; do
  ds="${spec%%:*}"; seeds="${spec#*:}"
  python -u -m scripts.build_scrubbed_eval_view --verify_only --datasets "${ds}" --seeds ${seeds//,/ }
done
for spec in ${BUILD}; do
  ds="${spec%%:*}"; seeds="${spec#*:}"
  todo=()
  for s in ${seeds//,/ }; do
    v=$(ls -d src/data/amazon_data/${ds}_sens_*_1e-4_seed${s}_scrubtestview 2>/dev/null || true)
    if [ -n "${v}" ]; then echo "exists, skip build: ${v}"; else todo+=("${s}"); fi
  done
  [ ${#todo[@]} -gt 0 ] && python -u -m scripts.build_scrubbed_eval_view --datasets "${ds}" --seeds "${todo[@]}"
done
} 2>&1 | tee "${LOG}"
echo "[$(date -Is)] done -> ${LOG}"
