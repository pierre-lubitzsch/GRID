#!/usr/bin/env bash
#SBATCH --job-name=rbv_batch
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=04:00:00
#
# Rebuttal variants (scripts/plan_rebuttal_variants.py): reference-capped forgetting on
# Beauty spam and the retain-side anchor on unwanted items. Thin wrapper around the
# existing packers; every row is run by run_valsel_one.sh (unlearn + test + validation
# eval) with VALSEL_KEEP_CKPT=1.
#
#   smoke:  sbatch run_rebuttal_variants_batch.sh smoke <spam.tsv> <sens.tsv>
#           runs the two rows one after the other on GPU 0 (one spam, one unwanted)
#   spam:   sbatch run_rebuttal_variants_batch.sh spam <batch.tsv> <guard_tag>...
#   sens:   sbatch run_rebuttal_variants_batch.sh sens <batch.tsv> <guard_tag>...
#           exit without running anything unless every guard tag (the smoke rows) has a
#           run dir with test and validation metrics, i.e. the new code ran end to end.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
export VALSEL_KEEP_CKPT=1
MODE="${1:?mode smoke|spam|sens}"; shift

has_metrics() {  # tag -> 0 if its run dir has test and validation metrics
  local d; d="$(ls -d logs/unlearn/runs/job*_bs1_"$1" 2>/dev/null | head -1)"
  [ -n "${d}" ] && [ -s "${d}/eval/csv/version_0/metrics.csv" ] && [ -s "${d}/eval_val/csv/version_0/metrics.csv" ]
}

case "${MODE}" in
  smoke)
    G0="$(cut -d, -f1 <<<"${CUDA_VISIBLE_DEVICES:-0}")"
    J="${SLURM_JOB_ID:-local}"
    CUDA_VISIBLE_DEVICES="${G0}" bash run_valsel_batch.sh "$1"
    # both packers name their row logs logs/valsel_<job>_lane0_1.*: keep the first pair
    for f in logs/valsel_"${J}"_lane0_1.out logs/valsel_"${J}"_lane0_1.runner.txt; do
      [ -e "${f}" ] && mv "${f}" "${f/_lane0_1/_smokespam}"
    done
    CUDA_VISIBLE_DEVICES="${G0}" bash run_valsel_batch_sens.sh "$2"
    ;;
  spam|sens)
    BATCH="$1"; shift
    for t in "$@"; do
      if ! has_metrics "${t}"; then
        echo "[$(date -Is)] GUARD: smoke run ${t} has no test+validation metrics; not running ${BATCH}" >&2
        exit 1
      fi
    done
    echo "[$(date -Is)] guard passed ($*)"
    if [ "${MODE}" = spam ]; then bash run_valsel_batch.sh "${BATCH}"; else bash run_valsel_batch_sens.sh "${BATCH}"; fi
    ;;
  *) echo "unknown mode ${MODE}" >&2; exit 2 ;;
esac
