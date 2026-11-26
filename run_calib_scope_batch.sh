#!/usr/bin/env bash
#SBATCH --job-name=calib_scope
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=08:00:00
#
# Calibration check (critique follow-up addendum 2): score every held-out row (--neighbor-def none
# --all-unrelated, per-query NDCG and next-item id) on Beauty 15 cells, test and validation, for
# clean, start, canonical NAU / Forget+Repair, the NAU weight grid and the k sweep (and, with
# EXTRA_TAGS at plan time, step-2 lambda_n sweep arms). Plan: scripts/plan_calibration_scope.py ->
# logs/calib_scope/plan.tsv (dataset stratum seed poison 'arm=path;...').
# Outputs logs/diagnose/${STEM:-calib_scope_none}{,_val}/beauty_<stratum>_seed<s>.json (mid: beauty_seed<s>.json).
# Rows i with i % NJOBS == JOB are handled by this job, on 2 lanes (one GPU each). Idempotent.
#   JOB=0 NJOBS=3 sbatch run_calib_scope_batch.sh
# Step 2 (lambda_n sweep arms only, after the unlearn runs):
#   JOB=0 NJOBS=1 MAKE_PLAN=1 PLAN=logs/calib_scope/plan_ln.tsv STEM=calib_scope_none_ln \
#     EXTRA_TAGS=ln0p001=n0p001,ln0p003=n0p003 ONLY=forget_repair,nau,ln0p001,ln0p003 sbatch --export=ALL --dependency=afterany:<ids> run_calib_scope_batch.sh  (values contain commas: env, not --export)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
JOB="${JOB:?}"; NJOBS="${NJOBS:-3}"; PLAN="${PLAN:-logs/calib_scope/plan.tsv}"; STEM="${STEM:-calib_scope_none}"
# MAKE_PLAN=1 (single job only): build PLAN here, after the runs it needs exist (EXTRA_TAGS / ONLY passed through)
if [ "${MAKE_PLAN:-0}" = "1" ]; then PLAN="${PLAN}" python3 scripts/plan_calibration_scope.py || exit 1; fi
mapfile -t ROWS < <(awk -v j="${JOB}" -v n="${NJOBS}" '(NR-1)%n==j' "${PLAN}")
lane() {
  local L="$1"; local i=0
  for row in "${ROWS[@]}"; do
    if [ $((i % 2)) -eq "${L}" ]; then
      IFS=$'\t' read -r DS ST SD POISON ARMS <<< "${row}"
      CK=(); IFS=';' read -ra AA <<< "${ARMS}"; for a in "${AA[@]}"; do CK+=(--ckpt "${a}"); done
      NAME="${DS}_${ST}_seed${SD}.json"; [ "${ST}" = mid ] && NAME="${DS}_seed${SD}.json"
      for split in test val; do
        if [ "${split}" = test ]; then S="${STEM}"; V=(); else S="${STEM}_val"; V=(--data-view _valview); fi
        OUT="logs/diagnose/${S}/${NAME}"
        if [ -s "${OUT}" ]; then echo "[lane ${L}] exists ${OUT}"; continue; fi
        echo "[lane ${L}] $(date +%T) ${DS} ${ST} s${SD} ${split}"
        CUDA_VISIBLE_DEVICES="${L}" python -u scripts/run_fig1_js_eval.py --dataset "${DS}" --seed "${SD}" \
          --stratum "${ST}" --metric scope --all-unrelated --neighbor-def none --out-stem "${S}" "${V[@]}" \
          --poison-ckpt "${POISON}" "${CK[@]}" \
          > "logs/calib_scope/${STEM}_${DS}_${ST}_s${SD}_${split}.log" 2>&1 || echo "[lane ${L}] FAILED ${OUT}"
        [ -s "${OUT}" ] || echo "[lane ${L}] MISSING ${OUT}"
      done
    fi
    i=$((i + 1))
  done
}
lane 0 & P0=$!
lane 1 & P1=$!
wait "${P0}"; wait "${P1}"
echo "[calib_scope] job ${JOB} done $(date)"
