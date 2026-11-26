#!/usr/bin/env bash
#SBATCH --job-name=scope_none
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=06:00:00
#
# Critique follow-up (rebuttal item 2): score every held-out row (--neighbor-def none --all-unrelated,
# per-query NDCG and next-item id) for the unpopular/popular spam cells, test and validation, with
# clean, start (poison), canonical NAU and canonical Forget+Repair. Plan: scripts/plan_scope_none.py
# -> logs/scope_none/plan.tsv. Outputs logs/diagnose/scope_none_orig{,_val}/<ds>_<stratum>_seed<s>.json.
# Rows i with i % NJOBS == JOB are handled by this job, on 2 lanes (one GPU each). Idempotent.
#   JOB=0 NJOBS=3 sbatch run_scope_none_batch.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
JOB="${JOB:?}"; NJOBS="${NJOBS:-3}"; PLAN="${PLAN:-logs/scope_none/plan.tsv}"
mapfile -t ROWS < <(awk -v j="${JOB}" -v n="${NJOBS}" '(NR-1)%n==j' "${PLAN}")
lane() {
  local L="$1"; local i=0
  for row in "${ROWS[@]}"; do
    if [ $((i % 2)) -eq "${L}" ]; then
      IFS=$'\t' read -r DS ST SD POISON NAU FR <<< "${row}"
      for split in test val; do
        if [ "${split}" = test ]; then S=scope_none_orig; V=(); else S=scope_none_orig_val; V=(--data-view _valview); fi
        OUT="logs/diagnose/${S}/${DS}_${ST}_seed${SD}.json"
        if [ -s "${OUT}" ]; then echo "[lane ${L}] exists ${OUT}"; continue; fi
        echo "[lane ${L}] $(date +%T) ${DS} ${ST} s${SD} ${split}"
        CUDA_VISIBLE_DEVICES="${L}" python -u scripts/run_fig1_js_eval.py --dataset "${DS}" --seed "${SD}" \
          --stratum "${ST}" --metric scope --all-unrelated --neighbor-def none --out-stem "${S}" "${V[@]}" \
          --poison-ckpt "${POISON}" --ckpt "nau=${NAU}" --ckpt "forget_repair=${FR}" \
          > "logs/scope_none/${DS}_${ST}_s${SD}_${split}.log" 2>&1 || echo "[lane ${L}] FAILED ${OUT}"
        [ -s "${OUT}" ] || echo "[lane ${L}] MISSING ${OUT}"
      done
    fi
    i=$((i + 1))
  done
}
lane 0 & P0=$!
lane 1 & P1=$!
wait "${P0}"; wait "${P1}"
echo "[scope_none] job ${JOB} done $(date)"
