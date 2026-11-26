#!/usr/bin/env bash
#SBATCH --job-name=scope_rr
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=04:00:00
#
# Retrained-vs-retrained reference for the per-query agreement analysis (R3 Q1,
# paper/plotting/per_query_agreement.py). Beauty data seed 2 has two further retrained (clean)
# TIGER models besides the canonical one (job9096928): job9465564 and job9642098, same config,
# same data, same seed (they differ by GPU nondeterminism only). Score them with the Figure 1/2
# scope evaluator (all prefix-disjoint unrelated rows, prefix1 neighborhood) on the identical
# test queries of each Beauty stratum, next to the canonical clean and poisoned models (which
# the evaluator re-scores in every JSON).
# Output: logs/diagnose/fig1_scope_rr/beauty[_<stratum>]_seed2.json (arms clean, poison, clean_b, clean_c)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
STEM=fig1_scope_rr
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "logs/diagnose/${STEM}" logs/scope_rr
CB=logs/train/runs/2026-06-12/14-14-06_job9465564_beauty_clean/checkpoints/checkpoint_epoch=000_step=003800.ckpt
CC=logs/train/runs/2026-06-18/10-59-59_job9642098_beauty_clean/checkpoints/checkpoint_epoch=000_step=003800.ckpt
one() {
  local st="$1" gpu="$2" name
  name="beauty_${st}_seed2.json"; [ "${st}" = mid ] && name="beauty_seed2.json"
  if [ -s "logs/diagnose/${STEM}/${name}" ]; then echo "[${st}] exists, skip"; return 0; fi
  echo "[${st}] [$(date -Is)] start on GPU ${gpu}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_fig1_js_eval.py --dataset beauty --seed 2 --stratum "${st}" \
    --metric scope --all-unrelated --out-stem "${STEM}" \
    --ckpt "clean_b=${CB}" --ckpt "clean_c=${CC}" > "logs/scope_rr/${SLURM_JOB_ID:-local}_${st}.out" 2>&1
  local rc=$?
  echo "[${st}] [$(date -Is)] exit ${rc}"
  [ "${rc}" = 0 ] && [ -s "logs/diagnose/${STEM}/${name}" ]
}
one mid "${GPUS[0]}" & p0=$!
( one unpopular "${GPUS[$(( 1 % ${#GPUS[@]} ))]}"; one popular "${GPUS[$(( 1 % ${#GPUS[@]} ))]}" ) & p1=$!
rc=0; wait "${p0}" || rc=1; wait "${p1}" || rc=1
echo "[$(date -Is)] finished rc=${rc}"; ls -la "logs/diagnose/${STEM}/"
exit "${rc}"
