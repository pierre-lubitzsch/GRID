#!/usr/bin/env bash
#SBATCH --job-name=fig1fix_scope
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=03:00:00
#
# Figure 1 neighborhood evaluation of the Figure 1 fix runs (run_fig1_fix_launch.sh): for each
# Beauty mid seed, the _canonck checkpoints of Table 1's Forget only (lambda_f=1, lambda_r=0) and
# Fanchuan t0.07, with the same evaluator and options that produced
# logs/diagnose/fig1_scope_allunrel (run_fig1_js_eval.sh defaults with ALL_UNRELATED=1; clean and
# poisoned models are re-scored too, as a consistency check against the existing JSONs).
# Output: logs/diagnose/${OUT_STEM:-fig1_scope_allunrel_fix}/beauty_seed<N>.json
# Seeds run in 2 lanes (one per GPU); a seed whose JSON exists is skipped. Exits non-zero if
# any seed lacks a checkpoint or fails.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
STEM="${OUT_STEM:-fig1_scope_allunrel_fix}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "logs/diagnose/${STEM}" logs/fig1_fix

ck() {  # latest _canonck run of <arm> for seed <sd> with a resolvable checkpoint
  local arm="$1" sd="$2" sfx="" d
  [ "${sd}" != "2" ] && sfx="_seed${sd}"
  d="$(for x in logs/unlearn/runs/job*_bs1_bw_tgtmid_${arm}_canonck${sfx}; do [ -d "$x" ] || continue; j="${x##*/job}"; echo "${j%%_*} $x"; done | sort -n | tail -1 | cut -d' ' -f2)"
  [ -n "${d}" ] && [ -e "${d}/checkpoints/unlearned.ckpt" ] || return 1
  readlink -f "${d}/checkpoints/unlearned.ckpt"
}
one() {
  local sd="$1" gpu="$2" fo fa
  if [ -s "logs/diagnose/${STEM}/beauty_seed${sd}.json" ]; then echo "[seed ${sd}] exists, skip"; return 0; fi
  fo="$(ck unified_abl_lf1.0_ls0.0_ln0.0_lr0.0 "${sd}")" || { echo "[seed ${sd}] MISSING forget_only checkpoint"; return 1; }
  fa="$(ck fanchuan_t0.07 "${sd}")" || { echo "[seed ${sd}] MISSING fanchuan checkpoint"; return 1; }
  echo "[seed ${sd}] [$(date -Is)] forget_only=${fo} fanchuan=${fa}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_fig1_js_eval.py --dataset beauty --seed "${sd}" --stratum mid \
    --metric scope --all-unrelated --out-stem "${STEM}" \
    --ckpt "forget_only=${fo}" --ckpt "fanchuan=${fa}" > "logs/fig1_fix/scope_${SLURM_JOB_ID:-local}_seed${sd}.out" 2>&1
  local rc=$?
  echo "[seed ${sd}] [$(date -Is)] exit ${rc} (log logs/fig1_fix/scope_${SLURM_JOB_ID:-local}_seed${sd}.out)"
  [ "${rc}" = 0 ] && [ -s "logs/diagnose/${STEM}/beauty_seed${sd}.json" ]
}
lane() {
  local l="$1" rc=0; shift
  for sd in "$@"; do one "${sd}" "${GPUS[$(( l % ${#GPUS[@]} ))]}" || rc=1; done
  return "${rc}"
}
lane 0 2 5 11 & p0=$!
lane 1 3 7 & p1=$!
rc=0; wait "${p0}" || rc=1; wait "${p1}" || rc=1
echo "[$(date -Is)] finished, rc=${rc}"; ls -la "logs/diagnose/${STEM}/"
exit "${rc}"
