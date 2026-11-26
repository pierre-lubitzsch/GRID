#!/usr/bin/env bash
#SBATCH --job-name=scope_nbrreh
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=06:00:00
#
# Neighborhood (Figure 2 protocol: first-SID-token neighborhood, all prefix-disjoint unrelated rows)
# of the neighborhood-rehearsal F+R runs (run_nbr_rehearsal_launch.sh) together with canonical NAU
# and canonical Forget+Repair, all 15 Beauty cells, SPLIT=test (default) or SPLIT=val (_valview).
# Arms per cell: nau (ledger nau_ls0.01_ln0.01), forget_repair (ledger forget_repair_lf0.1),
# fr_nbrreh25 / fr_nbrreh50 (the _canonck rehearsal runs). Adapted from run_scope_frtie.sh.
# Output: logs/diagnose/fig1_scope_nbrreh[_val]/beauty[_<stratum>]_seed<N>.json
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SPLIT="${SPLIT:-test}"
case "${SPLIT}" in
  test) STEM=fig1_scope_nbrreh; VIEW=() ;;
  val)  STEM=fig1_scope_nbrreh_val; VIEW=(--data-view _valview) ;;
  *) echo "bad SPLIT ${SPLIT}"; exit 2 ;;
esac
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
mkdir -p "logs/diagnose/${STEM}" logs/nbr_rehearsal

ck() {  # <tag> -> realpath of the latest run's unlearned.ckpt
  local d
  d="$(for x in logs/unlearn/runs/job*_bs1_$1; do [ -d "$x" ] || continue; j="${x##*/job}"; echo "${j%%_*} $x"; done | sort -n | tail -1 | cut -d' ' -f2)"
  [ -n "${d}" ] && [ -e "${d}/checkpoints/unlearned.ckpt" ] || return 1
  readlink -f "${d}/checkpoints/unlearned.ckpt"
}
one() {
  local st="$1" sd="$2" gpu="$3" sfx="" name args=() a t p
  [ "${sd}" != 2 ] && sfx="_seed${sd}"
  name="beauty_${st}_seed${sd}.json"; [ "${st}" = mid ] && name="beauty_seed${sd}.json"
  if [ -s "logs/diagnose/${STEM}/${name}" ]; then echo "[${st} ${sd}] exists, skip"; return 0; fi
  for a in "nau=nau_s0p01_n0p01_keepck" "forget_repair=unified_lf0.1_ls0.0_ac0_ln0.0_sepgen_keepck" \
           "fr_nbrreh25=fr_nbrrehearse25_canonck" "fr_nbrreh50=fr_nbrrehearse50_canonck"; do
    t="bw_tgt${st}_${a#*=}${sfx}"
    p="$(ck "${t}")" || { echo "[${st} ${sd}] MISSING checkpoint ${t}"; return 1; }
    args+=(--ckpt "${a%%=*}=${p}")
  done
  echo "[${st} ${sd}] [$(date -Is)] ${args[*]}"
  CUDA_VISIBLE_DEVICES="${gpu}" python -u scripts/run_fig1_js_eval.py --dataset beauty --seed "${sd}" --stratum "${st}" \
    --metric scope --all-unrelated --out-stem "${STEM}" "${VIEW[@]}" "${args[@]}" \
    > "logs/nbr_rehearsal/scope_${SPLIT}_${SLURM_JOB_ID:-local}_${st}_${sd}.out" 2>&1
  local rc=$?
  echo "[${st} ${sd}] [$(date -Is)] exit ${rc}"
  [ "${rc}" = 0 ] && [ -s "logs/diagnose/${STEM}/${name}" ]
}
CELLS=(); for st in unpopular mid popular; do for sd in 2 3 5 7 11; do CELLS+=("${st}:${sd}"); done; done
lane() {
  local l="$1" i=0 rc=0 c
  for c in "${CELLS[@]}"; do
    if [ $(( i % ${#GPUS[@]} )) -eq "${l}" ]; then one "${c%%:*}" "${c##*:}" "${GPUS[${l}]}" || rc=1; fi
    i=$((i+1))
  done
  return "${rc}"
}
pids=(); for ((l=0; l<${#GPUS[@]}; l++)); do lane "${l}" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
echo "[$(date -Is)] finished ${SPLIT}, rc=${rc}"; ls -la "logs/diagnose/${STEM}/"
exit "${rc}"
