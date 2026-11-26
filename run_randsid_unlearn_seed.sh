#!/usr/bin/env bash
#SBATCH --job-name=randsid_unl
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=03:00:00
#
# Random-SID control: the four unlearning arms (Finetune, Forget+Repair lf0.1, +L_sep ls0.1,
# NAU ls0.01/ln0.01) of ONE data seed, from logs/randsid/plan.tsv (scripts/plan_randsid.py).
# Resolves the random-SID poisoned model of the seed from its training label (last.ckpt, the
# convention of the canonical runs), then runs the rows through run_valsel_batch.sh ->
# run_valsel_one.sh (unlearn + test eval + validation eval) with VALSEL_KEEP_CKPT=1.
# Fails unless all four run dirs end with test eval, validation eval and checkpoint.
#
#   SEED=3 sbatch run_randsid_unlearn_seed.sh
#   DATASET=sports SEED=3 sbatch run_randsid_unlearn_seed.sh   # plan logs/randsid/plan_sports.tsv (also toys)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SD="${SEED:?}"
DS="${DATASET:-beauty}"
if [ "${DS}" = beauty ]; then
  PLAN="logs/randsid/plan.tsv"; OUT="logs/randsid/batch_seed${SD}_${SLURM_JOB_ID:-local}.tsv"
else
  PLAN="logs/randsid/plan_${DS}.tsv"; OUT="logs/randsid/batch_${DS}_seed${SD}_${SLURM_JOB_ID:-local}.tsv"
fi
[ -s "${PLAN}" ] || { echo "missing ${PLAN}" >&2; exit 1; }
: > "${OUT}"
while IFS=$'\t' read -r ds st sd useed tag ckpt rest; do
  [ "${sd}" = "${SD}" ] && [ "${ds}" = "${DS}" ] || continue
  label="${ckpt#label:}"
  mapfile -t dirs < <(ls -1d logs/train/runs/*/*_job*_"${label}" 2>/dev/null)
  [ "${#dirs[@]}" -eq 1 ] || { echo "expected one training run for ${label}, got ${#dirs[@]}: ${dirs[*]:-}" >&2; exit 1; }
  last="${dirs[0]}/checkpoints/last.ckpt"
  [ -f "${last}" ] || { echo "missing ${last}" >&2; exit 1; }
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" "${last}" "${rest}" >> "${OUT}"
done < "${PLAN}"
[ "$(wc -l < "${OUT}")" -eq 4 ] || { echo "expected 4 rows for seed ${SD}" >&2; exit 1; }
echo "[randsid] ${DS} seed ${SD}: $(cut -f6 "${OUT}" | sort -u)"
export VALSEL_KEEP_CKPT=1 POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1
bash run_valsel_batch.sh "${OUT}"
bad=0
while IFS=$'\t' read -r ds st sd useed tag rest; do
  d="$(ls -1d logs/unlearn/runs/*_bs1_"${tag}" 2>/dev/null | tail -1)"
  if [ -z "${d}" ] || [ ! -s "${d}/eval/csv/version_0/metrics.csv" ] || [ ! -s "${d}/eval_val/csv/version_0/metrics.csv" ] \
     || [ ! -e "${d}/checkpoints/unlearned.ckpt" ]; then
    echo "INCOMPLETE ${tag}: ${d:-<no run dir>}" >&2; bad=1
  else
    echo "ok ${tag}: ${d}"
  fi
done < "${OUT}"
exit "${bad}"
