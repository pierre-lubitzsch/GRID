#!/usr/bin/env bash
#SBATCH --job-name=scope_sep001
#SBATCH --output=logs/canonck_sep001/%x_%j.out
#SBATCH --error=logs/canonck_sep001/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=08:00:00
#
# Figure 2 neighborhood (first-token, test split) evaluation of the 25 `_canonck` +L_sep
# lambda_s=0.01 re-runs (run_canonck_sep001_launch.sh), with the canonical Forget+Repair checkpoint
# of the same cell in the same file. Output: logs/diagnose/fig1_scope_sep001/<ds>_<st>_seed<N>.json
# (arms sep_ln0p01, forget_repair, clean, poison). Idempotent.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
STEM=fig1_scope_sep001
while IFS=$'\t' read -r ds st sd _useed tag _rest; do
  name="${ds}_${st}_seed${sd}.json"; [ "${st}" = mid ] && name="${ds}_seed${sd}.json"
  out="logs/diagnose/${STEM}/${name}"
  [ -e "${out}" ] && { echo "skip ${out}"; continue; }
  run=$(ls -d logs/unlearn/runs/job*_bs1_${tag} 2>/dev/null | tail -n 1)
  sep="${run}/checkpoints/unlearned.ckpt"
  [ -n "${run}" ] && [ -e "${sep}" ] || { echo "FAIL no checkpoint for ${tag}"; continue; }
  fr=$(awk -F'\t' -v d="${ds}" -v s="${st}" -v e="${sd}" '$1=="spam" && $3=="tiger+rkmeans" && $4=="forget_repair_lf0.1" && $2==d && $6==s && $7==e {print $14}' paper/plotting/canonical_runs.tsv)
  [ -f "${fr}" ] || { echo "FAIL no F+R checkpoint for ${ds} ${st} ${sd}"; continue; }
  echo "[$(date -Is)] ${ds} ${st} ${sd}: $(readlink -f "${sep}")"
  python -u scripts/run_fig1_js_eval.py --dataset "${ds}" --seed "${sd}" --stratum "${st}" \
    --metric scope --all-unrelated --out-stem "${STEM}" \
    --ckpt "sep_ln0p01=$(readlink -f "${sep}")" --ckpt "forget_repair=${fr}" </dev/null || echo "FAIL ${ds} ${st} ${sd}"
done < logs/canonck_sep001/plan.tsv
echo "done $(date -Is)"
