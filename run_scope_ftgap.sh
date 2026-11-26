#!/usr/bin/env bash
#SBATCH --job-name=scope_ftgap
#SBATCH --output=logs/sasrec_nau_v2/slurm/%x_%j.out
#SBATCH --error=logs/sasrec_nau_v2/slurm/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=08:00:00
#
# TIGER Figure 2 neighborhood evaluation of the canonical Finetune checkpoints that no scope
# file covers yet (Sports/Toys unpopular/popular, the `_canonck` re-runs), with the canonical
# Forget+Repair checkpoint of the same cell in the same file (for the SASRec vs TIGER
# contrast "forget term's cost" = F+R minus Finetune). Plan: logs/sasrec_nau_v2/tiger_ftgap_plan.tsv.
# Output: logs/diagnose/fig1_scope_ftgap/<ds>_<stratum>_seed<N>.json. Idempotent.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
STEM=fig1_scope_ftgap
tail -n +2 logs/sasrec_nau_v2/tiger_ftgap_plan.tsv | while IFS=$'\t' read -r ds st sd ft fr; do
  out="logs/diagnose/${STEM}/${ds}_${st}_seed${sd}.json"
  [ -e "${out}" ] && { echo "skip ${out}"; continue; }
  echo "[$(date -Is)] ${ds} ${st} ${sd}"
  python -u scripts/run_fig1_js_eval.py --dataset "${ds}" --seed "${sd}" --stratum "${st}" \
    --metric scope --all-unrelated --out-stem "${STEM}" \
    --ckpt "retain_only=${ft}" --ckpt "forget_repair=${fr}" </dev/null || echo "FAIL ${ds} ${st} ${sd}"
done
echo "done $(date -Is)"
