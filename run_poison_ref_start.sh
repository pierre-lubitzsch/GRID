#!/usr/bin/env bash
#SBATCH --job-name=poison_ref_start
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
# Table 1's "Poisoned (no unlearning)" row evaluated the poisoned model at its newest
# checkpoint_epoch=* file (scripts/plan_reference_evals.ckpt_of), but the spam runs start from
# last.ckpt in 42 of 45 cells. This evaluates, on the TEST split with each cell's manifest, the
# exact file the runs start from (logs/poison_ref_start_plan.tsv, taken from the NAU validation
# runs' overrides). Output: logs/diagnose/poison_ref_start/<ds>_<stratum>_s<seed>/
#
#   DATASET=beauty sbatch run_poison_ref_start.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DS="${DATASET:?}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
while IFS=$'\t' read -r ds st sd ck dd sid; do
  [ "${ds}" = "${DS}" ] || continue
  out="logs/diagnose/poison_ref_start/${ds}_${st}_s${sd}"
  [ -s "${out}/csv/version_0/metrics.csv" ] && { echo "skip ${out}"; continue; }
  echo "== ${ds} ${st} s${sd}: ${ck}"
  python -u -m scripts.eval_ckpt_on_test experiment="tiger_train_flat" data_dir="src/data/amazon_data/${ds}" \
    "semantic_id_path='${sid}'" "ckpt_path='${ck}'" "spam_forget_manifest='${dd}/forget_manifest.json'" \
    num_hierarchies=4 seed="${sd}" train=False test=True \
    trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
    trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
    hydra.run.dir="${out}" || echo "FAILED ${out}"
done < logs/poison_ref_start_plan.tsv
