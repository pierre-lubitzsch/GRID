#!/usr/bin/env bash
#SBATCH --job-name=valsens_refs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=04:00:00
#
# Validation-split references for unwanted-item selection: the ORIGINAL (base) and
# RETRAINED models of every (dataset, seed), exactly the checkpoints and deletion
# manifests of the newest test-split reference (logs/eval/sensitive_reference/*/<tok>/
# coordinates.json), evaluated on src/data/amazon_data/<ds>_valview.
# Output: logs/diagnose/valsel_sens_refs/<ds>_<role>_s<seed>/
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
for ds in beauty sports toys; do
  for sd in 2 3 5 7 11; do
    for role in base retrain; do
      tok="${ds}_${role}"; [ "${sd}" != "2" ] && tok="${tok}_seed${sd}"
      coord="$(ls -t logs/eval/sensitive_reference/*/${tok}/coordinates.json 2>/dev/null | head -1)"
      [ -n "${coord}" ] || { echo "no reference for ${tok}" >&2; continue; }
      read -r ck man sid < <(python3 -c "import json,sys; c=json.load(open(sys.argv[1])); print(c['ckpt'], c['manifest'], c['semantic_id_path'])" "${coord}")
      out="logs/diagnose/valsel_sens_refs/${ds}_${role}_s${sd}"
      if [ -s "${out}/csv/version_0/metrics.csv" ]; then echo "skip ${out}"; continue; fi
      echo "== ${ds} s${sd} ${role}: ${ck}"
      python -u -m scripts.eval_ckpt_on_test experiment="tiger_train_flat" data_dir="src/data/amazon_data/${ds}_valview" \
        "semantic_id_path='${sid}'" "ckpt_path='${ck}'" "spam_forget_manifest='${man}'" \
        num_hierarchies=4 seed="${sd}" train=False test=True trainer.devices=1 trainer.strategy=auto \
        trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true \
        callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir="${out}" || echo "FAILED ${out}"
    done
  done
done
