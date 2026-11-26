#!/usr/bin/env bash
#SBATCH --job-name=filter_target
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
# "Just block the item at inference": the poisoned model the unlearning runs start from
# (logs/poison_ref_start_plan.tsv), evaluated on the TEST split with a decode mask that forbids
# ONLY the cell's promoted target item(s) for every user. The repo's filter baseline
# (algorithm=filter, deletion_spec=session) instead forbids every item of the fake users'
# sessions (about 1,000 items incl. the popular filler items), which is not what "block the
# item" means. No weights change. Output: logs/diagnose/filter_target/<ds>_<stratum>_s<seed>/
#
#   DATASET=beauty sbatch run_filter_target.sh
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DS="${DATASET:?}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES%%,*}"
while IFS=$'\t' read -r ds st sd ck dd sid; do
  [ "${ds}" = "${DS}" ] || continue
  out="logs/diagnose/filter_target/${ds}_${st}_s${sd}"
  [ -s "${out}/csv/version_0/metrics.csv" ] && { echo "skip ${out}"; continue; }
  mkdir -p "${out}"
  python3 - "${dd}/forget_manifest.json" "${out}/filter_mask.json" <<'EOF'
import json, sys
m = json.load(open(sys.argv[1]))
json.dump({"filter_mode": "global", "deletion_spec": "target_only",
           "forbidden_item_ids": [int(i) for i in m["target_items"]], "user_forget_items": None},
          open(sys.argv[2], "w"))
EOF
  echo "== ${ds} ${st} s${sd}: ${ck}; blocked $(python3 -c "import json;print(json.load(open('${out}/filter_mask.json'))['forbidden_item_ids'])")"
  python -u -m scripts.eval_ckpt_on_test experiment="tiger_train_flat" data_dir="src/data/amazon_data/${ds}" \
    "semantic_id_path='${sid}'" "ckpt_path='${ck}'" "spam_forget_manifest='${dd}/forget_manifest.json'" \
    "decode_filter_mask='${out}/filter_mask.json'" \
    num_hierarchies=4 seed="${sd}" train=False test=True \
    trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
    trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
    hydra.run.dir="${out}" || echo "FAILED ${out}"
done < logs/poison_ref_start_plan.tsv
