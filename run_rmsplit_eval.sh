#!/usr/bin/env bash
#SBATCH --job-name=rmsplit_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
# Evaluations of the retraining-matched split runs (run_rmsplit_sens_launch.sh):
#   1. cleaned-view (protocol A) test evals: scripts/plan_rmsplit_scrub_eval.py ->
#      logs/eval/sensitive_scrub/plan_rmsplit.tsv, then run_sensitive_scrub_eval.sh (CELL=all);
#   2. membership inference on the deleted category (same sets and scorer as
#      scripts/launch_mia_unwanted.py): retrained reference + NAU / Forget+Repair with the new split,
#      -> logs/diagnose/mia_unwanted_rmsplit/<ds>_seed<N>.json (methods nau_rm, forget_repair_rm,
#      retrained). Original / current-split rows stay in logs/diagnose/mia_unwanted{,_lastckpt}/.
# Idempotent: finished evals / existing JSONs are skipped.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
python scripts/plan_rmsplit_scrub_eval.py
CELL=all PLAN=logs/eval/sensitive_scrub/plan_rmsplit.tsv PAR="${PAR:-3}" VIEWS=scrub bash run_sensitive_scrub_eval.sh

declare -A CAT=([beauty]=hairloss [sports]=guns [toys]=toyweapons)
OUTD=logs/diagnose/mia_unwanted_rmsplit
mkdir -p "${OUTD}"
for ds in beauty sports toys; do
  for sd in 2 3 5 7 11; do
    out="${OUTD}/${ds}_seed${sd}.json"
    [ -s "${out}" ] && { echo "MIA done already: ${out}"; continue; }
    sfx=""; [ "${sd}" = "2" ] || sfx="_seed${sd}"
    nau=$(ls -d logs/unlearn/runs/job*_bs1_sens_${ds}_${CAT[$ds]}rm_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_rmsplit_canonck${sfx} 2>/dev/null | tail -1)
    fr=$(ls -d logs/unlearn/runs/job*_bs1_sens_${ds}_${CAT[$ds]}rm_unified_abl_lf0.1_ls0.0_ln0.0_rmsplit_canonck${sfx} 2>/dev/null | tail -1)
    tok="${ds}_retrain${sfx}"
    rc=$(ls -t logs/eval/sensitive_reference/*/${tok}/coordinates.json 2>/dev/null | head -1)
    if [ -z "${nau}" ] || [ -z "${fr}" ] || [ -z "${rc}" ]; then echo "MIA skip ${ds} s${sd}: missing run or reference" >&2; continue; fi
    rck=$(python -c "import json,sys;print(json.load(open(sys.argv[1]))['ckpt'])" "${rc}")
    sets="logs/diagnose/mia_unwanted/sets/${ds}_s${sd}"
    python -u -m src.measure_membership experiment=tiger_unlearn_scif_flat \
      data_dir="src/data/amazon_data/${ds}_sens_${CAT[$ds]}_1e-4_seed${sd}" \
      semantic_id_path="embeddings/${ds}/merged_predictions_tensor.pt" \
      "ckpt_path='${rck}'" num_hierarchies=4 seed="${sd}" \
      +unlearning.js_dataset="${ds}" +unlearning.mia_member_dir="${sets}/members" \
      +unlearning.mia_nonmember_dir="${sets}/nonmembers" +unlearning.scope_output="${out}" \
      "+unlearning.js_retrained_ckpt='${rck}'" \
      "+unlearning.js_nau_rm_ckpt='${nau}/checkpoints/unlearned.ckpt'" \
      "+unlearning.js_forget_repair_rm_ckpt='${fr}/checkpoints/unlearned.ckpt'" \
      hydra.run.dir="${OUTD}/hydra/${ds}_s${sd}" > "${OUTD}/${ds}_s${sd}.log" 2>&1 \
      && echo "MIA ok ${out}" || echo "MIA FAILED ${ds} s${sd} (see ${OUTD}/${ds}_s${sd}.log)" >&2
  done
done
echo "[$(date -Is)] rmsplit_eval finished"
