#!/usr/bin/env bash
#SBATCH --job-name=canon_val
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=06:00:00
#
# Validation re-evaluation of canonical checkpoints (paper/revision/canonical_runs_plan.md,
# step 2). Each row of the plan chunk (scripts/plan_canonical_val_eval.py) is EXACTLY the
# validation-eval block of run_valsel_one.sh, applied to an existing run's
# checkpoints/unlearned.ckpt, written to <run>/eval_val. No unlearning, nothing deleted.
# One lane per visible GPU (LANES_PER_GPU=2 runs two per GPU). A row is skipped when
# <run>/eval_val/csv/version_0/metrics.csv exists or another job holds its lock
# (logs/canon_val/locks/<run>); a failed row's partial eval_val is renamed, not removed.
#
#   submitted by run_canonical_val_eval_launch.sh (sbatch run_canonical_val_eval.sh <chunk.tsv>)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CHUNK="${1:?plan chunk tsv required}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
LPG="${LANES_PER_GPU:-1}"
NL=$(( ${#GPUS[@]} * LPG ))
LOCKS="logs/canon_val/locks"; mkdir -p "${LOCKS}"
echo "[$(date -Is)] chunk ${CHUNK}: $(wc -l < "${CHUNK}") rows, ${NL} lane(s) on GPUs ${CUDA_VISIBLE_DEVICES:-0}"

run_lane() {
  local lane="$1" gpu="$2" i=0
  while IFS=$'\t' read -r OUT ds VALVIEW SID DATA_DIR SEED cfg src; do
    if [ $(( i % NL )) -ne "${lane}" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    local name; name="$(basename "${OUT}")"
    if [ -s "${OUT}/eval_val/csv/version_0/metrics.csv" ]; then echo "[lane ${lane}] skip ${name}: eval_val exists"; continue; fi
    if ! mkdir "${LOCKS}/${name}" 2>/dev/null; then echo "[lane ${lane}] skip ${name}: locked by $(cat "${LOCKS}/${name}/owner" 2>/dev/null)"; continue; fi
    echo "${SLURM_JOB_ID:-local} lane ${lane}" > "${LOCKS}/${name}/owner"
    local CKPT="${OUT}/checkpoints/unlearned.ckpt"
    if [ ! -e "${CKPT}" ]; then echo "[lane ${lane}] FAILED ${name}: checkpoint does not resolve"; rm -rf "${LOCKS:?}/${name}"; continue; fi
    if [ -e "${OUT}/eval_val" ]; then mv "${OUT}/eval_val" "${OUT}/eval_val.partial_${SLURM_JOB_ID:-local}_$(date +%s)"; fi
    local LOG="logs/canon_val/${SLURM_JOB_ID:-local}_lane${lane}_${i}.out"
    echo "[lane ${lane}] [$(date -Is)] start ${name} (${cfg}, ${src}) -> ${LOG}"
    # --- the validation-eval block of run_valsel_one.sh (experiment = tiger_train_flat) ---
    CUDA_VISIBLE_DEVICES="${gpu}" python -u -m scripts.eval_ckpt_on_test \
      experiment=tiger_train_flat \
      data_dir="${VALVIEW}" \
      "semantic_id_path='${SID}'" \
      "ckpt_path='${CKPT}'" \
      "spam_forget_manifest='${DATA_DIR}/forget_manifest.json'" \
      num_hierarchies=4 \
      seed="${SEED}" \
      train=False test=True \
      trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
      trainer.deterministic=true \
      callbacks.model_checkpoint=null callbacks.early_stopping=null \
      hydra.run.dir="${OUT}/eval_val" > "${LOG}" 2>&1
    if [ -s "${OUT}/eval_val/csv/version_0/metrics.csv" ]; then
      echo "[lane ${lane}] [$(date -Is)] done ${name}"
    else
      echo "[lane ${lane}] [$(date -Is)] FAILED ${name} (see ${LOG})"
      [ -e "${OUT}/eval_val" ] && mv "${OUT}/eval_val" "${OUT}/eval_val.failed_${SLURM_JOB_ID:-local}_$(date +%s)"
      rm -rf "${LOCKS:?}/${name}"
    fi
  done < "${CHUNK}"
}
mkdir -p logs/canon_val
for ((l=0; l<NL; l++)); do run_lane "${l}" "${GPUS[$(( l % ${#GPUS[@]} ))]}" & done
wait
echo "[$(date -Is)] chunk finished: $(awk -F'\t' '{print $1"/eval_val/csv/version_0/metrics.csv"}' "${CHUNK}" | xargs -I{} sh -c 'test -s {} && echo ok || echo missing' | sort | uniq -c | tr '\n' ' ')"
