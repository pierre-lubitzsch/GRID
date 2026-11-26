#!/usr/bin/env bash
#SBATCH --job-name=valsens_batch
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=02:00:00
#
# UNWANTED-ITEM variant of run_valsel_batch.sh: the runner gets the sensitive data dir as
# its dataset argument and UNLEARN_EVAL_DATA_DIR=<clean ds dir>; no POISON_* env.
# Several validation-selection candidates in ONE allocation: one lane per visible
# GPU (2 on pgpu, 1 on gpu), each lane running its share of the rows in sequence.
# Every row is exactly what run_valsel_launch.sh would have submitted as one job
# (same run_valsel_one.sh, same env and arguments); only the packing differs.
# A row whose run dir already exists is skipped, so a requeue cannot duplicate.
#
#   sbatch run_valsel_batch.sh logs/valsel_batches/<batch>.tsv   (via run_valsel_batch_launch.sh)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
BATCH="${1:?batch tsv required}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
NL=${#GPUS[@]}
echo "[$(date -Is)] batch ${BATCH}: $(wc -l < "${BATCH}") rows, ${NL} lane(s) on GPUs ${CUDA_VISIBLE_DEVICES:-0}"

run_lane() {
  local lane="$1" i=0
  while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
    if [ $(( i % NL )) -ne "${lane}" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "[lane ${lane}] skip ${tag}: run dir exists"; continue; fi
    echo "[lane ${lane}] [$(date -Is)] start ${tag}"
    # shellcheck disable=SC2206
    ov=(${extra})
    CUDA_VISIBLE_DEVICES="${GPUS[$lane]}" VALSEL_LOG_SUFFIX="_lane${lane}_${i}" VALSEL_DATASET="${ds}" \
      UNLEARN_EVAL_DATA_DIR="src/data/amazon_data/${ds}" UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED="${useed}" \
      bash run_valsel_one.sh "${ckpt}" "${data_dir}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${ov[@]}" \
      > "logs/valsel_${SLURM_JOB_ID:-local}_lane${lane}_${i}.out" 2>&1 \
      && echo "[lane ${lane}] [$(date -Is)] done ${tag}" \
      || echo "[lane ${lane}] [$(date -Is)] FAILED ${tag} (see logs/valsel_${SLURM_JOB_ID:-local}_lane${lane}_${i}.out)"
  done < "${BATCH}"
}
for ((l=0; l<NL; l++)); do run_lane "${l}" & done
wait
echo "[$(date -Is)] batch finished"
