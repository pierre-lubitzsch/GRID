#!/usr/bin/env bash
#SBATCH --job-name=randsid_train
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=12:00:00
#
# Random-SID control (NAU rebuttal): TIGER on Beauty with the permuted SIDs of
# embeddings/beauty_randsid (scripts/make_random_sids.py), otherwise exactly the recorded
# RQ-KMeans Beauty trainings (experiment=tiger_train_flat, num_hierarchies=4, config defaults;
# seed=N for every seed; trainer.deterministic=warn only where the recorded run had it, i.e.
# the seed-2 poisoned model job9872117).
#   task 0-4: poisoned (bandwagon 1%, one mid-popularity target), data seeds 2,3,5,7,11
#   task 5-9: clean (= retrained reference), training seeds 2,3,5,7,11
# Labels: beauty_poison_pct1_n1[_seedN]_tgtmid_randsid, beauty_clean[_seedN]_randsid.
#
# DATASET=sports|toys (default beauty): the same replay of that dataset's recorded RQ-KMeans
# trainings (sports jobs 10464996-10465019, toys 10396605/10418649/10464976-10464993: all
# experiment=tiger_train_flat, num_hierarchies=4, config defaults; seed=N, or no seed override
# = config default 2 for the toys seed-2 runs; trainer.deterministic=warn only on the toys
# seed-2 poisoned model job10418649; no sports run had it). SIDs embeddings/<ds>_randsid.
#
#   sbatch --array=0-9%4 run_randsid_train_array.sh
#   DATASET=sports sbatch --job-name=randsid_train_sports --array=0-9%4 run_randsid_train_array.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
SEEDS=(2 3 5 7 11)
T="${SLURM_ARRAY_TASK_ID:?array task id}"
SD="${SEEDS[$(( T % 5 ))]}"
DS="${DATASET:-beauty}"
case "${DS}" in beauty|toys) DET_SEED=2 ;; sports) DET_SEED=none ;; *) echo "unknown DATASET ${DS}" >&2; exit 1 ;; esac
SID="embeddings/${DS}_randsid/merged_predictions_tensor.pt"
[ -f "${SID}" ] || { echo "missing ${SID}" >&2; exit 1; }
if [ "${T}" -lt 5 ]; then
  VARIANT=poison
  export POISON_SEED="${SD}" TARGET_STRATEGY=mid POISON_METHOD=bandwagon RUN_LABEL_SUFFIX=_randsid
  [ "${SD}" = "${DET_SEED}" ] && export DETERMINISTIC=1
else
  VARIANT=clean
  if [ "${SD}" = "2" ]; then export RUN_LABEL_SUFFIX=_randsid; else export RUN_LABEL_SUFFIX="_seed${SD}_randsid"; fi
fi
echo "[randsid] ${DS} task ${T}: ${VARIANT} seed ${SD} deterministic=${DETERMINISTIC:-0}"
# positions 4,5 = poisoning ratio, n targets; 6+ = extra Hydra overrides
bash run_tiger_train.sh "${DS}" "${VARIANT}" "${SID}" 0.01 1 "seed=${SD}"
