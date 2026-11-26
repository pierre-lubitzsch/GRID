#!/usr/bin/env bash
# Retrained references for the larger-request sensitive variant (budget 5e-4): 3 datasets x 5 seeds
# = 15 TIGER (RQ-KMeans) trainings on <ds>_retrain_<cat>_5e-4_seed<N>.
# Every knob is lifted from the recorded 1e-4 retrain trainings (e.g. job 10463505, 10465000,
# 10506652; retrain_toyweapons.sh): run_tiger_train.sh <ds> clean embeddings/<ds>/merged_predictions_tensor.pt
# with TIGER_DATA_DIR=<retrain dir>, TRAIN_SEED=<N>, pgpu 2x H200, 12 h. The resolved Hydra config
# was diffed against the recorded config.yaml (only inert new keys differ).
# Label: <ds>_clean[_seed<N>]_retrain5e-4_<cat> (NOT `_retrain_<cat>_1e-4`, so plan_reference_evals,
# extract_hpsearch and write_experiment_progress never see these runs).
# At most 4 concurrent: job i depends afterany on job i-4; all depend afterok on the data build.
#   BUILD_JID=<jid> bash run_sens5e4_train_launch.sh            # dry run
#   BUILD_JID=<jid> DRYRUN=0 bash run_sens5e4_train_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
RATIO=5e-4
declare -A CAT=([beauty]=hairloss [sports]=guns [toys]=toyweapons)
jids=()
i=0
for ds in beauty sports toys; do
  for sd in 2 3 5 7 11; do
    data="src/data/amazon_data/${ds}_retrain_${CAT[$ds]}_${RATIO}_seed${sd}"
    dep=()
    [ -n "${BUILD_JID:-}" ] && dep+=("afterok:${BUILD_JID}")
    [ "${i}" -ge 4 ] && dep+=("afterany:${jids[$((i-4))]}")
    depopt=""; [ ${#dep[@]} -gt 0 ] && depopt="--dependency=$(IFS=,; echo "${dep[*]}")"
    cmd=(sbatch --parsable --nice=500 --job-name="s5e4_retr_${ds}_s${sd}" --partition=pgpu
         --gres=gpu:nvidia_h200:2 --exclude=s-sc-pgpu01,s-sc-pgpu02 --time=12:00:00 ${depopt}
         run_tiger_train.sh "${ds}" clean "embeddings/${ds}/merged_predictions_tensor.pt")
    env=(TIGER_DATA_DIR="${data}" TRAIN_SEED="${sd}" RUN_LABEL_SUFFIX="_retrain${RATIO}_${CAT[$ds]}")
    if [ "${DRYRUN}" = "1" ]; then
      printf '%s ' "${env[@]}"; printf '%q ' "${cmd[@]}"; echo; jids+=("DRY${i}")
    else
      j=$(env "${env[@]}" "${cmd[@]}"); jids+=("${j}"); echo "${ds} s${sd}: ${j}"
      echo -e "${ds}\t${sd}\t${j}" >> logs/sens5e4/train_jobs.tsv
    fi
    i=$((i+1))
  done
done
echo "jobs: ${jids[*]}"
