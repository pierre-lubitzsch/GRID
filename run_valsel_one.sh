#!/usr/bin/env bash
#SBATCH --job-name=valsel
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=01:00:00
#
# ONE hyperparameter-selection candidate on ONE (dataset, stratum, seed) cell:
#   1. run_tiger_unlearn_sequential.sh with the given arguments (its usual post-eval
#      on the clean TEST split -> <run>/eval/),
#   2. evaluate the SAME unlearned checkpoint on the VALIDATION split
#      (src/data/amazon_data/<ds>_valview, whose testing/ is ../<ds>/evaluation)
#      -> <run>/eval_val/,
#   3. delete the unlearned checkpoint (metrics are kept).
# Arguments are passed through to the runner unchanged; env (POISON_*, TARGET_STRATEGY,
# UNLEARN_RUN_TAG, UNLEARN_SEED, VALSEL_DATASET) is set by run_valsel_launch.sh.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
: "${VALSEL_DATASET:?}"
VALVIEW="src/data/amazon_data/${VALSEL_DATASET}_valview"
[ -d "${VALVIEW}/testing" ] || { echo "missing ${VALVIEW}/testing" >&2; exit 1; }

LOG="logs/valsel_${SLURM_JOB_ID:-local}${VALSEL_LOG_SUFFIX:-}.runner.txt"
UNLEARN_KEEP_CKPT=1 bash run_tiger_unlearn_sequential.sh "$@" 2>&1 | tee "${LOG}"
OUT="$(grep -m1 '^Using output_dir=' "${LOG}" | sed 's/^Using output_dir=//')"
DATA_DIR="$(grep -m1 '^Using data_dir=' "${LOG}" | sed 's/^Using data_dir=//')"
SID="$(grep -m1 '^Using semantic_id_path=' "${LOG}" | sed 's/^Using semantic_id_path=//')"
CKPT="${OUT}/checkpoints/unlearned.ckpt"
[ -e "${CKPT}" ] || { echo "no unlearned checkpoint at ${CKPT}" >&2; exit 1; }
[ -s "${OUT}/eval/csv/version_0/metrics.csv" ] || { echo "test eval missing in ${OUT}" >&2; exit 1; }

# The model's own train config (MODEL=letter -> letter_train_flat), as the runner's
# test eval does; identity for MODEL=tiger, so TIGER validation evals are unchanged.
# shellcheck source=scripts/resolve_model.sh
source "${GRID_DIR:-$PWD}/scripts/resolve_model.sh"
resolve_model "${MODEL:-tiger}" >/dev/null || exit 1
VAL_EXPERIMENT="$(grid_model_config tiger_train_flat)" || exit 1
echo "[$(date -Is)] validation eval of ${CKPT} on ${VALVIEW} (experiment=${VAL_EXPERIMENT})"
python -u -m scripts.eval_ckpt_on_test \
  experiment="${VAL_EXPERIMENT}" \
  data_dir="${VALVIEW}" \
  "semantic_id_path='${SID}'" \
  "ckpt_path='${CKPT}'" \
  "spam_forget_manifest='${DATA_DIR}/forget_manifest.json'" \
  num_hierarchies=4 \
  seed="${UNLEARN_SEED:-2}" \
  train=False test=True \
  trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
  trainer.deterministic=true \
  callbacks.model_checkpoint=null callbacks.early_stopping=null \
  hydra.run.dir="${OUT}/eval_val"
[ -s "${OUT}/eval_val/csv/version_0/metrics.csv" ] || { echo "validation eval failed" >&2; exit 1; }

# metrics are extracted; drop the weights (checkpoint retention policy),
# unless VALSEL_KEEP_CKPT=1 (the tie-break runs need them for the scope eval)
if [ "${VALSEL_KEEP_CKPT:-0}" = "1" ]; then echo "[$(date -Is)] keeping checkpoint: ${CKPT}"; echo "[$(date -Is)] done: ${OUT}"; exit 0; fi
find "${OUT}/checkpoints" -maxdepth 1 -type f -name 'unlearned_*.ckpt' -delete
find "${OUT}/checkpoints" -maxdepth 1 -xtype l -name 'unlearned*.ckpt' -delete
echo "[$(date -Is)] done: ${OUT}"
