#!/usr/bin/env bash
#SBATCH --job-name=retrain_floor
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
# Retraining noise floor (reviewer question: how noisy is the retrained reference?).
# The five clean models of a dataset are retrained references trained on IDENTICAL data
# with different training seeds. Scoring clean model s' on the spam targets of seed s
# (s' != s) gives the spread of SH@10 between two equally valid retrained models, i.e.
# the noise floor of Delta_SH. The diagonal (s' == s) already exists in the references.
#
# Exactly the reference evaluation (logs/eval/clean_baseline_bandwagon*/): same checkpoints
# (best-step, from the reference provenance), same overrides, only the manifest differs.
# One job = one (dataset, model seed): 4 target seeds x 3 strata = 12 evals, 2 GPU lanes.
#
#   sbatch run_retrain_floor.sh <dataset> <model_seed>
#   DRYRUN=1 bash run_retrain_floor.sh <dataset> <model_seed>
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DS="${1:?dataset}"; MS="${2:?model seed}"
OUT_ROOT=logs/diagnose/retrain_floor
CKPT="$(python3 -c "
import json, sys
ds, sd = '${DS}', int('${MS}')
p = 'logs/eval/spam_references/' + ds + ('' if sd == 2 else f'_seed{sd}') + '.json'
prov = json.load(open(p))['provenance']['clean_vs_mid']
import yaml
ov = yaml.safe_load(open(prov + '/.hydra/overrides.yaml'))
print([o.split('=', 1)[1].strip(\"'\") for o in ov if o.startswith('ckpt_path=')][0])")"
[ -f "${CKPT}" ] || { echo "missing clean checkpoint for ${DS} seed ${MS}: ${CKPT}" >&2; exit 1; }
echo "clean model ${DS} seed ${MS}: ${CKPT}"

ROWS=()
for TS in 2 3 5 7 11; do
  [ "${TS}" = "${MS}" ] && continue
  for ST in unpopular mid popular; do
    if [ "${ST}" = unpopular ]; then PD="src/data/amazon_data/${DS}_spam_seed${TS}_pct1_n1"
    else PD="src/data/amazon_data/${DS}_spam_tgt${ST}_seed${TS}_pct1_n1"; fi
    [ -f "${PD}/forget_manifest.json" ] || { echo "missing manifest ${PD}" >&2; exit 1; }
    ROWS+=("${TS}|${ST}|${PD}/forget_manifest.json")
  done
done
echo "${#ROWS[@]} evaluations"

IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"; NL=${#GPUS[@]}
run_lane() {
  local lane="$1" i=0
  for row in "${ROWS[@]}"; do
    if [ $(( i % NL )) -ne "${lane}" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    IFS='|' read -r TS ST MAN <<< "${row}"
    OUT="${OUT_ROOT}/${DS}_m${MS}_t${TS}_${ST}"
    if [ -s "${OUT}/csv/version_0/metrics.csv" ]; then echo "[lane ${lane}] skip ${OUT}"; continue; fi
    if [ "${DRYRUN:-0}" = "1" ]; then echo "[lane ${lane}] would eval -> ${OUT} (${MAN})"; continue; fi
    CUDA_VISIBLE_DEVICES="${GPUS[$lane]}" python -u -m scripts.eval_ckpt_on_test \
      experiment=tiger_train_flat data_dir="src/data/amazon_data/${DS}" \
      "semantic_id_path='embeddings/${DS}/merged_predictions_tensor.pt'" \
      "ckpt_path='${CKPT}'" "spam_forget_manifest='${MAN}'" \
      num_hierarchies=4 seed=2 train=False test=True \
      trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
      trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
      hydra.run.dir="${OUT}" > "${OUT_ROOT}/logs/${DS}_m${MS}_t${TS}_${ST}.log" 2>&1 \
      && echo "[lane ${lane}] done ${OUT}" || echo "[lane ${lane}] FAILED ${OUT}"
  done
}
mkdir -p "${OUT_ROOT}/logs"
for ((l=0; l<NL; l++)); do run_lane "${l}" & done
wait
