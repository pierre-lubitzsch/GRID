#!/usr/bin/env bash
#SBATCH --job-name=pkm_slots
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=pgpu
#SBATCH --time=02:00:00
#
# PKM memory-slot access diagnosis (read-only): do forget and retain
# interactions route to disjoint memory slots? Gates top-t slot selection.
#
# Usage: PKM_ARCH=e23d01 TARGET_STRATEGY=mid sbatch run_diagnose_pkm_slots.sh <ckpt_run_dir>
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"; mkdir -p logs
CKPT_IN="${1:?pass the training run dir or a .ckpt}"
DATASET="${DATASET:-beauty}"
SID="${SID:-embeddings/${DATASET}/merged_predictions_tensor.pt}"
PKM_ARCH="${PKM_ARCH:-e23d01}"
case "${PKM_ARCH}" in
  e23d01) PKM_OVR="model.pkm_layers={encoder:[2,3],decoder:[0,1]}" ;;
  allv2)  PKM_OVR="model.pkm_layers={encoder:all,decoder:all}" ;;
  d01)    PKM_OVR="model.pkm_layers={encoder:null,decoder:[0,1]}" ;;
  d0)     PKM_OVR="model.pkm_layers={encoder:null,decoder:[0]}" ;;
  d1)     PKM_OVR="model.pkm_layers={encoder:null,decoder:[1]}" ;;
  d2)     PKM_OVR="model.pkm_layers={encoder:null,decoder:[2]}" ;;
  custom)
    # Generic passthrough: give PKM_ENC / PKM_DEC in the same format as
    # run_tiger_train.sh's PKM_ENCODER / PKM_DECODER ("0,1" | "all" | "none").
    _sel() { case "${1:-}" in ""|none|off|false) echo "null";; all|null) echo "$1";; *) echo "[$1]";; esac; }
    PKM_OVR="model.pkm_layers={encoder:$(_sel "${PKM_ENC:-}"),decoder:$(_sel "${PKM_DEC:-}")}" ;;
  *) echo "PKM_ARCH must be e23d01|allv2|d01|d0|d1|d2|custom"; exit 1 ;;
esac
if [ -d "${CKPT_IN}" ]; then
  CK="${CKPT_IN%/}"; [ -d "${CK}/checkpoints" ] && CK="${CK}/checkpoints"
  CKPT="$(ls -t "${CK}"/*.ckpt 2>/dev/null | head -1)"
else CKPT="${CKPT_IN}"; fi
[ -f "${CKPT}" ] || { echo "no ckpt found from '${CKPT_IN}'"; exit 1; }
source scripts/resolve_grid_dataset.sh
resolve_grid_dataset "${DATASET}"
OUT="logs/diagnose/pkm_slots/$(date +%Y-%m-%d_%H-%M-%S)_${PKM_ARCH}_tgt${TARGET_STRATEGY:-mid}"
echo "[$(date -Is)] PKM slot diagnosis: ckpt=${CKPT}"
echo "  data=${GRID_POISON_DATA_DIR}  arch=${PKM_ARCH}  out=${OUT}"
python -u -m src.diagnose_pkm_slots \
  experiment=tiger_unlearn_scif_flat \
  data_dir="${GRID_POISON_DATA_DIR}" \
  "semantic_id_path='${SID}'" \
  "ckpt_path='${CKPT}'" \
  num_hierarchies="${NUM_HIER:-4}" \
  ${PKM_OVR} model.pkm_mode=${PKM_MODE:-replace} \
  hydra.run.dir="${OUT}"
echo "[$(date -Is)] done -> ${OUT}/pkm_slot_diagnostics.json"
