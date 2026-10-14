#!/usr/bin/env bash
# Follow-ups to the switch load-balancing result (BALANCE_E1_bnsw10 = query_norm=
# batchnorm + pkm_balance={coef:0.1,kind:switch} fixed trained-in collapse: 165k
# of 262k slots read at eval with full retain, vs 32 for p0, at equal N@10).
# Each training job gets its eval slot diagnostic CHAINED (afterok), so both
# finish unattended.
#
# Sets (4000 steps, replace, DETERMINISTIC=1, 2x A100 pgpu, like the sweeps):
#   seeds   beauty E1, TRAIN_SEED 3 and 4  x {p0, bnsw10}          (replication)
#   dec     beauty D01 bnsw10    (D01 p0 = EXPLORE_D01_p0 already exists)
#   coef    beauty E1 bnsw30 / bnsw100   (coef 0.3 / 1.0: is stronger better?)
#   sports  sports 1% tgtmid E1 {p0, bnsw10}                       (2nd dataset)
# Every arm carries a pkm_explore p=0 block for the training-time slot trend.
#
# Usage:  ./run_pkm_balance_followup.sh          DRY=1 ./run_pkm_balance_followup.sh
#         SETS="seeds" ./run_pkm_balance_followup.sh
set -euo pipefail

SETS="${SETS:-seeds dec coef sports}"
STEPS="${STEPS:-4000}"
PARTITION="${PARTITION:-pgpu}"
GRES="${GRES:-gpu:nvidia_a100-sxm4-80gb:2}"
TIME="${TIME:-16:00:00}"
BN=model.pkm_params.query_norm=batchnorm

# arm spec: name dataset pos coef(0=off) seed
specs=()
for g in $SETS; do
  case "$g" in
    seeds)  specs+=("p0 beauty E1 0 3" "bnsw10 beauty E1 0.1 3" "p0 beauty E1 0 4" "bnsw10 beauty E1 0.1 4") ;;
    dec)    specs+=("bnsw10 beauty D01 0.1 2") ;;
    coef)   specs+=("bnsw30 beauty E1 0.3 2" "bnsw100 beauty E1 1.0 2") ;;
    sports) specs+=("p0 sports E1 0 2" "bnsw10 sports E1 0.1 2") ;;
    *) echo "unknown group $g"; exit 1 ;;
  esac
done

# chained eval diagnostic for training job $1 (uses the loop's ds/DATA/SID/diag/dovr)
chain_diag() {
  local tj="$1"
  local wrap="set -e; run=\$(ls -d logs/train/runs/*/*_job${tj}_* | head -1); bash run_diagnose_pkm_slots.sh \"\$run/checkpoints/last.ckpt\""
  dj=$(env "${diag[@]}" DATASET="$ds" TARGET_STRATEGY=mid GRID_POISON_DATA_DIR="$DATA" SID="$SID" \
       EXTRA_OVR="$dovr" sbatch --parsable --dependency=afterok:"$tj" --partition=pgpu --gres=gpu:2 \
       --time=02:00:00 --job-name="pkm_slots_${tag}_s${seed}" --output=logs/%j.out --wrap "$wrap")
}

for spec in "${specs[@]}"; do
  read -r arm ds pos coef seed <<< "$spec"
  case "$ds" in
    beauty) SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt ;;
    sports) SID=embeddings/sports/merged_predictions_tensor.pt ;;
  esac
  DATA=src/data/amazon_data/${ds}_spam_tgtmid_seed2_pct1_n1
  [ -d "$DATA" ] && [ -f "$SID" ] || { echo "missing $DATA or $SID"; exit 1; }
  case "$pos" in
    E1)  penc=1;    pdec=none; diag=(PKM_ARCH=custom PKM_ENC=1 PKM_DEC=none) ;;
    D01) penc=none; pdec=0,1;  diag=(PKM_ARCH=d01) ;;
  esac
  ovr=("model.pkm_explore={p:0.0,util_window:100}"); dovr=""
  if [ "$coef" != "0" ]; then
    ovr+=("model.pkm_balance={coef:$coef,kind:switch}" "$BN"); dovr="$BN"
  fi
  seedenv=(); [ "$seed" != "2" ] && seedenv=(TRAIN_SEED="$seed")
  tag="BALANCE_${pos}_${arm}"; [ "$ds" != beauty ] && tag="${tag}_${ds}"
  echo "[$(date -Is)] $tag seed=$seed: ${ovr[*]}"
  train=(env PKM_MODE=replace PKM_ENCODER=$penc PKM_DECODER=$pdec DETERMINISTIC=1 "${seedenv[@]}"
         TIGER_DATA_DIR="$DATA" RUN_LABEL_SUFFIX="_tgtmid_${tag}" N_TARGET_ITEMS=1
         sbatch --parsable --partition="$PARTITION" --gres="$GRES" --time="$TIME"
         run_tiger_train.sh "$ds" poison "$SID" 0.01 1 trainer.max_steps=${STEPS} "${ovr[@]}")
  if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "${train[@]}"; echo; continue; fi
  # EXISTING="TAG_sSEED=jobid ...": reuse an already-submitted training job (no
  # resubmission) and only chain its diagnostic
  tj=""
  for e in ${EXISTING:-}; do [ "${e%%=*}" = "${tag}_s${seed}" ] && tj="${e#*=}"; done
  if [ -n "$tj" ]; then echo "    reusing existing train job $tj"; else tj=$("${train[@]}"); fi
  # chained eval diagnostic: resolve the run dir by job id at RUN time (it is
  # timestamped, so unknown now), then run the diagnostic script in-allocation
  chain_diag "$tj"
  echo "    train=$tj  diag=$dj (afterok)"
done
