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
#   mdrop   beauty E1 bnsw10 + memory_dropout 0.1 / 0.3       (MUNKEY pathway dropout)
#   islots  beauty E1 bnsw10 + item_slots 16                  (item-addressed slots)
#   sw2     raw-score P ablation; place: D01 coef .3, E23D01, ALL; cap: n_keys 256/1024
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
    # memory (pathway) dropout on top of the collapse fix (MUNKEY-style)
    # switch-loss annealing: coef 0.1 -> 0 over 500 steps from step 1000 / 2000
    anneal) specs+=("bnsw10an1k beauty E1 0.1 2 model.pkm_balance={coef:0.1,kind:switch,anneal_start:1000,anneal_steps:500}"
                    "bnsw10an2k beauty E1 0.1 2 model.pkm_balance={coef:0.1,kind:switch,anneal_start:2000,anneal_steps:500}") ;;
    # switch ablation: P on raw scores instead of per-token standardised ones
    sw2)    specs+=("bnsw10raw beauty E1 0.1 2 model.pkm_balance={coef:0.1,kind:switch,standardize:false}") ;;
    # placements: stronger coef on D01 (only 21-24k slots at 0.1), multi-layer layouts
    place)  specs+=("bnsw30 beauty D01 0.3 2" "bnsw10 beauty E23D01 0.1 2" "bnsw10 beauty ALL 0.1 2") ;;
    # all 8 FFNs replaced, per-layer pressure matched to E1: the balance loss is the MEAN
    # over memories, so coef 0.8 over 8 memories = 0.1 per layer (ALL bnsw10 had 1/8)
    allmatch) specs+=("bnsw80 beauty ALL 0.8 2") ;;
    # the same per-layer pressure via the per-memory (sum) reduction: coef 0.1 x 8 memories
    allsum)   specs+=("bnsw10sum beauty ALL 0.1 2 model.pkm_balance={coef:0.1,kind:switch,reduction:sum}") ;;
    # capacity: 256^2 = 65k and 1024^2 = 1M slots (default 512^2 = 262k)
    cap256seeds) specs+=("bnsw10nk256 beauty E1 0.1 3 model.pkm_params.n_keys=256"
                         "bnsw10nk256 beauty E1 0.1 4 model.pkm_params.n_keys=256") ;;
    cap)    specs+=("bnsw10nk256 beauty E1 0.1 2 model.pkm_params.n_keys=256"
                    "bnsw10nk1024 beauty E1 0.1 2 model.pkm_params.n_keys=1024") ;;
    # item-addressed PKM slots: 16 private slots per item next to the shared table
    islots) specs+=("bnsw10is16 beauty E1 0.1 2 model.pkm_params.item_slots=16") ;;
    mdrop)  specs+=("bnsw10md10 beauty E1 0.1 2 model.pkm_params.memory_dropout=0.1"
                    "bnsw10md30 beauty E1 0.1 2 model.pkm_params.memory_dropout=0.3") ;;
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
  read -r arm ds pos coef seed xovr <<< "$spec"
  case "$ds" in
    beauty) SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt ;;
    sports) SID=embeddings/sports/merged_predictions_tensor.pt ;;
  esac
  DATA=src/data/amazon_data/${ds}_spam_tgtmid_seed2_pct1_n1
  [ -d "$DATA" ] && [ -f "$SID" ] || { echo "missing $DATA or $SID"; exit 1; }
  case "$pos" in
    E1)  penc=1;    pdec=none; diag=(PKM_ARCH=custom PKM_ENC=1 PKM_DEC=none) ;;
    D01) penc=none; pdec=0,1;  diag=(PKM_ARCH=d01) ;;
    E23D01) penc=2,3; pdec=0,1; diag=(PKM_ARCH=e23d01) ;;
    ALL) penc=all;  pdec=all;  diag=(PKM_ARCH=allv2) ;;
  esac
  ovr=("model.pkm_explore={p:0.0,util_window:100}"); dovr=""
  if [ "$coef" != "0" ]; then
    # an extra override that sets pkm_balance itself replaces the default one
    # reduction:mean reproduces every arm recorded before 2026-10-26 (Hydra MERGES dicts, so
    # without it they would inherit the new default sum); BALRED=sum for new runs
    [[ "${xovr:-}" == *pkm_balance=* ]] || ovr+=("model.pkm_balance={coef:$coef,kind:switch,reduction:${BALRED:-mean}}")
    ovr+=("$BN"); dovr="$BN"
    # PKM architecture overrides must reach the chained diagnostic too
    for t in ${xovr:-}; do [[ "$t" == model.pkm_params.* ]] && dovr="$dovr $t"; done
  fi
  [ -n "${xovr:-}" ] && ovr+=($xovr)
  seedenv=(); [ "$seed" != "2" ] && seedenv=(TRAIN_SEED="$seed")
  # p0 arms are the UNBALANCED control (the switch loss is the default since 2026-10-20)
  [ "$coef" = "0" ] && seedenv+=(PKM_BALANCE=off)
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
