#!/usr/bin/env bash
# Explore / exploit training for TRAINED-IN PKM collapse, beauty (TIGER).
#
# With probability p per optimizer step every PKM layer reads -- and so updates
# -- ALL n_keys**2 slots (exact factorised dense read, PKMExploreController /
# HashingMemory.set_explore); otherwise the usual sparse top-knn read. Question:
# does periodically pushing gradient into every slot keep the router from
# collapsing onto 32-128 slots, and does p need to decay once the memory is used?
#
# Setup matches run_pkm_warmup_beauty.sh (beauty poison tgtmid, replace, 4000
# steps, DETERMINISTIC=1), so the WARMUP_base (96 slots) / WARMUP_bn (2742)
# references compare directly. Every arm sets a pkm_explore block, so all of
# them (incl. p=0) log pkm/util_mean and pkm/live_slots/* during training.
#
# Placements (POS):  E1  = encoder block 1 (prevention side, collapse reference)
#                    D01 = decoder blocks 0,1 (removal side, default layout)
# Arms:
#   p0      p=0 (no exploration; utilisation logging only = matched baseline)
#   d05     dense, p=0.05
#   d20     dense, p=0.2
#   d50     dense, p=0.5
#   st20    straight_through, p=0.2 (sparse forward, dense gradient)
#   ada     adaptive: p=0.5 -> floor 0.02 as utilisation approaches 0.3
#   lin     linear: p=0.5 -> 0.02 over steps 0..3000
#   bnd20   query_norm=batchnorm + dense p=0.2 (stack on the one partial fix)
#
# After training, run run_diagnose_pkm_slots.sh on each run for coverage /
# Jaccard / forget-exclusive slots, as for the earlier arms.
#
# Usage:  ./run_pkm_explore_beauty.sh
#         POS="E1" ARMS="p0 d20" ./run_pkm_explore_beauty.sh
#         DRY=1 ./run_pkm_explore_beauty.sh
set -euo pipefail

DATA=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
STEPS="${STEPS:-4000}"
POS="${POS:-E1 D01}"
ARMS="${ARMS:-p0 d05 d20 d50 st20 ada lin}"
PARTITION="${PARTITION:-pgpu}"
GRES="${GRES:-gpu:nvidia_a100-sxm4-80gb:2}"
WIN="${WIN:-100}"
# Explore steps cost extra: the smoke run (p=0.5, E1+D01) took ~13 s/step vs
# ~6.5 s/step before, so a 4000-step d50 arm needs ~14 h > run_tiger_train.sh's 12 h.
TIME="${TIME:-24:00:00}"

[ -d "$DATA" ] || { echo "missing $DATA"; exit 1; }
[ -f "$SID" ]  || { echo "missing $SID";  exit 1; }

for pos in $POS; do
  case "$pos" in
    E1)  penc=1;    pdec=none ;;
    D01) penc=none; pdec=0,1  ;;
    *) echo "unknown position $pos"; exit 1 ;;
  esac
  for a in $ARMS; do
    bn=()
    case "$a" in
      p0)    ex="p:0.0" ;;
      d05)   ex="p:0.05,forward:dense" ;;
      d20)   ex="p:0.2,forward:dense" ;;
      d50)   ex="p:0.5,forward:dense" ;;
      st20)  ex="p:0.2,forward:straight_through" ;;
      ada)   ex="p:0.5,schedule:adaptive,p_final:0.02,util_target:0.3,forward:dense" ;;
      lin)   ex="p:0.5,schedule:linear,p_final:0.02,decay_steps:3000,forward:dense" ;;
      bnd20) ex="p:0.2,forward:dense"; bn=(model.pkm_params.query_norm=batchnorm) ;;
      *) echo "unknown arm $a"; exit 1 ;;
    esac
    tag="EXPLORE_${pos}_${a}"
    echo "[$(date -Is)] $tag: {$ex,util_window:$WIN} ${bn[*]}"
    cmd=(env PKM_MODE=replace PKM_ENCODER=$penc PKM_DECODER=$pdec DETERMINISTIC=1
         TIGER_DATA_DIR="$DATA" RUN_LABEL_SUFFIX="_tgtmid_${tag}" N_TARGET_ITEMS=1
         sbatch --partition="$PARTITION" --gres="$GRES" --time="$TIME"
         run_tiger_train.sh beauty poison "$SID" 0.01 1
         trainer.max_steps=${STEPS} "model.pkm_explore={$ex,util_window:$WIN}" "${bn[@]}")
    if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
  done
done
