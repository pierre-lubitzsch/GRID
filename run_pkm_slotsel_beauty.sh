#!/usr/bin/env bash
# Slot-selection unlearning on a HEALTHY trained-in PKM, beauty tgtmid (TIGER, E1).
#
# The switch load-balancing run (BALANCE_E1_bnsw10: query_norm=batchnorm +
# pkm_balance={coef:0.1,kind:switch}) is the first trained-in model with BOTH a
# spread memory (92k of 262k slots read at eval, Jaccard 0.43, ~16.7k
# forget-exclusive slots on the retain subset) AND the spam still present.
# This is the testbed that top-t slot selection never had.
#
# Unlearning recipe = the recorded best pkm_only operating point (job 10280110;
# positional: neighborhood_aware=false, request_batch_size=1, sample_rate=0.0;
# unified lf0.01 ls1.0, sep_negatives=forget_target_only, adaptive_codes=false,
# adam lr 1e-3, n_epochs 100, 10 chunks), retain_source=subset (the default) as recorded.
#
# Arms (on bnsw10 unless noted):
#   full       update_scope=all (full-model unified, reference)
#   pkm        update_scope=pkm_only, all memory params, no selection
#   vals       pkm_only, VALUES only (keys + query frozen), no selection
#   afihf_T    values only, top-T slots by AF-IHF,           T in TS
#   gradc_T    values only, top-T slots by grad_combined,    T in TS
#   p0pkm      pkm_only on the COLLAPSED EXPLORE_E1_p0 model (control)
#   fullrec    full model with its OWN recorded recipe (FULLREC; ignores EPOCHS)
#   fullrec_p0 same on the collapsed p0 model
# Selection masks only values.weight rows, so the selection arms freeze keys and
# query; 'vals' is their no-selection reference.
#
# Usage:  ./run_pkm_slotsel_beauty.sh      DRY=1 ./run_pkm_slotsel_beauty.sh
#         ARMS="pkm afihf_1000" ./run_pkm_slotsel_beauty.sh
set -euo pipefail

DATA=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
TS="${TS:-100 1000 10000}"
ARMS="${ARMS:-full pkm vals $(for t in $TS; do echo -n "afihf_$t gradc_$t "; done)p0pkm}"
PARTITION="${PARTITION:-pgpu}"
GRES="${GRES:-gpu:2}"
TIME="${TIME:-04:00:00}"

BN_RUN=$(ls -d logs/train/runs/*/*_BALANCE_E1_bnsw10 | head -1)
P0_RUN=$(ls -d logs/train/runs/*/*_EXPLORE_E1_p0 | head -1)
PKM_ARCH_OVR="model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace"
# EPOCHS: memory-update budget (recorded pkm_only best = 100). Smaller values
# unsaturate the test (at 100 every healthy-memory arm reached SH 0); tagged _ep<N>.
EPOCHS="${EPOCHS:-100}"
BASE=(unlearning.n_unlearning_chunks=10 unlearning.lambda_f=0.01 unlearning.lambda_s=1.0
      unlearning.sep_negatives=forget_target_only unlearning.adaptive_codes=false)
RECIPE=("${BASE[@]}" unlearning.unified_lr=0.001 unlearning.n_epochs=${EPOCHS} unlearning.unified_optimizer=adam)
# Full-model reference with ITS OWN recorded recipe (job 10268039: E23D01 mid,
# SH 0.01248 -> 0.00411): n_epochs=4, default lr/optimizer. The pkm_only recipe
# above destroyed the full model (N@10 0.0039, job 11137260).
FULLREC=("${BASE[@]}" unlearning.n_epochs=4 unlearning.update_scope=all)
# NOT retain_source=full: with unified it makes every step iterate the full retain
# set (~100 min per chunk vs ~1.7 min, jobs 11131204-13 cancelled for that). The
# recorded operating point used the default subset.
VALS_ONLY=(unlearning.update_scope=pkm_only unlearning.pkm_update_keys=false unlearning.pkm_update_query=false)

for a in $ARMS; do
  run="$BN_RUN"; qn="model.pkm_params.query_norm=batchnorm"
  case "$a" in
    full)     ovr=(unlearning.update_scope=all) ;;
    fullrec)  ovr=() ;;  # recipe replaced below
    fullrec_p0) ovr=(); run="$P0_RUN"; qn="" ;;
    pkm)      ovr=(unlearning.update_scope=pkm_only) ;;
    vals)     ovr=("${VALS_ONLY[@]}") ;;
    afihf_*)  ovr=("${VALS_ONLY[@]}" unlearning.slot_selection=af_ihf unlearning.slot_top_t=${a#afihf_}) ;;
    gradc_*)  ovr=("${VALS_ONLY[@]}" unlearning.slot_selection=grad_combined unlearning.slot_top_t=${a#gradc_}) ;;
    p0pkm)    ovr=(unlearning.update_scope=pkm_only); run="$P0_RUN"; qn="" ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac
  ckpt="$run/checkpoints/last.ckpt"
  [ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
  model_ovr="$PKM_ARCH_OVR${qn:+ $qn}"
  recipe=("${RECIPE[@]}"); tag="SLOTSEL_${a}"
  if [[ "$a" == fullrec* ]]; then recipe=("${FULLREC[@]}")
  elif [ "$EPOCHS" != "100" ]; then tag="${tag}_ep${EPOCHS}"; fi
  echo "[$(date -Is)] $tag: $(basename "$run") ${ovr[*]}"
  cmd=(env UNLEARN_RUN_TAG="bw_tgtmid_E1_${tag}" EVAL_EXTRA_OVERRIDES="$model_ovr"
       N_TARGET_ITEMS=1
       sbatch --partition="$PARTITION" --gres="$GRES" --time="$TIME"
       run_tiger_unlearn_sequential.sh "$ckpt" "$DATA" unified "$SID" false 1 0.0
       "${recipe[@]}" "${ovr[@]}" $model_ovr)
  if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
