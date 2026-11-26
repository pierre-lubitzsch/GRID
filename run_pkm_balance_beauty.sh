#!/usr/bin/env bash
# Routing balance loss for TRAINED-IN PKM collapse, beauty (TIGER), encoder block 1.
#
# The explore/exploit sweep (run_pkm_explore_beauty.sh) showed that collapse is a
# ROUTING failure: dense value updates did not stop it. Every arm read 32-160 slots
# at eval, with 0 forget-exclusive slots. This sweep puts the pressure on the router:
#   L = loss + coef * mean_layers( log n_keys - H(mean_b p_b) + mean_b H(p_b) )
# per head and sub-key half (p_b = softmax over the 512 sub-keys for token b),
# i.e. log n_keys minus the token/sub-key mutual information (model.pkm_balance).
#
# Same setup as the explore sweep (beauty poison tgtmid, replace, E1, 4000 steps,
# DETERMINISTIC=1, 2x A100), so EXPLORE_E1_p0 is the matched no-balance baseline.
# A pkm_explore block with p=0 is kept only for the training-time live-slot trend.
# Judge collapse with run_diagnose_pkm_slots.sh (eval mode); the training-time
# live_slots overstate it (dropout 0.15).
#
# Arms:
#   bal01    coef 0.01
#   bal10    coef 0.1
#   bnbal01  query_norm=batchnorm + coef 0.01
#   bnbal10  query_norm=batchnorm + coef 0.1
#   sw01 / sw10 / bnsw01 / bnsw10: same grid with kind=switch (Switch-style load
#            balancing on the hard top-knn sub-key assignments, standardised P).
#            Added after the mi arms proved inert (coef .01) or sharpening (.1).
#
# Usage:  ./run_pkm_balance_beauty.sh            DRY=1 ./run_pkm_balance_beauty.sh
#         ARMS="bal10" ./run_pkm_balance_beauty.sh
set -euo pipefail

# These runs predate the switch-loss default (2026-10-20): train them unbalanced,
# as recorded. Arms that set pkm_balance / query_norm themselves still do so.
export PKM_BALANCE="${PKM_BALANCE:-off}"

DATA=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
STEPS="${STEPS:-4000}"
ARMS="${ARMS:-bal01 bal10 bnbal01 bnbal10}"
PARTITION="${PARTITION:-pgpu}"
GRES="${GRES:-gpu:nvidia_a100-sxm4-80gb:2}"
TIME="${TIME:-16:00:00}"

[ -d "$DATA" ] || { echo "missing $DATA"; exit 1; }
[ -f "$SID" ]  || { echo "missing $SID";  exit 1; }

for a in $ARMS; do
  bn=(); kind=mi
  case "$a" in
    bal01)   coef=0.01 ;;
    bal10)   coef=0.1  ;;
    bnbal01) coef=0.01; bn=(model.pkm_params.query_norm=batchnorm) ;;
    bnbal10) coef=0.1;  bn=(model.pkm_params.query_norm=batchnorm) ;;
    sw01)    coef=0.01; kind=switch ;;
    sw10)    coef=0.1;  kind=switch ;;
    bnsw01)  coef=0.01; kind=switch; bn=(model.pkm_params.query_norm=batchnorm) ;;
    bnsw10)  coef=0.1;  kind=switch; bn=(model.pkm_params.query_norm=batchnorm) ;;
    *) echo "unknown arm $a"; exit 1 ;;
  esac
  tag="BALANCE_E1_${a}"
  echo "[$(date -Is)] $tag: coef=$coef kind=$kind ${bn[*]}"
  cmd=(env PKM_MODE=replace PKM_ENCODER=1 PKM_DECODER=none DETERMINISTIC=1
       TIGER_DATA_DIR="$DATA" RUN_LABEL_SUFFIX="_tgtmid_${tag}" N_TARGET_ITEMS=1
       sbatch --partition="$PARTITION" --gres="$GRES" --time="$TIME"
       run_tiger_train.sh beauty poison "$SID" 0.01 1
       trainer.max_steps=${STEPS} "model.pkm_balance={coef:$coef,kind:$kind}"
       "model.pkm_explore={p:0.0,util_window:100}" "${bn[@]}")
  if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "${cmd[@]}"; echo; else "${cmd[@]}"; fi
done
