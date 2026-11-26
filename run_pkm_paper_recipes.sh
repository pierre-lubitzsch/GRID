#!/usr/bin/env bash
# How much does a NON-collapsed (balanced) PKM help unlearning? Collapsed (p0) vs balanced
# (switch loss + batchnorm) PKM in encoder block 1, and item memory + balanced PKM
# (impkmsc), under the PAPER's unlearning algorithms (Forget + Repair, NAU), each with:
#   pre      zero-step eval (before deletion)
#   full     full-model unlearning (recipe defaults: lr 1e-4, 4 epochs, 10 requests)
#   vals     memory-only: only the PKM value table moves (keys / query frozen),
#            Adam lr 1e-3, 4 epochs (the memory-only setting of the 2026-10-13 sweep)
#   afihf    memory-only restricted to the top-1000 value slots by AF-IHF
#   boost    full unlearning, but the top-1000 forget-dominated (AF-IHF) value slots per
#            memory move at 10x the rate, each drift-capped at 0.5 of its norm (no deletion)
# The architecture is what this (PKM) paper changes; F+R / NAU are only the unlearning
# algorithms run on top (NAU is the other paper).
#
# Usage: SPECS="p0_s2:full bnsw10_s2:vals ..." ./run_pkm_paper_recipes.sh   (DRY=1 to preview)
set -euo pipefail

DATA=src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
EMB=embeddings/beauty_merged_predictions_tensor_latest.pt
ARCH="model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace"
SCORE="score_loss_weight:1.0,score_weight:1.0,score_negatives:255,score_tau:1.0"
IMS="model.item_memory={embedding_path:$EMB,p_drop:0.1,p_nbr:0.2,max_neighbors:4,tau:0.1,$SCORE}"
run() { if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "$@"; echo; else "$@"; fi; }

# recipes: identical to run_item_memory_beauty.sh (diffed against the recorded paper runs)
recipe_args() {
  case "$1" in
    fr)  LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0
             unlearning.sep_loss_type=cosine); rtok=_fr_lf0.1 ;;
    nau) LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01
             unlearning.coherence_loss_type=mass unlearning.coherence_neighbor_method=embedding
             "unlearning.embedding_path=$EMB" unlearning.neighborhood_count=8
             unlearning.coherence_checkpoint=true unlearning.sep_checkpoint=true); rtok=_nau ;;
    *) echo "unknown recipe $1"; exit 1 ;;
  esac
}
RECIPES="${RECIPES:-fr nau}"
BASE0=(unlearning.n_unlearning_chunks=10 unlearning.sep_negatives=forget_target_only
       unlearning.adaptive_codes=false unlearning.n_epochs=4)
MEM=(unlearning.update_scope=pkm_only unlearning.pkm_update_keys=false
     unlearning.pkm_update_query=false unlearning.unified_lr=0.001 unlearning.unified_optimizer=adam)

for spec in ${SPECS:?SPECS=\"model:arm ...\"}; do
  model="${spec%%:*}"; arm="${spec#*:}"
  case "$model" in   # run dirs of the trained models
    p0_s2)       pat="*_tgtmid_EXPLORE_E1_p0" ;;
    bnsw10_s2)   pat="*pkmreplace_tgtmid_BALANCE_E1_bnsw10" ;;
    p0_s3)       pat="*seed3_tgtmid_BALANCE_E1_p0" ;;
    bnsw10_s3)   pat="*seed3_tgtmid_BALANCE_E1_bnsw10" ;;
    impkmsc_s2)  pat="*_tgtmid_ITEMMEM_impkmsc" ;;
    impkmsc_s3)  pat="*_tgtmid_ITEMMEM_impkmsc_s3" ;;
    all80_s2)    pat="*pkmreplace_res_tgtmid_BALANCE_ALL_bnsw80" ;;
    *) echo "unknown model $model"; exit 1 ;;
  esac
  run_dir=$(ls -d logs/train/runs/*/$pat 2>/dev/null | tail -1)
  ckpt="$run_dir/checkpoints/last.ckpt"
  [ -f "$ckpt" ] || { echo "missing $ckpt ($pat)"; exit 1; }
  # query_norm: auto in the unlearn config, and adopted from the ckpt at eval
  arch="$ARCH"; [[ "$model" == all80* ]] && arch="model.pkm_layers={encoder:all,decoder:all} model.pkm_mode=replace"
  mo="$arch"; [[ "$model" == impkmsc* ]] && mo="$IMS $arch"
  algo=unified; x=()
  case "$arm" in
    pre)   algo=finetune; x=(unlearning.finetune_steps=0 unlearning.n_epochs=null) ;;
    full)  x=() ;;
    vals)  x=("${MEM[@]}") ;;
    afihf) x=("${MEM[@]}" unlearning.slot_selection=af_ihf unlearning.slot_top_t=1000) ;;
    boost) x=(unlearning.slot_boost=10.0 unlearning.slot_boost_top_t=1000 unlearning.slot_boost_max_drift=0.5) ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  rlist="$RECIPES"; [ "$arm" = pre ] && rlist=none
  for rc in $rlist; do
    if [ "$rc" = none ]; then LS=(); rtok=""; else recipe_args "$rc"; fi
    tag="bw_tgtmid_PKMREC_${model}_${arm}${rtok}"
    echo "[$(date -Is)] $tag"
    run env UNLEARN_RUN_TAG="$tag" EVAL_EXTRA_OVERRIDES="$mo" N_TARGET_ITEMS=1 \
      sbatch --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
      run_tiger_unlearn_sequential.sh "$ckpt" "$DATA" "$algo" "$SID" false 1 0.0 \
      "${BASE0[@]}" "${LS[@]}" "${x[@]}" $mo
  done
done
