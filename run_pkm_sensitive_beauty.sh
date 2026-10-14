#!/usr/bin/env bash
# Sensitive-category unlearning with trained-in PKMs (beauty, "Hair Loss Products",
# forget ratio 1e-4, seed 2), in ERASE's setting: real users' real interactions,
# deleted per user (deletion_spec=item_pairs, sequential, 5 chunks).
#
# Question: on the spam attack, full-model unlearning matched the retrained model
# while memory-only unlearning cost 6-11% N@10. Does memory-only locality pay off
# when the deletion is per user and the concept's neighbours must also go?
#
# BASE models are the CLEAN PKM runs (they still contain the category):
#   p0     = BALANCE_E1_p0_CLEAN     (job 11177823, collapsed memory)
#   bnsw10 = BALANCE_E1_bnsw10_CLEAN (job 11177824, switch balance + batchnorm)
# REFERENCES = the same architectures retrained on the data minus the forget pairs
# (src/data/amazon_data/beauty_retrain_hairloss_1e-4_seed2), submitted by
# STAGE=retrain. SH@10 in every eval is Sensitive@10 against the hairloss manifest.
#
# Unlearning arms (STAGE=unlearn):
#   pre        finetune with 0 steps = eval of the base model (Sensitive@10 before)
#   full       unified, whole model, the sensitive grid's pinned lambda_f=0.1,
#              lambda_s=0, lambda_n=0, n_epochs 4 (sweep default), adaptive_codes off
#   vals       memory values only, adam lr 1e-3, n_epochs EP_MEM (default 10)
#   afihf_T    values only, top-T slots by AF-IHF, same budget
# on bnsw10: pre full vals afihf_100 afihf_1000; on p0: pre full vals.
#
# Usage:  STAGE=retrain ./run_pkm_sensitive_beauty.sh
#         STAGE=unlearn ./run_pkm_sensitive_beauty.sh     (DRY=1 to preview)
set -euo pipefail

STAGE="${STAGE:?STAGE=retrain|unlearn}"
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
SENS=src/data/amazon_data/beauty_sens_hairloss_1e-4_seed2
RETRAIN=src/data/amazon_data/beauty_retrain_hairloss_1e-4_seed2
MAN=src/data/amazon_data/beauty/forget_manifest_hairloss_1e-4_seed2.json
EMB=embeddings/beauty_merged_predictions_tensor_latest.pt
EP_MEM="${EP_MEM:-10}"
BN_OVR=(model.pkm_params.query_norm=batchnorm)
BAL_OVR=("model.pkm_balance={coef:0.1,kind:switch}")
ARCH="model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace"
run() { if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "$@"; echo; else "$@"; fi; }

if [ "$STAGE" = retrain ]; then
  for m in p0 bnsw10; do
    ovr=("model.pkm_explore={p:0.0,util_window:100}" "spam_forget_manifest=$MAN")
    [ $m = bnsw10 ] && ovr+=("${BAL_OVR[@]}" "${BN_OVR[@]}")
    echo "[$(date -Is)] retrain $m"
    run env PKM_MODE=replace PKM_ENCODER=1 PKM_DECODER=none DETERMINISTIC=1 N_TARGET_ITEMS=1 \
      TIGER_DATA_DIR="$RETRAIN" RUN_LABEL_SUFFIX="_BALANCE_E1_${m}_RETRAIN_hairloss" \
      sbatch --partition=pgpu --gres=gpu:nvidia_a100-sxm4-80gb:2 --time=16:00:00 \
      run_tiger_train.sh beauty clean "$SID" 0.01 1 trainer.max_steps=4000 "${ovr[@]}"
  done
  exit 0
fi

SENS_BASE=(unlearning.deletion_spec=item_pairs unlearning.include_context_rows=true
           unlearning.request_user_order=sorted unlearning.n_unlearning_chunks=5
           "unlearning.embedding_path=$EMB" unlearning.lambda_f=0.1 unlearning.lambda_s=0.0
           unlearning.lambda_n=0.0 unlearning.sep_negatives=forget_target_only
           unlearning.adaptive_codes=false)
MEM=(unlearning.unified_lr=0.001 unlearning.unified_optimizer=adam unlearning.n_epochs=$EP_MEM
     unlearning.update_scope=pkm_only unlearning.pkm_update_keys=false unlearning.pkm_update_query=false)

for spec in "bnsw10:11177824:pre full vals afihf_100 afihf_1000" "p0:11177823:pre full vals"; do
  IFS=: read -r m jid arms <<< "$spec"
  mo="$ARCH"; [ $m = bnsw10 ] && mo="$ARCH ${BN_OVR[*]}"
  for a in $arms; do
    algo=unified; extra=()
    case "$a" in
      pre)      algo=finetune; extra=(unlearning.finetune_steps=0 unlearning.update_scope=pkm_only) ;;
      full)     extra=("${SENS_BASE[@]}" unlearning.n_epochs=4 unlearning.update_scope=all) ;;
      vals)     extra=("${SENS_BASE[@]}" "${MEM[@]}") ;;
      afihf_*)  extra=("${SENS_BASE[@]}" "${MEM[@]}" unlearning.slot_selection=af_ihf unlearning.slot_top_t=${a#afihf_}) ;;
      *) echo "unknown arm $a"; exit 1 ;;
    esac
    [ "$a" = pre ] && extra+=(unlearning.deletion_spec=item_pairs unlearning.include_context_rows=true)
    tag="sens_beauty_hairloss_E1_${m}_${a}"; [[ "$a" == vals || "$a" == afihf_* ]] && tag="${tag}_ep${EP_MEM}"
    echo "[$(date -Is)] $tag (base job $jid)"
    run env UNLEARN_RUN_TAG="$tag" UNLEARN_SEED=2 UNLEARN_EVAL_DATA_DIR=src/data/amazon_data/beauty \
      EVAL_EXTRA_OVERRIDES="$mo" \
      sbatch --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
      run_tiger_unlearn_sequential.sh "job:$jid" "$SENS" "$algo" "$SID" false 1 0.0 \
      "spam_forget_manifest=$MAN" "${extra[@]}" $mo
  done
done
