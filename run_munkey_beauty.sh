#!/usr/bin/env bash
# MUNKEY baseline (ported from the authors' code in ../MUNKEY/, see
# src/models/components/network_blocks/munkey_memory.py) on beauty spam
# (tgtmid, 1%), plain TIGER (no PKM). Instances = training users; keys = frozen
# text-embedding mean of the history; one 128-d token per user prepended to the
# encoder input; eval always retrieves the top-4 OTHER users (softmax(sim/0.07));
# unlearning = deleting the forget users' keys (zero-shot).
#
# STAGE=keys   user-key files (CPU): spam training users, clean beauty users
# STAGE=train  (4000 steps, DETERMINISTIC=1, 2x A100 pgpu), arms:
#   mkAB   p_retrieval 0.2 (paper), pathway dropout p_token_drop = 0.A,
#          p_hist_drop = 0.B, e.g. mk00 / mk11 / mk22. MUNKEY picks (A, B) by
#          pathway sensitivity P_s; the grid stands in for that selection.
#   SEED=<n> other training seeds (label suffix _s<n>)
#   CLEAN=1  train on clean beauty with the clean-user keys (the retrained-
#            MUNKEY reference, "Retrain (M)" in the paper); label suffix _clean
#   RETUD=1  train on the whole-user-deletion RETRAIN data (2% of users removed,
#            beauty_userdel_2pct_seed2_retrain) with its own user keys: the
#            Retrain (M) reference for STAGE=userdel; label suffix _retud
# STAGE=userdel  MUNKEY's home setting: a stratified-like random 2% of whole users
#   deleted (beauty_userdel_2pct_seed2, 447 users, from the clean data). Every
#   SPEC "model:arm" is evaluated on BOTH scoped test sets (deleted users only /
#   all other users). arms: pre (0-step eval) | zs (key deletion, no update).
#   SELF=1 (default) evaluates as MUNKEY does: the query may retrieve its OWN
#   instance (their forget images are queried with themselves), so before deletion
#   a deleted user reads its own token and after deletion it cannot. SELF=0 keeps
#   the spam-setting default (own instance excluded).
#   e.g. SPECS="mk11_clean:pre mk11_clean:zs mk11_retud:pre"
# STAGE=unlearn  SPECS="mk11:zs mk11:zsfull ..." with arms
#   zs      MUNKEY: delete the forget (fake) users' keys, NO weight update
#   zsfull  key deletion + the recorded full-model unified recipe (job 10268039)
#   full    the unified recipe alone (no key deletion)
#
# Usage:  STAGE=keys ./run_munkey_beauty.sh        DRY=1 to preview
#         STAGE=train ARMS="mk00 mk11 mk22" ./run_munkey_beauty.sh
#         STAGE=unlearn SPECS="mk11:zs mk11:zsfull mk11:full" ./run_munkey_beauty.sh
set -euo pipefail

STAGE="${STAGE:?STAGE=keys|train|userdel|unlearn}"
D=src/data/amazon_data
# Paper grid (2026-11-24): DSEED (2,3,5,7,11) and STRATUM (unpopular|mid|popular) as in
# run_item_memory_beauty.sh; model names carry _up / _pop and _d<N> (mk11_pop_d7), keys
# per spam dataset (embeddings/user_keys/<data dir>.pt). Defaults = the original cell.
DSEED="${DSEED:-2}"; STRATUM="${STRATUM:-mid}"
spam_dir() {  # $1 data seed, $2 stratum
  case "$2" in
    unpopular) echo "$D/beauty_spam_seed${1}_pct1_n1" ;;
    mid)       echo "$D/beauty_spam_tgtmid_seed${1}_pct1_n1" ;;
    popular)   echo "$D/beauty_spam_tgtpopular_seed${1}_pct1_n1" ;;
    *) echo "unknown stratum $2" >&2; exit 1 ;;
  esac
}
st_tok() { case "$1" in unpopular) echo _up ;; popular) echo _pop ;; *) echo "" ;; esac; }
keys_of() { echo "embeddings/user_keys/$(basename "$1").pt"; }
DATA=$(spam_dir "$DSEED" "$STRATUM")
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
EMB=embeddings/beauty_merged_predictions_tensor_latest.pt
KEYS_SPAM=$(keys_of "$DATA")
KEYS_CLEAN=embeddings/user_keys/beauty_clean.pt
UD=$D/beauty_userdel_2pct_seed2
KEYS_UDRET=embeddings/user_keys/beauty_userdel_2pct_seed2_retrain.pt
run() { if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "$@"; echo; else "$@"; fi; }
# MK <key_file> <p_token_drop> <p_hist_drop>
MK() { echo "model.munkey_memory={key_file:$1,item_embedding_path:$EMB,token_dim:128,topk:${TOPK:-4},tau:0.07,p_retrieval:${P_RET:-0.2},p_token_drop:$2,p_hist_drop:$3${4:+,eval_exclude_self:$4}}"; }
arm_drops() {  # mkAB -> "0.A 0.B"
  local a="${1#mk}"; a="${a%%_*}"
  [[ "$a" =~ ^[0-9][0-9]$ ]] || { echo "bad arm $1 (want mkAB)" >&2; exit 1; }
  echo "0.${a:0:1} 0.${a:1:1}"
}

if [ "$STAGE" = keys ]; then
  specs=("$DATA:$KEYS_SPAM" "$D/beauty:$KEYS_CLEAN" "${UD}_retrain:$KEYS_UDRET")
  [ "${GRID_KEYS:-0}" = 1 ] && specs=("$DATA:$KEYS_SPAM")
  for spec in "${specs[@]}"; do
    data="${spec%%:*}"; out="${spec#*:}"
    [ -f "$out" ] && { echo "exists: $out"; continue; }
    run sbatch --partition=compute --mem=32G -c 8 --time=01:00:00 --output=logs/%j.out \
      --job-name=munkey_keys --wrap "python -m scripts.build_user_neighbors \
        --history $data/training --universe $data/training --emb $EMB --k 4 --save_keys --out $out"
  done
  exit 0
fi

if [ "$STAGE" = train ]; then
  SEED="${SEED:-2}"; stok=""; seedenv=()
  [ "$SEED" != 2 ] && { stok="_s$SEED"; seedenv=(TRAIN_SEED=$SEED); }
  data=$DATA; keys=$KEYS_SPAM; variant=poison; ctok=""
  dtok=""; [ "$DSEED" != 2 ] && { [ "$SEED" != 2 ] && { echo "set DSEED or SEED, not both"; exit 1; }
                                 dtok="_d$DSEED"; seedenv=(TRAIN_SEED=$DSEED); }
  [ "${CLEAN:-0}" = 1 ] && { data=$D/beauty; keys=$KEYS_CLEAN; variant=clean; ctok="_clean"; }
  [ "${RETUD:-0}" = 1 ] && { data=${UD}_retrain; keys=$KEYS_UDRET; variant=clean; ctok="_retud"; }
  for arm in ${ARMS:-mk00 mk11 mk22}; do
    read -r pt ph <<< "$(arm_drops "$arm")"
    echo "[$(date -Is)] train MUNKEY_${arm}${ctok}${stok}"
    run env DETERMINISTIC=1 "${seedenv[@]}" TIGER_DATA_DIR="$data" N_TARGET_ITEMS=1 \
      RUN_LABEL_SUFFIX="_tgt${STRATUM}_MUNKEY_${arm}${ctok}$(st_tok "$STRATUM")${dtok}${stok}" \
      sbatch ${DEP:+--dependency=afterok:$DEP} --partition=pgpu --gres=${TRAIN_GRES:-gpu:nvidia_a100-sxm4-80gb:2} --time=16:00:00 \
      run_tiger_train.sh beauty "$variant" "$SID" 0.01 1 trainer.max_steps=4000 \
      "$(MK "$keys" "$pt" "$ph")"
  done
  exit 0
fi

if [ "$STAGE" = userdel ]; then
  self="${SELF:-1}"; ex=false; [ "$self" = 0 ] && ex=true
  for spec in ${SPECS:?SPECS=\"mk11_clean:pre mk11_clean:zs mk11_retud:pre\"}; do
    model="${spec%%:*}"; arm="${spec#*:}"
    run_dir=$(ls -d logs/train/runs/*/*_tgtmid_MUNKEY_${model} 2>/dev/null | tail -1)
    ckpt="$run_dir/checkpoints/last.ckpt"
    [ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
    keys=$KEYS_CLEAN; [[ "$model" == *_retud* ]] && keys=$KEYS_UDRET
    read -r pt ph <<< "$(arm_drops "$model")"
    mo="$(MK "$keys" "$pt" "$ph" "$ex")"
    case "$arm" in
      pre) algo=finetune; x=(unlearning.finetune_steps=0 unlearning.n_epochs=null) ;;
      zs)  algo=unified;  x=(unlearning.munkey_delete=true unlearning.zero_shot=true
                             unlearning.n_unlearning_chunks=1) ;;
    esac
    # whole-user deletion has no target item: the default sep_negatives
    # (forget_target_only) refuses to run (jobs 11275324-27, 11276969/70); no arm here
    # takes a separation step, so the negatives only have to be constructible
    x+=(unlearning.sep_negatives=forget)
    case "$arm" in pre|zs) ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    for scope in forget retain; do
      tag="ud_beauty_MUNKEY_${model}_${arm}_${scope}_self${self}"
      echo "[$(date -Is)] $tag"
      run env UNLEARN_RUN_TAG="$tag" EVAL_EXTRA_OVERRIDES="$mo" N_TARGET_ITEMS=1 \
        UNLEARN_EVAL_DATA_DIR="${UD}_eval${scope}" \
        sbatch --partition=pgpu --gres=gpu:2 --time=04:00:00 --job-name="ul_$tag" \
        run_tiger_unlearn_sequential.sh "$ckpt" "$UD" "$algo" "$SID" false 1 0.0 "${x[@]}" "$mo"
    done
  done
  exit 0
fi

# unlearning: the recorded full-model unified recipe (job 10268039) as the base
# RECIPES (space-separated list, each request runs once per recipe; default "fr nau"):
#   fr     the paper's Forget + Repair (valsel-selected unified_abl_lf0.1_ls0.0_ln0.0):
#          lambda_f 0.1, lambda_s = lambda_n = 0                      tag _fr_lf0.1
#   nau    the paper's NAU (nau_s0p01_n0p01_keepck, job 10870227): lambda_f 0.1,
#          lambda_s 0.01, lambda_n 0.01, embedding neighbourhood (8, mass)  tag _nau
#   fr001  forget + repair at lambda_f 0.01 (the 2026-10-21 runs; removes nothing)  tag _fr
#   sep    the earlier recipe, lambda_f 0.01, lambda_s 1.0                 untagged
# Both paper recipes were diffed key by key against the recorded runs' config_tree.log.
recipe_args() {  # $1 = recipe, $2 = item embedding file -> sets LS, rtok
  case "$1" in
    fr)    LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0
               unlearning.sep_loss_type=cosine); rtok=_fr_lf0.1 ;;
    nau)   LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01
               unlearning.coherence_loss_type=mass unlearning.coherence_neighbor_method=embedding
               "unlearning.embedding_path=$2" unlearning.neighborhood_count=8
               # gradient checkpointing (ported from main): memory only, bit-identical
               # gradients (checked 2026-10-22); without it NAU on the score models ran
               # out of memory on 40 GB A100s (jobs 11268362-80)
               unlearning.coherence_checkpoint=true unlearning.sep_checkpoint=true); rtok=_nau ;;
    fr001) LS=(unlearning.lambda_f=0.01 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0); rtok=_fr ;;
    sep)   LS=(unlearning.lambda_f=0.01 unlearning.lambda_s=1.0); rtok="" ;;
    *) echo "unknown recipe $1 (fr|nau|fr001|sep)"; exit 1 ;;
  esac
}
RECIPES="${RECIPES:-fr nau}"
BASE0=(unlearning.n_unlearning_chunks=10 unlearning.sep_negatives=forget_target_only
       unlearning.adaptive_codes=false unlearning.n_epochs=4)
for spec in ${SPECS:?SPECS=\"mk11:zs ...\"}; do
  model="${spec%%:*}"; arm="${spec#*:}"
  mdseed=2; [[ "$model" =~ _d([0-9]+)$ ]] && mdseed="${BASH_REMATCH[1]}"
  mst=mid; [[ "$model" =~ _up(_d[0-9]+)?$ ]] && mst=unpopular; [[ "$model" =~ _pop(_d[0-9]+)?$ ]] && mst=popular
  mdata=$(spam_dir "$mdseed" "$mst"); mkeys=$(keys_of "$mdata")
  run_dir=$(ls -d logs/train/runs/*/*_tgt${mst}_MUNKEY_${model} 2>/dev/null | tail -1) || true
  ckpt="$run_dir/checkpoints/last.ckpt"
  dep=(); [ -n "${DEP:-}" ] && dep=(--dependency=afterok:$DEP)
  [ -n "$run_dir" ] && { [ -f "$ckpt" ] || [ -n "${DEP:-}" ]; } || { echo "missing $ckpt"; exit 1; }
  read -r pt ph <<< "$(arm_drops "$model")"
  mo="$(MK "$mkeys" "$pt" "$ph")"
  algo=unified
  case "$arm" in
    # zero-step eval of the trained model (the before-deletion reference)
    pre)    algo=finetune; x=(unlearning.finetune_steps=0 unlearning.n_epochs=null) ;;
    zs)     x=(unlearning.munkey_delete=true unlearning.zero_shot=true) ;;
    zsfull) x=(unlearning.munkey_delete=true) ;;
    full)   x=() ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  # zs takes no gradient step, so the lambdas are inert: one run, untagged
  rlist="$RECIPES"; [[ "$arm" == zs || "$arm" == pre ]] && rlist=none
  for rc in $rlist; do
    if [ "$rc" = none ]; then LS=(); rtok=""; else recipe_args "$rc" "$EMB"; fi
    tag="bw_tgt${mst}_MUNKEY_${model}_${arm}${rtok}"
    echo "[$(date -Is)] $tag${DEP:+ (after $DEP)}"
    run env UNLEARN_RUN_TAG="$tag" EVAL_EXTRA_OVERRIDES="$mo" N_TARGET_ITEMS=1 \
      sbatch "${dep[@]}" --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
      run_tiger_unlearn_sequential.sh "$ckpt" "$mdata" "$algo" "$SID" false 1 0.0 \
      "${BASE0[@]}" "${LS[@]}" "${x[@]}" "$mo"
  done
done
