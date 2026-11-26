#!/usr/bin/env bash
# User-keyed exemplar memory (MUNKEY-style, users as instances), TIGER, beauty.
# One learnable token per training user, prepended to the encoder input; neighbour
# users by frozen history embedding (scripts/build_user_neighbors.py, built by
# scripts/prep_user_memory_data.sh). Test users are the training users, so each
# uses its own token.
#
# MAIN: per-user SENSITIVE deletion (hair-loss products, forget ratio 1e-3:
# 139 requesting users, 154 interactions). With the backbone frozen and only the
# requesting users' tokens updated, every other user's recommendations are
# unchanged BY CONSTRUCTION -- the full model removed the category for everyone
# (SHR -37/-42% at 1e-4).
# SIDE CHECK: spam prevention. Fake users are training-only, so editing their
# tokens changes no test prediction; the question is only whether user tokens
# absorb the bandwagon pattern instead of the backbone (SH vs ITEMMEM_plain).
#
# STAGE=train (4000 steps, DETERMINISTIC=1, 2x A100 pgpu, no PKM):
#   plainclean  clean beauty, no user memory            (sensitive base, baseline)
#   umclean     clean beauty + user memory              (sensitive base, ours)
#   plainret    retrain data (forget pairs removed)     (reference for plainclean)
#   umret       retrain data + user memory              (reference for umclean)
#   umspam      poisoned beauty + user memory            (spam side check)
#   umhclean    umclean + history dropout p_hist (MUNKEY's p_i; P_HIST, default 0.2)
#   plainretud / umhretud  retrain references for WHOLE-USER deletion (2% users)
#   umhspam     umspam + history dropout (spam prevention)
#   umg_<h>_<d> MUNKEY P_s grid: p_hist=h, p_drop=d on clean beauty
#   DEP=<jobid> makes the jobs wait for the data prep.
# STAGE=unlearn (sensitive 1e-3 data; SH@10 = Sensitive@10, SHF/SHR scoped):
#   plainclean: full
#   umclean:    full | uonly (user tokens of the requesting users only, x10,
#               backbone frozen) | ureset (zero-shot: requesting users' tokens set
#               to their neighbour average, neighbours from the RETAINED histories)
#               | uresetft (reset alpha 0.5, then uonly)
set -euo pipefail

STAGE="${STAGE:?STAGE=train|unlearn}"
D=src/data/amazon_data
SID=embeddings/beauty_rqvae/merged_predictions_tensor.pt
SENS=$D/beauty_sens_hairloss_1e-3_seed2
RET=$D/beauty_retrain_hairloss_1e-3_seed2
MAN=$SENS/forget_manifest.json
SPAM=$D/beauty_spam_tgtmid_seed2_pct1_n1
NB=embeddings/user_neighbors
UM() { echo "model.user_memory={neighbor_file:$1,p_drop:${P_DROP:-0.1},p_nbr:${P_NBR:-0.2},tau:0.1${2:+,p_hist:$2}}"; }
run() { if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "$@"; echo; else "$@"; fi; }

if [ "$STAGE" = train ]; then
  dep=(); [ -n "${DEP:-}" ] && dep=(--dependency=afterok:$DEP)
  for arm in ${ARMS:-plainclean umclean plainret umret umspam}; do
    case "$arm" in
      plainclean) data=$D/beauty; variant=clean;  ovr=("spam_forget_manifest=$MAN") ;;
      umclean)    data=$D/beauty; variant=clean;  ovr=("spam_forget_manifest=$MAN" "$(UM $NB/beauty_clean.pt)") ;;
      plainret)   data=$RET;      variant=clean;  ovr=("spam_forget_manifest=$MAN") ;;
      umret)      data=$RET;      variant=clean;  ovr=("spam_forget_manifest=$MAN" "$(UM $NB/beauty_retrain_hairloss_1e-3_seed2.pt)") ;;
      umspam)     data=$SPAM;     variant=poison; ovr=("$(UM $NB/beauty_spam_tgtmid_seed2_pct1_n1.pt)") ;;
      # spam prevention with history dropout: can the token absorb the fake pattern?
      umhspam)    data=$SPAM;     variant=poison; ovr=("$(UM $NB/beauty_spam_tgtmid_seed2_pct1_n1.pt ${P_HIST:-0.2})") ;;
      # MUNKEY 2 grid over (p_hist, p_drop): arm name umg_<p_hist>_<p_drop>, e.g. umg_0.3_0.1
      umg_*)      data=$D/beauty; variant=clean; h=$(echo $arm | cut -d_ -f2); dd=$(echo $arm | cut -d_ -f3)
                  ovr=("spam_forget_manifest=$MAN" "$(P_DROP=$dd UM $NB/beauty_clean.pt $h)") ;;
      # MUNKEY's history-pathway dropout (p_hist): the token must carry the user
      umhclean)   data=$D/beauty; variant=clean;  ovr=("spam_forget_manifest=$MAN" "$(UM $NB/beauty_clean.pt ${P_HIST:-0.2})") ;;
      # whole-user deletion references (2% of users removed); deleted users are
      # never neighbours, and at test they get the neighbour-average token
      plainretud) data=$D/beauty_userdel_2pct_seed2_retrain; variant=clean; ovr=() ;;
      umhretud)   data=$D/beauty_userdel_2pct_seed2_retrain; variant=clean; ovr=("$(UM $NB/beauty_userdel_2pct_seed2_excl.pt ${P_HIST:-0.2})") ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    echo "[$(date -Is)] train USERMEM_$arm"
    run env DETERMINISTIC=1 TIGER_DATA_DIR="$data" RUN_LABEL_SUFFIX="_USERMEM_${arm}" N_TARGET_ITEMS=1 \
      sbatch "${dep[@]}" --partition=pgpu --gres=gpu:nvidia_a100-sxm4-80gb:2 --time=16:00:00 \
      run_tiger_train.sh beauty "$variant" "$SID" 0.01 1 trainer.max_steps=4000 "${ovr[@]}"
  done
  exit 0
fi

# unlearning: the sensitive grid's pinned unified settings (lambda_f 0.1, no sep /
# neighbourhood terms, 4 epochs, 5 sequential per-user chunks)
SENS_BASE=(unlearning.deletion_spec=item_pairs unlearning.include_context_rows=true
           unlearning.request_user_order=sorted unlearning.n_unlearning_chunks=5
           unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0
           unlearning.sep_negatives=forget_target_only unlearning.adaptive_codes=false
           unlearning.n_epochs=4)
for spec in ${SPECS:-plainclean:full umclean:full umclean:uonly umclean:ureset umclean:uresetft}; do
  model="${spec%%:*}"; arm="${spec#*:}"
  run_dir=$(ls -d logs/train/runs/*/*_USERMEM_${model} 2>/dev/null | tail -1)
  ckpt="$run_dir/checkpoints/last.ckpt"
  [ -f "$ckpt" ] || { echo "missing $ckpt"; exit 1; }
  # at unlearning time the user neighbours come from the RETAINED histories
  mo=""; [ "$model" = umclean ] && mo="$(UM $NB/beauty_retrain_hairloss_1e-3_seed2.pt)"
  case "$arm" in
    full)     x=() ;;
    uonly)    x=(unlearning.user_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.0
                 unlearning.user_memory_rows=forget) ;;
    ureset)   x=(unlearning.user_memory_rows=forget unlearning.user_memory_reset_alpha=0.0
                 unlearning.zero_shot=true) ;;
    uresetft) x=(unlearning.user_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.0
                 unlearning.user_memory_rows=forget unlearning.user_memory_reset_alpha=0.5) ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  tag="sens_beauty_hairloss1e-3_USERMEM_${model}_${arm}"
  echo "[$(date -Is)] $tag"
  run env UNLEARN_RUN_TAG="$tag" UNLEARN_SEED=2 UNLEARN_EVAL_DATA_DIR=$D/beauty \
    EVAL_EXTRA_OVERRIDES="$mo" \
    sbatch --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
    run_tiger_unlearn_sequential.sh "$ckpt" "$SENS" unified "$SID" false 1 0.0 \
    "spam_forget_manifest=$MAN" "${SENS_BASE[@]}" "${x[@]}" $mo
done
