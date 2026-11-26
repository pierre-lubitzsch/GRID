#!/usr/bin/env bash
# Item-keyed exemplar memory (MUNKEY-style, see src/models/components/network_blocks/
# item_memory.py) on the beauty spam setup (tgtmid, 1%), TIGER.
#
# The item stays in the catalogue: its memory entry is not deleted but UPDATED,
# with the unlearning step weighted towards the forget items' entries and away
# from the shared backbone (unlearning.item_memory_lr_scale / backbone_lr_scale /
# item_memory_rows).
#
# STAGE=train (4000 steps, DETERMINISTIC=1, 2x A100 pgpu; same setup as the PKM sweeps):
#   plain   no PKM, no item memory            (reference with the same SIDs)
#   im      item memory INSTEAD of a PKM       (the user's "more realistic" option)
#   impkm   item memory + balanced PKM in E1   (switch coef 0.1 + batchnorm)
#   ims     im + output-side item score (sampled softmax; reranks the beams)
#   imsc    ims + consistency distillation (KL own -> neighbour-substituted)
#   impkmsc imsc + balanced PKM
#   imscclean / imscret  imsc on clean beauty / the hairloss 1e-3 retrain data
#   sportsplain / sportsimsc  second dataset (sports spam tgtmid 1%)
#   immk / imscmk  im / imsc with MUNKEY's own recipe (item_memory.py recipe=munkey)
#   SEED=<n> trains another seed (label suffix _s<n>)
#   imc / plainscore: controls (consistency without score; score head without input memory)
# STAGE=unlearn (recorded unified recipe of job 10268039 as the base):
#   on plain: full
#   on im / impkm: full, emph (item lr x10, backbone x0.1, rows afihf top 1000),
#                  imonly (backbone frozen, rows = forget items); idea-4 variants:
#                  emphsoft (afihf_soft rows), emphreset (+ warm start alpha 0.5),
#                  emphlocal (+ ADDITIONAL local repair term, lambda_local 1.0;
#                  INERT: subset rows), emphlocalfull (local rows from the full retain split)
#   (the PKM in impkm* moves with the backbone scale)
#   other seeds: model name with its _s<n> suffix (imsc_s3:emph)
#   sports: sportsplain:full, sportsimsc:{full,emph}  (sports data / SIDs)
#   sensitive 1e-3 (imscclean, imscret): the plain baseline's pinned recipe
#     (item_pairs, 5 per-user chunks, lambda_f 0.1, lambda_s/n 0, 4 epochs);
#     arm pre = 0-step eval (imscret:pre = the retrain reference)
#   DEP=<jobid>: submit with --dependency=afterok (the training job), so the
#     checkpoint need not exist yet
#
# Usage:  STAGE=train ./run_item_memory_beauty.sh      DRY=1 to preview
#         STAGE=unlearn ./run_item_memory_beauty.sh
set -euo pipefail

STAGE="${STAGE:?STAGE=train|unlearn|ref}"
# DSEED: the paper's DATA seed (2, 3, 5, 7, 11): its own spam dataset (different target
# item and fake users, same item catalogue, so the same SIDs / text embeddings).
# Models trained on DSEED != 2 carry _d<N> and are also trained with TRAIN_SEED=N, as
# the main project's data-seed runs are; the unlearn stage reads _d<N> from the name.
DSEED="${DSEED:-2}"
# STRATUM: target popularity stratum of the paper grid: unpopular (the untagged
# beauty_spam_seed<N> dirs, target_strategy=unpopular) | mid | popular. Model names carry
# _up / _pop (mid: none, so the existing names stay valid), e.g. imsc_pop_d7.
STRATUM="${STRATUM:-mid}"
# DS: dataset of the poisoned grid (beauty | toys). Toys models carry ITEMMEM_toys_ in the
# label / unlearn tag (DSTOK), use the toys SIDs / item embeddings, and evaluate on toys.
DS="${DS:-beauty}"
case "$DS" in beauty) DSTOK="" ;; toys) DSTOK="_toys" ;; *) echo "unknown DS $DS"; exit 1 ;; esac
spam_dir() {  # $1 data seed, $2 stratum
  case "${2:-mid}" in
    unpopular) echo "src/data/amazon_data/${DS}_spam_seed${1}_pct1_n1" ;;
    mid)       echo "src/data/amazon_data/${DS}_spam_tgtmid_seed${1}_pct1_n1" ;;
    popular)   echo "src/data/amazon_data/${DS}_spam_tgtpopular_seed${1}_pct1_n1" ;;
    *) echo "unknown stratum $2" >&2; exit 1 ;;
  esac
}
st_tok() { case "$1" in unpopular) echo _up ;; popular) echo _pop ;; *) echo "" ;; esac; }
DATA=$(spam_dir "$DSEED" "$STRATUM")
# ATTACK=clone_append: second attack of the grid (beauty, mid target, all 5 data seeds): fake
# users clone real users' sessions and append the target. Model names carry _ca (im_ca_d3).
CA_DIR() { echo "src/data/amazon_data/beauty_spam_clone_append_tgtmid_seed${1}_pct1_n1"; }
if [ "${ATTACK:-bandwagon}" = clone_append ]; then
  [ "$DS" = beauty ] && [ "$STRATUM" = mid ] || { echo "clone_append: beauty, mid only"; exit 1; }
  DATA=$(CA_DIR "$DSEED")
fi
[ -d "$DATA" ] || { echo "missing $DATA"; exit 1; }
# ATTACK=clone_inject: the clone-inject spam (226 fake users that CLONE real users'
# sessions with the target injected, unpopular target; same catalogue). Tag _ci.
CI_DIR=src/data/amazon_data/beauty_spam_clone_inject_seed2_pct1_n1
SID=embeddings/${DS}_rqvae/merged_predictions_tensor.pt
EMB=embeddings/${DS}_merged_predictions_tensor_latest.pt
IMBASE="embedding_path:$EMB,p_drop:${P_DROP:-0.1},p_nbr:${P_NBR:-0.2},max_neighbors:4,tau:0.1"
IM="model.item_memory={$IMBASE}"
SCORE="score_loss_weight:1.0,score_weight:1.0,score_negatives:255,score_tau:1.0"
IMS="model.item_memory={$IMBASE,$SCORE}"
IMSC="model.item_memory={$IMBASE,$SCORE,consistency_weight:0.1}"
# controls for "is the gain just the reranking?": consistency without the score, and the
# score head WITHOUT the input memory (input_term:false; plain TIGER + a reranker)
IMC="model.item_memory={$IMBASE,consistency_weight:0.1}"
PSCORE="model.item_memory={$IMBASE,$SCORE,input_term:false}"
# MUNKEY's own recipe (item_memory.py recipe=munkey): per-batch draws, K'~U{2..4},
# tau 0.07, eval always retrieves over [self + neighbours]
MKBASE="embedding_path:$EMB,p_drop:${P_DROP:-0.1},p_nbr:${P_NBR:-0.2},max_neighbors:4,tau:0.07,recipe:munkey"
IMMK="model.item_memory={$MKBASE}"
IMSMK="model.item_memory={$MKBASE,$SCORE}"
IMSCMK="model.item_memory={$MKBASE,$SCORE,consistency_weight:0.1}"
# category memory keyed by the first SID level (GroupMemory): 32 / 128 slots per group
GM="model.group_memory={slots:32,knn:4,key_dim:64}"
GM128="model.group_memory={slots:128,knn:4,key_dim:64}"
PKM_BAL=("model.pkm_balance={coef:0.1,kind:switch}" model.pkm_params.query_norm=batchnorm
         "model.pkm_explore={p:0.0,util_window:100}")
run() { if [ "${DRY:-0}" = "1" ]; then printf '  %q ' "$@"; echo; else "$@"; fi; }

# STAGE=ref: retrained references for a grid cell (DSEED, STRATUM): the CLEAN-trained plain
# and item-memory (imsc) models, zero-step eval on the cell's spam data (SH against ITS
# target). CPU jobs on `compute` (eval only). Tags ref_spam_<stratum>_d<N>_{plain,imsc}clean.
if [ "$STAGE" = ref ]; then
  # beauty: the existing clean models (plain from the USERMEM launcher, imsc, im = arm imclean);
  # other datasets: arms cleanref_<m> (STAGE=train DS=<ds> ARMS="cleanref_plain ...")
  if [ "$DS" = beauty ]; then
    REF_PLAIN=$(ls -d logs/train/runs/*/*USERMEM_plainclean | tail -1)/checkpoints/last.ckpt
    REF_IMSC=$(ls -d logs/train/runs/*/*ITEMMEM_imscclean | tail -1)/checkpoints/last.ckpt
    REF_IM=$(ls -d logs/train/runs/*/*_ITEMMEM_imclean 2>/dev/null | tail -1 || true)/checkpoints/last.ckpt
  else
    for m in plain im imsc; do
      r=$(ls -d logs/train/runs/*/*_ITEMMEM${DSTOK}_cleanref_$m 2>/dev/null | tail -1 || true)/checkpoints/last.ckpt
      eval "REF_${m^^}=$r"
    done
  fi
  CPU="trainer.accelerator=cpu trainer.devices=1"
  envr=(); [ "$DS" != beauty ] && envr=(UNLEARN_EVAL_DATA_DIR=src/data/amazon_data/$DS)
  atok=""; [ "${ATTACK:-bandwagon}" = clone_append ] && atok=_ca
  # REF_MODELS: which clean models to evaluate (default plain imsc)
  for m in ${REF_MODELS:-plain imsc}; do
    ck=$REF_PLAIN; mo=""; [ $m = imsc ] && { ck=$REF_IMSC; mo="$IMS"; }
    [ $m = im ] && { ck=$REF_IM; mo="$IM"; }
    [ -f "$ck" ] || { echo "missing reference model $ck"; exit 1; }
    tag="ref_spam${DSTOK}${atok}_${STRATUM}_d${DSEED}_${m}clean"
    run env UNLEARN_RUN_TAG="$tag" EVAL_EXTRA_OVERRIDES="$mo $CPU trainer.strategy=auto" \
      N_TARGET_ITEMS=1 CUDA_VISIBLE_DEVICES= OMP_NUM_THREADS=16 "${envr[@]}" \
      sbatch --partition=compute --gres=none -c 16 --mem=64G --time=06:00:00 --job-name="$tag" \
      --output=logs/%j.out \
      run_tiger_unlearn_sequential.sh "$ck" "$DATA" finetune "$SID" false 1 0.0 \
      unlearning.finetune_steps=0 unlearning.n_epochs=null unlearning.n_unlearning_chunks=1 \
      $CPU $mo
  done
  exit 0
fi

if [ "$STAGE" = train ]; then
  # SEED: training seed (default 2). Other seeds get _s<seed> in the label so the
  # unlearn stage's run-dir glob never mixes seeds.
  SEED="${SEED:-2}"; stok=""; seedenv=()
  [ "$SEED" != 2 ] && { stok="_s$SEED"; seedenv=(TRAIN_SEED=$SEED); }
  if [ "$DSEED" != 2 ]; then
    [ "$SEED" != 2 ] && { echo "set DSEED or SEED, not both"; exit 1; }
    stok="_d$DSEED"; seedenv=(TRAIN_SEED=$DSEED)
  fi
  [ "${ATTACK:-bandwagon}" = clone_append ] && stok="_ca${stok}"
  if [ "${ATTACK:-bandwagon}" = clone_inject ]; then
    [ "$DSEED" != 2 ] && { echo "ATTACK=clone_inject exists for data seed 2 only"; exit 1; }
    DATA=$CI_DIR; stok="${stok}_ci"
  fi
  for arm in ${ARMS:-plain im impkm}; do
    envs=(DETERMINISTIC=1 "${seedenv[@]}" TIGER_DATA_DIR="$DATA" N_TARGET_ITEMS=1
          RUN_LABEL_SUFFIX="_tgt${STRATUM}_ITEMMEM${DSTOK}_${arm}$(st_tok "$STRATUM")${stok}")
    ds=$DS; sid="$SID"; variant=poison
    ovr=()
    case "$arm" in
      plain) ;;
      im)    ovr=("$IM") ;;
      impkm) envs+=(PKM_MODE=replace PKM_ENCODER=1 PKM_DECODER=none); ovr=("$IM" "${PKM_BAL[@]}") ;;
      # + output-side item score (idea 1): trained with a sampled softmax, used to rerank
      ims)   ovr=("$IMS") ;;
      # + score + consistency distillation (idea 3)
      imsc)  ovr=("$IMSC") ;;
      # controls: consistency only (no score), and plain + score head (no input memory)
      imc)   ovr=("$IMC") ;;
      plainscore) ovr=("$PSCORE") ;;
      impkmsc) envs+=(PKM_MODE=replace PKM_ENCODER=1 PKM_DECODER=none); ovr=("$IMSC" "${PKM_BAL[@]}") ;;
      # MUNKEY's own training / inference recipe on the item memory
      immk)   ovr=("$IMMK") ;;
      gm)     ovr=("$GM") ;;
      gm128)  ovr=("$GM128") ;;
      imscgm) ovr=("$IMSC" "$GM") ;;
      imscmk) ovr=("$IMSCMK") ;;
      # clean-trained im: the retrained reference for every im grid cell (STAGE=ref REF_MODELS=im)
      imclean) envs=("${envs[@]/TIGER_DATA_DIR=*/TIGER_DATA_DIR=src/data/amazon_data/beauty}"); variant=clean
               ovr=("$IM") ;;
      # sensitive 1e-3 base (clean beauty; also the clean reference) and its retrain
      imscclean) envs=("${envs[@]/TIGER_DATA_DIR=*/TIGER_DATA_DIR=src/data/amazon_data/beauty}"); variant=clean
                 ovr=("$IMSC" "spam_forget_manifest=src/data/amazon_data/beauty_sens_hairloss_1e-3_seed2/forget_manifest.json") ;;
      imscret)   envs=("${envs[@]/TIGER_DATA_DIR=*/TIGER_DATA_DIR=src/data/amazon_data/beauty_retrain_hairloss_1e-3_seed2}"); variant=clean
                 ovr=("$IMSC" "spam_forget_manifest=src/data/amazon_data/beauty_sens_hairloss_1e-3_seed2/forget_manifest.json") ;;
      # second dataset: sports spam (tgtmid, 1%), plain vs item memory + score + consistency
      sportsplain|sportsimsc)
                 ds=sports; sid=embeddings/sports_rqvae/merged_predictions_tensor.pt
                 envs=("${envs[@]/TIGER_DATA_DIR=*/TIGER_DATA_DIR=src/data/amazon_data/sports_spam_tgtmid_seed2_pct1_n1}")
                 [ "$arm" = sportsimsc ] && ovr=("model.item_memory={embedding_path:embeddings/sports_merged_predictions_tensor_latest.pt,p_drop:0.1,p_nbr:0.2,max_neighbors:4,tau:0.1,$SCORE,consistency_weight:0.1}") ;;
      # clean retrained references for STAGE=ref on non-beauty datasets
      cleanref_plain|cleanref_im|cleanref_imsc)
                 envs=("${envs[@]/TIGER_DATA_DIR=*/TIGER_DATA_DIR=src/data/amazon_data/$DS}"); variant=clean
                 case $arm in cleanref_im) ovr=("$IM") ;; cleanref_imsc) ovr=("$IMSC") ;; esac ;;
      *) echo "unknown arm $arm"; exit 1 ;;
    esac
    echo "[$(date -Is)] train ITEMMEM_$arm"
    run env "${envs[@]}" sbatch --partition=pgpu --gres=${TRAIN_GRES:-gpu:nvidia_a100-sxm4-80gb:2} --time=16:00:00 \
      run_tiger_train.sh "$ds" "$variant" "$sid" 0.01 1 trainer.max_steps=4000 "${ovr[@]}"
  done
  exit 0
fi

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
# VALSEL=1: validation-based selection of n_epochs (ported from main, mode=run: the whole
# sequence is re-run for n_epochs = 1, 2, ... and each final model scored on the
# validation split; patience 2, max 10). Tags get _vs; OFF by default.
VS=(); vtok=""
[ "${VALSEL:-0}" = 1 ] && { VS=(unlearning.val_selection.enabled=true); vtok=_vs; }
# CHUNKS: number of sequential deletion requests (recorded recipe 10); other values
# (e.g. the 50-request stress test) are tagged _ch<N>.
CHUNKS="${CHUNKS:-10}"; ctok=""; [ "$CHUNKS" != 10 ] && ctok="_ch$CHUNKS"
BASE0=(unlearning.n_unlearning_chunks=$CHUNKS unlearning.sep_negatives=forget_target_only
       unlearning.adaptive_codes=false unlearning.n_epochs=4)
D=src/data/amazon_data
SENS=$D/beauty_sens_hairloss_1e-3_seed2
SENS_BASE=(unlearning.deletion_spec=item_pairs unlearning.include_context_rows=true
           unlearning.request_user_order=sorted unlearning.n_unlearning_chunks=5
           unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0
           unlearning.sep_negatives=forget_target_only unlearning.adaptive_codes=false
           unlearning.n_epochs=4)
for spec in ${SPECS:-plain:full im:full im:emph im:imonly impkm:full impkm:emph impkm:imonly}; do
  model="${spec%%:*}"; arm="${spec#*:}"
  base=$(echo "$model" | sed -E 's/_d[0-9]+$//; s/_(up|pop)$//; s/_ci$//; s/_ca$//; s/_s[0-9]+$//')
  # _d<N> / _up / _pop: the model's data seed and stratum -> its dataset
  mdseed=2; [[ "$model" =~ _d([0-9]+)$ ]] && mdseed="${BASH_REMATCH[1]}"
  mst=mid; [[ "$model" =~ _up(_d[0-9]+)?$ ]] && mst=unpopular; [[ "$model" =~ _pop(_d[0-9]+)?$ ]] && mst=popular
  run_dir=$(ls -d logs/train/runs/*/*_tgt${mst}_ITEMMEM${DSTOK}_${model} 2>/dev/null | tail -1)
  ckpt="$run_dir/checkpoints/last.ckpt"
  dep=(); [ -n "${DEP:-}" ] && dep=(--dependency=afterok:$DEP)
  [ -n "$run_dir" ] && { [ -f "$ckpt" ] || [ -n "${DEP:-}" ]; } || { echo "missing $ckpt"; exit 1; }
  # model overrides at unlearn/eval time. Score models keep the score (so the
  # forget term acts on it and the eval reranks); consistency is training-only.
  data=$(spam_dir "$mdseed" "$mst"); sid="$SID"; envx=(); pre=(); algo=unified
  # non-beauty data paths: the eval dir must be explicit (see the sports note below)
  [ "$DS" != beauty ] && envx=(UNLEARN_EVAL_DATA_DIR=$D/$DS)
  [[ "$model" == *_ci ]] && data=$CI_DIR
  [[ "$model" =~ _ca(_d[0-9]+)?$ ]] && data=$(CA_DIR "$mdseed")
  case "$base" in
    plain)         mo="" ;;
    im|impkm)      mo="$IM" ;;
    ims|imsc|impkmsc) mo="$IMS" ;;
    # controls: consistency is training-only (imc evaluates like im); plainscore keeps its score
    imc)           mo="$IM" ;;
    plainscore)    mo="model.item_memory={$IMBASE,$SCORE,input_term:false}" ;;
    immk)          mo="$IMMK" ;;
    gm)            mo="$GM" ;;
    gm128)         mo="$GM128" ;;
    imscgm)        mo="$IMS $GM" ;;
    imscmk)        mo="$IMSMK" ;;
    # sports: the eval dir must be set explicitly (run_tiger_unlearn_sequential
    # only resolves it for short dataset names; a data PATH falls back to beauty,
    # which silently scored jobs 11260179-81 on beauty's test set)
    sportsplain)   mo=""; data=$D/sports_spam_tgtmid_seed2_pct1_n1; sid=embeddings/sports_rqvae/merged_predictions_tensor.pt
                   envx=(UNLEARN_EVAL_DATA_DIR=$D/sports) ;;
    sportsimsc)    mo="model.item_memory={embedding_path:embeddings/sports_merged_predictions_tensor_latest.pt,p_drop:${P_DROP:-0.1},p_nbr:${P_NBR:-0.2},max_neighbors:4,tau:0.1,$SCORE}"
                   data=$D/sports_spam_tgtmid_seed2_pct1_n1; sid=embeddings/sports_rqvae/merged_predictions_tensor.pt
                   envx=(UNLEARN_EVAL_DATA_DIR=$D/sports) ;;
    imscclean|imscret)
                   mo="$IMS"; data=$SENS; envx=(UNLEARN_SEED=2 UNLEARN_EVAL_DATA_DIR=$D/beauty)
                   pre=("spam_forget_manifest=$SENS/forget_manifest.json" "${SENS_BASE[@]}") ;;
    *) echo "unknown model $model"; exit 1 ;;
  esac
  [[ "$model" == impkm* ]] && mo="$mo model.pkm_layers={encoder:[1],decoder:null} model.pkm_mode=replace model.pkm_params.query_norm=batchnorm"
  case "$arm" in
    full)   x=() ;;
    emph)   x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
               unlearning.item_memory_rows=afihf unlearning.item_memory_top_t=1000) ;;
    imonly) x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.0
               unlearning.item_memory_rows=forget) ;;
    # idea 4 variants on top of emph
    emphsoft)  x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
                  unlearning.item_memory_rows=afihf_soft) ;;
    emphreset) x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
                  unlearning.item_memory_rows=afihf unlearning.item_memory_top_t=1000
                  unlearning.item_memory_reset_alpha=0.5) ;;
    emphlocal) x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
                  unlearning.item_memory_rows=afihf unlearning.item_memory_top_t=1000
                  unlearning.lambda_local=1.0 unlearning.local_repair_items=targets) ;;
    # emphlocal never fired (0 of the 16 subset rows were affected); this draws
    # up to 256 affected rows from the FULL retain split instead
    emphlocalfull) x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
                  unlearning.item_memory_rows=afihf unlearning.item_memory_top_t=1000
                  unlearning.lambda_local=1.0 unlearning.local_repair_items=targets
                  unlearning.local_repair_source=full unlearning.local_repair_max_rows=256) ;;
    # n_epochs=null: n_epochs overrides finetune_steps, and the recipes set it
    # (the first pre runs 11260175/78 fine-tuned 4 epochs instead of 0 steps)
    pre)       algo=finetune; x=(unlearning.finetune_steps=0 unlearning.n_epochs=null) ;;
    # zero-shot item deletion: the target items' keys deleted, entries -> neighbour
    # mix (input term and output score), NO weight update; emphdel = deletion + emph
    zsdel)     x=(unlearning.item_memory_delete=targets unlearning.zero_shot=true) ;;
    # full-strength backbone + boosted item entries (forget-read rows, x10), with the
    # anti-nuking drift cap (each entry stays within c * ||v_0|| of its pre-unlearning
    # value over the whole sequence): boost10 uncapped, boost10cap05 / boost10cap02
    boost10)      x=(unlearning.item_memory_lr_scale=10.0 unlearning.item_memory_rows=afihf
                     unlearning.item_memory_top_t=1000) ;;
    boost10cap05) x=(unlearning.item_memory_lr_scale=10.0 unlearning.item_memory_rows=afihf
                     unlearning.item_memory_top_t=1000 unlearning.item_memory_max_drift=0.5) ;;
    boost10cap02) x=(unlearning.item_memory_lr_scale=10.0 unlearning.item_memory_rows=afihf
                     unlearning.item_memory_top_t=1000 unlearning.item_memory_max_drift=0.2) ;;
    emphdel)   x=(unlearning.item_memory_lr_scale=10.0 unlearning.backbone_lr_scale=0.1
                  unlearning.item_memory_rows=afihf unlearning.item_memory_top_t=1000
                  unlearning.item_memory_delete=targets) ;;
    *) echo "unknown arm $arm"; exit 1 ;;
  esac
  emb_ds=$EMB; [[ "$base" == sports* ]] && emb_ds=embeddings/sports_merged_predictions_tensor_latest.pt
  # pre evaluates the model as it is: one run, no recipe
  rlist="$RECIPES"; [[ "$arm" == pre || "$arm" == zsdel ]] && rlist=none
  for rc in $rlist; do
    if [ "$rc" = none ]; then LS=(); rtok=""; else recipe_args "$rc" "$emb_ds"; fi
    tag="bw_tgt${mst}_ITEMMEM${DSTOK}_${model}_${arm}${rtok}"; recipe=("${BASE0[@]}" "${LS[@]}")
    [ "$rc" != none ] && { tag="${tag}${vtok}${ctok}"; recipe+=("${VS[@]}"); }
    # sensitive: the pinned per-user recipe replaces the spam one
    [ ${#pre[@]} -gt 0 ] && { tag="sens_beauty_hairloss1e-3_ITEMMEM_${model}_${arm}"; recipe=("${pre[@]}"); }
    echo "[$(date -Is)] $tag${DEP:+ (after $DEP)}"
    run env UNLEARN_RUN_TAG="$tag" OFF_EVALS="${OFF_EVALS:-1}" EVAL_EXTRA_OVERRIDES="$mo" N_TARGET_ITEMS=1 "${envx[@]}" \
      sbatch "${dep[@]}" --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
      run_tiger_unlearn_sequential.sh "$ckpt" "$data" "$algo" "$sid" false 1 0.0 \
      "${recipe[@]}" "${x[@]}" $mo
  done
done
