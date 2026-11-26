#!/usr/bin/env bash
# Item memory vs plain TIGER on Beauty, Sports and Toys with the SAME tokenizer per dataset
# (embeddings/<ds>_rqvae), the paper's data seeds and recipes. Generalizes
# run_item_memory_beauty.sh (which stays the launcher of the existing Beauty runs).
#
# Architectures (ARCH):
#   plain  TIGER                  im    + item memory (input side)
#   imc    im + consistency       ims   im + output score (reranks the beam)
#   imsc   ims + consistency        imfree  capacity control: the same per-item vectors as im
#                                           but no pathway dropout and no neighbour substitution
# Stages:
#   train   STAGE=train DATASET=sports ARCH=im DSEED=3 [STRATUM=mid] [CLEAN=1]
#           poisoned model on <ds>_spam_tgt<st>_seed<N>_pct1_n1, or with CLEAN=1 the clean-trained
#           model (the retrained reference; one per dataset and architecture).
#           Label: _tgt<st>_ITEMMEM_<arch>[_d<N>] | _ITEMMEM_<arch>clean[_es][_d<N>]
#           Length: MAXSTEPS=<n> caps training (trainer.max_steps); default 4000 for poisoned
#           models, and for CLEAN=1 no cap: early stopping on val (patience 10), label _es
#   unlearn STAGE=unlearn DATASET=... ARCH=... DSEED=... [RECIPES="fr nau"]
#           the 0-step eval (pre) plus the paper's Forget+Repair (fr) and NAU (nau), checkpoints
#           kept (needed by STAGE=ablate). Tags <ds>_tgt<st>_ITEMMEM_<arch>[_d<N>]_{pre,fr,nau}_keep
#   ref     STAGE=ref DATASET=... ARCH=... DSEED=...   clean model, 0-step eval on the cell's data
#   ablate  STAGE=ablate DATASET=... ARCH=... DSEED=... [WHICH="pre fr nau clean"]
#           eval-only: the model as trained, memory off (item_memory.eval_pathway=nulltoken), score off
#           (score_weight=0), both off. Tags ..._<which>_abl_<variant>
#   RECIPE=tiger (with CLEAN=1): the TIGER paper's training recipe instead of GRID's
#           (Rajput et al. 2023): batch 256 (2 GPUs x 32 x accumulate 4), Adam lr 0.01 constant
#           for 10k steps then inverse square-root decay, 200k steps (100k on Toys), no early
#           stopping (best-val and last checkpoints both kept), validation every 2000 steps.
#           Also TIGER's model settings (dropout 0.1, 2000 hashed user tokens) and TIGER's tokenizer
#           (embeddings/<ds>_tiger_rqvae). Label _tigr; three chained 2-day segments.
# DRY=1 prints instead of submitting.
set -euo pipefail
cd "$(dirname "$(readlink -f "$0")")"   # always this tree (an interactive job sets SLURM_SUBMIT_DIR elsewhere)
STAGE="${STAGE:?train|unlearn|ref|ablate}"; DS="${DATASET:?beauty|sports|toys}"; ARCH="${ARCH:?plain|im|imc|ims|imsc|imfree}"
DSEED="${DSEED:-2}"; ST="${STRATUM:-mid}"; D=src/data/amazon_data
SID=embeddings/${DS}_rqvae/merged_predictions_tensor.pt
# RECIPE=tiger also uses TIGER's tokenizer: Sentence-T5 embeddings -> RQ-VAE, 3 levels + collision
# code (scripts/sentence_t5_embeddings.py, scripts/tiger_rqvae.py; logs/tigertok_jobs.txt)
# TOKENIZER=tiger (with RECIPE=tiger) switches to it; default: our RQ-VAE SIDs for both recipes
TOKENIZER="${TOKENIZER:-grid}"
[ "$TOKENIZER" = tiger ] && SID=embeddings/${DS}_tiger_rqvae/merged_predictions_tensor.pt
# TOKENIZER=rkmeans: GRID's default RK-Means SIDs (flan-T5-XL, 3 x 256 + dedup digit), the
# tokenizer of NAU's TIGER models (embeddings/<ds>/, identical to the main tree's). Label suffix rk.
[ "$TOKENIZER" = rkmeans ] && SID=embeddings/${DS}/merged_predictions_tensor.pt
# TOKENIZER=letter: LETTER (arXiv 2405.07314, MODEL=letter): its learned identifier space
# (embeddings/<ds>_letter, paper mode: 4 x 256 codes, no dedup digit, 0.03-0.3% items collide)
# on TIGER's backbone (temperature 1 = the released recipe). Label suffix lt.
MODEL_ENV=(); EVAL_EXP=tiger_train_flat
[ "$TOKENIZER" = letter ] && { SID=embeddings/${DS}_letter/merged_predictions_tensor.pt; MODEL_ENV=(MODEL=letter); EVAL_EXP=letter_train_flat; }
# item-memory neighbours / init: the same flan-T5 content embeddings for both recipes
EMB=embeddings/${DS}_merged_predictions_tensor_latest.pt
for f in "$SID" "$EMB"; do [ -f "$f" ] || { echo "missing $f"; exit 1; }; done
run() { if [ "${DRY:-0}" = 1 ]; then printf '  %q' "$@"; echo; else "$@"; fi; }
spam_dir() { case "$ST" in unpopular) echo "$D/${DS}_spam_seed${DSEED}_pct1_n1" ;; *) echo "$D/${DS}_spam_tgt${ST}_seed${DSEED}_pct1_n1" ;; esac; }
DATA=$(spam_dir); dtok=""; [ "$DSEED" != 2 ] && dtok="_d$DSEED"
if [ "${CLEAN:-0}" = 1 ]; then MAXSTEPS="${MAXSTEPS-}"; else MAXSTEPS="${MAXSTEPS-4000}"; fi
estok=""; [ -z "$MAXSTEPS" ] && estok="_es"   # clean, early-stopped (uncapped) models
RECIPE="${RECIPE:-grid}"
# the TIGER paper's model settings that differ from GRID's: dropout 0.1 (GRID 0.15) and 2000
# hashed user-id tokens prepended to the encoder input (GRID: none). Needed at evaluation too
# (else the eval model has no user_embedding and the checkpoint loads without it).
TIGER_ARCH=(); [ "$RECIPE" = tiger ] && TIGER_ARCH=(model.num_user_bins=2000
  model.huggingface_model.config.dropout_rate=0.1 model.decoder.config.dropout_rate=0.1)
if [ "$RECIPE" = tiger ]; then
  [ "${CLEAN:-0}" = 1 ] || { echo "RECIPE=tiger is for CLEAN=1 models"; exit 1; }
  estok="_tigr"; [ "$TOKENIZER" = tiger ] && estok="_tigrtok"; TSTEPS=200000; [ "$DS" = toys ] && TSTEPS=100000
  TIGER_OVR=(trainer.max_steps=$TSTEPS trainer.max_epochs=-1 trainer.accumulate_grad_batches=4
             trainer.val_check_interval=8000 callbacks.early_stopping=null optim.optimizer.lr=0.01
             "optim.scheduler={_target_:src.components.scheduler.ConstantThenInverseSqrt,_partial_:true,constant_steps:10000}"
             "${TIGER_ARCH[@]}")
  # TIGER_OPT=adafactor: TIGER's lr/schedule is the T5X default, which pairs with Adafactor
  # (Adam at lr 0.01 learns far too slowly, pilot 2026-10-29). T5X: no weight decay. Label _tigraf.
  if [ "${TIGER_OPT:-adam}" = adafactor ]; then
    estok="${estok/_tigr/_tigraf}"
    TIGER_OVR+=(optim.optimizer._target_=transformers.optimization.Adafactor optim.optimizer.weight_decay=0.0
                +optim.optimizer.scale_parameter=true +optim.optimizer.relative_step=false
                +optim.optimizer.warmup_init=false)
  fi
fi
# RECIPE=gridpaper: the GRID paper's TIGER setting (arXiv 2507.22224, Sec. 4): Adam lr 5e-4,
# weight decay 1e-6, batch 256 (2 GPUs x 32 x accumulate 4), early stopping after 10 validations
# of 100 optimizer steps without val NDCG@10 improvement (best checkpoint by val NDCG@10).
# Label _gp. (RECIPE=grid = this repo's config = NAU's: lr 1e-3, wd 1e-4, batch 1024, val R@5.)
GP_OVR=()
if [ "$RECIPE" = gridpaper ]; then
  estok="_gp"
  GP_OVR=(optim.optimizer.lr=0.0005 optim.optimizer.weight_decay=0.000001
          trainer.accumulate_grad_batches=4 trainer.val_check_interval=400
          callbacks.model_checkpoint.monitor=val/ndcg@10)
  # GP_PATIENCE=40: the paper's 10 x 100 steps at batch 256 stopped plain Beauty while val was still
  # rising (test .0244 vs the paper's .0347, 2026-10-29); 40 x 100 steps = the samples of our _es
  # patience at batch 1024. Label _gp40.
  if [ "${GP_PATIENCE:-10}" != 10 ]; then
    estok="_gp${GP_PATIENCE}"; GP_OVR+=(callbacks.early_stopping.patience=$GP_PATIENCE)
  fi
fi
[ "$TOKENIZER" = rkmeans ] && estok="${estok}rk"
[ "$TOKENIZER" = letter ] && estok="${estok}lt"
# 2 GPUs on pgpu, any type with >= 80 GB (A100-80G, H100, H200; s-sc-pgpu01-02 are A100-40G)
GPUS=(--partition=pgpu --gres=gpu:2 --exclude=s-sc-pgpu[01-02])

IMBASE="embedding_path:$EMB,p_drop:0.1,p_nbr:0.2,max_neighbors:4,tau:0.1"
IMFREE="embedding_path:$EMB,p_drop:0.0,p_nbr:0.0,max_neighbors:4,tau:0.1"
SCORE="score_loss_weight:1.0,score_weight:1.0,score_negatives:255,score_tau:1.0"
# 2026-11-25 controls and memory-size study (clean data):
#   plainwide   parameter-matched TIGER: FFN width 1024 -> 1792 = +1.573M parameters, the
#               item memory's +1.57M (values 12101 x 128 + adapter), all 8 layers
#   imc_r<R>    imc with R-dimensional rows (adapter R -> 128), e.g. imc_r32
#   imc_h<P>    imc with rows only for the top P% items by training count (others: mix of
#               their nearest head rows), e.g. imc_h25 (counts: scripts/item_train_counts.py)
WIDE="model.huggingface_model.config.d_ff=1792"
size_opts() {  # imc_r32 / imc_h25 -> extra item_memory keys
  case "$1" in imc_r*) echo ",row_dim:${1#imc_r}" ;;
    imc_h*) echo ",head_frac:0.${1#imc_h},counts_path:embeddings/item_counts/${DS}.pt" ;; esac; }
train_mo() { case "$1" in plain) echo "" ;; plainwide) echo "$WIDE" ;; im) echo "model.item_memory={$IMBASE}" ;;
  imc_r*|imc_h*) echo "model.item_memory={$IMBASE,consistency_weight:0.1$(size_opts "$1")}" ;;
  imc) echo "model.item_memory={$IMBASE,consistency_weight:0.1}" ;; ims) echo "model.item_memory={$IMBASE,$SCORE}" ;;
  imsc) echo "model.item_memory={$IMBASE,$SCORE,consistency_weight:0.1}" ;;
  imfree) echo "model.item_memory={$IMFREE}" ;; *) echo "bad arch $1" >&2; exit 1 ;; esac; }
# at unlearn / eval time: consistency is training-only (as in run_item_memory_beauty.sh)
eval_mo() { local extra="${2:-}"; case "$1" in plain) echo "" ;; plainwide) echo "$WIDE" ;;
  imc_r*|imc_h*) echo "model.item_memory={$IMBASE$(size_opts "$1")${extra:+,$extra}}" ;;
  im|imc) echo "model.item_memory={$IMBASE${extra:+,$extra}}" ;;
  imfree) echo "model.item_memory={$IMFREE${extra:+,$extra}}" ;;
  ims|imsc) echo "model.item_memory={$IMBASE,$SCORE${extra:+,$extra}}" ;; esac; }

model_dir() {  # newest training run of this model
  local lab; if [ "${1:-}" = clean ]; then lab="ITEMMEM_${ARCH}clean${estok}${dtok}"; else lab="tgt${ST}_ITEMMEM_${ARCH}${dtok}"; fi
  # newest run dir of this label that saved a last.ckpt (a resume segment started after training
  # already finished may leave a dir without one)
  local d; for d in $(ls -d logs/train/runs/*/*_${DS}_*${lab} 2>/dev/null | grep -E "_${lab}$" | sort -r); do
    [ -f "$d/checkpoints/last.ckpt" ] && { echo "$d"; return; }; done
}

if [ "$STAGE" = train ]; then
  envs=(DETERMINISTIC=1 N_TARGET_ITEMS=1 "${MODEL_ENV[@]}"); variant=poison
  [ "$DSEED" != 2 ] && envs+=(TRAIN_SEED=$DSEED)
  # clean models: DSEED is the training seed (label _d<N>, so seeds do not hit the duplicate guard)
  if [ "${CLEAN:-0}" = 1 ]; then envs+=(TIGER_DATA_DIR=$D/$DS RUN_LABEL_SUFFIX="_ITEMMEM_${ARCH}clean${estok}${dtok}"); variant=clean
  else [ -d "$DATA" ] || { echo "missing $DATA"; exit 1; }; envs+=(TIGER_DATA_DIR=$DATA RUN_LABEL_SUFFIX="_tgt${ST}_ITEMMEM_${ARCH}${dtok}"); fi
  mo=$(train_mo "$ARCH")
  # letter_train_flat has no item_memory key (tiger_train_flat declares it null): append it
  [ "$TOKENIZER" = letter ] && mo="${mo/model.item_memory=/+model.item_memory=}"
  ovr=(); [ -n "$MAXSTEPS" ] && ovr+=(trainer.max_steps=$MAXSTEPS); [ -n "$mo" ] && ovr+=("$mo")
  ovr+=("${GP_OVR[@]}")
  if [ "$RECIPE" = tiger ]; then
    # segment 1, then segment 2 resuming from it (FORCE: the duplicate guard would skip it)
    # EXTEND_FROM=<job>: only append one more resume segment after that job
    ovr+=("${TIGER_OVR[@]}"); jn="itm_${DS}_${ARCH}${estok}${dtok}"
    seg() {  # $1 previous job ('' = first segment)
      if [ -z "$1" ]; then run env "${envs[@]}" sbatch --parsable "${GPUS[@]}" --time=2-00:00:00 --job-name="$jn" \
          run_tiger_train.sh "$DS" "$variant" "$SID" 0.01 1 "${ovr[@]}"
      else run env "${envs[@]}" FORCE=1 RESUME_FROM=job:$1 sbatch --parsable "${GPUS[@]}" --time=2-00:00:00 \
          --dependency=afterany:$1 --job-name="${jn}_r" run_tiger_train.sh "$DS" "$variant" "$SID" 0.01 1 "${ovr[@]}"; fi; }
    if [ -n "${EXTEND_FROM:-}" ]; then seg "$EXTEND_FROM"; exit 0; fi
    p=""; for i in 1 2 3; do p=$(seg "$p"); echo "$p"; [ "${DRY:-0}" = 1 ] && p=SEG$i; done   # 3 x 2 days
    exit 0
  fi
  # EXTEND_FROM=<job>: resume a timed-out run (one more 36 h segment after that job)
  [ -n "${EXTEND_FROM:-}" ] && envs+=(FORCE=1 RESUME_FROM=job:$EXTEND_FROM)
  run env "${envs[@]}" sbatch --parsable "${GPUS[@]}" --time=36:00:00 \
    --job-name="itm_${DS}_${ARCH}${estok}${dtok}" run_tiger_train.sh "$DS" "$variant" "$SID" 0.01 1 "${ovr[@]}"
  exit 0
fi

BASE0=(unlearning.n_unlearning_chunks=10 unlearning.sep_negatives=forget_target_only
       unlearning.adaptive_codes=false unlearning.n_epochs=4)
recipe() { case "$1" in
  fr)  LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.sep_loss_type=cosine) ;;
  nau) LS=(unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01
           unlearning.coherence_loss_type=mass unlearning.coherence_neighbor_method=embedding
           "unlearning.embedding_path=$EMB" unlearning.neighborhood_count=8
           unlearning.coherence_checkpoint=true unlearning.sep_checkpoint=true) ;;
  *) echo "bad recipe $1"; exit 1 ;; esac; }
base_tag="${DS}_tgt${ST}_ITEMMEM_${ARCH}${dtok}"
submit_eval() {  # $1 tag, $2 ckpt, $3 eval model overrides  (0-step eval on the cell's data)
  run env UNLEARN_RUN_TAG="$1" EVAL_EXTRA_OVERRIDES="$3" N_TARGET_ITEMS=1 UNLEARN_EVAL_DATA_DIR=$D/$DS UNLEARN_SEED=$DSEED \
    sbatch --parsable --partition=pgpu --gres=gpu:2 --time=02:00:00 --job-name="ie_$1" \
    run_tiger_unlearn_sequential.sh "$2" "$DATA" finetune "$SID" false 1 0.0 \
    unlearning.finetune_steps=0 unlearning.n_epochs=null unlearning.n_unlearning_chunks=1 $3
}

if [ "$STAGE" = unlearn ]; then
  md=$(model_dir); ck="$md/checkpoints/last.ckpt"; [ -f "$ck" ] || { echo "missing model for $base_tag ($md)"; exit 1; }
  mo=$(eval_mo "$ARCH")
  submit_eval "${base_tag}_pre_keep" "$ck" "$mo"
  for rc in ${RECIPES:-fr nau}; do
    recipe "$rc"; tag="${base_tag}_${rc}_keep"
    run env UNLEARN_RUN_TAG="$tag" EVAL_EXTRA_OVERRIDES="$mo" N_TARGET_ITEMS=1 UNLEARN_EVAL_DATA_DIR=$D/$DS \
      UNLEARN_SEED=$DSEED UNLEARN_KEEP_CKPT=1 \
      sbatch --parsable --partition=pgpu --gres=gpu:2 --time=06:00:00 --job-name="ul_$tag" \
      run_tiger_unlearn_sequential.sh "$ck" "$DATA" unified "$SID" false 1 0.0 "${BASE0[@]}" "${LS[@]}" $mo
  done
  exit 0
fi

if [ "$STAGE" = ref ]; then
  md=$(model_dir clean); ck="$md/checkpoints/last.ckpt"; [ -f "$ck" ] || { echo "missing clean model ($md)"; exit 1; }
  submit_eval "${base_tag}_ref_clean" "$ck" "$(eval_mo "$ARCH")"
  exit 0
fi

if [ "$STAGE" = ablate ]; then
  [ "$ARCH" = plain ] && { echo "plain has no pathway to switch off"; exit 0; }
  variants=(full "null:eval_pathway:nulltoken")
  case "$ARCH" in ims|imsc) variants+=("noscore:score_weight:0.0" "both:eval_pathway:nulltoken,score_weight:0.0") ;; esac
  for which in ${WHICH:-pre fr nau clean}; do
    case "$which" in
      pre)   ck="$(model_dir)/checkpoints/last.ckpt" ;;
      clean) ck="$(model_dir clean)/checkpoints/last.ckpt" ;;
      *)     ck=$(ls -d logs/unlearn/runs/*_bs1_${base_tag}_${which}_keep 2>/dev/null | tail -1)/checkpoints/unlearned.ckpt ;;
    esac
    [ -e "$ck" ] || { echo "skip $which: no checkpoint ($ck)"; continue; }
    for v in "${variants[@]}"; do
      name="${v%%:*}"; extra=""; [ "$v" != full ] && extra="${v#*:}"
      [ "$name" = full ] && extra=""
      submit_eval "${base_tag}_${which}_abl_${name}" "$(readlink -f "$ck")" "$(eval_mo "$ARCH" "$extra")"
    done
  done
  exit 0
fi
# STAGE=cleaneval: the clean-trained model (training seed DSEED) on the clean TEST split, NDCG@10:
# as trained (full) and, for memory models, with the memory off at inference (null). One GPU.
# Checkpoint: the best-by-val one (checkpoint_epoch=*), the early-stopping selection; CKPT=last
# for last.ckpt (what the capped 4000-step runs were evaluated on). Output dir gets _last then.
# Output: logs/diagnose/itemmem_clean/<ds>_<arch>_s<seed>_<variant>/
if [ "$STAGE" = cleaneval ]; then
  md=$(model_dir clean)
  if [ "${CKPT:-best}" = last ]; then ck="$md/checkpoints/last.ckpt"; ctok="_last"
  elif lab="ITEMMEM_${ARCH}clean${estok}${dtok}"; segs=($(ls -d logs/train/runs/*/*_${DS}_*${lab} | grep -E "_${lab}$"))
       [ "$RECIPE" = tiger ] || [ "${#segs[@]}" -gt 1 ]; then
    # chained segments (TIGER chains, resumed timeouts): each resume segment has its own run dir
    # and Lightning resets the best score when the dirpath changes, so take the best over all
    # segments' last.ckpt
    ck=$(python3 -m scripts.best_ckpt_across_segments "${segs[@]}" 2>/dev/null)
    ctok="_xseg"
  else ck=$(ls "$md"/checkpoints/checkpoint_epoch=*.ckpt 2>/dev/null | tail -1); ctok=""; fi
  [ -f "$ck" ] || { echo "missing clean model ($md)"; exit 1; }
  # 'full' on the best checkpoint repeats the end-of-training test of train.py (same checkpoint,
  # verified .03215 vs .03216), so only the final checkpoint (CKPT=last) and the cross-segment
  # best of chained TIGER runs (_xseg; train.py only tests its own segment's best) get it
  vs=(); [ -n "$ctok" ] && vs+=(full); [[ "$ARCH" != plain* ]] && vs+=(null)
  # VARIANTS overrides the list, e.g. VARIANTS="int8 fp16": post-hoc row quantization (eval only)
  # BUCKET=1|2|3: the test set restricted to the rarest / mid / most popular third of targets by
  # training count (scripts/build_popularity_buckets.py, data dir <ds>_popb<k>); full (+ null)
  edir=$D/$DS; btok=""
  if [ -n "${BUCKET:-}" ]; then
    edir=$D/${DS}_popb$BUCKET; btok="_popb$BUCKET"; vs=(full); [[ "$ARCH" != plain* ]] && vs+=(null)
  fi
  [ -n "${VARIANTS:-}" ] && read -r -a vs <<< "$VARIANTS"
  for v in "${vs[@]}"; do
    extra=""; [ "$v" = null ] && extra="eval_pathway:nulltoken"
    [ "$v" = int8 ] || [ "$v" = fp16 ] && extra="eval_quant:$v"
    mo=$(eval_mo "$ARCH" "$extra"); [ "$TOKENIZER" = letter ] && mo="${mo/model.item_memory=/+model.item_memory=}"; out="logs/diagnose/itemmem_clean/${DS}_${ARCH}${estok}_s${DSEED}_${v}${ctok}${btok}"
    # single-quoted for the --wrap shell: unquoted, its {a,b,...} is brace-expanded into separate
    # words and the model gets a broken item_memory (every memory-off eval failed before 10-28)
    mq=""; [ -n "$mo" ] && mq="'$mo'"
    [ -s "$out/csv/version_0/metrics.csv" ] && { echo "skip $out"; continue; }
    run sbatch --parsable --partition=gpu --gpus-per-node=1 -c 8 --mem=64G --time=02:00:00 --output=logs/%j.out \
      --job-name="ce_${DS}_${ARCH}_s${DSEED}_${v}" --wrap "cd $PWD && python -u -m scripts.eval_ckpt_on_test \
      experiment=$EVAL_EXP data_dir=$edir \"semantic_id_path='$SID'\" \"ckpt_path='$ck'\" num_hierarchies=4 \
      seed=$DSEED train=False test=True trainer.devices=1 trainer.strategy=auto trainer.num_nodes=1 \
      trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null $mq ${TIGER_ARCH[*]} hydra.run.dir=$out"
  done
  exit 0
fi
echo "unknown STAGE $STAGE"; exit 1
