#!/usr/bin/env bash
# Table-1 methods on RSC15 (the 10% subsample test_rsc15_seed_2, seed 2): every paper
# method at its configuration as selected on BEAUTY (paper/plotting/valsel_selected.csv,
# beauty,rkmeans; RSC15 has no validation selection of its own), on all three target
# strata, plus the clean / poisoned reference evaluations PGR and UR need.
#
# The unlearning.* overrides are copied verbatim from the Beauty mid seed-2 template
# runs of each arm (their .hydra/overrides.yaml), with ONE substitution: NAU's
# embedding_path points at the RSC15 item embeddings instead of Beauty's.
# TRACER is not included: it needs its own tokenizer, which RSC15 does not have.
#
# Poisoned models (bandwagon 1%, one target, 8072 fake users of 807k):
#   unpopular job10730768, mid job10730769, popular job10730770; clean job10730767.
# Measured runtime of a unified run on this data: ~1.3 h.
#
#   bash run_rsc15_paper_methods.sh            # DRY RUN (default)
#   DRYRUN=0 bash run_rsc15_paper_methods.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
DS="test_rsc15_seed_2"
SID="embeddings/test_rsc15_seed_2/merged_predictions_tensor.pt"
EMB="embeddings/test_rsc15_seed_2_item_embeddings.pt"
# SETTING=n1  : one target per stratum (the 2026-10-13 batch; the attack did NOT land)
# SETTING=n10 : ten unpopular targets at 1% and 10% poisoning (the 2026-05 models;
#               poisoned SH@10 0.00025 / 0.00390 vs clean 0, logs/rsc15_refs/*_n10/)
SETTING="${SETTING:-n1}"
if [ "${SETTING}" = "n10" ]; then
  STRATS="${STRATS:-pct1 pct10}"
else
  STRATS="${STRATS:-unpopular mid popular}"
fi
SLURM=(--partition=pgpu --gres=${GRES:-gpu:2} --exclude=s-sc-pgpu01,s-sc-pgpu02 --time=08:00:00)
declare -A CKJOB=([unpopular]=10730768 [mid]=10730769 [popular]=10730770)
declare -A N10CK=([pct1]="logs/train/runs/2026-05-26/16-42-47/checkpoints/checkpoint_epoch=000_step=008900.ckpt"
                  [pct10]="$(ls -1 logs/train/runs/*/*job9056494_*/checkpoints/*.ckpt 2>/dev/null | head -1)")
declare -A N10RATIO=([pct1]=0.01 [pct10]=0.1)
CLEAN_JOB=10730767

U="unlearning.n_unlearning_chunks=10"
UNI="unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4 unlearning.adaptive_codes=false"
NAU_NBR="unlearning.sep_loss_type=generative unlearning.coherence_loss_type=mass unlearning.coherence_neighbor_method=embedding unlearning.coherence_rows=target_only unlearning.coherence_embedding_metric=cosine unlearning.neighborhood_count=8 unlearning.embedding_path=${EMB}"
# arm (paper tag, so run_tags.family_of parses it) | algorithm | overrides
ARMS=(
  "unified_abl_lf0.0_ls0.0_ln0.0|unified|${U} unlearning.lambda_f=0.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0 ${UNI}"
  "unified_abl_lf1.0_ls0.0_ln0.0_lr0.0|unified|${U} unlearning.lambda_f=1.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=0.0 ${UNI}"
  "unified_abl_lf0.1_ls0.0_ln0.0|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0 ${UNI}"
  "scif|scif|${U}"
  "seif_std0.06|seif|${U} unlearning.seif_erase_std=0.06"
  "kookmin_ir0.001|kookmin|${U} unlearning.kookmin_init_rate=0.001"
  "fanchuan_t0.07|fanchuan|${U} unlearning.fanchuan_contrastive_temperature=0.07"
  "nau_s0p01_n0p01|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01 ${UNI} ${NAU_NBR}"
  "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.1 unlearning.lambda_n=0.01 ${UNI} ${NAU_NBR}"
  # L_nbr without L_sep (paper ablation arm nosep_ln0p1, and its lambda_n=0.01 sibling from the
  # validation grid). Template: Beauty bw_tgtmid_nosep_ln0p1_keepck (no sep_loss_type override).
  "nosep_ln0p01|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.01 ${UNI} ${NAU_NBR/unlearning.sep_loss_type=generative /}"
  "nosep_ln0p1|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.1 ${UNI} ${NAU_NBR/unlearning.sep_loss_type=generative /}"
)

[ -f "${SID}" ] && [ -f "${EMB}" ] || { echo "missing ${SID} or ${EMB}" >&2; exit 1; }
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
n=0; skip=0
for st in ${STRATS}; do
  if [ "${SETTING}" = "n10" ]; then
    ckpt="${N10CK[$st]}"; RATIO="${N10RATIO[$st]}"; NT=10; STRAT_ENV=unpopular
    TAGSTEM="${DS}_bw_${st}_n10"
  else
    rd="$(ls -1d logs/train/runs/*/*job${CKJOB[$st]}_* | head -1)"
    ckpt="${rd}/checkpoints/last.ckpt"; RATIO=0.01; NT=1; STRAT_ENV="${st}"
    TAGSTEM="${DS}_bw_tgt${st}"
  fi
  [ -f "${ckpt}" ] || { echo "missing ${ckpt}" >&2; exit 1; }
  for spec in "${ARMS[@]}"; do
    IFS='|' read -r arm algo ov <<< "${spec}"
    tag="${TAGSTEM}_${arm}"
    if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  skip ${tag}: run dir exists"; skip=$((skip+1)); continue; fi
    if grep -qxF "r15_${tag}" <<<"${QUEUED}"; then echo "  skip ${tag}: queued"; skip=$((skip+1)); continue; fi
    # shellcheck disable=SC2206
    ova=(${ov})
    cmd=(sbatch --job-name="r15_${tag}" "${SLURM[@]}" run_tiger_unlearn_sequential.sh
         "${ckpt}" "${DS}" "${algo}" "${SID}" false 1 0.0 "${ova[@]}")
    if [ "${DRYRUN}" = "1" ]; then
      echo "== ${tag}"; printf '   %q' "${cmd[@]}"; printf '\n'
    else
      POISON_METHOD=bandwagon POISONING_RATIO="${RATIO}" N_TARGET_ITEMS="${NT}" POISON_SEED=2 \
      TARGET_STRATEGY="${STRAT_ENV}" UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED=2 "${cmd[@]}"
    fi
    n=$((n+1))
  done
done

# Reference evaluations on the clean test split, against each stratum's target set:
# the poisoned model (SH_poison) and the clean model (SH_clean, NDCG_clean).
REFDIR="logs/rsc15_refs/${DS}"
if [ "${SETTING}" = "n10" ]; then
  echo "  references for n10: logs/rsc15_refs/${DS}_n10/ (already evaluated)"
elif [ -s "${REFDIR}/done" ]; then
  echo "  skip references: ${REFDIR}/done exists"
else
  crd="$(ls -1d logs/train/runs/*/*job${CLEAN_JOB}_* | head -1)"
  W="set -euo pipefail; cd '${PWD}'"
  for st in ${STRATS}; do
    tok="_tgt${st}"; [ "${st}" = "unpopular" ] && tok=""
    man="src/data/erase_data/${DS}_spam${tok}_seed2_pct1_n1/forget_manifest.json"
    prd="$(ls -1d logs/train/runs/*/*job${CKJOB[$st]}_* | head -1)"
    for role in poison clean; do
      ck="${prd}/checkpoints/last.ckpt"; [ "${role}" = "clean" ] && ck="${crd}/checkpoints/last.ckpt"
      W+="; python -u -m scripts.eval_ckpt_on_test experiment=tiger_train_flat data_dir=src/data/erase_data/${DS} \"semantic_id_path='${SID}'\" \"ckpt_path='${ck}'\" \"spam_forget_manifest='${man}'\" num_hierarchies=4 sequence_length=120 seed=2 train=False test=True trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir=${REFDIR}/${st}_${role}"
    done
  done
  W+="; date -Is > ${REFDIR}/done"
  if [ "${DRYRUN}" = "1" ]; then
    echo "== references -> ${REFDIR} (6 evals in one job)"
  else
    mkdir -p "${REFDIR}"
    sbatch --job-name="r15_refs" "${SLURM[@]}" --cpus-per-task=8 --mem=96G --output=logs/%j.out --wrap "${W}"
  fi
fi
echo "${n} unlearning job(s) $([ "${DRYRUN}" = "1" ] && echo 'would be submitted' || echo submitted); ${skip} skipped"
