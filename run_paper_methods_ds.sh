#!/usr/bin/env bash
# Table-1 methods on the extra datasets with meaningful item semantics (MovieLens-10M,
# Amazon 2023 Grocery food_k5), for TIGER and LETTER: every paper method at its
# configuration as selected on BEAUTY (paper/plotting/valsel_selected.csv, beauty,rkmeans),
# plus the lambda_s=0 ablation arms, plus the clean / poisoned reference evaluations
# PGR and UR need. Mid-popularity target, bandwagon 1%, n_target 1 or 10, seed 2.
#
# Overrides are those of run_rsc15_paper_methods.sh (copied from the Beauty mid seed-2
# templates) with NAU's embedding_path pointing at the dataset's own item embeddings.
# LETTER uses the same overrides; its unified tags carry the "_nauform" suffix like the
# Beauty LETTER runs. Generative L_sep at n_target=10 needs an H200 (OOM on 80 GB).
#
#   DS=ml10m MODEL=tiger NT=1 bash run_paper_methods_ds.sh            # DRY RUN (default)
#   DS=ml10m MODEL=tiger NT=1 DRYRUN=0 bash run_paper_methods_ds.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
DS="${DS:?DS=ml10m|food_k5}"
MODEL="${MODEL:-tiger}"
NT="${NT:-1}"
case "${DS}" in
  ml10m)   EMB="embeddings/ml10m_merged_predictions_tensor_latest.pt"
           BASE="src/data/erase_data/ml10m" ;;
  food_k5) EMB="logs/inference/runs/embeddings/food_k5_10462481/pickle/merged_predictions_tensor.pt"
           BASE="src/data/amazon_data/food_k5" ;;
  *) echo "DS must be ml10m|food_k5" >&2; exit 1 ;;
esac
SFX=""; SID="embeddings/${DS}/merged_predictions_tensor.pt"
if [ "${MODEL}" = "letter" ]; then SFX="_nauform"; SID="embeddings/${DS}_letter/merged_predictions_tensor.pt"; fi
# Repair data. ERASE's subset rule (16*|D_f| retain rows that mention NO forget item) leaves
# only ~505 niche users on ML-10M (fake users touch ~1,300 popular movies per request), and
# repair memorises them: utility drops and repair-only methods AMPLIFY the target (2026-10-27,
# jobs 11326175/76). full = repair batches from the whole retain split, same optimizer steps.
# SCIF keeps the subset (its influence step is scaled by the subset size; it is a no-op here).
RETAIN_SOURCE="${RETAIN_SOURCE:-full}"
RS_OV=""; RS_SFX=""
if [ "${RETAIN_SOURCE}" = full ]; then RS_OV=" unlearning.retain_source=full"; RS_SFX="_rfull"; fi
PDIR="${BASE}_spam_tgtmid_seed2_pct1_n${NT}"

# Training segments of each chain, in order (segments were appended after the
# 2026-10-21 submissions when runs had not early-stopped). run_ckpt takes the
# FIRST segment that COMPLETED (early stopping fired, later segments only reload
# it), else the last segment that has a checkpoint.
declare -A CKJOB=(
  [tiger_food_k5_clean]="11257688 11257705 11257706 11257707 11316851 11339536 11339537"
  [tiger_food_k5_1]="11257692 11257708 11257709 11257710 11316852 11339538 11339539"
  [tiger_food_k5_10]="11257697 11257711 11257712 11257713 11316853 11339540 11339541"
  [tiger_ml10m_clean]="11257749 11257750 11314641"
  [tiger_ml10m_1]="11257751 11257752"
  [tiger_ml10m_10]="11257753 11257754 11314642"
  [tiger_ml10m_5]="11326163 11326164 11326165"
  [letter_ml10m_clean]="11267928 11267929 11314643"
  [letter_ml10m_1]="11267930 11267931"
  [letter_ml10m_10]="11267932 11267933"
)
run_ckpt() {
  local j d pick=""
  for j in ${CKJOB[$1]}; do
    d="$(ls -1d logs/train/runs/*/*job${j}_* 2>/dev/null | head -1)"
    [ -n "${d}" ] && [ -f "${d}/checkpoints/last.ckpt" ] || continue
    pick="${d}"
    [ "$(sacct -j "${j}" -X -n -o State 2>/dev/null | tr -d ' ')" = COMPLETED ] && break
  done
  echo "${pick}/checkpoints/last.ckpt"
}
# CKPT_OVERRIDE: start from a specific poisoned checkpoint (e.g. a longer-trained resume);
# pair it with TAG_EXTRA so its runs and references get their own names.
CKPT="${CKPT_OVERRIDE:-$(run_ckpt "${MODEL}_${DS}_${NT}")}"
TAG_EXTRA="${TAG_EXTRA:-}"
CLEAN_CKPT="$(run_ckpt "${MODEL}_${DS}_clean")"

SLURM=(--partition=pgpu --exclude=s-sc-pgpu01,s-sc-pgpu02 --time=${TIME:-36:00:00})
GRES_STD="${GRES:-gpu:2}"
GRES_BIG="${GRES_BIG:-gpu:nvidia_h200:2}"

U="unlearning.n_unlearning_chunks=10"
UNI="unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4 unlearning.adaptive_codes=false"
NBR="unlearning.coherence_loss_type=mass unlearning.coherence_neighbor_method=embedding unlearning.coherence_rows=target_only unlearning.coherence_embedding_metric=cosine unlearning.neighborhood_count=8 unlearning.embedding_path=${EMB}"
GEN="unlearning.sep_loss_type=generative"
# arm (paper tag) | algorithm | overrides
ARMS=(
  "unified_abl_lf0.0_ls0.0_ln0.0|unified|${U} unlearning.lambda_f=0.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0 ${UNI}"
  "unified_abl_lf1.0_ls0.0_ln0.0_lr0.0|unified|${U} unlearning.lambda_f=1.0 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=0.0 ${UNI}"
  "unified_abl_lf0.1_ls0.0_ln0.0|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.0 unlearning.lambda_r=1.0 ${UNI}"
  "scif|scif|${U}"
  "seif_std0.06|seif|${U} unlearning.seif_erase_std=0.06"
  "kookmin_ir0.001|kookmin|${U} unlearning.kookmin_init_rate=0.001"
  "fanchuan_t0.07|fanchuan|${U} unlearning.fanchuan_contrastive_temperature=0.07"
  "nau_s0p01_n0p01|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.01 unlearning.lambda_n=0.01 ${UNI} ${GEN} ${NBR}"
  "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.1 unlearning.lambda_n=0.01 ${UNI} ${GEN} ${NBR}"
  "nosep_ln0p01|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.01 ${UNI} ${NBR}"
  "nosep_ln0p1|unified|${U} unlearning.lambda_f=0.1 unlearning.lambda_s=0.0 unlearning.lambda_n=0.1 ${UNI} ${NBR}"
)

for f in "${SID}" "${EMB}" "${CKPT}" "${CLEAN_CKPT}" "${PDIR}/forget_manifest.json"; do
  [ -f "${f}" ] || { echo "missing ${f}" >&2; exit 1; }
done
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
TAGSTEM="${DS}_bw_tgtmid_n${NT}"
n=0; skip=0
for spec in "${ARMS[@]}"; do
  IFS='|' read -r arm algo ov <<< "${spec}"
  tag="${TAGSTEM}_${arm}"
  [[ "${algo}" = unified && -n "${SFX}" ]] && tag="${tag}${SFX}"
  if [ "${algo}" != scif ]; then ov="${ov}${RS_OV}"; tag="${tag}${RS_SFX}"; fi
  tag="${tag}${TAG_EXTRA}"
  if compgen -G "logs/unlearn/runs/*_bs1_*${tag}" >/dev/null; then echo "  skip ${tag}: run dir exists"; skip=$((skip+1)); continue; fi
  if grep -qxF "pm_${MODEL}_${tag}" <<<"${QUEUED}"; then echo "  skip ${tag}: queued"; skip=$((skip+1)); continue; fi
  gres="${GRES_STD}"; [[ "${NT}" = 10 && "${ov}" == *"${GEN}"* ]] && gres="${GRES_BIG}"
  # shellcheck disable=SC2206
  ova=(${ov})
  cmd=(sbatch --job-name="pm_${MODEL}_${tag}" "${SLURM[@]}" --gres="${gres}" run_tiger_unlearn_sequential.sh
       "${CKPT}" "${DS}" "${algo}" "${SID}" false 1 0.0 "${ova[@]}")
  if [ "${DRYRUN}" = "1" ]; then
    echo "== ${tag} (${gres})"; printf '   %q' "${cmd[@]}"; printf '\n'
  else
    MODEL="${MODEL}" POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS="${NT}" POISON_SEED=2 \
    TARGET_STRATEGY=mid UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED=2 "${cmd[@]}"
  fi
  n=$((n+1))
done

# Reference evaluations on the clean test split against this setting's target set.
REFDIR="logs/ds_refs/${MODEL}_${TAGSTEM}${TAG_EXTRA}"
if [ -s "${REFDIR}/done" ]; then
  echo "  skip references: ${REFDIR}/done exists"
else
  # shellcheck source=scripts/resolve_model.sh
  source scripts/resolve_model.sh
  resolve_model "${MODEL}"
  EXP="$(grid_model_config tiger_train_flat)"
  W="set -euo pipefail; cd '${PWD}'"
  for role in poison clean; do
    ck="${CKPT}"; [ "${role}" = clean ] && ck="${CLEAN_CKPT}"
    W+="; python -u -m scripts.eval_ckpt_on_test experiment=${EXP} data_dir=${BASE} \"semantic_id_path='${SID}'\" \"ckpt_path='${ck}'\" \"spam_forget_manifest='${PDIR}/forget_manifest.json'\" num_hierarchies=4 seed=2 train=False test=True trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir=${REFDIR}/${role}"
  done
  W+="; date -Is > ${REFDIR}/done"
  if [ "${DRYRUN}" = "1" ]; then
    echo "== references -> ${REFDIR} (experiment=${EXP})"
  else
    mkdir -p "${REFDIR}"
    sbatch --job-name="pm_refs_${MODEL}_${TAGSTEM}" "${SLURM[@]}" --gres="${GRES_STD}" --cpus-per-task=8 --mem=96G --output=logs/%j.out --wrap "${W}"
  fi
fi
echo "${n} unlearning job(s) $([ "${DRYRUN}" = "1" ] && echo 'would be submitted' || echo submitted); ${skip} skipped"
