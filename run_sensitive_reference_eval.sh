#!/usr/bin/env bash
#SBATCH --job-name=sens_ref_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:1
#SBATCH --partition=pgpu
#SBATCH --time=08:00:00
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# -----------------------------------------------------------------------------
# Score the BASE and RETRAIN reference models against a SENSITIVE item set.
#
# Why this exists: training runs are evaluated with `spam_forget_manifest`
# pointing at their own (spam) manifest, or none at all, so SH/ASI/TPM come out
# NaN for the sensitive category. The unlearned models DO have Sensitive@k, which
# makes them uninterpretable on their own -- 0.032 could be a big drop or a rise.
#
# ERASE's metrics need exactly these two references:
#   Sensitive@k                 measured here for base and retrain
#   RelItems@k = Sens_retrain@k - Sens_unlearn@k     (needs retrain)
#   RelEff@k   = nDCG_unlearn@k / nDCG_retrain@k - 1 (needs retrain)
# The BASE model is the "before" that shows how much there was to remove.
#
# The evaluation split is identical across base / retrain / unlearned (the
# retrain builder copies evaluation/ and testing/ verbatim), so the three are
# directly comparable.
#
# Usage:
#   sbatch run_sensitive_reference_eval.sh                  # beauty + toys
#   DATASETS=beauty sbatch run_sensitive_reference_eval.sh
# -----------------------------------------------------------------------------
GRID_DIR="${GRID_DIR:-$PWD}"
DATASETS="${DATASETS:-beauty toys sports food_k5}"
NUM_HIER="${NUM_HIER:-4}"
EVAL_SEED="${EVAL_SEED:-2}"
DRYRUN="${DRYRUN:-0}"

# Grid coordinates. The category / manifest / SID all follow from
# (dataset, seed, quantizer), so nothing here is per-dataset hardcoded any more:
# the old version pinned two datasets, seed-2 manifests and four job ids, which
# cannot express 4 datasets x 5 seeds x 3 tracks.
#
# MODELS maps "<dataset>:<seed>:<base_jid>:<retrain_jid>", exactly as in
# run_unlearn_sweep_sensitive.sh. The base model is the one that still contains
# the category; the retrain model is the reference RelEff/RelItems divide by.
MODELS="${MODELS:-beauty:2:9096928:10449165 toys:2:10396605:10449166}"
SEEDS="${SEEDS:-2}"
FORGET_RATIO="${FORGET_RATIO:-1e-4}"
MODEL="${MODEL:-tiger}"
QUANTIZER="${QUANTIZER:-rkmeans}"
# Opt-in for other item sets (e.g. the takedown: CATEGORY_TOK=takedown_elf FORGET_RATIO=0).
# Give such sets their own OUT_ROOT: readers pick the NEWEST <ds>_base / <ds>_retrain
# under logs/eval/sensitive_reference regardless of the item set.
CATEGORY_TOK="${CATEGORY_TOK:-}"
OUT_ROOT="${OUT_ROOT:-logs/eval/sensitive_reference}"
# BASE_CKPT=last evaluates the base model at last.ckpt, the checkpoint the unlearning runs
# start from (all cells but beauty seed 2); default (newest checkpoint_epoch=*) unchanged.
# ROLES restricts which references are evaluated (default: base retrain).
BASE_CKPT="${BASE_CKPT:-epoch}"
ROLES="${ROLES:-base retrain}"

# Must stay identical to category_of() in run_unlearn_sweep_sensitive.sh: a
# reference measured against a different I_s than the unlearning runs used is
# not a reference, and the mismatch is invisible in the numbers.
category_of() {
  case "$1" in
    beauty)  echo "hairloss" ;;
    toys)    echo "toyweapons" ;;
    sports)  echo "guns" ;;
    food_k5) echo "meat" ;;
    *) echo "unknown" ;;
  esac
}

sid_of() {
  if [ "${QUANTIZER}" = "rkmeans" ]; then
    echo "embeddings/$1/merged_predictions_tensor.pt"
  else
    echo "embeddings/$1_${QUANTIZER}/merged_predictions_tensor.pt"
  fi
}

jobids_for() {   # dataset seed -> "<base_jid> <retrain_jid>"
  local key="$1:$2"
  for kv in ${MODELS}; do
    case "${kv}" in
      "${key}:"*) local r="${kv#${key}:}"; echo "${r%%:*} ${r#*:}"; return ;;
    esac
  done
}

# Track/seed tokens on the output dir. Sensitive references are per identifier
# space and per seed for the same reason the spam ones are: a retrain model is
# trained on ONE seed's forget set in ONE identifier space.
role_tok() {
  local t="$1"
  [ "${MODEL}" != "tiger" ] && t="${t}_${MODEL}"
  [ "${QUANTIZER}" != "rkmeans" ] && t="${t}_${QUANTIZER}"
  [ "$2" != "2" ] && t="${t}_seed$2"
  echo "${t}"
}

ckpt_for() {  # job id -> newest epoch checkpoint, excluding other models' runs
  # shellcheck source=scripts/resolve_model.sh
  source "${GRID_DIR}/scripts/resolve_model.sh"
  ls -t logs/train/runs/*/*job${1}_*/checkpoints/checkpoint_epoch=*.ckpt 2>/dev/null \
    | grep -vE "$(grid_other_model_regex)" | head -1 || true
}

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
for ds in ${DATASETS}; do
  cat="${CATEGORY_TOK:-$(category_of "${ds}")}"
  if [ "${cat}" = "unknown" ]; then
    echo "no sensitive category for dataset '${ds}'" >&2; exit 1
  fi
  SID="$(sid_of "${ds}")"
  DATA_DIR="src/data/amazon_data/${ds}"
  for sd in ${SEEDS}; do
    MANIFEST="${DATA_DIR}/forget_manifest_${cat}_${FORGET_RATIO}_seed${sd}.json"
    read -r BASE_JID RETRAIN_JID <<< "$(jobids_for "${ds}" "${sd}")"
    if [ -z "${BASE_JID:-}" ]; then
      echo "## ${ds} seed ${sd}: no job ids in MODELS, skipping (pass MODELS='${ds}:${sd}:<base>:<retrain>')" >&2
      continue
    fi
    [ -f "${MANIFEST}" ] || { echo "manifest missing: ${MANIFEST}" >&2; continue; }
    for role in ${ROLES}; do
      jid=$([ "${role}" = "base" ] && echo "${BASE_JID}" || echo "${RETRAIN_JID}")
      [ -n "${jid}" ] || { echo "   no ${role} job id for ${ds} seed ${sd}, skip" >&2; continue; }
      ck="$(ckpt_for "${jid}")"
      if [ "${role}" = "base" ] && [ "${BASE_CKPT}" = "last" ]; then
        ck="$(ls -d logs/train/runs/*/*job${jid}_*/checkpoints/last.ckpt 2>/dev/null | head -1 || true)"
      fi
      out="${OUT_ROOT}/${STAMP}/$(role_tok "${ds}_${role}" "${sd}")"
      echo "== ${ds} seed ${sd} ${role} (job ${jid}) -> ${out}"
      echo "   ckpt: ${ck:-<MISSING>}"
      echo "   I_s:  ${MANIFEST}"
      [ -n "${ck}" ] || { echo "   no checkpoint, skip" >&2; continue; }
      [ -f "${out}/csv/version_0/metrics.csv" ] && { echo "   already done, skip"; continue; }
      [ "${DRYRUN}" = "1" ] && continue
      mkdir -p "${out}"
      cat > "${out}/coordinates.json" <<JSON
{
  "dataset": "${ds}",
  "role": "${role}",
  "category": "${cat}",
  "model": "${MODEL}",
  "quantizer": "${QUANTIZER}",
  "seed": ${sd},
  "forget_ratio": "${FORGET_RATIO}",
  "train_job": ${jid},
  "ckpt": "${ck}",
  "manifest": "${MANIFEST}",
  "semantic_id_path": "${SID}"
}
JSON
      python -u -m scripts.eval_ckpt_on_test \
        experiment=tiger_train_flat \
        data_dir="${DATA_DIR}" \
        "semantic_id_path='${SID}'" \
        "ckpt_path='${ck}'" \
        "spam_forget_manifest='${MANIFEST}'" \
        num_hierarchies="${NUM_HIER}" seed="${EVAL_SEED}" \
        train=False test=True \
        trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false \
        trainer.num_nodes=1 trainer.deterministic=true \
        callbacks.model_checkpoint=null callbacks.early_stopping=null \
        hydra.run.dir="${out}" || echo "   FAILED: ${ds}_${role}"
    done
  done
done
echo "[$(date -Is)] sensitive reference eval done -> ${OUT_ROOT}/${STAMP}/"
