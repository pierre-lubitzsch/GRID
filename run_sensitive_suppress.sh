#!/usr/bin/env bash
# Sensitive-category arms that vary ONLY the neighbourhood term's form/sign.
#
# This is a SUBMITTER, not an sbatch job. It dispatches through
# run_tiger_unlearn_sequential.sh, which is what the recorded sensitive runs
# used. Calling `python -m src.unlearn` directly (as an earlier version did)
# reproduces the overrides but NOT the pipeline: it writes to a date-stamped
# run dir instead of the flat `_bs1_<tag>` one every extractor globs, and it
# never runs the post-unlearn evaluation, so the run completes with exit 0 and
# no metrics.csv at all. Jobs 10454216-19 died that way, silently.
#
# WHY THESE ARMS. For SPAM, letting suppressed mass flow to a deleted item's
# neighbours is repair. For a SENSITIVE deletion the neighbours ARE the concept,
# and repair undoes the deletion: measured on beauty, lambda_n=0.1 with `mass`
# returns Sensitive@10 to 0.03844 against a pre-unlearning 0.03840.
#   suppress            bounded drain, -log(1 - sum_p p)
#   lambda_n < 0        drains by sign flip; same intent, but unbounded below
#                       (gradient constant in log-mass), so it can look healthy
#                       for many steps and then diverge. Run to settle it.
#
#   DRYRUN=1 ./run_sensitive_suppress.sh
#   LOSS_TYPE=mass LAMBDA_N=-0.1 ./run_sensitive_suppress.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRYRUN="${DRYRUN:-0}"
DATASETS="${DATASETS:-beauty toys}"
LOSS_TYPE="${LOSS_TYPE:-suppress}"
LAMBDA_N="${LAMBDA_N:-0.1}"
SEEDS="${SEEDS:-2}"   # unlearning seeds; every arm was 1 seed until 2026-08-19
# "-" in a run-dir name breaks the extractors' `ln([0-9.]+)` regexes.
case "${LAMBDA_N}" in
  -*) LN_TOK="lnneg${LAMBDA_N#-}" ;;
  *)  LN_TOK="ln${LAMBDA_N}" ;;
esac

COUNT=0
for ds in ${DATASETS}; do
  case "${ds}" in
    beauty)
      DATA="src/data/amazon_data/beauty_sens_hairloss_1e-4_seed2"
      SID="embeddings/beauty/merged_predictions_tensor.pt"
      CKPT="logs/train/runs/2026-05-29/14-06-14_job9096928_beauty_clean/checkpoints/checkpoint_epoch=000_step=004600.ckpt"
      EMB="embeddings/beauty_merged_predictions_tensor_latest.pt"
      MAN="src/data/amazon_data/beauty/forget_manifest_hairloss_1e-4_seed2.json"
      EVAL_DIR="src/data/amazon_data/beauty"
      TAG="sens_beauty_hairloss_unified_${LN_TOK}_${LOSS_TYPE}_embedding"
      ;;
    toys)
      DATA="src/data/amazon_data/toys_sens_babytoys_1e-4_seed2"
      SID="embeddings/toys/merged_predictions_tensor.pt"
      # Same checkpoint as the recorded toys ln0 / mass arms (last.ckpt, NOT the
      # selected epoch checkpoint: N@10 0.0332 vs 0.0359). Kept so the toys arms
      # stay comparable to each other; do not compare them to evaluations of the
      # epoch checkpoint without re-running all of them.
      CKPT="logs/train/runs/2026-08-12/17-26-53_job10396605_toys_clean/checkpoints/last.ckpt"
      EMB="logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"
      MAN="src/data/amazon_data/toys/forget_manifest_babytoys_1e-4_seed2.json"
      EVAL_DIR="src/data/amazon_data/toys"
      TAG="sens_toys_babytoys_unified_${LN_TOK}_${LOSS_TYPE}_embedding"
      ;;
    *) echo "unknown dataset '${ds}'" >&2; exit 1 ;;
  esac
  for _sd in ${SEEDS}; do
  _seedtok=""; [ "${_sd}" != "2" ] && _seedtok="_useed${_sd}"
  TAG_S="${TAG}${_seedtok}"
  if [ -n "$(find logs/unlearn/runs -maxdepth 1 -name "*_bs1_${TAG_S}" 2>/dev/null | head -1)" ]; then
    echo "[skip] exists: ${TAG_S}"; continue
  fi
  COUNT=$((COUNT+1))
  echo "[$COUNT] ${ds}: ${TAG_S}"
  [ "${DRYRUN}" = "1" ] && continue
  UNLEARN_RUN_TAG="${TAG_S}" UNLEARN_SEED="${_sd}" \
  UNLEARN_EVAL_DATA_DIR="${EVAL_DIR}" \
    # --job-name carries the arm so a PENDING job is identifiable: a run dir only
    # appears once the job starts, so without this a partially submitted batch
    # cannot be reconciled and a blind re-run duplicates whatever did land.
    # The retry checks by that name first, because when the controller is
    # unhealthy sbatch reports `Socket timed out` AFTER accepting the job.
    # Skip what is already there. The run-dir check catches finished work; the
    # squeue-by-name check catches work still PENDING, which no run dir reveals.
    # Without both, re-driving this submitter after a partial failure duplicates
    # whatever did land.
    if ls -d logs/unlearn/runs/*_bs1_"${TAG}${_seedtok}" >/dev/null 2>&1; then
      echo "  [have] ${TAG}${_seedtok}"; continue
    fi
    if [ -n "$(squeue -u "$USER" -n "ul_${TAG}${_seedtok}" -h 2>/dev/null)" ]; then
      echo "  [queued] ${TAG}${_seedtok}"; continue
    fi
    _try=1
    while :; do
      sbatch --job-name="ul_${TAG}${_seedtok}" \
        --partition="${PART:-pgpu}" --time="${UNLEARN_TIME:-02:00:00}" \
        --gres="${UNLEARN_GRES:-gpu:nvidia_h200:2}" \
      run_tiger_unlearn_sequential.sh \
      "${CKPT}" "${DATA}" unified "${SID}" \
      false 1 0.0 \
      "spam_forget_manifest=${MAN}" \
      unlearning.deletion_spec=item_pairs \
      unlearning.include_context_rows=true \
      unlearning.request_user_order=sorted \
      unlearning.n_unlearning_chunks=5 \
      unlearning.lambda_f=0.1 unlearning.lambda_s=0.1 \
      unlearning.lambda_n="${LAMBDA_N}" \
      unlearning.sep_negatives=forget_target_only \
      unlearning.n_epochs=4 \
      unlearning.adaptive_codes=false \
      unlearning.neighborhood_count=8 \
      unlearning.coherence_loss_type="${LOSS_TYPE}" \
      unlearning.coherence_neighbor_method=embedding \
      unlearning.coherence_rows=target_only \
      "unlearning.embedding_path=${EMB}" \
      unlearning.coherence_embedding_metric=cosine && break || true
      sleep 5
      if [ -n "$(squeue -u "$USER" -n "ul_${TAG}${_seedtok}" -h 2>/dev/null)" ]; then
        echo "  sbatch errored for ${TAG}${_seedtok} but the job IS queued; not retrying" >&2
        break
      fi
      if [ "${_try}" -ge "${SUBMIT_TRIES:-5}" ]; then
        echo "  sbatch failed ${_try}x for ${TAG}${_seedtok}; giving up" >&2
        break
      fi
      echo "  sbatch failed for ${TAG}${_seedtok}; retry ${_try}/${SUBMIT_TRIES:-5} in $((_try * 10))s" >&2
      sleep $((_try * 10))
      _try=$((_try + 1))
    done
  done
done
echo "submitted ${COUNT} (loss_type=${LOSS_TYPE}, lambda_n=${LAMBDA_N})"
