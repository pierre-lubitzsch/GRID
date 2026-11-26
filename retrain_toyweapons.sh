#!/usr/bin/env bash
# Retrain reference models for the toys "toyweapons" sensitive concept.
#
# 5 seeds x 2 tracks = 10 runs. These are what RelEff@10 / RelItems@10 divide by:
# a model trained on the data with the forget interactions removed. Without them
# toys has no sensitive reference at all and cannot be scored.
#
# Labels match what plan_reference_evals.py and extract_sensitive.py expect:
#   toys_clean[_seed<N>][_<quantizer>]_retrain_toyweapons_1e-4
# which run_tiger_train.sh builds as <ds>_clean + seed token + quantizer token
# + RUN_LABEL_SUFFIX.
# WALLTIME: run_tiger_train.sh's header asks for 2 days, which the 2026-08-27
# maintenance reservation refuses outright ("ReqNodeNotAvail, Reserved for
# maintenance") -- the same trap that stalled the reference evals. Measured toys
# training is ~8.7h (max 08:46:33 over the recent runs), so 12h is ample and
# still schedules before the window.
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-0}"
# QUANTIZERS restricts the tracks. rkmeans is the track the paper reports, so
# that is the default; pass QUANTIZERS="rkmeans rqvae" for both.
for q in ${QUANTIZERS:-rkmeans}; do
  if [ "$q" = "rkmeans" ]; then SID="embeddings/toys/merged_predictions_tensor.pt"; QTOK=""
  else SID="embeddings/toys_rqvae/merged_predictions_tensor.pt"; QTOK="_rqvae"; fi
  for sd in 2 3 5 7 11; do
    DATA="src/data/amazon_data/toys_retrain_toyweapons_1e-4_seed${sd}"
    if [ ! -d "${DATA}/training" ]; then echo "  MISSING ${DATA}; skip"; continue; fi
    _seedtok=""; [ "${sd}" != "2" ] && _seedtok="_seed${sd}"
    echo "== seed${sd} ${q} -> label toys_clean${_seedtok}${QTOK}_retrain_toyweapons_1e-4"
    [ "${DRYRUN}" = "1" ] && continue
    # run_tiger_train.sh does NOT read QUANTIZER and appends no quantizer token
    # (verified: grep QUANTIZER = 0 hits). The track token has to ride in
    # RUN_LABEL_SUFFIX, AFTER the _seed<N> token the script adds itself, so the
    # label matches plan_reference_evals.label():
    #   toys_clean[_seed<N>][_rqvae]_retrain_toyweapons_1e-4
    # Without this both tracks build the SAME label, the duplicate guard skips
    # the second, and 5 models silently stand in for 10.
    TIGER_DATA_DIR="${DATA}" \
    TRAIN_SEED="${sd}" \
    RUN_LABEL_SUFFIX="${QTOK}_retrain_toyweapons_1e-4" \
      sbatch --time="${TRAIN_TIME:-12:00:00}" run_tiger_train.sh toys clean "${SID}"
    sleep 5
  done
done
echo "RETRAIN_SUBMIT_DONE"
