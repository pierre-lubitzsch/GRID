#!/usr/bin/env bash
# Unlearning sweep over the clone_inject poisoned beauty models (n=1 and n=3).
#
# Exact analog of run_unlearn_sweep_clone_flood.sh, retargeted to the clone_inject
# poisoned checkpoints (POISON_METHOD=clone_inject). Each run starts FROM the
# poisoned checkpoint and unlearns the 226 injected spam users (training_forget/)
# while preserving training_retain/, via run_tiger_unlearn_sequential.sh, which
# also runs a post-unlearn clean-test eval (recall/ndcg + SH@k/ASI@k vs I_t).
#
# Checkpoints are AUTO-DISCOVERED from the clone_inject (no-PKM) training run dirs
#   logs/train/runs/*/*_beauty_poison_clone_inject_pct${PCT}_<ntag>/checkpoints/*.ckpt
# (the latest by mtime). Override per model with CKPT_n1=... / CKPT_n3=... env vars.
# NOTE: the poisoned models must have FINISHED training (a .ckpt must exist) before
# you submit for real. Use DRYRUN=1 to preview the plan while they are still running.
#
# Usage:
#   ./run_unlearn_sweep_clone_inject.sh           # submit all jobs
#   DRYRUN=1 ./run_unlearn_sweep_clone_inject.sh  # print the sbatch calls only
#   MODELS="n3" ./run_unlearn_sweep_clone_inject.sh           # restrict to one model
#   CKPT_n3=logs/.../checkpoints/foo.ckpt ./run_unlearn_sweep_clone_inject.sh
#
# Global unlearning params: all DEFAULT except
#   neighborhood_aware_sample_rate=0.0  (positional arg 7)
#   unlearning.n_unlearning_chunks=10   (extra hydra override)
#   neighborhood_aware=false            (positional arg 5)
# (request_batch_size positional=1 is a dummy; n_unlearning_chunks takes priority.)

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

SID="embeddings/beauty/merged_predictions_tensor.pt"
POISON_SEED="${POISON_SEED:-2}"
POISONING_RATIO="${POISONING_RATIO:-0.01}"
DRYRUN="${DRYRUN:-0}"
MODELS="${MODELS:-n1 n3}"          # which poison models to sweep (n1 n3 [n5 n10])

PCT="$(python3 -c "print(int(round(${POISONING_RATIO} * 100)))")"

# Auto-discover the latest clone_inject (no-PKM) checkpoint for a model tag.
# The trailing "/checkpoints" anchor excludes the _pkmreplace/_pkmadd run dirs and
# avoids n1 matching n10 (dir is ..._n10/checkpoints, not ..._n1/checkpoints).
discover_ckpt() {
  local ntag="$1"
  ls -t logs/train/runs/*/*_beauty_poison_clone_inject_pct${PCT}_${ntag}/checkpoints/*.ckpt \
    2>/dev/null | head -n1 || true
}

# unified (lambda_f, lambda_s) grid
UNIFIED_PAIRS=(
  "0.0 0.0" "0.01 0.0" "0.01 1.0" "0.01 0.1"
  "0.001 0.0" "0.001 0.1" "0.0 1.0" "0.0 0.1" "0.0 10.0"
)

COUNT=0

# submit <ntag:n1|n3> <ckpt> <algorithm> <run_tag_suffix> [extra hydra overrides...]
submit() {
  local ntag="$1" ckpt="$2" algo="$3" suffix="$4"; shift 4
  local n="${ntag#n}"                      # n1 -> 1, n3 -> 3
  local tag="ci_${ntag}_${suffix}"
  COUNT=$((COUNT + 1))
  echo "[$COUNT] N_TARGET_ITEMS=$n algo=$algo tag=$tag  extra=[$*]"
  if [ "$DRYRUN" = "1" ]; then return; fi
  POISON_METHOD=clone_inject POISONING_RATIO="$POISONING_RATIO" N_TARGET_ITEMS="$n" POISON_SEED="$POISON_SEED" \
  UNLEARN_RUN_TAG="$tag" \
    sbatch run_tiger_unlearn_sequential.sh \
      "$ckpt" beauty "$algo" "$SID" \
      false 1 0.0 \
      unlearning.n_unlearning_chunks=10 "$@"
}

for ntag in $MODELS; do
  # Resolve the poisoned checkpoint: explicit CKPT_<ntag> env wins, else discover.
  eval "ckpt=\${CKPT_${ntag}:-}"
  if [ -z "$ckpt" ]; then ckpt="$(discover_ckpt "$ntag")"; fi

  if [ -z "$ckpt" ]; then
    if [ "$DRYRUN" = "1" ]; then
      ckpt="<ckpt-pending:${ntag} still training>"
      echo "## MODEL ${ntag}: NO checkpoint yet (training in progress) -> ${ckpt}"
    else
      echo "MISSING ckpt for ${ntag}: no .ckpt under logs/train/runs/*/*_beauty_poison_clone_inject_pct${PCT}_${ntag}/checkpoints/" >&2
      echo "  The poisoned model is probably still training. Wait for it, or set CKPT_${ntag}=..." >&2
      exit 1
    fi
  else
    echo "## MODEL ${ntag}: using ckpt ${ckpt}"
  fi

  # scif — default params
  submit "$ntag" "$ckpt" scif "scif"

  # seif — seif_erase_std in [0.006, 0.06]
  for std in 0.006 0.06; do
    submit "$ntag" "$ckpt" seif "seif_std${std}" unlearning.seif_erase_std="$std"
  done

  # kookmin — kookmin_init_rate in [0.01, 0.001, 0.0001]
  for ir in 0.01 0.001 0.0001; do
    submit "$ntag" "$ckpt" kookmin "kookmin_ir${ir}" unlearning.kookmin_init_rate="$ir"
  done

  # fanchuan — fanchuan_contrastive_temperature in [0.07, 0.007]
  for t in 0.07 0.007; do
    submit "$ntag" "$ckpt" fanchuan "fanchuan_t${t}" unlearning.fanchuan_contrastive_temperature="$t"
  done

  # unified — (lambda_f, lambda_s) grid x adaptive_codes{false, true}
  #   sep_negatives=forget_target_only, n_epochs=4.
  #   stable_codes is only meaningful with adaptive_codes=true -> set it (=2)
  #   ONLY in the adaptive arm; the non-adaptive arm uses the config default.
  for pair in "${UNIFIED_PAIRS[@]}"; do
    read -r lf ls <<< "$pair"
    for ac in false true; do
      acflag=$([ "$ac" = "true" ] && echo 1 || echo 0)
      uni_extra=(
        unlearning.lambda_f="$lf" unlearning.lambda_s="$ls"
        unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4
        unlearning.adaptive_codes="$ac"
      )
      if [ "$ac" = "true" ]; then uni_extra+=(unlearning.stable_codes=2); fi
      submit "$ntag" "$ckpt" unified "unified_lf${lf}_ls${ls}_ac${acflag}" "${uni_extra[@]}"
    done
  done

  # neg_train — default params
  submit "$ntag" "$ckpt" neg_train "negtrain"

  # filter — filter_mode in [global, user_dependent]
  for fm in global user_dependent; do
    submit "$ntag" "$ckpt" filter "filter_${fm}" unlearning.filter_mode="$fm"
  done
done

echo "----"
echo "Total jobs: $COUNT  (DRYRUN=$DRYRUN, MODELS='$MODELS', method=clone_inject)"
