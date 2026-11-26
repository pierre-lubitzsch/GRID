#!/usr/bin/env bash
# Which RQ code POSITION is a usable unlearning interface? Beauty, bandwagon n=1.
#
# WHY THIS SWEEP EXISTS
# ---------------------
# `adaptive_codes=true` (ac1: freeze the coarse prefix c1,c2; train the fine tail)
# removes essentially NOTHING -- measured over 9 (lambda_f, lambda_s) pairs, mean
# removal (poison-SH)/poison: unpopular -5.1%, mid -1.3%, popular +0.9%, i.e. on
# two strata SH ends up ABOVE the poisoned checkpoint. Three reinforcing reasons:
#
#   1. It freezes where the forget signal is. Per-code forget-gradient norms
#      (tables/bandwagon_rqid_beauty.tex): c1 1.93-2.17, c2 1.77-1.94,
#      c3 0.88-1.18, c4 0.44-0.59 -- c1+c2 carry 70-74% of the norm.
#   2. It cannot stop the model ENTERING the target's subtree. Decoding is
#      autoregressive c1->c2->c3->c4; with p(c1),p(c2) frozen you can only
#      redistribute WITHIN the depth-2 bucket, and that bucket averages 3.16
#      items with 30.6% of items singletons. The `mid` target has ZERO depth-2
#      neighbours -- it IS a singleton, so no c3/c4 edit can avoid it. That is
#      why mid is ac1's worst stratum.
#   3. The backbone is frozen too (`adaptive_update_backbone` defaults false), so
#      the encoder cannot adapt its representation of the spam history either.
#      This is a CONFOUND, not a property of position restriction -- hence the
#      BACKBONES dimension below.
#
# c4 is deliberately excluded from the single-position arms: it is the dedup digit
# (16 values, 1.20 bits entropy) and depth-3 buckets are 78.3% singletons, so
# editing only c4 has essentially nothing to redistribute among.
#
# WHAT WAS ALREADY RUN, AND WHY IT IS NOT A FAIR COMPARISON
# The existing `_acp0` study (update_positions=[0], 81 runs, jobs 9934760-9934840)
# only covered lambda_f in {0.0, 0.001, 0.01} -- it NEVER ran lambda_f=0.1, the
# operating point at which the unrestricted ac0 update drives SH@10 to ~0. So the
# position-restricted scopes were judged at a handicapped lambda_f, exactly the
# understatement that made tables/bandwagon_unlearn_beauty*.tex credit unified with
# 41-53% of its actual removal. This sweep therefore runs every scope at
# lambda_f=0.1, lambda_s=0.1 -- the 3-seed-selected operating point.
#
# Reference points at that operating point (already on disk, tag *_ac0_ln0.0):
#   unified ac0: SH@10 0.00000 / 0.00034 / 0.00000, UR 0.977 / 0.966 / 0.937
#   poisoned start: SH@10 0.01400 / 0.01270 / 0.00180 (unpop / mid / popular)
#
# Usage:
#   DRYRUN=1 ./run_unlearn_positions_beauty.sh          # inspect (33 jobs)
#   ./run_unlearn_positions_beauty.sh                   # submit
#   POSITIONS="0 1 2" BACKBONES=on ./run_unlearn_positions_beauty.sh
#   STRATS=mid ./run_unlearn_positions_beauty.sh        # one stratum first (11 jobs)
#
# Reading the result: if a single coarse position recovers most of ac0's removal at
# comparable UR, the RQ hierarchy IS a local unlearning interface and the
# stable/adaptive split was simply inverted (coarse codes are the interface, not
# the fine ones). If no restriction comes close even with the backbone free, the
# honest conclusion is that spam removal in TIGER is not expressible as a
# code-local edit.

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DATASET="${DATASET:-beauty}"
SID="${SID:-embeddings/${DATASET}/merged_predictions_tensor.pt}"
# DATA seeds (own poison generation, clean and poisoned models), tagged
# `_seed<N>`. Distinct from the legacy `_useed<N>` replicate axis, which reuses
# ONE seed-2 poisoned model; set REPLICATES to reproduce that instead.
SEEDS="${SEEDS:-2 3 5 7 11}"
REPLICATES="${REPLICATES:-}"
POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET="${N_TARGET:-1}"
STRATS="${STRATS:-unpopular mid popular}"
DRYRUN="${DRYRUN:-0}"

# The 3-seed-selected operating point. Held fixed so the only variable is scope.
LF="${LF:-0.1}"
LS="${LS:-0.1}"
N_EPOCHS="${N_EPOCHS:-4}"

# Position subsets to try. Space-separated, commas within a subset.
#   0     c1 only (coarsest; where the forget gradient is largest)
#   1     c2 only
#   2     c3 only
#   0,1   c1+c2 -- the EXACT INVERSE of ac1's trainable set
#   0,1,2 everything except the dedup digit (the user's proposal)
POSITIONS="${POSITIONS:-0 1 2 0,1 0,1,2}"
# off = backbone frozen (matches the historical _acp0 convention)
# on  = backbone free, which isolates "which code position" from "can the encoder
#       adapt at all" -- reason 3 above.
BACKBONES="${BACKBONES:-off on}"
# Also run ac1 at THIS operating point, since the published ac1 numbers stop at
# lambda_f=0.01 and so are not comparable to anything here.
INCLUDE_AC1="${INCLUDE_AC1:-1}"

[ -f "${SID}" ] || { echo "SID tensor missing: ${SID}" >&2; exit 1; }

COUNT=0
SKIPPED=0

# submit <strategy> <rundir> <tag_suffix> [extra hydra overrides...]
submit() {
  local strat="$1" rundir="$2" suffix="$3" dseed="$4" useed="$5"; shift 5
  local seedtok=""
  if [ -n "${REPLICATES}" ]; then
    [ "${useed}" != "2" ] && seedtok="_useed${useed}"
  else
    [ "${dseed}" != "2" ] && seedtok="_seed${dseed}"
  fi
  local tag="bw_tgt${strat}_unified_lf${LF}_ls${LS}_${suffix}${seedtok}"
  # Duplicate guard: run dirs are flat, logs/unlearn/runs/job<jid>_<ts>_bs1_<tag>.
  # A second run under an existing tag is double-counted by every extractor that
  # averages over seeds. The `_bs1_` anchor stops rsc15-prefixed tags matching.
  if [ "${FORCE:-0}" != "1" ]; then
    local existing
    existing="$(find logs/unlearn/runs -maxdepth 1 -name "*_bs1_${tag}" 2>/dev/null | head -1 || true)"
    if [ -n "${existing}" ]; then
      SKIPPED=$((SKIPPED + 1))
      echo "[skip] exists: $(basename "${existing}")"
      return
    fi
  fi
  COUNT=$((COUNT + 1))
  echo "[$COUNT] ${strat} tag=${tag}"
  [ "$DRYRUN" = "1" ] && return
  env POISON_METHOD=bandwagon POISONING_RATIO="$POISONING_RATIO" \
      N_TARGET_ITEMS="$N_TARGET" POISON_SEED="$dseed" \
      TARGET_STRATEGY="$strat" UNLEARN_RUN_TAG="$tag" UNLEARN_SEED="$useed" \
      ${REPLICATES:+ALLOW_SEED_MISMATCH=1} \
    sbatch run_tiger_unlearn_sequential.sh \
      "$rundir" "$DATASET" unified "$SID" \
      false 1 0.0 \
      unlearning.n_unlearning_chunks=10 \
      unlearning.lambda_f="$LF" unlearning.lambda_s="$LS" \
      unlearning.sep_negatives=forget_target_only \
      unlearning.n_epochs="$N_EPOCHS" \
      "$@"
}

# Resolved from the run-dir LABEL, so a new data seed needs no edit here. The
# seed token sits BEFORE the stratum token in the label.
poison_rundir() {  # <stratum> <data seed>
  local sfx=""; [ "$2" != "2" ] && sfx="_seed$2"
  ls -1d logs/train/runs/*/*_job*_"${DATASET}"_poison_pct1_n"${N_TARGET}""${sfx}"_tgt"$1" \
    2>/dev/null | head -1
}

for strat in ${STRATS}; do
 for dseed in ${SEEDS}; do
  rundir="$(poison_rundir "${strat}" "${dseed}")"
  [ -n "${rundir}" ] || { echo "No poisoned run for ${strat}/seed${dseed}" >&2; exit 1; }
  [ -n "$(ls -1 "${rundir}"/checkpoints/*.ckpt 2>/dev/null | head -1)" ] \
    || { echo "No checkpoint in ${rundir}" >&2; exit 1; }
  for useed in ${REPLICATES:-$dseed}; do
  echo "## MODEL ${strat} seed${dseed}: $(basename "${rundir}")"

  for pos in ${POSITIONS}; do
    postok="${pos//,/}"
    for bb in ${BACKBONES}; do
      bbtok=""; bbval=false
      [ "${bb}" = "on" ] && { bbtok="bb"; bbval=true; }
      # Optional code-lr scales ON TOP of the position restriction.
      #   ACLR  scales the ADAPTIVE tail [stable_codes, H). Only meaningful when
      #         update_positions keeps positions on BOTH sides of stable_codes
      #         (e.g. [0,1,2]); unified.py refuses [0,1] (nothing to scale) and
      #         [2,3] (uniform = a plain lr change).
      #   CLR   scales EVERY code param uniformly. Well defined for any position
      #         set, so it is the right knob when ACLR is refused.
      SCALE_OVR=(); scaletok=""
      if [ "${ACLR:-1.0}" != "1.0" ]; then
        SCALE_OVR+=(unlearning.adaptive_code_lr_scale="${ACLR}")
        scaletok="${scaletok}_aclr${ACLR}"
      fi
      if [ "${CLR:-1.0}" != "1.0" ]; then
        SCALE_OVR+=(unlearning.code_lr_scale="${CLR}")
        scaletok="${scaletok}_clr${CLR}"
      fi
      submit "$strat" "$rundir" "acp${postok}${bbtok}${scaletok}" "$dseed" "$useed" \
        unlearning.update_positions="[${pos}]" \
        unlearning.update_positions_backbone="${bbval}" \
        unlearning.adaptive_codes=false \
        ${SCALE_OVR[@]+"${SCALE_OVR[@]}"}
    done
  done

  if [ "${INCLUDE_AC1}" = "1" ]; then
    # ac1 at this operating point, for a like-for-like reference against the
    # position arms (the published ac1 numbers stop at lambda_f=0.01).
    submit "$strat" "$rundir" "ac1" "$dseed" "$useed" \
      unlearning.adaptive_codes=true unlearning.stable_codes=2
  fi
  done   # useed (replicate)
 done    # dseed (data seed)
done     # strat

echo "----"
echo "Total unlearn jobs: ${COUNT}  (skipped ${SKIPPED} existing; FORCE=1 to re-run)"
echo "  operating point: lambda_f=${LF} lambda_s=${LS} n_epochs=${N_EPOCHS}"
echo "  POSITIONS='${POSITIONS}'  BACKBONES='${BACKBONES}'  STRATS='${STRATS}'"
echo "Disk estimate: ~$((COUNT * 190 / 1024)) GB"
echo
echo "Compare against (same operating point, already on disk):"
echo "  ac0 (unrestricted): tag bw_tgt<strat>_unified_lf0.1_ls0.1_ac0_ln0.0"
echo "  SH@10 0.00000 / 0.00034 / 0.00000, UR 0.977 / 0.966 / 0.937"
