#!/usr/bin/env bash
set -euo pipefail
# -----------------------------------------------------------------------------
# Minimal validation of the REFINED ID SPACE as an L_n neighbour source.
# TIGER, beauty (width 256), bandwagon n_target=1, all four unified terms active.
#
#   ./run_unlearn_sweep_latent.sh            # 15 jobs (3 strata x 5 arms)
#   DRYRUN=1 ./run_unlearn_sweep_latent.sh   # list, submit nothing
#
# PREREQUISITE: ./run_latent_refiner.sh (Step 2) must have written a z tensor per
# stratum under ${LATENT_ROOT}. This script refuses to submit a latent arm
# without one rather than silently falling back to the embedding neighbourhood.
#
# WHAT IS AND IS NOT VARIED
# -------------------------
# Everything except the NEIGHBOUR SOURCE is pinned at the selected operating
# point, so a difference between arms is attributable to N(t) alone:
#
#   L = L_R + lambda_f L_F + lambda_s L_sep + lambda_n L_n
#       lambda_f = 0.1   lambda_s = 0.1   lambda_n = 0.1    (all four ACTIVE)
#       adaptive_codes = false, n_epochs = 4, neighborhood_count = 8
#       coherence_rows = target_only, coherence_loss_type = mass
#
# (lambda_f, lambda_s) = (0.1, 0.1) is the 3-seed-selected point from the 432-run
# grid, confirmed on held-out seeds (WORKFLOW.md, HOLDOUT_HEAD2HEAD). lambda_n =
# 0.1 is the middle of the post-fix sweep's {0.01, 0.1, 1.0} and the value its
# rows control used. `mass` and `target_only` are the FIXED L_n: `nll`'s optimum
# needs every neighbour at probability 1 (infeasible for count > 1) and
# `rows=all` spent >=91.7% of the gradient budget on popular filler items.
#
# THE ARMS  (Step 4 of the plan: compare N_emb, N_z, and N_emb u N_z)
# -----------------------------------------------------------------
#   ln0.0                 L_n off. The reference: does ANY neighbourhood help?
#   embedding             N_emb  — top-k in the pre-quantization space (incumbent)
#   latent                N_z    — top-k in the refined space z
#   emblatent (full)      N_emb u N_z, each source contributing count=8 (<=16)
#   emblatent (matched)   N_emb u N_z inside ONE count=8 budget (4+4)
#
# Both union variants are run on purpose. `full` is the plan's literal union, but
# it hands that arm up to twice the neighbours, so a win there is confounded with
# simply teacher-forcing more items; `matched` removes the confound. Read them
# together — full-only win = budget effect, both win = the union is genuinely
# better per neighbour.
#
# NOT varied here (deliberately out of scope for the minimal version): no
# decoding-time guidance, no latent-space unlearning term (beta L_latent), no
# alpha reranking. Standard decoding only, exactly as Step 5 specifies.
#
# Runs land in THIS checkout's logs/unlearn/runs, which is a fresh tree — the
# 2280 recorded runs live in the /sc-scratch checkout. Tags keep the historical
# naming so the existing extractors parse them; the `embedding` and `ln0.0` arms
# therefore reproduce recorded configurations, which is intended: this batch is
# self-contained and comparable within itself.
# -----------------------------------------------------------------------------

# Repo root -- resolved from THIS SCRIPT's own location, not from the
# environment. There are two GRID checkouts on this machine and an interactive
# session here is itself a SLURM job, so SLURM_SUBMIT_DIR is already set and has
# been observed pointing at the OTHER (live, /sc-scratch) checkout. Trusting it
# would silently run this sweep out of the wrong repo -- submitting jobs into the
# tree this branch exists to stay out of. Checking that the target merely "looks
# like a checkout" (.project-root) is NOT enough: both checkouts have one.
#
# Directly executed: BASH_SOURCE is the script inside its checkout -> correct.
# Under sbatch: SLURM copies the script to a spool dir with no .project-root
# beside it, so we fall back to SLURM_SUBMIT_DIR, which sbatch sets to the
# directory the job was submitted FROM -- also correct.
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "${_ROOT}" ] && [ -f "${_ROOT}/.project-root" ] || _ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${_ROOT}"
echo "## repo root: ${_ROOT}"

DATASET="${DATASET:-beauty}"
POISON_SEED="${POISON_SEED:-2}"
UNLEARN_SEED="${UNLEARN_SEED:-2}"
POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET="${N_TARGET:-1}"
DRYRUN="${DRYRUN:-0}"
STRATS="${STRATS:-unpopular mid popular}"

# Operating point (see header for provenance).
LF="${LF:-0.1}"
LS="${LS:-0.1}"
LN="${LN:-0.1}"
N_EPOCHS="${N_EPOCHS:-4}"
NEIGHBORHOOD_COUNT="${NEIGHBORHOOD_COUNT:-8}"
PREFIX_LENGTH="${PREFIX_LENGTH:-2}"
COHERENCE_LOSS_TYPE="${COHERENCE_LOSS_TYPE:-mass}"
COHERENCE_ROWS="${COHERENCE_ROWS:-target_only}"
COHERENCE_EMBEDDING_METRIC="${COHERENCE_EMBEDDING_METRIC:-cosine}"

# Which arms to run. Space-separated subset of:
#   ln0 prefix embedding latent emblatent_full emblatent_matched
# `prefix` is the SHIPPED config default (shared-SID-prefix neighbourhood) and so
# the baseline any N_z claim has to beat. It is width-dependent and can be EMPTY
# (30.6% of beauty items have no prefix-2 neighbour), which is the whole reason
# the embedding method was introduced -- but that makes it the "before" state,
# not something to omit.
ARMS="${ARMS:-ln0 prefix embedding latent emblatent_full emblatent_matched}"

# SLURM placement. Testing runs go on the ORDINARY `gpu` partition, not `pgpu`:
# pgpu is the contended H200/A100-SXM pool the production sweeps live on, and
# this branch exists to stay out of their way.
#
# The gres MUST change with the partition. run_tiger_unlearn_sequential.sh hard-
# codes `--gres=gpu:nvidia_h200:2`, and h200 exists ONLY on pgpu — on `gpu` that
# request can never be satisfied and the job would sit PENDING forever. So we ask
# for one GPU of a type that exists on `gpu` (see GRES below for which).
#
# GPU COUNT IS PARTITION-DEPENDENT, and getting it wrong FAILS SILENTLY.
#
# The unlearn config pins trainer.devices=1 with strategy=auto, so only ONE GPU is
# ever used. But `pgpu` is a multi-GPU partition and the site enforces:
#
#   sbatch: Request for single GPU Tres on a multi-GPU partition found
#   sbatch: Forcing single-GPU partition and default GPU TRES
#
# i.e. asking for :1 on pgpu does NOT queue -- sbatch REWRITES the job onto
# partition `gpu` with the DEFAULT gres (a100-pcie-40gb, 40 GB), which then OOMs
# the L_n arms. It still returns a job id, and `slurm:` below still echoes what
# you asked for, so the only way to notice is `scontrol show job`. Measured:
# jobs 10524322-32 requested pgpu/sxm4-80gb:1 and every one landed on
# gpu/a100-pcie-40gb; three OOMed before being cancelled.
#
# So: on `pgpu` request >= 2 GPUs (that is what the `:2` in the production header
# is for -- a queueing requirement, not a compute one). On `gpu`, request :1.
# ALWAYS verify with:
#   scontrol show job <id> | grep -oE 'Partition=[^ ]*|TresPerNode=[^ ]*'
PARTITION="${PARTITION:-gpu}"
# The `gpu` partition is NOT uniform, and for the L_n arms only ONE class of node
# has enough memory. Measured here, all on identical config:
#   s-sc-gpu[002-021] (20)  a100-pcie-40gb  39.5 GiB  OOM at 39.49 GiB
#   s-sc-gpu029       (1)   l40s            44.4 GiB  OOM at 44.35 GiB
#   s-sc-gpu[022-028] (7)   a100_80gb_pcie  80   GiB  OK
#
# CAUTION on reading those numbers: the ln0.0 arm (L_n OFF) DID complete on the
# l40s in 6m23s. That is not evidence the l40s is big enough -- turning L_n on
# teacher-forces neighborhood_count neighbours per scored row, which is what
# pushes peak memory past 44.4 GiB. So do not size the request off an ln0.0 run.
#
# Hence pin the gres to the 80 GB type rather than excluding node ranges: it is
# the only class that fits every arm, and pinning survives new nodes being added
# to the partition (an exclude list would silently let a new 40 GB node through).
GRES="${GRES:-gpu:nvidia_a100_80gb_pcie:1}"
EXCLUDE_NODES="${EXCLUDE_NODES:-}"

# The emblatent_full arm runs within a couple of hundred MiB of the 80 GB card
# (measured: 77.95 GiB allocated by PyTorch, died asking for 194 MiB more with
# 62 MiB free), because alloc=2*count doubles the teacher-forced neighbour
# tensor. expandable_segments lets the caching allocator give back
# reserved-but-unallocated blocks instead of fragmenting, which is exactly this
# failure mode. Harmless for the arms that already fit.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Walltime. run_tiger_unlearn_sequential.sh asks for 1-00:00:00 because the
# production sweeps run much bigger configs; THESE runs measure 6m23s-7m54s
# end to end (13 completed runs, incl. the post-unlearn eval).
#
# Asking for 24h when you need 8min is not just impolite, it is the difference
# between running today and running in two days: SLURM's backfill scheduler can
# slot a 40-minute job into the gaps before the next big reservation, but never a
# 24-hour one. Measured on this partition with all seven 80 GB nodes held by
# other users' 2-day jobs: estimated start 2026-08-23T10:24 at 24h, immediate
# backfill at 40min.
#
# 2h, NOT the 40min originally chosen as "5x the observed worst case". That
# reasoning was wrong: across 12 lambda_f=0.01 runs, 11 finished in 7:13-8:15 and
# one took 40:29 on s-sc-gpu024 and was KILLED at the 40min limit (job 10527695).
# The runtime tail is much heavier than the median suggests -- a co-tenant job on
# the same node is enough -- and a TIMEOUT presents as a crash, leaves a
# metric-less run dir that the duplicate guard then SKIPS on retry, and silently
# drops a seed from the table. 2h still backfills far better than the wrapper's
# 24h default while covering a 15x outlier.
TIME_LIMIT="${TIME_LIMIT:-02:00:00}"

NUM_HIER="${NUM_HIER:-4}"
SID="${SID:-embeddings/beauty/merged_predictions_tensor.pt}"
EMBEDDING_PATH="${EMBEDDING_PATH:-embeddings/beauty_merged_predictions_tensor_latest.pt}"
LATENT_ROOT="${LATENT_ROOT:-embeddings/latent}"
LATENT_DIM="${LATENT_DIM:-128}"
# Explicit z override, for comparing refiners trained at different loss weights.
# The default refiner (w_nbr 1.0 / w_sid 0.1) produces a z whose neighbourhood
# overlaps N_emb at 0.778 -- only ~1.8 of 8 neighbours differ, which bounds any
# downstream effect to ~nothing. run_latent_weight_sweep.sh (job 10524127) found
# w_nbr=0.0 / w_sid=1.0 gives overlap 0.422 (4.9% of items fully disjoint) at
# UNCHANGED reconstruction loss, so the low-overlap z is the one worth testing.
#
# LATENT_TAG is appended to the arm tag so the two z's never share a run dir --
# without it the duplicate guard would treat them as the same arm and skip.
LATENT_PATH="${LATENT_PATH:-}"
LATENT_TAG="${LATENT_TAG:-}"

MODELS_MAP="${MODELS_MAP:-unpopular:9872116 mid:9872117 popular:9872118}"
jobid_for() { for kv in ${MODELS_MAP}; do [ "${kv%%:*}" = "$1" ] && { echo "${kv##*:}"; return; }; done; }

latent_path_for() {
  if [ -n "${LATENT_PATH}" ]; then echo "${LATENT_PATH}"; return; fi
  echo "${LATENT_ROOT}/${DATASET}_bw_tgt${1}_n${N_TARGET}_seed${POISON_SEED}_dz${LATENT_DIM}.pt"
}

has_arm() { case " ${ARMS} " in *" $1 "*) return 0;; *) return 1;; esac; }

[ -f "${SID}" ] || { echo "SID tensor missing: ${SID}" >&2; exit 1; }
[ -f "${EMBEDDING_PATH}" ] || { echo "dense embeddings missing: ${EMBEDDING_PATH}" >&2; exit 1; }

COUNT=0
SKIPPED=0

# submit <strategy> <rundir> <tag_suffix> [extra hydra overrides...]
submit() {
  local strat="$1" rundir="$2" suffix="$3"; shift 3
  local seedtok=""
  [ "${UNLEARN_SEED}" != "2" ] && seedtok="_useed${UNLEARN_SEED}"
  local tag="bw_tgt${strat}_${suffix}${seedtok}"
  # Duplicate guard: a second run under one tag is double-counted by every
  # extractor that averages over seeds.
  if [ "${FORCE:-0}" != "1" ]; then
    local existing
    existing="$(find logs/unlearn/runs -maxdepth 1 -name "*_bs1_${tag}" 2>/dev/null | head -1 || true)"
    if [ -n "${existing}" ]; then
      SKIPPED=$((SKIPPED + 1))
      echo "[skip] tag already exists, reusing: $(basename "${existing}")"
      return
    fi
  fi
  COUNT=$((COUNT + 1))
  echo "[$COUNT] strat=${strat} tag=${tag}"
  echo "        extra=[$*]"
  if [ "$DRYRUN" = "1" ]; then return; fi
  env POISON_METHOD=bandwagon POISONING_RATIO="$POISONING_RATIO" \
      N_TARGET_ITEMS="$N_TARGET" POISON_SEED="$POISON_SEED" \
      TARGET_STRATEGY="$strat" UNLEARN_RUN_TAG="$tag" \
      UNLEARN_SEED="$UNLEARN_SEED" NUM_HIER="$NUM_HIER" \
    sbatch --partition="${PARTITION}" --gres="${GRES}" --time="${TIME_LIMIT}" \
      ${EXCLUDE_NODES:+--exclude="${EXCLUDE_NODES}"} \
      run_tiger_unlearn_sequential.sh \
      "$rundir" "$DATASET" unified "$SID" \
      false 1 0.0 \
      unlearning.n_unlearning_chunks=10 "$@"
}

# Overrides shared by every arm: the operating point itself.
base_overrides() {
  printf '%s ' \
    "unlearning.lambda_f=${LF}" "unlearning.lambda_s=${LS}" \
    "unlearning.sep_negatives=forget_target_only" \
    "unlearning.n_epochs=${N_EPOCHS}" "unlearning.adaptive_codes=false" \
    "unlearning.neighborhood_count=${NEIGHBORHOOD_COUNT}" \
    "unlearning.neighborhood_prefix_length=${PREFIX_LENGTH}"
}

# Overrides for one L_n neighbour source.
ln_overrides() {  # <method> <latent_path> [union_size]
  local method="$1" lat="$2" usize="${3:-}"
  printf '%s ' \
    "unlearning.lambda_n=${LN}" \
    "unlearning.coherence_loss_type=${COHERENCE_LOSS_TYPE}" \
    "unlearning.coherence_rows=${COHERENCE_ROWS}" \
    "unlearning.coherence_neighbor_method=${method}"
  case "${method}" in
    embedding|embedding+latent)
      printf '%s ' \
        "unlearning.embedding_path=${EMBEDDING_PATH}" \
        "unlearning.coherence_embedding_metric=${COHERENCE_EMBEDDING_METRIC}" ;;
  esac
  case "${method}" in
    latent|embedding+latent)
      printf '%s ' "unlearning.coherence_latent_path=${lat}" ;;
  esac
  [ -n "${usize}" ] && printf '%s ' "unlearning.coherence_union_size=${usize}"
}

echo "#### REFINED ID SPACE, minimal validation"
echo "     dataset=${DATASET} H=${NUM_HIER} strata='${STRATS}' arms='${ARMS}'"
echo "     slurm: partition=${PARTITION} gres=${GRES} time=${TIME_LIMIT} exclude=${EXCLUDE_NODES:-<none>}"
echo "     operating point: lf=${LF} ls=${LS} ln=${LN} ac0 n_epochs=${N_EPOCHS}"
echo "                      count=${NEIGHBORHOOD_COUNT} loss=${COHERENCE_LOSS_TYPE} rows=${COHERENCE_ROWS}"
echo "     SID=${SID}"
echo "     x  =${EMBEDDING_PATH}"
echo "     z  =${LATENT_PATH:-${LATENT_ROOT}/${DATASET}_bw_tgt<strat>_n${N_TARGET}_seed${POISON_SEED}_dz${LATENT_DIM}.pt} tag=${LATENT_TAG:-<none>}"
echo

for strat in ${STRATS}; do
  jid="$(jobid_for "${strat}")"
  [ -n "${jid}" ] || { echo "No job id for strategy '${strat}' in '${MODELS_MAP}'" >&2; exit 1; }
  rundir="$(ls -1d logs/train/runs/*/*job${jid}_* 2>/dev/null | head -1 || true)"
  [ -n "${rundir}" ] || { echo "No training run dir for job ${jid} (${strat})" >&2; exit 1; }
  pdir="$(POISON_METHOD=bandwagon POISONING_RATIO="$POISONING_RATIO" N_TARGET_ITEMS="$N_TARGET" \
          POISON_SEED="$POISON_SEED" TARGET_STRATEGY="$strat" bash -c "
            source scripts/resolve_grid_dataset.sh
            resolve_grid_dataset '${DATASET}' >/dev/null 2>&1 && printf '%s' \"\${GRID_POISON_DATA_DIR}\"")"
  [ -d "${pdir}/training_forget" ] || \
    echo "WARNING: ${pdir}/training_forget missing — run split_forget_retain first." >&2

  lat="$(latent_path_for "${strat}")"
  needs_latent=0
  for a in latent emblatent_full emblatent_matched; do has_arm "$a" && needs_latent=1; done
  if [ "${needs_latent}" = "1" ] && [ ! -f "${lat}" ]; then
    echo "Refined latent tensor missing for ${strat}: ${lat}" >&2
    echo "Run Step 2 first:  STRATS=${strat} sbatch run_latent_refiner.sh" >&2
    exit 1
  fi

  echo "## MODEL ${strat}: train job ${jid}, run dir ${rundir}"

  # (a) L_n off — the reference point for "does a neighbourhood help at all".
  if has_arm ln0; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" "unified_lf${LF}_ls${LS}_ac0_ln0.0" \
      $(base_overrides) "unlearning.lambda_n=0.0"
  fi

  # (a2) prefix — the shipped default neighbourhood.
  if has_arm prefix; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" \
      "unified_lf${LF}_ls${LS}_ac0_ln${LN}_${COHERENCE_LOSS_TYPE}_prefix" \
      $(base_overrides) $(ln_overrides prefix "")
  fi

  # (b) N_emb — the incumbent neighbourhood.
  if has_arm embedding; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" \
      "unified_lf${LF}_ls${LS}_ac0_ln${LN}_${COHERENCE_LOSS_TYPE}_embedding" \
      $(base_overrides) $(ln_overrides embedding "")
  fi

  # (c) N_z — the refined space.
  if has_arm latent; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" \
      "unified_lf${LF}_ls${LS}_ac0_ln${LN}_${COHERENCE_LOSS_TYPE}_latent_dz${LATENT_DIM}${LATENT_TAG}" \
      $(base_overrides) $(ln_overrides latent "${lat}")
  fi

  # (d) N_emb u N_z, each source at full count.
  if has_arm emblatent_full; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" \
      "unified_lf${LF}_ls${LS}_ac0_ln${LN}_${COHERENCE_LOSS_TYPE}_emblatent_dz${LATENT_DIM}${LATENT_TAG}_full" \
      $(base_overrides) $(ln_overrides embedding+latent "${lat}" full)
  fi

  # (e) N_emb u N_z inside one count budget — the size-matched control.
  if has_arm emblatent_matched; then
    # shellcheck disable=SC2046
    submit "$strat" "$rundir" \
      "unified_lf${LF}_ls${LS}_ac0_ln${LN}_${COHERENCE_LOSS_TYPE}_emblatent_dz${LATENT_DIM}${LATENT_TAG}_matched" \
      $(base_overrides) $(ln_overrides embedding+latent "${lat}" matched)
  fi
done

echo
echo "submitted=${COUNT} skipped=${SKIPPED} (DRYRUN=${DRYRUN})"
