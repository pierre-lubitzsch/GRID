#!/usr/bin/env bash
# Unlearning sweep over the bandwagon n_target=1 poisoned beauty models, one per
# target-popularity strategy (unpopular / mid / popular). Mirrors the algorithm
# grid of run_unlearn_sweep_clone_inject.sh.
#
# Tailored for models that are STILL TRAINING:
#   * Each unlearn job is submitted with --dependency=afterok:<train_jobid> so it
#     only starts once its poisoned model finishes training successfully.
#   * The checkpoint is passed as the training RUN DIR (stable, contains the job
#     id) and resolved to the FINAL *.ckpt at RUNTIME by
#     run_tiger_unlearn_sequential.sh — so a changing/rewritten checkpoint
#     filename during training doesn't matter.
#   * TARGET_STRATEGY + POISON_METHOD=bandwagon + N_TARGET_ITEMS=1 make
#     resolve_grid_dataset point at that strategy's poison dir (forget manifest);
#     the datasets must already have training_forget/ + training_retain/ (run
#     split_forget_retain first — this script checks and warns).
#
# MODELS maps strategy -> training job id (override to retarget):
#   default: unpopular:9872116 mid:9872117 popular:9872118  (the non-PKM runs)
#
# Usage:
#   ./run_unlearn_sweep_bandwagon.sh              # submit the full grid
#   DRYRUN=1 ./run_unlearn_sweep_bandwagon.sh     # print the sbatch calls only
#   STRATS="mid" ./run_unlearn_sweep_bandwagon.sh
#   ALGOS="scif seif kookmin neg_train filter" ./run_unlearn_sweep_bandwagon.sh  # core subset
#   # PKM models (training already finished => NO_DEP=1):
#   NO_DEP=1 PKM_MODE=replace PKM_ARCH=allv2  ./run_unlearn_sweep_bandwagon.sh
#   NO_DEP=1 PKM_MODE=replace PKM_ARCH=e23d01 ./run_unlearn_sweep_bandwagon.sh
#
# Global unlearning params match the clone_flood/clone_inject sweeps:
#   neighborhood_aware=false (arg5), request_batch_size=1 (arg6, dummy),
#   neighborhood_aware_sample_rate=0.0 (arg7), unlearning.n_unlearning_chunks=10.

#
# CHECKPOINTS ARE DELETED BY DEFAULT. run_tiger_unlearn_sequential.sh removes the
# unlearned checkpoints at the end of every run (UNLEARN_KEEP_CKPT defaults to 0,
# gated on metrics.csv existing). That is right for the tables, which read only
# metrics.csv, but it makes the PLOTS impossible after the fact: drift and
# label-logprob stats come from a LATER diagnostic pass
# (src.measure_collateral_drift, src.label_logprob) that loads the unlearned
# checkpoint to score items. A cell whose checkpoint is gone cannot be plotted
# and has to be re-run.
#
# So pass UNLEARN_KEEP_CKPT=1 for any cell you intend to plot:
#
#   UNLEARN_KEEP_CKPT=1 ALGOS=unified ./run_unlearn_sweep_bandwagon.sh
#
# and reap the checkpoints AFTER the diagnostic pass, not on completion. This is
# not hypothetical: the whole beauty seed-2 ablation set finished, was reaped, and
# had to be re-submitted for exactly this reason (2026-09-04).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"

# Dataset short name (beauty default; rsc15 etc. — anything resolve_grid_dataset
# knows). SID defaults to the dataset's canonical tensor; override via SID env.
DATASET="${DATASET:-beauty}"
# Tokenizer track. rkmeans installs its tensor at embeddings/<ds>/, every other
# quantizer at embeddings/<ds>_<quantizer>/ (run_rkmeans_inference.sh), so the
# SID default follows QUANTIZER. QTOK keeps the two tracks in separate run dirs;
# it is EMPTY for rkmeans so every historical tag stays byte-identical.
QUANTIZER="${QUANTIZER:-rkmeans}"
if [ "${QUANTIZER}" = "rkmeans" ]; then
  SID="${SID:-embeddings/${DATASET}/merged_predictions_tensor.pt}"
  QTOK=""
else
  SID="${SID:-embeddings/${DATASET}_${QUANTIZER}/merged_predictions_tensor.pt}"
  QTOK="${QUANTIZER}_"
fi
# GRID_MODEL_TAG ("" for tiger, "diger_" otherwise). run_tiger_unlearn_sequential.sh
# prefixes UNLEARN_RUN_TAG with it automatically, so the duplicate guard below has
# to add it too -- guarding on the unprefixed tag would make every diger cell look
# like an already-run tiger cell and skip the whole sweep.
# shellcheck source=scripts/resolve_model.sh
source "${PWD}/scripts/resolve_model.sh"
resolve_model "${MODEL:-tiger}" || exit 1
POISON_SEED="${POISON_SEED:-2}"
# Unlearning-side seed (retain sampling + request ordering). Distinct from
# POISON_SEED, which selects the poisoned DATASET/model. Vary this one for
# replicates of a fixed attack.
# In the main grid the unlearning seed IS the data seed: one draw of the
# experiment owns its poison generation, its models and its unlearning runs, so
# there is no separate unlearn-seed axis to cross. UNLEARN_SEED therefore
# defaults to POISON_SEED and a mismatch is refused, because the two used to be
# independent and a stale UNLEARN_SEED=2 alongside POISON_SEED=7 silently
# produced a seed-2-tagged run dir against a seed-7 model.
#
# ALLOW_SEED_MISMATCH=1 restores the old behaviour for reproducing the 2026-08-04
# replicate sweeps, which held the poisoned model at seed 2 and varied only the
# retain sampling; those runs are tagged `_useed<N>` (see the seed token below).
UNLEARN_SEED="${UNLEARN_SEED:-${POISON_SEED}}"
if [ "${UNLEARN_SEED}" != "${POISON_SEED}" ] && [ "${ALLOW_SEED_MISMATCH:-0}" != "1" ]; then
  echo "UNLEARN_SEED=${UNLEARN_SEED} != POISON_SEED=${POISON_SEED}. In the grid the" >&2
  echo "unlearning seed is the data seed. Set both to the same value, or pass" >&2
  echo "ALLOW_SEED_MISMATCH=1 to run an unlearning-side replicate of the" >&2
  echo "POISON_SEED=${POISON_SEED} model (tagged _useed${UNLEARN_SEED})." >&2
  exit 1
fi
POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET="${N_TARGET:-1}"
DRYRUN="${DRYRUN:-0}"
STRATS="${STRATS:-unpopular mid popular}"
# Which algorithm families to run (default = full grid, like the clone sweeps).
# The 8 headline algorithms plus `filter` (no-weight-update reference).
# tracer needs a surviving codebook checkpoint, so it is only in the default list
# when one is configured -- see TRACER_CKPT below.
ALGOS="${ALGOS:-scif seif kookmin fanchuan unified neg_train finetune filter}"

# PKM mode of the poisoned models to unlearn: none (default) | replace | add.
# Each mode has its own training job ids + needs the PKM arch overrides at both
# unlearn and post-eval time (else the PKM checkpoint won't load).
PKM_MODE="${PKM_MODE:-none}"
# PKM_ARCH selects the LAYER LAYOUT of the checkpoints being unlearned. It must
# match how they were trained — the overrides below are what makes the ckpt
# loadable at all. Presets also carry their own train-job ids and a run-tag
# token so the arch variants don't collide in logs/unlearn/runs:
#   d01     encoder:null,    decoder:[0,1]  historical default (tag: pkm<mode>_)
#   allv2   encoder:all,     decoder:all    jobs 9985371/72/73  (tag: pkmreplaceallv2_)
#   e23d01  encoder:[2,3],   decoder:[0,1]  jobs 9985375/76/77  (tag: pkmreplacee23d01_)
# allv2/e23d01 are the 2026-07-14 reruns with the fixed encoder-FFN detection
# (the 07-10 v1 jobs either crashed or silently trained decoder-only).
# PKM_ENCODER / PKM_DECODER override the preset's selections directly.
PKM_ARCH="${PKM_ARCH:-d01}"
case "${PKM_ARCH}" in
  d01)    ARCH_ENC="null"  ; ARCH_DEC="[0,1]"; ARCH_TOK=""
          ARCHMODELS_replace="unpopular:9872119 mid:9872121 popular:9872123"
          ARCHMODELS_add="unpopular:9872120 mid:9872122 popular:9872124" ;;
  allv2)  ARCH_ENC="all"   ; ARCH_DEC="all"  ; ARCH_TOK="allv2"
          ARCHMODELS_replace="unpopular:9985371 mid:9985372 popular:9985373"
          ARCHMODELS_add="" ;;
  e23d01) ARCH_ENC="[2,3]" ; ARCH_DEC="[0,1]"; ARCH_TOK="e23d01"
          ARCHMODELS_replace="unpopular:9985375 mid:9985376 popular:9985377"
          ARCHMODELS_add="" ;;
  *) echo "PKM_ARCH must be d01|allv2|e23d01, got '${PKM_ARCH}'" >&2; exit 1 ;;
esac
PKM_ENCODER="${PKM_ENCODER:-${ARCH_ENC}}"
PKM_DECODER="${PKM_DECODER:-${ARCH_DEC}}"
# strategy -> training job id, per PKM mode
case "${PKM_MODE}" in
  none)    DEFMODELS="unpopular:9872116 mid:9872117 popular:9872118" ;;
  replace) DEFMODELS="${ARCHMODELS_replace}" ;;
  add)     DEFMODELS="${ARCHMODELS_add}" ;;
  *) echo "PKM_MODE must be none|replace|add, got '${PKM_MODE}'" >&2; exit 1 ;;
esac
if [ "${PKM_MODE}" != "none" ] && [ -z "${DEFMODELS}" ] && [ -z "${MODELS:-}" ]; then
  echo "No trained models for PKM_ARCH=${PKM_ARCH} PKM_MODE=${PKM_MODE} — pass MODELS='unpopular:<jid> ...'" >&2
  exit 1
fi
MODELS="${MODELS:-${DEFMODELS}}"
# The DEFMODELS job ids are the BEAUTY bandwagon-n1 models — for any other
# dataset an explicit strategy->train-jobid map must be passed.
if [ "${DATASET}" != "beauty" ] && [ "${MODELS}" = "${DEFMODELS}" ]; then
  echo "DATASET=${DATASET}: pass MODELS='unpopular:<jid> mid:<jid> popular:<jid>' (defaults are the beauty train jobs)" >&2
  exit 1
fi
jobid_for() { for kv in ${MODELS}; do [ "${kv%%:*}" = "$1" ] && { echo "${kv##*:}"; return; }; done; }

# PKM arch overrides (empty for none). Applied to BOTH the unlearn call (trailing
# hydra args) and the post-unlearn eval (via EVAL_EXTRA_OVERRIDES env).
PKM_OVR=()
PKM_TAG=""
if [ "${PKM_MODE}" != "none" ]; then
  PKM_OVR=("model.pkm_layers={encoder:${PKM_ENCODER},decoder:${PKM_DECODER}}" "model.pkm_mode=${PKM_MODE}")
  # ARCH_TOK is empty for d01 so the historical bw_tgt*_pkm<mode>_* tags are
  # preserved; allv2/e23d01 get their own namespace.
  PKM_TAG="pkm${PKM_MODE}${ARCH_TOK}_"
  # SCIF is incompatible with PKM: its Hessian/HVP needs double-backward, but
  # PKM's HashingMemory uses EmbeddingBag whose 2nd derivative is unimplemented
  # (NotImplementedError: _embedding_bag_backward). Drop it (all other algos are
  # first-order and work fine). Override ALLOW_SCIF_PKM=1 to force-keep it.
  if [ "${ALLOW_SCIF_PKM:-0}" != "1" ] && printf ' %s ' ${ALGOS} | grep -q ' scif '; then
    echo "NOTE: dropping 'scif' for PKM_MODE=${PKM_MODE} (HVP double-backward unsupported by PKM EmbeddingBag)." >&2
    ALGOS="$(printf '%s\n' ${ALGOS} | grep -vx scif | tr '\n' ' ')"
  fi
fi

# TRACER needs the codebook checkpoint the semantic ids were built from, plus the
# pre-quantization embeddings. The width-256 rkmeans codebook for beauty is gone,
# which is why the grid runs tracer on the RQ-VAE track; any dataset whose codebook
# checkpoint survives can also run it on rkmeans by passing TRACER_CKPT explicitly.
#
# `phi=0` must reproduce the stored ids exactly or the run refuses to start
# (tracer_tokenizer.assert_reproduces_sids). If it does, regenerate the tensor with
#   python -m scripts.assign_sids_exact --codebook_ckpt ... --embedding_path ... --out ...
# BEFORE training anything on that space.
TRACER_CKPT="${TRACER_CKPT:-}"
# Pre-quantization embeddings for tracer. The old default
# embeddings/<ds>_merged_predictions_tensor_latest.pt exists for BEAUTY ONLY, so
# tracer on any other dataset was queued against a path that does not exist and
# died on it (the module raises), leaving a run dir that then blocks its own
# resubmission. run_unlearn_sweep_sensitive.sh::emb_of resolves both layouts;
# this uses the shared helper so there is one place that knows them.
# shellcheck source=scripts/resolve_item_embeddings.sh
source "${PWD}/scripts/resolve_item_embeddings.sh"
if [ -z "${TRACER_EMB:-}" ]; then
  if resolve_item_embeddings "${DATASET}" 2>/dev/null; then
    TRACER_EMB="${GRID_ITEM_EMBEDDINGS}"
  else
    TRACER_EMB=""
  fi
fi
if [ -z "${TRACER_CKPT}" ] && [ "${QUANTIZER}" != "rkmeans" ]; then
  # Newest codebook run for this dataset+quantizer that actually kept a ckpt.
  for _d in $(ls -1dt logs/train/runs/codebook/${DATASET}_${QUANTIZER}_* 2>/dev/null); do
    _c="$(ls -1t "${_d}"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)"
    [ -n "${_c}" ] && { TRACER_CKPT="${_c}"; break; }
  done
fi
if printf ' %s ' ${ALGOS} | grep -q ' tracer ' && [ -z "${TRACER_EMB}" ]; then
  echo "algo 'tracer' requested but no pre-quantization embeddings for ${DATASET}." >&2
  echo "Run ./generate_embeddings.sh ${DATASET} or pass TRACER_EMB=<path>." >&2
  exit 1
fi
if printf ' %s ' ${ALGOS} | grep -q ' tracer '; then
  if [ -z "${TRACER_CKPT}" ]; then
    echo "algo 'tracer' requested but no codebook checkpoint found for ${DATASET}/${QUANTIZER}; pass TRACER_CKPT=<ckpt>" >&2
    exit 1
  fi
  echo "## tracer codebook: ${TRACER_CKPT}"
  echo "## tracer embeddings: ${TRACER_EMB}"
fi

# unified (lambda_f, lambda_s) grid — same as the clone sweeps
# (lambda_f, lambda_s) pairs for algo=unified, one pair per line.
# NOTE the default grid stops at lambda_f=0.01 and UNDERSTATES the method by
# 41-53% on PGR -- the 3-seed held-out result puts the operating point at
# lambda_f=0.1 (PGR@10 1.000 unpopular / 0.973 mid vs 0.406 / 0.512 here). It is
# kept as the default only so previously recorded sweeps reproduce byte for byte.
# For NEW datasets override it, e.g. UNIFIED_PAIRS_STR="0.1 0.1".
# The historical 9-pair grid is kept ONLY for the cell it was recorded on
# (tiger + rkmeans + beauty), so those sweeps still reproduce byte for byte.
# Everywhere else it would burn 18 runs per cell on a grid that tops out at
# lambda_f=0.01, i.e. below the operating point the 3-seed head-to-head selected.
if [ -z "${UNIFIED_PAIRS_STR:-}" ] && \
   { [ "${DATASET}" != "beauty" ] || [ "${QUANTIZER}" != "rkmeans" ] || [ "${MODEL:-tiger}" != "tiger" ]; }; then
  UNIFIED_PAIRS_STR="0.1 0.1"
  ADAPTIVE_CODES="${ADAPTIVE_CODES:-false}"
  echo "## unified: single operating point lambda_f=0.1 lambda_s=0.1 (override with UNIFIED_PAIRS_STR)"
fi
UNIFIED_PAIRS_STR="${UNIFIED_PAIRS_STR:-0.0 0.0
0.01 0.0
0.01 1.0
0.01 0.1
0.001 0.0
0.001 0.1
0.0 1.0
0.0 0.1
0.0 10.0}"
UNIFIED_PAIRS=()
while IFS= read -r _line; do
  [ -n "${_line}" ] && UNIFIED_PAIRS+=("${_line}")
done <<< "${UNIFIED_PAIRS_STR}"

# Which adaptive_codes settings to run per pair. ac0 = every code position
# trainable; ac1 = FREEZE levels [0, stable_codes) and train only the fine tail.
# Measured: the fine tail addresses ~1.15 items per code (90.4% of c0..c2
# prefixes are singletons) while a level-0 code is shared by ~47 items, so ac1 is
# the weak direction. Default keeps both for back-compat; ADAPTIVE_CODES="false"
# halves the job count on new datasets.
ADAPTIVE_CODES="${ADAPTIVE_CODES:-false true}"

# Snapshot of my queued/running unlearn job names, taken ONCE. The run-dir
# duplicate guard below cannot see a job that is QUEUED but has not started --
# its dir does not exist yet -- so re-running the sweep to add cells silently
# resubmits every not-yet-started arm of the cells already queued. Observed
# 2026-09-15: a 45-cell loop duplicated the 20 arms of the one cell submitted a
# moment earlier. Same fix as run_unlearn_sweep_sensitive.sh.
QUEUED_NAMES=""
if [ "${DRYRUN}" != "1" ]; then
  QUEUED_NAMES="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"
fi

COUNT=0

# submit <strategy> <jobid> <rundir> <algorithm> <tag_suffix> [extra hydra overrides...]

# GPU request per arm. run_tiger_unlearn_sequential.sh pins
# --gres=gpu:nvidia_h200:2 in its header, which queues every job behind the
# H200s even though almost nothing needs them: measured on the 2026-09-07 batch,
# 189 of 206 jobs completed on the 80GB H100 node and the only 17 failures were
# ALL sep_loss_type=generative arms, which OOM at 80GB (the generative
# separation loss scores full sequence log-probs; H200 has 141GB). pgpu01/02 are
# 40GB and stay excluded.
gres_for_arm() {
  # GEN_ANY_GPU=1 lets generative-separation arms use any >=80GB card (H100,
  # H200, A100-80) instead of pinning them to the scarcer H200s. The pin was
  # added after a batch where the only failures were sepgen arms, attributed to
  # OOM at 80GB. Re-checked 2026-09-18 over 400 generative-sep failures: ZERO
  # were CUDA OOM (all were a missing unlearning.embedding_path), and three
  # generative arms had completed on the 80GB A100 nodes pgpu06/pgpu07. The pin
  # roughly halves the eligible node pool, so it is worth lifting when queue
  # time matters -- but it stays the default until an OOM-free full batch
  # confirms it, since an OOM wastes the whole arm.
  case " $* " in
    *sepgen*|*sep_loss_type=generative*)
      if [ "${GEN_ANY_GPU:-0}" = "1" ]; then
        printf '%s' "--gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02"
      else
        printf '%s' "--gres=gpu:nvidia_h200:2"
      fi ;;
    *)
      printf '%s' "--gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02" ;;
  esac
}

submit() {
  local strat="$1" jid="$2" rundir="$3" algo="$4" suffix="$5"; shift 5
  # Tag prefix: empty for beauty (keeps the historical bw_tgt* tags); other
  # datasets get '<dataset>_' so their runs don't collide in logs/unlearn/runs.
  local dstok=""
  [ "${DATASET}" != "beauty" ] && dstok="${DATASET}_"
  # Seed token. `_seed<N>` is the DATA seed (its own poison generation, clean and
  # poisoned models); `_useed<N>` is the legacy unlearning-side replicate of the
  # seed-2 model. They must stay distinct tokens: 450 beauty run dirs already
  # carry `_useed3` / `_useed5`, so emitting `_useed3` for data seed 3 would make
  # the duplicate guard below skip every seed-3 cell as "already run", and the
  # tables would then score a seed-2 poisoned model in the seed-3 column.
  # Seed 2 emits no token at all, keeping every historical tag byte-identical.
  local seedtok=""
  if [ "${ALLOW_SEED_MISMATCH:-0}" = "1" ] && [ "${UNLEARN_SEED}" != "${POISON_SEED}" ]; then
    seedtok="_useed${UNLEARN_SEED}"
  elif [ "${POISON_SEED}" != "2" ]; then
    seedtok="_seed${POISON_SEED}"
  fi
  # Optional trailing token, empty by default. Exists so a re-run made purely to
  # KEEP a checkpoint (UNLEARN_KEEP_CKPT=1, for collateral-drift measurement)
  # lands on its own arm name instead of the sweep's. Without it, FORCE=1 would
  # write a second run dir under the SAME arm tag and the head-to-head would
  # average it in as an extra unlearning replicate -- silently shifting the
  # published numbers for exactly the cells being re-measured.
  local tag="${QTOK}${dstok}bw_tgt${strat}_${PKM_TAG}${suffix}${seedtok}${EXTRA_TAG_SUFFIX:-}"
  # Guard on the MODEL-PREFIXED tag: run_tiger_unlearn_sequential.sh prefixes
  # UNLEARN_RUN_TAG with GRID_MODEL_TAG, so the run dir carries it and an
  # unprefixed guard would match another model's run.
  # Pass the MODEL-PREFIXED tag to the runner, not the bare one. The runner adds
  # GRID_MODEL_TAG only when the tag does not already start with it -- a plain
  # string-prefix test. With QUANTIZER=letter the QTOK is "letter_", identical to
  # GRID_MODEL_TAG, so the bare tag "letter_bw_..." looked already-prefixed and
  # the MODEL token was silently dropped: 39 run dirs parsed as
  # track=letter+rkmeans instead of letter+letter. Handing over the prefixed tag
  # makes that branch the intended path for every model, and is a no-op for
  # tiger (empty tag => guard == tag) and for diger.
  local guard="${GRID_MODEL_TAG}${tag}"
  if [ "${FORCE:-0}" != "1" ] && \
     [ -n "$(find logs/unlearn/runs -maxdepth 1 -name "*_bs1_${guard}" 2>/dev/null | head -1)" ]; then
    echo "[skip] tag already exists: ${guard}"
    return
  fi
  # ...and skip if an identically-named job is already QUEUED but not started.
  if [ "${FORCE:-0}" != "1" ] && \
     printf '%s\n' "${QUEUED_NAMES}" | grep -qxF "ul_${tag}"; then
    echo "[skip] already queued: ul_${tag}"
    return
  fi
  COUNT=$((COUNT + 1))
  local depmsg="${DEP_TYPE:-afterok}:${jid}"
  [ "${NO_DEP:-0}" = "1" ] && depmsg="none(NO_DEP=1)"
  echo "[$COUNT] strat=${strat} pkm=${PKM_MODE} dep=${depmsg} algo=${algo} tag=${tag} extra=[$*]"
  if [ "$DRYRUN" = "1" ]; then return; fi
  # Dependency on the training job so unlearning waits for it. Omit it (NO_DEP=1)
  # when the training job has already finished — SLURM rejects afterok on a job
  # that has aged out of the controller ("Job dependency problem").
  local dep=()
  # DEP_TYPE (default afterok): use afterany when the parent train job is a
  # 2-day-wall resume segment — TIMEOUT is its expected exit, and afterok would
  # strand the dependents as DependencyNeverSatisfied (seen 2026-07-11 on the
  # first rsc15 chain). The job: ckpt reference resolves whatever checkpoint
  # the segment left behind, so afterany is safe.
  [ "${NO_DEP:-0}" != "1" ] && dep=(--dependency="${DEP_TYPE:-afterok}:${jid}")
  # PKM models: pass arch overrides to the unlearn call AND to the post-eval.
  local eval_ovr=""
  [ "${PKM_MODE}" != "none" ] && eval_ovr="${PKM_OVR[*]}"
  # The controller intermittently answers `Socket timed out on send/recv
  # operation`. Under `set -e` that killed the whole sweep mid-way and left a
  # partially submitted batch, which then cannot simply be re-run: the run-dir
  # duplicate guard is blind to QUEUED jobs, so a blind retry double-submits
  # everything that did land. Retrying the one call instead keeps the sweep
  # atomic. SUBMIT_TRIES=1 restores the old fail-fast behaviour.
  # See gres_for_arm: only sep_loss_type=generative arms need the 141GB H200;
  # everything else runs on any >=80GB card, which is most of the pool.
  local _gres; _gres="$(gres_for_arm "${tag}" "$@")"
  local _resources=()
  [ -n "${UNLEARN_MEM:-}" ] && _resources+=(--mem="${UNLEARN_MEM}")
  [ -n "${UNLEARN_PARTITION:-}" ] && _resources+=(--partition="${UNLEARN_PARTITION}")
  local try=1 rc=0
  while :; do
    POISON_METHOD=bandwagon POISONING_RATIO="$POISONING_RATIO" N_TARGET_ITEMS="$N_TARGET" \
    POISON_SEED="$POISON_SEED" TARGET_STRATEGY="$strat" UNLEARN_RUN_TAG="$guard" \
    UNLEARN_SEED="$UNLEARN_SEED" \
    EVAL_EXTRA_OVERRIDES="${eval_ovr}" \
      sbatch ${_gres} "${_resources[@]}" --job-name="ul_${tag}" --time="${UNLEARN_TIME:-02:00:00}" "${dep[@]}" run_tiger_unlearn_sequential.sh \
        "$rundir" "$DATASET" "$algo" "$SID" \
        false 1 0.0 \
        unlearning.n_unlearning_chunks=10 "$@" ${PKM_OVR[@]+"${PKM_OVR[@]}"} && rc=0 || rc=$?
    [ "${rc}" = "0" ] && break
    # A socket timeout is reported by sbatch AFTER the controller may already
    # have accepted the job, so a blind retry duplicates it (that is exactly how
    # bw_tgtpopular_seif_std0.000006_useed3 ended up submitted twice, as jobs
    # 10505711 and 10505712). The --job-name above makes a queued job
    # identifiable, so check before retrying.
    sleep 5
    if [ -n "$(squeue -u "$USER" -n "ul_${tag}" -h 2>/dev/null)" ]; then
      echo "  sbatch reported rc=${rc} for ${tag} but the job IS queued; not retrying" >&2
      break
    fi
    if [ "${try}" -ge "${SUBMIT_TRIES:-5}" ]; then
      echo "  sbatch failed ${try}x for ${tag}; giving up" >&2
      return "${rc}"
    fi
    echo "  sbatch failed (rc=${rc}) for ${tag}; retry ${try}/${SUBMIT_TRIES:-5} in $((try * 10))s" >&2
    sleep $((try * 10))
    try=$((try + 1))
  done
}

has_algo() { case " ${ALGOS} " in *" $1 "*) return 0;; *) return 1;; esac; }

for strat in ${STRATS}; do
  jid="$(jobid_for "${strat}")"
  if [ -z "${jid}" ]; then echo "No job id for strategy '${strat}' in MODELS='${MODELS}'" >&2; exit 1; fi
  # '|| true': under pipefail a non-matching glob makes ls kill the whole
  # script before the pending-job fallback below can run.
  rundir="$(ls -1d logs/train/runs/*/*job${jid}_* 2>/dev/null | head -1 || true)"
  if [ -z "${rundir}" ]; then
    if [ "${NO_DEP:-0}" = "1" ]; then
      echo "No training run dir for job ${jid} (strategy ${strat})" >&2; exit 1
    fi
    # Train job still PENDING (dir is created when it starts). Pass the job-id
    # reference instead; run_tiger_unlearn_sequential.sh resolves it to the run
    # dir + final ckpt at RUNTIME, after the afterok dependency fires.
    rundir="job:${jid}"
    echo "## train job ${jid} pending — passing '${rundir}' (resolved at unlearn runtime)"
  fi
  # Poison dir sanity via the resolver (dataset + strategy aware): needs the
  # forget/retain split (warn, don't hard-fail — a chained gen job may still
  # be creating it).
  pdir="$(POISON_METHOD=bandwagon POISONING_RATIO="$POISONING_RATIO" N_TARGET_ITEMS="$N_TARGET" \
          POISON_SEED="$POISON_SEED" TARGET_STRATEGY="$strat" bash -c "
            source scripts/resolve_grid_dataset.sh
            resolve_grid_dataset '${DATASET}' >/dev/null 2>&1 && printf '%s' \"\${GRID_POISON_DATA_DIR}\"")"
  if [ ! -d "${pdir}/training_forget" ]; then
    echo "WARNING: ${pdir}/training_forget missing — run split_forget_retain first (unlearn will fail without it, unless a dependency-chained poison job creates it)." >&2
  fi
  echo "## MODEL ${strat}: train job ${jid}, run dir ${rundir}"

  # CUSTOM_ARMS_FILE names arms explicitly, one per line:
  #
  #     <algo> <TAB> <tag> <TAB> <hydra override> [<hydra override> ...]
  #
  # This exists because the grids below OWN their tag format, so they cannot
  # reproduce an arm that was recorded under a different one. Concretely:
  # `seif_std0.000006` was run before the seif grid gained its `_rlr` suffix, and
  # re-running it through the grid would emit `seif_std0.000006_rlr0.0007` -- a
  # new arm name for an identical config, which then fails to pool with the runs
  # already on disk. Lines starting with # and blank lines are skipped. Takes
  # precedence over HPSEARCH and the default arms for this stratum.
  if [ -n "${CUSTOM_ARMS_FILE:-}" ]; then
    if [ ! -f "${CUSTOM_ARMS_FILE}" ]; then
      echo "CUSTOM_ARMS_FILE=${CUSTOM_ARMS_FILE} not found" >&2; exit 1
    fi
    while IFS=$'\t' read -r _algo _tag _ovr; do
      case "${_algo}" in ''|'#'*) continue ;; esac
      if [ -z "${_tag}" ] || [ -z "${_ovr}" ]; then
        echo "CUSTOM_ARMS_FILE: malformed line for algo='${_algo}' (need algo<TAB>tag<TAB>overrides)" >&2
        exit 1
      fi
      has_algo "${_algo}" || continue
      # shellcheck disable=SC2086
      submit "$strat" "$jid" "$rundir" "${_algo}" "${_tag}" ${_ovr}
    done < "${CUSTOM_ARMS_FILE}"
    continue
  fi

  # HPSEARCH=1 replaces the fixed per-algorithm arms below with env-overridable
  # GRIDS, for the baseline hyper-parameter search. Gated rather than merged so
  # every previously recorded sweep reproduces byte for byte (the default arms
  # carry bare tags like "scif" that the extractors key on).
  if [ "${HPSEARCH:-0}" = "1" ]; then
    if has_algo scif; then
      for dmp in ${SCIF_DAMPINGS:-0.001 0.01 0.1}; do
        for umn in ${SCIF_MAX_NORMS:-0.5 1.0 5.0}; do
          submit "$strat" "$jid" "$rundir" scif "scif_d${dmp}_n${umn}" \
            unlearning.damping="$dmp" unlearning.update_max_norm="$umn"
        done
      done
    fi
    if has_algo seif; then
      for std in ${SEIF_STDS:-0.006 0.06 0.6}; do
        for rlr in ${SEIF_REPAIR_LRS:-0.0007 0.007}; do
          submit "$strat" "$jid" "$rundir" seif "seif_std${std}_rlr${rlr}" \
            unlearning.seif_erase_std="$std" unlearning.seif_repair_lr="$rlr"
        done
      done
    fi
    if has_algo kookmin; then
      for ir in ${KOOKMIN_IRS:-0.0001 0.001 0.01}; do
        for rlr in ${KOOKMIN_RETAIN_LRS:-0.001 0.01}; do
          submit "$strat" "$jid" "$rundir" kookmin "kookmin_ir${ir}_rlr${rlr}" \
            unlearning.kookmin_init_rate="$ir" unlearning.kookmin_retain_lr="$rlr"
        done
      done
    fi
    if has_algo fanchuan; then
      for t in ${FANCHUAN_TEMPS:-0.007 0.07 1.15}; do
        for flr in ${FANCHUAN_LRS:-0.001 0.01}; do
          submit "$strat" "$jid" "$rundir" fanchuan "fanchuan_t${t}_lr${flr}" \
            unlearning.fanchuan_contrastive_temperature="$t" \
            unlearning.fanchuan_lr="$flr"
        done
      done
    fi
    if has_algo neg_train; then
      # NEG_TRAIN_STEPS exists so the step BUDGET can be matched to unified.
      # unified's budget is n_epochs * min(n_forget_batches, n_retain_batches),
      # and on spam the forget set is one target item that fits in a single
      # batch -- so unified takes 4 optimizer steps while neg_train's default is
      # 200 and finetune's is 500. The baselines were being run for 50-125x more
      # optimization on the same objective, which is most of why their UR
      # collapses. Unset keeps the historical tag (no _s token) byte-identical.
      for lr in ${NEG_TRAIN_LRS:-0.0001 0.001 0.01}; do
        for ev in ${NEG_RETAIN_EVERY:-1 5}; do
          for nst in ${NEG_TRAIN_STEPS:-}; do :; done
          if [ -z "${NEG_TRAIN_STEPS:-}" ]; then
            submit "$strat" "$jid" "$rundir" neg_train "negtrain_lr${lr}_re${ev}" \
              unlearning.neg_train_lr="$lr" unlearning.neg_retain_every="$ev"
          else
            for nst in ${NEG_TRAIN_STEPS}; do
              submit "$strat" "$jid" "$rundir" neg_train \
                "negtrain_lr${lr}_re${ev}_s${nst}" \
                unlearning.neg_train_lr="$lr" unlearning.neg_retain_every="$ev" \
                unlearning.neg_train_steps="${nst}"
            done
          fi
        done
      done
    fi
    if has_algo finetune; then
      for lr in ${FINETUNE_LRS:-0.0001 0.001 0.01}; do
        for st in ${FINETUNE_STEPS:-200 500}; do
          submit "$strat" "$jid" "$rundir" finetune "finetune_lr${lr}_s${st}" \
            unlearning.finetune_lr="$lr" unlearning.finetune_steps="$st"
        done
      done
    fi
    if has_algo unified; then
      # 2026-08-26 REDESIGN. The previous grid searched lambda_f x lambda_s and
      # then lambda_n, but left `sep_loss_type` at its config default `cosine`
      # for all 37 arms. Cosine scores a pooled-encoder geometry the decoder
      # never reads, so the model satisfies it by moving embeddings without
      # changing what it generates: measured on beauty/mid, lambda_s over four
      # orders of magnitude moves PGR only 0.979 -> 0.991, whereas `generative`
      # (InfoNCE on the model's own sequence log-probs) reaches PGR 1.000 at
      # lambda_s >= 0.1 at the same UR. The old search therefore selected
      # lambda_s=0.01 on a term that does essentially nothing.
      #
      # So sep_loss_type is now a SEARCHED axis, and lambda_f is pinned at 0.1,
      # the value the old lf x ls grid picked unanimously on all three datasets.
      for sep in ${SEP_LOSS_TYPES:-cosine generative}; do
        case "${sep}" in
          generative) _septok="gen"; _sepovr=(unlearning.sep_loss_type=generative
                                              unlearning.sep_gen_temperature=1.0) ;;
          cosine)     _septok="cos"; _sepovr=(unlearning.sep_loss_type=cosine) ;;
          # cosine at tau=1 instead of its 0.07 default. Without this the
          # cosine-vs-generative comparison confounds TWO changes, the score
          # function and the temperature; this arm isolates the score function
          # by matching generative's tau. Note the temperatures are separate on
          # purpose: cosine logits are bounded to +-1/tau by L2 normalisation
          # while sequence log-probs are unbounded below, so tau=1 makes cosine
          # logits 14x flatter than at 0.07 and may simply weaken the term.
          cosine_t1)  _septok="cos1"; _sepovr=(unlearning.sep_loss_type=cosine
                                               unlearning.sep_temperature=1.0) ;;
          *) echo "unknown SEP_LOSS_TYPES entry '${sep}'" >&2; exit 1 ;;
        esac
        for ls in ${UNIFIED_LSS:-0.0 0.01 0.1 1.0}; do
          for ln in ${UNIFIED_LNS:-0.0 0.01 0.1 1.0}; do
            # lambda_n > 0 with `mass` is the REPAIR direction, which is what
            # spam wants: probability freed by forgetting flows back to the
            # target's embedding neighbours. Sensitive uses the opposite sign
            # and lives in run_unlearn_sweep_sensitive.sh.
            _nb=()
            if [ "${ln}" != "0.0" ] && [ "${ln}" != "0" ]; then
              if [ -z "${TRACER_EMB}" ]; then
                echo "  unified lambda_n=${ln} needs item embeddings for ${DATASET}; skipping" >&2
                continue
              fi
              _nb=(unlearning.lambda_n="${ln}"
                   unlearning.coherence_loss_type="${LOSS_TYPE:-mass}"
                   unlearning.coherence_neighbor_method=embedding
                   unlearning.coherence_rows=target_only
                   unlearning.coherence_embedding_metric=cosine
                   unlearning.neighborhood_count="${NBR_COUNT:-8}"
                   "unlearning.embedding_path=${TRACER_EMB}")
            else
              _nb=(unlearning.lambda_n=0.0)
            fi
            submit "$strat" "$jid" "$rundir" unified \
              "unified_hp2_sep${_septok}_lf${UNIFIED_LF:-0.1}_ls${ls}_ln${ln}" \
              unlearning.lambda_f="${UNIFIED_LF:-0.1}" unlearning.lambda_s="$ls" \
              unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4 \
              unlearning.adaptive_codes=false \
              "${_sepovr[@]}" ${_nb[@]+"${_nb[@]}"}
          done
        done
      done
    fi

    if has_algo tracer; then
      # TRACER_EPOCHS matches TRACER's step budget to unified's. Both compute
      # steps = n_epochs * min(n_forget_batches, n_retain_batches) from the same
      # formula (tracer.py per_pass == unified.py optim_steps_per_pass), so
      # n_epochs=4 is exactly unified's 4-step budget. Left unset TRACER uses
      # tracer_steps=500, i.e. 125x more optimization than the method it is
      # being compared against. Unset keeps the historical tag unchanged.
      _tep=(); _teptok=""
      if [ -n "${TRACER_EPOCHS:-}" ]; then
        _tep=(unlearning.n_epochs="${TRACER_EPOCHS}")
        _teptok="_ep${TRACER_EPOCHS}"
      fi
      for l1 in ${TRACER_LFS:-0.1 1.0}; do
        for l2 in ${TRACER_LCS:-0.1 1.0}; do
          for plr in ${TRACER_PHI_LRS:-0.001 0.01}; do
            submit "$strat" "$jid" "$rundir" tracer \
              "tracer_lf${l1}_lc${l2}_plr${plr}${_teptok}" \
              ${_tep[@]+"${_tep[@]}"} \
              "unlearning.tracer_codebook_ckpt=${TRACER_CKPT}" \
              "unlearning.embedding_path=${TRACER_EMB}" \
              unlearning.tracer_lambda_forget="$l1" \
              unlearning.tracer_lambda_coherence="$l2" \
              unlearning.tracer_phi_lr="$plr" ${TRACER_ARGS:-}
          done
        done
      done
    fi
    if has_algo filter; then
      # No weight update, so nothing to tune beyond the mode; kept in the search
      # so every algorithm has a documented selection.
      for fm in global user_dependent; do
        submit "$strat" "$jid" "$rundir" filter "filter_${fm}" unlearning.filter_mode="$fm"
      done
    fi
  else
  has_algo scif      && submit "$strat" "$jid" "$rundir" scif "scif"
  if has_algo seif; then
    # SEIF_STDS: noise stds to run (env-overridable, e.g. SEIF_STDS="0.0006 0.00006"
    # to extend an existing sweep downward without resubmitting the defaults).
    for std in ${SEIF_STDS:-0.006 0.06}; do submit "$strat" "$jid" "$rundir" seif "seif_std${std}" unlearning.seif_erase_std="$std"; done
  fi
  if has_algo kookmin; then
    for ir in 0.01 0.001 0.0001; do submit "$strat" "$jid" "$rundir" kookmin "kookmin_ir${ir}" unlearning.kookmin_init_rate="$ir"; done
  fi
  if has_algo fanchuan; then
    for t in 0.07 0.007; do submit "$strat" "$jid" "$rundir" fanchuan "fanchuan_t${t}" unlearning.fanchuan_contrastive_temperature="$t"; done
  fi
  fi  # end HPSEARCH gate
  if [ "${HPSEARCH:-0}" = "1" ]; then
    continue   # the fixed arms below are the frozen-config sweep, not the search
  fi
  if has_algo unified; then
    # UNIFIED_CONFIGS: one "ls ln sep" triple per line, the frozen operating
    # points selected by the sep-scoring search. sep is the sep_loss_type
    # (generative or cosine) and rides in the tag so the two are separable in
    # the tables. Unset falls back to the historical UNIFIED_PAIRS behaviour.
    # UNIFIED_ABLATION: "lf ls ln lr" quadruples, one per line. Runs the
    # ascent/descent baselines THROUGH this objective, so the comparison
    # against unified carries no implementation difference -- same batches,
    # same batch size, same optimizer, same step budget, one accumulated
    # opt.step() instead of alternating updates. lambda_r is the retain weight.
    #   0.0 0.0 0.0 1.0  -> retain only            (= finetune)
    #   0.1 0.0 0.0 1.0  -> ascent + repair        (= neg_train, simultaneous)
    #   0.1 0.0 0.0 0.0  -> ascent, no repair      (= pure gradient ascent)
    if [ -n "${UNIFIED_ABLATION:-}" ]; then
      while IFS= read -r _q; do
        [ -z "${_q}" ] && continue
        read -r _lf _ls _ln _lr <<< "${_q}"
        _rtok=""
        [ "${_lr}" != "1.0" ] && _rtok="_lr${_lr}"
        # ABL_NAU_FORM=1 runs the grid in the HEADLINE NAU configuration
        # (generative separation score + embedding/mass neighbourhood) instead
        # of the Hydra defaults (cosine + prefix/nll) this branch otherwise
        # inherits. Those defaults are inert wherever the corresponding weight
        # is 0, so only the 16 of 20 grid points with lambda_s>0 or
        # lambda_n!=0 actually differ -- but for those the tag alone does not
        # say which objective was run, which is how a cosine/nll grid ended up
        # reported beside generative NAU rows. The token keeps the two sets in
        # separate run dirs.
        _nau=()
        _nautok=""
        if [ "${ABL_NAU_FORM:-0}" = "1" ]; then
          _nautok="_nauform"
          _nau=(unlearning.sep_loss_type=generative)
          if [ "${_ln}" != "0.0" ] && [ "${_ln}" != "0" ]; then
            # embedding neighbours need the pre-quantization item embeddings;
            # without unlearning.embedding_path the run dies in
            # _build_coherence_neighbors AFTER loading the model, so the whole
            # arm is wasted. Same guard the UNIFIED_CONFIGS branch uses.
            if [ -z "${TRACER_EMB}" ]; then
              echo "  NAU-form lambda_n=${_ln} needs item embeddings for ${DATASET}; skipping" >&2
              continue
            fi
            _nau+=(unlearning.coherence_loss_type="${LOSS_TYPE:-mass}"
                   unlearning.coherence_neighbor_method=embedding
                   unlearning.coherence_rows=target_only
                   unlearning.coherence_embedding_metric=cosine
                   unlearning.neighborhood_count="${NBR_COUNT:-8}"
                   "unlearning.embedding_path=${TRACER_EMB}")
          fi
        fi
        submit "$strat" "$jid" "$rundir" unified \
          "unified_abl_lf${_lf}_ls${_ls}_ln${_ln}${_rtok}${_nautok}" \
          unlearning.lambda_f="${_lf}" unlearning.lambda_s="${_ls}" \
          unlearning.lambda_n="${_ln}" unlearning.lambda_r="${_lr}" \
          unlearning.sep_negatives=forget_target_only \
          unlearning.n_epochs="${UNIFIED_EPOCHS:-4}" \
          unlearning.adaptive_codes=false \
          ${_nau[@]+"${_nau[@]}"}
      done <<< "${UNIFIED_ABLATION}"
    elif [ -n "${UNIFIED_CONFIGS:-}" ]; then
      while IFS= read -r _cfg; do
        [ -z "${_cfg}" ] && continue
        read -r _ls _ln _sep <<< "${_cfg}"
        _lf="${UNIFIED_LF:-0.1}"
        uni_extra=(
          unlearning.lambda_f="${_lf}" unlearning.lambda_s="${_ls}"
          unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4
          unlearning.adaptive_codes=false
          unlearning.sep_loss_type="${_sep}"
        )
        _septok="_sep${_sep}"
        if [ "${_ln}" != "0" ] && [ "${_ln}" != "0.0" ]; then
          if [ -z "${TRACER_EMB}" ]; then
            echo "  unified lambda_n=${_ln} needs item embeddings for ${DATASET}; skipping" >&2
            continue
          fi
          uni_extra+=(
            unlearning.lambda_n="${_ln}"
            unlearning.coherence_loss_type="${LOSS_TYPE:-mass}"
            unlearning.coherence_neighbor_method=embedding
            unlearning.coherence_rows=target_only
            unlearning.coherence_embedding_metric=cosine
            unlearning.neighborhood_count="${NBR_COUNT:-8}"
            "unlearning.embedding_path=${TRACER_EMB}"
          )
        else
          uni_extra+=(unlearning.lambda_n=0.0)
        fi
        submit "$strat" "$jid" "$rundir" unified \
          "unified_lf${_lf}_ls${_ls}_ac0_ln${_ln}${_septok}" "${uni_extra[@]}"
      done <<< "${UNIFIED_CONFIGS}"
    else
      for pair in "${UNIFIED_PAIRS[@]}"; do
        read -r lf ls <<< "$pair"
        for ac in ${ADAPTIVE_CODES}; do
          acflag=$([ "$ac" = "true" ] && echo 1 || echo 0)
          uni_extra=(
            unlearning.lambda_f="$lf" unlearning.lambda_s="$ls"
            unlearning.sep_negatives=forget_target_only unlearning.n_epochs=4
            unlearning.adaptive_codes="$ac"
          )
          [ "$ac" = "true" ] && uni_extra+=(unlearning.stable_codes=2)
          # UNIFIED_LN freezes the neighbourhood term for the grid sweep. The
          # frozen arm had no lambda_n hook at all, so it could not express the
          # configuration the search selects. Unset (the default) passes nothing
          # and keeps every historical tag byte-identical
          # (`unified_lf0.1_ls0.1_ac0`); set, it appends `_ln<val>` and the
          # embedding-neighbourhood overrides the term needs.
          _lntok=""
          if [ -n "${UNIFIED_LN:-}" ] && [ "${UNIFIED_LN}" != "0" ] && [ "${UNIFIED_LN}" != "0.0" ]; then
            if [ -z "${TRACER_EMB}" ]; then
              echo "  unified lambda_n=${UNIFIED_LN} needs item embeddings for ${DATASET}; skipping" >&2
              continue
            fi
            _lntok="_ln${UNIFIED_LN}"
            uni_extra+=(
              unlearning.lambda_n="${UNIFIED_LN}"
              unlearning.coherence_loss_type="${LOSS_TYPE:-mass}"
              unlearning.coherence_neighbor_method=embedding
              unlearning.coherence_rows=target_only
              unlearning.coherence_embedding_metric=cosine
              unlearning.neighborhood_count="${NBR_COUNT:-8}"
              "unlearning.embedding_path=${TRACER_EMB}"
            )
          elif [ -n "${UNIFIED_LN:-}" ]; then
            _lntok="_ln${UNIFIED_LN}"
            uni_extra+=(unlearning.lambda_n="${UNIFIED_LN}")
          fi
          submit "$strat" "$jid" "$rundir" unified "unified_lf${lf}_ls${ls}_ac${acflag}${_lntok}" "${uni_extra[@]}"
        done
      done
    fi
  fi
  has_algo neg_train && submit "$strat" "$jid" "$rundir" neg_train "negtrain"
  has_algo finetune  && submit "$strat" "$jid" "$rundir" finetune "finetune"
  if has_algo tracer; then
    # TRACER_EPOCHS has to reach the FROZEN arm too, not only the hpsearch grid
    # above. Without it this submit falls back to tracer_steps=500 while unified
    # uses n_epochs=4: with n_forget=n_retain=1 that is 500 passes over a single
    # batch, so TRACER both overfits (retain loss falls monotonically for 500
    # steps) and gets 125x the optimization of the method it is compared against,
    # which also makes its wall-clock column meaningless. The tag carries the
    # token so a corrected run does not collide with a 500-step one.
    _tepf=(); _tepftok=""
    if [ -n "${TRACER_EPOCHS:-}" ]; then
      _tepf=(unlearning.n_epochs="${TRACER_EPOCHS}")
      _tepftok="_ep${TRACER_EPOCHS}"
    fi
    submit "$strat" "$jid" "$rundir" tracer "tracer${_tepftok}" \
      ${_tepf[@]+"${_tepf[@]}"} \
      "unlearning.tracer_codebook_ckpt=${TRACER_CKPT}" \
      "unlearning.embedding_path=${TRACER_EMB}"
  fi
  if has_algo filter; then
    for fm in global user_dependent; do submit "$strat" "$jid" "$rundir" filter "filter_${fm}" unlearning.filter_mode="$fm"; done
  fi
done

echo "----"
echo "Total unlearn jobs: $COUNT  (DRYRUN=$DRYRUN, STRATS='$STRATS', ALGOS='$ALGOS')"
