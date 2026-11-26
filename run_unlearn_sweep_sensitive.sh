#!/usr/bin/env bash
# Sensitive-category unlearning sweep: the second setting of the main grid.
#
# Generalizes run_sensitive_suppress.sh, which was unified-only, two datasets and
# hardcoded checkpoint paths, to the full grid: 8 algorithms x 4 datasets x 5
# seeds x {tiger,diger} x {rkmeans,rqvae}. Mirrors run_unlearn_sweep_bandwagon.sh
# so the two settings share their conventions (MODELS map of seed -> train job id,
# duplicate guard on the model-prefixed tag, DRYRUN, FORCE, HPSEARCH).
#
# WHAT MAKES THIS DIFFERENT FROM SPAM. The forget set is real users' real
# interactions with a category, not injected fake users, so:
#   * the reference is a RETRAIN model (trained on the data minus exactly those
#     interactions), not the clean model -- see scripts/build_retrain_dataset.py;
#   * deletion_spec=item_pairs WITHOUT unlearn_whole_items, which is ERASE's
#     per-user request rather than a global ban;
#   * the neighbourhood IS the concept, so a repair-shaped neighbourhood term
#     undoes the deletion. Measured on beauty: lambda_n=0.1 with
#     coherence_loss_type=mass returns Sensitive@10 to 0.03844 against a
#     pre-unlearning 0.03840. `suppress` is the form that drains it instead.
#
# Usage:
#   DRYRUN=1 ./run_unlearn_sweep_sensitive.sh
#   DATASETS=beauty SEEDS=2 ALGOS="unified tracer" ./run_unlearn_sweep_sensitive.sh
#   HPSEARCH=1 DATASETS=beauty SEEDS=2 ./run_unlearn_sweep_sensitive.sh
#   MODEL=diger QUANTIZER=rqvae ./run_unlearn_sweep_sensitive.sh
#
# MODELS maps "<dataset>:<seed>:<base_train_jobid>:<retrain_jobid>". The base job
# is the model being unlearned (the clean model, which still contains the
# category); the retrain job is the reference the metrics are scored against.

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

DRYRUN="${DRYRUN:-0}"
DATASETS="${DATASETS:-beauty toys sports food_k5}"
SEEDS="${SEEDS:-2 3 5 7 11}"
ALGOS="${ALGOS:-scif seif kookmin fanchuan unified neg_train finetune tracer filter}"
FORGET_RATIO="${FORGET_RATIO:-1e-4}"
CHUNKS="${CHUNKS:-5}"

QUANTIZER="${QUANTIZER:-rkmeans}"
QTOK=""
[ "${QUANTIZER}" != "rkmeans" ] && QTOK="${QUANTIZER}_"

# GRID_MODEL_TAG ("" for tiger, "diger_" otherwise); the runner prefixes
# UNLEARN_RUN_TAG with it, so the guard below has to as well.
# shellcheck source=scripts/resolve_model.sh
source "${PWD}/scripts/resolve_model.sh"
resolve_model "${MODEL:-tiger}" || exit 1

# Sensitive category per dataset.
#
# toys uses "Baby & Toddler Toys" (user's call, 2026-08-19), which also makes the
# recorded seed-2 toys runs reusable. Note it is a much more recommendable concept
# than the others: 399 items / 3.3% of catalog and a base Sensitive@10 of ~0.177
# against beauty's ~0.032, so the two are NOT on the same scale and the removal
# percentages, not the absolute values, are what compare across datasets.
category_of() {
  case "$1" in
    beauty)  echo "hairloss" ;;   # Hair Loss Products, level 2, 106 items, 0.88%
    toys)    echo "toyweapons" ;; # keyword concept, 68 items, 0.37% of interactions
    sports)  echo "guns" ;;       # Guns & Rifles, level 3, 202 items, 1.10%
    # ERASE names two sensitive scenarios, meat for vegetarians and alcohol for
    # users with addiction issues. Alcohol is unusable on this catalog: after the
    # k-core-5 filter "Alcoholic Beverages" holds 20 items, 0.012% of 167,428, so
    # Sensitive@k has no headroom to show removal. Meat & Seafood is the other
    # ERASE scenario and is correctly sized at 1.14%.
    food_k5) echo "meat" ;;       # Meat & Seafood, level 1, 1904 items, 1.14%
    *) echo "unknown" ;;
  esac
}

data_root_of() { echo "src/data/amazon_data"; }

MODELS="${MODELS:-}"
jobids_for() {   # dataset seed -> "<base_jid> <retrain_jid>"
  local key="$1:$2"
  for kv in ${MODELS}; do
    case "${kv}" in
      "${key}:"*) local r="${kv#${key}:}"; echo "${r%%:*} ${r#*:}"; return ;;
    esac
  done
}

COUNT=0
SKIPPED=0

# Snapshot of my queued/running unlearn job names, taken ONCE (a squeue call per
# arm would be ~435 calls). The run-dir duplicate guard below cannot see a job
# that is queued but has not started -- its dir does not exist yet -- so
# re-running the sweep to fill in later cells silently resubmits every
# not-yet-started arm of the cells already queued. Guarding on the job name too
# closes that window.
QUEUED_NAMES="$(squeue -u "${USER}" -h -o '%j' 2>/dev/null || true)"


# GPU request per arm. run_tiger_unlearn_sequential.sh pins
# --gres=gpu:nvidia_h200:2 in its header, which queues every job behind the
# H200s even though almost nothing needs them: measured on the 2026-09-07 batch,
# 189 of 206 jobs completed on the 80GB H100 node and the only 17 failures were
# ALL sep_loss_type=generative arms, which OOM at 80GB (the generative
# separation loss scores full sequence log-probs; H200 has 141GB). So ask for
# H200 only for those and let everything else use any >=80GB card. pgpu01/02 are
# 40GB and stay excluded.
gres_for_arm() {
  case " $* " in
    *sepgen*|*sep_loss_type=generative*)
      printf '%s' "--gres=gpu:nvidia_h200:2" ;;
    *)
      printf '%s' "--gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02" ;;
  esac
}

# submit <dataset> <seed> <base_jid> <algo> <tag_suffix> [extra hydra overrides...]
submit() {
  local ds="$1" sd="$2" jid="$3" algo="$4" suffix="$5"; shift 5
  local cat; cat="$(category_of "${ds}")"
  local root; root="$(data_root_of)"
  # `_seed<N>` is the data seed; the legacy `_useed<N>` token means an
  # unlearning-side replicate of the seed-2 model and 16 such run dirs already
  # exist here, so the grid must not reuse that spelling (the duplicate guard
  # below would skip those cells and the table would mis-attribute them).
  local seedtok=""
  [ "${sd}" != "2" ] && seedtok="_seed${sd}"
  local tag="${QTOK}sens_${ds}_${cat}_${suffix}${seedtok}"
  # Guard on the MODEL-PREFIXED tag, and hand THAT to the runner (see below).
  local guard="${GRID_MODEL_TAG}${tag}"
  if [ "${FORCE:-0}" != "1" ] && \
     [ -n "$(find logs/unlearn/runs -maxdepth 1 -name "*_bs1_${guard}" 2>/dev/null | head -1)" ]; then
    echo "[skip] tag already exists: ${guard}"
    SKIPPED=$((SKIPPED + 1))
    return
  fi
  # ...and skip if an identically-named job is ALREADY QUEUED but not yet
  # started (no run dir to find). submit() names jobs "ul_<tag>".
  if [ "${FORCE:-0}" != "1" ] && \
     printf '%s\n' "${QUEUED_NAMES}" | grep -qxF "ul_${tag}"; then
    echo "[skip] already queued: ul_${tag}"
    SKIPPED=$((SKIPPED + 1))
    return
  fi

  local data="${root}/${ds}_sens_${cat}_${FORGET_RATIO}_seed${sd}"
  local man="${root}/${ds}/forget_manifest_${cat}_${FORGET_RATIO}_seed${sd}.json"
  local evaldir="${root}/${ds}"
  local sid; sid="$(sid_of "${ds}")"
  local emb; emb="$(emb_of "${ds}")"

  if [ ! -d "${data}" ]; then
    echo "  MISSING dataset ${data} -- build it with scripts/build_sensitive_manifest.py + build_retrain_dataset.py" >&2
    return
  fi

  COUNT=$((COUNT + 1))
  echo "[$COUNT] ds=${ds} seed=${sd} algo=${algo} tag=${tag} extra=[$*]"
  [ "${DRYRUN}" = "1" ] && return

  local _gres; _gres="$(gres_for_arm "${tag}" "$@")"
  # Optional SLURM dependency on the base training job, mirroring
  # run_unlearn_sweep_bandwagon.sh. The sensitive sweep historically assumed the
  # base model was ALREADY trained (it resolves `job:<id>` to a run dir at
  # runtime), which is true when re-running an existing cell but not when the
  # campaign trains and unlearns in one pass -- there the arm starts, finds no
  # checkpoint, and fails. DEP_JID is the training job to wait for; leave it
  # empty to keep the old no-dependency behaviour exactly.
  local dep=()
  [ -n "${DEP_JID:-}" ] && dep=(--dependency="${DEP_TYPE:-afterok}:${DEP_JID}")
  # shellcheck disable=SC2086  # _gres is intentionally word-split
  # Pass the MODEL-PREFIXED tag to the runner, not the bare one. The runner adds
  # GRID_MODEL_TAG only when the tag does not already start with it -- a plain
  # string-prefix test. With QUANTIZER=letter the QTOK is "letter_", identical to
  # GRID_MODEL_TAG, so the bare tag "letter_sens_..." looks already-prefixed and
  # the MODEL token is silently dropped: the run dir parses as
  # track=letter+rkmeans instead of letter+letter. Same fix as
  # run_unlearn_sweep_bandwagon.sh, where it cost 39 mis-parsed run dirs. No-op
  # for tiger (empty tag => guard == tag) and for diger.
  UNLEARN_RUN_TAG="${guard}" UNLEARN_SEED="${sd}" \
  UNLEARN_EVAL_DATA_DIR="${evaldir}" \
    sbatch ${_gres} --job-name="ul_${tag}" "${dep[@]}" run_tiger_unlearn_sequential.sh \
      "job:${jid}" "${data}" "${algo}" "${sid}" \
      false 1 0.0 \
      "spam_forget_manifest=${man}" \
      unlearning.deletion_spec=item_pairs \
      unlearning.include_context_rows=true \
      unlearning.request_user_order=sorted \
      unlearning.n_unlearning_chunks="${CHUNKS}" \
      "unlearning.embedding_path=${emb}" \
      "$@"
}

sid_of() {
  if [ "${QUANTIZER}" = "rkmeans" ]; then
    echo "embeddings/$1/merged_predictions_tensor.pt"
  else
    echo "embeddings/$1_${QUANTIZER}/merged_predictions_tensor.pt"
  fi
}

# Dense pre-quantization embeddings, which TRACER's coherence set and the
# embedding-neighbourhood term both need. Two layouts exist and only beauty has
# the first: a bare [N, D] tensor at embeddings/<ds>_..._latest.pt, and the
# {embeddings, item_ids} dict under logs/inference/runs/embeddings/<ds>_<jid>/.
# Resolving only the first silently handed toys, sports and food_k5 a path that
# does not exist.
emb_of() {
  local ds="$1" p
  [ -n "${TRACER_EMB_OVERRIDE:-}" ] && { echo "${TRACER_EMB_OVERRIDE}"; return 0; }
  p="embeddings/${ds}_merged_predictions_tensor_latest.pt"
  [ -e "${p}" ] && { echo "${p}"; return 0; }
  p="$(ls -1dt logs/inference/runs/embeddings/${ds}_*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -1 || true)"
  echo "${p}"
  return 0
}

codebook_of() {
  local ds="$1"
  [ -n "${TRACER_CKPT:-}" ] && { echo "${TRACER_CKPT}"; return; }
  # Auto-resolve ONLY for non-rkmeans tracks. A codebook is usable by TRACER only
  # if it is the one the SID tensor was built from, and for rkmeans that is not
  # something the directory name can establish: logs/train/runs/codebook/
  # beauty_rkmeans_10397484 exists but is the beauty_v2 rebuild, a DIFFERENT
  # identifier space from embeddings/beauty. Submitting against it would burn a
  # job that assert_reproduces_sids then refuses. Pass TRACER_CKPT explicitly to
  # run tracer on an rkmeans space whose checkpoint you have verified.
  [ "${QUANTIZER}" = "rkmeans" ] && return
  local d c
  for d in $(ls -1dt logs/train/runs/codebook/${ds}_${QUANTIZER}_* 2>/dev/null); do
    c="$(ls -1t "${d}"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)"
    [ -n "${c}" ] && { echo "${c}"; return; }
  done
}

has_algo() { case " ${ALGOS} " in *" $1 "*) return 0;; *) return 1;; esac; }

for ds in ${DATASETS}; do
  for sd in ${SEEDS}; do
    read -r base_jid retrain_jid <<< "$(jobids_for "${ds}" "${sd}")"
    if [ -z "${base_jid:-}" ]; then
      echo "## ${ds} seed ${sd}: no train job id in MODELS, skipping (pass MODELS='${ds}:${sd}:<base>:<retrain>')" >&2
      continue
    fi
    echo "## ${ds} seed ${sd}: base job ${base_jid}, retrain reference ${retrain_jid:-none}"
    tck="$(codebook_of "${ds}")"

    if [ "${HPSEARCH:-0}" = "1" ]; then
      for dmp in ${SCIF_DAMPINGS:-0.001 0.01 0.1}; do
        has_algo scif && submit "$ds" "$sd" "$base_jid" scif "scif_d${dmp}" unlearning.damping="$dmp"
      done
      for std in ${SEIF_STDS:-0.006 0.06 0.6}; do
        has_algo seif && submit "$ds" "$sd" "$base_jid" seif "seif_std${std}" unlearning.seif_erase_std="$std"
      done
      for ir in ${KOOKMIN_IRS:-0.0001 0.001 0.01}; do
        has_algo kookmin && submit "$ds" "$sd" "$base_jid" kookmin "kookmin_ir${ir}" unlearning.kookmin_init_rate="$ir"
      done
      for t in ${FANCHUAN_TEMPS:-0.007 0.07 1.15}; do
        has_algo fanchuan && submit "$ds" "$sd" "$base_jid" fanchuan "fanchuan_t${t}" unlearning.fanchuan_contrastive_temperature="$t"
      done
      for lr in ${NEG_TRAIN_LRS:-0.0001 0.001 0.01}; do
        has_algo neg_train && submit "$ds" "$sd" "$base_jid" neg_train "negtrain_lr${lr}" unlearning.neg_train_lr="$lr"
      done
      for lr in ${FINETUNE_LRS:-0.0001 0.001 0.01}; do
        has_algo finetune && submit "$ds" "$sd" "$base_jid" finetune "finetune_lr${lr}" unlearning.finetune_lr="$lr"
      done
      # UNIFIED_ABLATION expresses the ascent/descent BASELINES in the unified
      # objective, exactly as run_unlearn_sweep_bandwagon.sh does, so that no
      # implementation difference (step budget, batch construction, optimizer)
      # is confounded with the comparison. lambda_r is the retain weight:
      #   0.0 0.0 0.0 1.0  -> retain only        (= finetune)
      #   0.1 0.0 0.0 1.0  -> ascent + repair    (= neg_train)
      #   0.1 0.0 0.0 0.0  -> ascent, no repair  (= pure gradient ascent)
      # Without this the sensitive tables carried the standalone finetune and
      # neg_train implementations, which run their own step budgets and drove
      # utility to tau_P 0.04 and 0.00: a destroyed model, not a baseline.
      if [ -n "${UNIFIED_ABLATION:-}" ] && has_algo unified; then
        while IFS= read -r _q; do
          [ -z "${_q}" ] && continue
          read -r _lf _ls _ln _lr <<< "${_q}"
          _rtok=""
          [ "${_lr}" != "1.0" ] && _rtok="_lr${_lr}"
          submit "$ds" "$sd" "$base_jid" unified \
            "unified_abl_lf${_lf}_ls${_ls}_ln${_ln}${_rtok}" \
            unlearning.lambda_f="${_lf}" unlearning.lambda_s="${_ls}" \
            unlearning.lambda_n="${_ln}" unlearning.lambda_r="${_lr}" \
            unlearning.sep_negatives=forget_target_only \
            unlearning.n_epochs="${UNIFIED_EPOCHS:-4}" \
            unlearning.adaptive_codes=false
        done <<< "${UNIFIED_ABLATION}"
      elif has_algo unified; then
        # 2026-08-26 REDESIGN, mirroring the spam sweep: `sep_loss_type` is now a
        # SEARCHED axis. Every previous unified arm here left it at the config
        # default `cosine`, which acts on a pooled-encoder geometry the decoder
        # never reads; `generative` scores the model's own sequence log-probs and
        # is what actually moves the ranking. lambda_f is pinned at 0.1.
        #
        # lambda_n is NEGATIVE here. For a sensitive deletion the neighbours ARE
        # the concept, so the mass must be drained, not repaired; lambda_n > 0
        # with `mass` measurably undoes the deletion (beauty: Sensitive@10 back
        # to 0.03844 against a pre-unlearn 0.03840). A negative lambda_n on
        # `mass` is the drain direction and is what the measured arms used
        # (beauty 0.02809 vs 0.02985 for the bounded `suppress` form).
        for sep in ${SEP_LOSS_TYPES:-cosine generative}; do
          case "${sep}" in
            # generative costs 1 + |I_f| teacher-forced decoder passes per
            # batch. For SPAM |I_f| = 1, so 2 passes and no issue. For SENSITIVE
            # the forget set is the CATEGORY -- 106 items for hairloss, 202 for
            # guns -- so it is 107 to 203 passes and the 2026-08-26 arms died
            # with CUDA OOM at 138.73 GiB of a 139.83 GiB H200. The batch size is
            # the lever that scales with those passes, so drop it here only;
            # cosine is unaffected and keeps the default.
            generative) _septok="gen"; _sepovr=(unlearning.sep_loss_type=generative
                                                unlearning.sep_gen_temperature=1.0
                                                unlearning.batch_size_per_device="${SEP_GEN_BATCH:-32}") ;;
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
            for ln in ${UNIFIED_LNS:--1.0 -0.1 0.0}; do
              _nb=()
              if [ "${ln}" != "0.0" ] && [ "${ln}" != "0" ]; then
                _nb=(unlearning.lambda_n="${ln}"
                     unlearning.coherence_loss_type=mass
                     unlearning.coherence_neighbor_method=embedding
                     unlearning.coherence_rows=target_only
                     unlearning.coherence_embedding_metric=cosine
                     unlearning.neighborhood_count="${NBR_COUNT:-8}")
              else
                _nb=(unlearning.lambda_n=0.0)
              fi
              # "-" in a run-dir name breaks the extractors' `ln([0-9.]+)` regex.
              case "${ln}" in -*) _lntok="lnneg${ln#-}" ;; *) _lntok="ln${ln}" ;; esac
              submit "$ds" "$sd" "$base_jid" unified \
                "unified_hp2_sep${_septok}_lf${UNIFIED_LF:-0.1}_ls${ls}_${_lntok}" \
                unlearning.lambda_f="${UNIFIED_LF:-0.1}" unlearning.lambda_s="$ls" \
                unlearning.sep_negatives=forget_target_only \
                unlearning.n_epochs=4 unlearning.adaptive_codes=false \
                "${_sepovr[@]}" ${_nb[@]+"${_nb[@]}"}
            done
          done
        done
      fi
      if has_algo tracer && [ -n "${tck}" ]; then
        # TRACER_EPOCHS matches TRACER's step budget to unified's, exactly as in
        # run_unlearn_sweep_bandwagon.sh. Both derive
        #   steps = n_epochs * min(n_forget_batches, n_retain_batches)
        # from the same formula (tracer.py per_pass == unified.py
        # optim_steps_per_pass), so the budget scales with the forget set the
        # same way for both. Left unset TRACER takes tracer_steps=500, i.e. 125x
        # more optimization than the method it is compared against, and it does
        # not converge there -- the forget term is unbounded ascent, so the extra
        # steps just run it further away. Unset keeps the historical tag.
        _tep=(); _teptok=""
        if [ -n "${TRACER_EPOCHS:-}" ]; then
          _tep=(unlearning.n_epochs="${TRACER_EPOCHS}")
          _teptok="_ep${TRACER_EPOCHS}"
        fi
        for l1 in ${TRACER_LFS:-0.1 1.0}; do
          for l2 in ${TRACER_LCS:-0.1 1.0}; do
            submit "$ds" "$sd" "$base_jid" tracer \
              "tracer_lf${l1}_lc${l2}${_teptok}" \
              ${_tep[@]+"${_tep[@]}"} \
              "unlearning.tracer_codebook_ckpt=${tck}" \
              unlearning.tracer_lambda_forget="$l1" \
              unlearning.tracer_lambda_coherence="$l2"
          done
        done
      fi
      # `filter` was missing from this branch entirely, so HPSEARCH=1 never
      # produced a single Filter run for the sensitive setting even though the
      # algorithm is in the ALGOS default and extract_sensitive.py reports it.
      # It matters most HERE: a decode-time ban on I_s is exactly the shape of a
      # sensitive deletion request, which is why Filter is reported for this
      # setting and excluded from the spam tables
      # (see h2h.SPAM_EXCLUDED_FAMILIES).
      if has_algo filter; then
        # No weight update, so nothing to tune beyond the mode.
        for fm in global user_dependent; do
          submit "$ds" "$sd" "$base_jid" filter "filter_${fm}" \
            unlearning.filter_mode="$fm"
        done
      fi
      continue
    fi

    # Frozen-config arms. Hyperparameters not named here come from
    # configs/experiment/tiger_unlearn_scif_flat.yaml; anything the search
    # selects should be pinned via the *_ARGS overrides below.
    has_algo scif      && submit "$ds" "$sd" "$base_jid" scif      "scif"      ${SCIF_ARGS:-}
    has_algo seif      && submit "$ds" "$sd" "$base_jid" seif      "seif"      ${SEIF_ARGS:-}
    has_algo kookmin   && submit "$ds" "$sd" "$base_jid" kookmin   "kookmin"   ${KOOKMIN_ARGS:-}
    has_algo fanchuan  && submit "$ds" "$sd" "$base_jid" fanchuan  "fanchuan"  ${FANCHUAN_ARGS:-}
    has_algo neg_train && submit "$ds" "$sd" "$base_jid" neg_train "negtrain"  ${NEG_TRAIN_ARGS:-}
    has_algo finetune  && submit "$ds" "$sd" "$base_jid" finetune  "finetune"  ${FINETUNE_ARGS:-}
    if has_algo unified; then
      submit "$ds" "$sd" "$base_jid" unified "unified" \
        unlearning.lambda_f="${UNIFIED_LF:-0.1}" unlearning.lambda_s="${UNIFIED_LS:-0.1}" \
        unlearning.lambda_n="${UNIFIED_LN:-0.1}" \
        unlearning.sep_negatives=forget_target_only \
        unlearning.n_epochs=4 unlearning.adaptive_codes=false \
        unlearning.neighborhood_count=8 \
        unlearning.coherence_loss_type="${LOSS_TYPE:-suppress}" \
        unlearning.coherence_neighbor_method=embedding \
        unlearning.coherence_rows=target_only \
        unlearning.coherence_embedding_metric=cosine
    fi
    if has_algo tracer; then
      if [ -z "${tck}" ]; then
        echo "  tracer skipped for ${ds}: no ${QUANTIZER} codebook checkpoint (pass TRACER_CKPT=)" >&2
      else
        # Same fix as the hpsearch branch above: without n_epochs this falls
        # back to tracer_steps=500, which on a sensitive cell (one forget batch)
        # is 500 passes over that batch. Measured on toys at 500 steps:
        # lf1.0 arms reach N@10 0.00000 (model destroyed) and lf0.1 arms sit at
        # N@10 0.011-0.015 against a base of 0.03583, i.e. UR 0.30-0.41.
        _tepf=(); _tepftok=""
        if [ -n "${TRACER_EPOCHS:-}" ]; then
          _tepf=(unlearning.n_epochs="${TRACER_EPOCHS}")
          _tepftok="_ep${TRACER_EPOCHS}"
        fi
        submit "$ds" "$sd" "$base_jid" tracer "tracer${_tepftok}" \
          ${_tepf[@]+"${_tepf[@]}"} \
          "unlearning.tracer_codebook_ckpt=${tck}" ${TRACER_ARGS:-}
      fi
    fi
    if has_algo filter; then
      # Seed 2 only: no weight update, so replicates would be identical.
      [ "${sd}" = "2" ] && submit "$ds" "$sd" "$base_jid" filter "filter_global" \
        unlearning.filter_mode=global
    fi
  done
done

echo "----"
echo "Total sensitive unlearn jobs: ${COUNT} (skipped ${SKIPPED} existing; DRYRUN=${DRYRUN})"
