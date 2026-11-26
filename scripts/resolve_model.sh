#!/usr/bin/env bash
# Resolve GRID experiment-config names from a short MODEL name.
#
# The recommender architecture is an experiment axis, exactly like the dataset
# and the identifier space. This is the shell counterpart of
# src/models/registry.py -- keep the two in sync; there is a test that asserts
# they agree (test_model_registry.py).
#
# Usage:
#   source scripts/resolve_model.sh
#   resolve_model tiger
#
# Exports:
#   GRID_TRAIN_EXPERIMENT      experiment= for python -m src.train
#   GRID_INFER_EXPERIMENT      experiment= for python -m src.inference
#   GRID_UNLEARN_EXPERIMENT    experiment= for python -m src.unlearn
#   GRID_MODEL_TARGET          model._target_ (informational; the experiment
#                              config already sets it)
#   GRID_MODEL_NAME            the resolved name, echoed back
#   GRID_MODEL_TAG             run-tag token ("" for tiger so existing run dirs
#                              and every recorded result keep their names)
#
# Adding a model: add one case block here, one entry in
# src/models/registry.py, and the three experiment configs it names.

resolve_model() {
  local name="${1:-${MODEL:-tiger}}"
  if [ -z "${name}" ]; then
    echo "resolve_model: model name required" >&2
    return 1
  fi

  local train="" infer="" unlearn="" target="" tag=""
  case "${name}" in
    tiger)
      train="tiger_train_flat"
      infer="tiger_inference_flat"
      unlearn="tiger_unlearn_scif_flat"
      target="src.models.modules.semantic_id.tiger_generation_model.SemanticIDEncoderDecoder"
      # EMPTY on purpose: tiger is the incumbent, and every recorded run dir,
      # extractor regex and table keys off tags with no model token.
      tag=""
      ;;
    diger)
      # DIGER (arXiv:2601.19711): TIGER backbone + a DIFFERENTIABLE semantic-ID
      # tokenizer trained jointly with the recommender.
      train="diger_train_flat"
      infer="diger_inference_flat"
      unlearn="diger_unlearn_scif_flat"
      target="src.models.modules.semantic_id.diger_generation_model.DigerEncoderDecoder"
      tag="diger_"
      ;;
    letter)
      # LETTER (arXiv:2405.07314): TIGER backbone, but the semantic IDs come from
      # LETTER's own tokenizer (RQ-VAE + collaborative alignment + diversity +
      # Sinkhorn). Build them with run_letter_sid.sh and pass the resulting
      # tensor as the semantic_id_path.
      train="letter_train_flat"
      infer="letter_inference_flat"
      unlearn="letter_unlearn_scif_flat"
      target="src.models.modules.semantic_id.letter_generation_model.LetterEncoderDecoder"
      tag="letter_"
      ;;
    *)
      echo "resolve_model: unknown model '${name}' (known: tiger diger letter)" >&2
      return 1
      ;;
  esac

  export GRID_MODEL_NAME="${name}"
  export GRID_TRAIN_EXPERIMENT="${GRID_TRAIN_EXPERIMENT:-${train}}"
  export GRID_INFER_EXPERIMENT="${GRID_INFER_EXPERIMENT:-${infer}}"
  export GRID_UNLEARN_EXPERIMENT="${GRID_UNLEARN_EXPERIMENT:-${unlearn}}"
  export GRID_MODEL_TARGET="${target}"
  export GRID_MODEL_TAG="${tag}"
  return 0
}

# --- run-dir naming, and why this helper exists -----------------------------
# Run labels are "<tag><dataset>_<variant>", e.g. beauty_clean vs
# diger_beauty_clean. A dataset-keyed glob absorbs the tag into its leading
# wildcard, so BOTH of these match `*_beauty_clean` and `*beauty_poison*`
# (verified). A DIGER run would then be silently picked up as the TIGER
# baseline -- no error, just the wrong model.
#
# Job-id-keyed lookups (`*job${jid}_*`, the majority) are unaffected. For the
# dataset-keyed ones, filter with:
#     ls -t <glob> | grep -vE "$(grid_other_model_regex)" | head -1
GRID_MODEL_TAGS_NON_DEFAULT="diger letter"

grid_other_model_regex() {
  # ERE matching the run labels of every model OTHER than the active one, for
  # `ls <glob> | grep -vE "$(grid_other_model_regex)"`.
  #
  # This used to emit every non-default tag unconditionally, which is right only
  # when the caller is selecting FOR tiger. With MODEL=letter it emitted
  # `_(diger|letter)_` and so filtered out letter's own runs: every lookup
  # returned nothing. run_sensitive_reference_eval.sh then printed
  # `ckpt: <MISSING>`, skipped both roles and still exited 0, so 15 jobs
  # reported COMPLETED having written nothing -- the sensitive references for
  # LETTER simply never existed. Drop the ACTIVE model from the exclusion list.
  # (tiger carries an empty tag and cannot be matched by a `_<tag>_` token
  # anyway; callers glob by job id, so this is a safety net, not the selector.)
  local active="${GRID_MODEL_NAME:-${MODEL:-tiger}}"
  local alt=""
  local m
  for m in ${GRID_MODEL_TAGS_NON_DEFAULT}; do
    [ "${m}" = "${active}" ] && continue
    alt="${alt:+${alt}|}${m}"
  done
  if [ -z "${alt}" ]; then
    # Nothing to exclude; emit a pattern that matches nothing so the caller's
    # `grep -vE` is a no-op rather than an error.
    printf '%s' '$^'
    return
  fi
  printf '_(%s)_' "${alt}"
}

# --- experiment-config name mapping ----------------------------------------
# Every model's experiment configs are the tiger ones with the model name in
# place of the `tiger_` prefix (tiger_unlearn_scif_sequential ->
# letter_unlearn_scif_sequential). Runners that hardcode a tiger config name can
# therefore stay readable and still be model-aware:
#
#   EXPERIMENT="$(grid_model_config tiger_unlearn_scif_sequential)"
#
# For MODEL=tiger this returns its argument unchanged, so every existing run
# reproduces byte-for-byte. A name without the `tiger_` prefix is passed through
# untouched (rqvae_*, rkmeans_*, ... are shared, not per-model).
#
# It FAILS LOUDLY when the mapped config does not exist. The alternative --
# letting Hydra report "Could not find 'experiment/diger_unlearn_unified_flat'"
# three screens into a job's log -- is how a model axis quietly acquires holes.
grid_model_config() {
  local base="${1:?grid_model_config: config name required}"
  local name="${GRID_MODEL_NAME:-${MODEL:-tiger}}"
  local mapped="${base}"
  case "${base}" in
    tiger_*) mapped="${name}_${base#tiger_}" ;;
  esac
  local root="${GRID_DIR:-$PWD}"
  if [ ! -f "${root}/configs/experiment/${mapped}.yaml" ]; then
    echo "grid_model_config: MODEL=${name} has no experiment config" \
         "'${mapped}' (mapped from '${base}')." >&2
    echo "  Add configs/experiment/${mapped}.yaml, or run this with MODEL=tiger." >&2
    return 1
  fi
  printf '%s' "${mapped}"
}
