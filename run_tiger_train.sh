#!/usr/bin/env bash
#SBATCH --job-name=tiger_train
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --partition=pgpu
#SBATCH --time=2-00:00:00

# Resource notes:
# * --gres=gpu:nvidia_h200:2 + --partition=pgpu: requests 2 H200s (s-sc-pgpu[11-16]).
#   Override at submit time with `sbatch --gres=gpu:nvidia_h200:N ...`.

set -euo pipefail

# -----------------------------------------------------------------------------
# 4. Train Generative Recommendation Model with Semantic IDs (README)
#
#   python -m src.train experiment=tiger_train_flat \
#       data_dir=data/amazon_data/beauty \
#       semantic_id_path=<output_path_from_step_3>/pickle/merged_predictions_tensor.pt \
#       num_hierarchies=4
#
# num_hierarchies=4: add 1 vs RKMeans (step 3) because the previous step appends one
# additional digit to de-duplicate semantic IDs (3 codebooks -> 4 hierarchies here).
# -----------------------------------------------------------------------------
#
# Wrapper usage:
#   sbatch run_tiger_train.sh [dataset] [clean|poison] [semantic_id_path]
#
# data_dir: README uses data/amazon_data/<dataset>. This checkout often has data under
# src/data/amazon_data/<dataset> — override with:  TIGER_DATA_DIR=src/data/amazon_data/beauty
# (default below follows the repo layout.)
#
# Progress bar: off in configs/experiment/tiger_train_flat.yaml. Local bar:
#   add trainer.enable_progress_bar=true

DATASET="${1:-beauty}"
ARG2="${2:-}"
ARG3="${3:-}"
POISONING_RATIO="${POISONING_RATIO:-${4:-0.01}}"
N_TARGET_ITEMS="${N_TARGET_ITEMS:-${5:-10}}"
VARIANT="clean"
SEMANTIC_ID_PATH=""

case "${ARG2}" in
  clean|poison)
    VARIANT="${ARG2}"
    SEMANTIC_ID_PATH="${ARG3}"
    ;;
  "")
    ;;
  *)
    if [[ "${ARG2}" == *.pt ]]; then
      SEMANTIC_ID_PATH="${ARG2}"
    else
      echo "Unknown arg '${ARG2}'. Use clean|poison or a .pt semantic_id_path."
      exit 1
    fi
    ;;
esac

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs

# MODEL selects the recommender ARCHITECTURE (tiger default | diger | ...). It
# resolves to the experiment config; see scripts/resolve_model.sh and
# src/models/registry.py. MODEL=tiger reproduces every previously recorded run
# byte for byte -- same experiment config, empty run-tag token.
# shellcheck source=scripts/resolve_model.sh
source "${GRID_DIR}/scripts/resolve_model.sh"
if ! resolve_model "${MODEL:-tiger}"; then
  exit 1
fi
# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
if ! resolve_grid_dataset "${DATASET}"; then
  exit 1
fi

if [ -n "${TIGER_DATA_DIR:-}" ]; then
  DATA_DIR="${TIGER_DATA_DIR}"
elif [ "${VARIANT}" = "poison" ]; then
  DATA_DIR="${GRID_POISON_DATA_DIR}"
else
  DATA_DIR="${GRID_DATA_DIR}"
fi

AUTO_SEMANTIC_ID_PATH="$(ls -t \
  logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt \
  embeddings/*/merged_predictions_tensor.pt \
  embeddings/merged_predictions_tensor.pt \
  2>/dev/null | head -n1 || true)"

if [ -n "${SEMANTIC_ID_PATH}" ] && [ ! -f "${SEMANTIC_ID_PATH}" ]; then
  echo "Provided semantic_id_path not found: ${SEMANTIC_ID_PATH}"
  if [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
    echo "Falling back to latest discovered tensor: ${AUTO_SEMANTIC_ID_PATH}"
    SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
  fi
fi

if [ -z "${SEMANTIC_ID_PATH}" ] && [ -f "${GRID_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${GRID_SEMANTIC_ID_PATH}"
fi
if [ -z "${SEMANTIC_ID_PATH}" ] && [ -n "${AUTO_SEMANTIC_ID_PATH}" ]; then
  SEMANTIC_ID_PATH="${AUTO_SEMANTIC_ID_PATH}"
fi

if [ ! -f "${SEMANTIC_ID_PATH}" ]; then
  echo "Semantic ID tensor not found: ${SEMANTIC_ID_PATH:-<empty>}"
  echo "Step 3 must produce pickle/merged_predictions_tensor.pt first."
  echo "Expected under logs/inference/runs/*/*/pickle/merged_predictions_tensor.pt"
  exit 1
fi

# Build a unique, informative run directory: date/time_jobID_dataset_variant[_pctX_nY]
JOB_ID="${SLURM_JOB_ID:-local$$}"
TS="$(date +%Y-%m-%d/%H-%M-%S)"
# GRID_MODEL_TAG is EMPTY for tiger, so every existing run-dir name is unchanged.
RUN_LABEL="${GRID_MODEL_TAG}${DATASET}_${VARIANT}"
if [ "${VARIANT}" = "poison" ]; then
  PCT_LABEL="$(python3 -c "r=${POISONING_RATIO}; print(f'pct{int(round(r*100))}')")"
  # Tag the poison method (empty for bandwagon) so runs are self-describing.
  POISON_METHOD="${POISON_METHOD:-bandwagon}"
  if [ "${POISON_METHOD}" = "bandwagon" ]; then MTOK=""; else MTOK="_${POISON_METHOD}"; fi
  # clone_inject with >1 injection per session: encode the count so run dirs
  # (and downstream eval/auto-discovery) distinguish K (matches dataset naming).
  if [ "${POISON_METHOD}" = "clone_inject" ] && [ "${CLONE_INJECT_COUNT:-1}" != "1" ]; then
    MTOK="${MTOK}x${CLONE_INJECT_COUNT}"
  fi
  RUN_LABEL="${RUN_LABEL}${MTOK}_${PCT_LABEL}_n${N_TARGET_ITEMS}"
fi
case "$(printf '%s' "${PKM_MODE:-}" | tr '[:upper:]' '[:lower:]')" in
  add|replace) RUN_LABEL="${RUN_LABEL}_pkm$(printf '%s' "${PKM_MODE}" | tr '[:upper:]' '[:lower:]')" ;;
esac
# Tag longer-ID length (L!=4) and item-token aggregation so run dirs self-describe.
if [ -n "${NUM_HIER:-}" ] && [ "${NUM_HIER}" != "4" ]; then
  RUN_LABEL="${RUN_LABEL}_L${NUM_HIER}"
fi
# Tag the item-history budget when pinned (distinguishes e.g. L8 @15 items
# [config default 120 codes] from L8 @30 items [sequence_length=240]).
if [ -n "${HISTORY_ITEMS:-}" ]; then
  RUN_LABEL="${RUN_LABEL}_h${HISTORY_ITEMS}"
fi
# Tag resumed segments (self-describing chain: _res marks a continuation run).
if [ -n "${RESUME_FROM:-}" ]; then
  RUN_LABEL="${RUN_LABEL}_res"
fi
case "$(printf '%s' "${ITEM_TOKEN_AGG:-}" | tr '[:upper:]' '[:lower:]')" in
  ""|none|null|off|false) : ;;
  mean)                    RUN_LABEL="${RUN_LABEL}_aggmean" ;;
  sum)                     RUN_LABEL="${RUN_LABEL}_aggsum" ;;
  *attentive*|*attention*|*merger*)
    # k latents/item is the compression ratio, and at L=4 the default k=4 means
    # NO compression — so the label must record k or the runs are indistinguishable.
    _k="$(printf '%s' "${ITEM_TOKEN_AGG}" | sed -n 's/.*num_query_tokens[: ]*\([0-9]\+\).*/\1/p')"
    RUN_LABEL="${RUN_LABEL}_aggattn${_k:+k${_k}}" ;;
esac
# TRAIN_SEED: draw a DIFFERENT training run for replication. The config pins
# seed: 2, and nothing here used to override it, so "re-run with another seed" was
# not expressible — a repeat submission just reproduced the same draw. Setting it
# passes `seed=N` to Hydra and (in deterministic mode) PYTHONHASHSEED=N.
# The label token is appended only when != 2, so every historical seed-2 run dir
# and every extractor keyed off those names stays byte-identical.
if [ -n "${TRAIN_SEED:-}" ] && [ "${TRAIN_SEED}" != "2" ]; then
  RUN_LABEL="${RUN_LABEL}_seed${TRAIN_SEED}"
fi
# Optional free-form label suffix (e.g. "_det_seed7") so multi-seed / variant
# sweeps produce self-describing, non-colliding run dirs.
RUN_LABEL="${RUN_LABEL}${RUN_LABEL_SUFFIX:-}"
# --- duplicate guard --------------------------------------------------------
# Training runs are expensive and were previously unguarded, so re-submitting the
# same configuration silently trained it twice and left two run dirs that every
# dataset-keyed lookup then had to disambiguate by hand. Skip when a run with the
# SAME label already produced a checkpoint. FORCE=1 overrides.
#
# The match is anchored on the FULL label, and other models' runs are filtered
# out: a tiger label like `beauty_clean` is also a suffix of
# `..._diger_beauty_clean`, so an unfiltered glob would make tiger skip because
# DIGER had already run.
#
# LIMITATION, stated because it bites: a run dir only exists once the job
# STARTS, so this cannot see an identical job still PENDING in the queue. Check
# `squeue` before re-submitting a config you may already have queued.
if [ "${FORCE:-0}" != "1" ]; then
  _dupes="$(ls -1d logs/train/runs/*/*_"${RUN_LABEL}" 2>/dev/null || true)"
  if [ -n "${GRID_MODEL_TAG}" ]; then
    _keep="${_dupes}"
  else
    _keep="$(printf '%s\n' "${_dupes}" | grep -vE "$(grid_other_model_regex)" || true)"
  fi
  for _d in ${_keep}; do
    if ls "${_d}"/checkpoints/*.ckpt >/dev/null 2>&1; then
      echo "[skip] a completed run with label '${RUN_LABEL}' already exists:"
      echo "       ${_d}"
      echo "       Set FORCE=1 to train it again."
      exit 0
    fi
  done
fi

HYDRA_RUN_DIR="logs/train/runs/${TS}_job${JOB_ID}_${RUN_LABEL}"

# Optional Hydra overrides. PKM is gated entirely by PKM_MODE:
#   PKM_MODE=replace|add  -> PKM ON. Default layer set = decoder blocks [0,1]
#                            (the historical default). Customize the targeted
#                            blocks with PKM_DECODER / PKM_ENCODER ("0,1", "all").
#   unset / none / null / off / "" -> PKM OFF (no pkm_layers override; the config
#                            default model.pkm_layers=null applies).
# Any trailing args (positions 6+) are forwarded verbatim as extra Hydra overrides.
EXTRA_OVERRIDES=()
PKM_MODE_LC="$(printf '%s' "${PKM_MODE:-}" | tr '[:upper:]' '[:lower:]')"
case "${PKM_MODE_LC}" in
  add|replace)
    # normalize a layer selector: ""->null, "all"/"null" passthrough, "0,1"->[0,1]
    pkm_sel() {
      case "${1:-}" in
        "")             echo "null" ;;
        all|null)       echo "$1" ;;
        # 'none'/'off' are the natural way to say "no PKM on this sub-tree" when
        # the OTHER sub-tree is being selected (e.g. PKM_ENCODER=2
        # PKM_DECODER=none). Without this they fell through to the [$1] branch
        # and produced an invalid 'decoder:[none]'.
        none|off|false) echo "null" ;;
        *)              echo "[$1]" ;;
      esac
    }
    PKM_ENC="$(pkm_sel "${PKM_ENCODER:-}")"
    if [ -n "${PKM_DECODER:-}" ]; then PKM_DEC="$(pkm_sel "${PKM_DECODER}")"; else PKM_DEC="[0,1]"; fi
    EXTRA_OVERRIDES+=("model.pkm_layers={encoder:${PKM_ENC},decoder:${PKM_DEC}}")
    EXTRA_OVERRIDES+=("model.pkm_mode=${PKM_MODE_LC}")
    ;;
  ""|none|null|off|false)
    : ;;  # PKM off — emit nothing
  *)
    echo "Unknown PKM_MODE='${PKM_MODE}' (expected add|replace|none)"; exit 1 ;;
esac

# Deterministic training (opt-in: DETERMINISTIC=1). Makes runs reproducible
# run-to-run for a fixed seed WITHOUT giving up the 2-GPU pgpu allocation: DDP
# all-reduce is reproducible on a fixed GPU model, seed_everything(workers=True)
# (already called in launcher_utils) seeds the dataloader workers so the collate's
# random sub-sequence sampling repeats, and the flags below kill the remaining
# CUDA nondeterminism (cuDNN autotuning + atomic embedding-grad scatter + cuBLAS).
#   DETERMINISTIC=1            -> on (trainer.deterministic=warn: best-effort, no
#                                 crash if an op lacks a deterministic kernel)
#   DETERMINISTIC_MODE=true    -> strict (trainer.deterministic=true; errors on
#                                 any nondeterministic op — use to audit)
# cuBLAS workspace MUST be set before CUDA init (i.e. here, before python).
# Seed override (see TRAIN_SEED above). Passed to Hydra so seed_everything picks
# it up; unset means the config default (2) applies untouched.
if [ -n "${TRAIN_SEED:-}" ]; then
  EXTRA_OVERRIDES+=("seed=${TRAIN_SEED}")
  echo "Training seed: ${TRAIN_SEED} (config default is 2)"
fi

case "$(printf '%s' "${DETERMINISTIC:-}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    # PYTHONHASHSEED must follow TRAIN_SEED, or a "different seed" run would keep
    # the seed-2 hash seed and stay partly correlated with the original draw.
    export PYTHONHASHSEED="${PYTHONHASHSEED:-${TRAIN_SEED:-2}}"
    DET_MODE="${DETERMINISTIC_MODE:-warn}"   # warn (safe) | true (strict)
    EXTRA_OVERRIDES+=("trainer.deterministic=${DET_MODE}")
    echo "Deterministic mode ON (trainer.deterministic=${DET_MODE}, CUBLAS_WORKSPACE_CONFIG=${CUBLAS_WORKSPACE_CONFIG}); 2-GPU DDP retained."
    ;;
esac

# Optimizer as a hyperparameter. Default keeps the config's Adam. OPTIMIZER=sgd
# switches the (shared) optimizer class to SGD; OPTIMIZER_MOMENTUM optionally
# sets momentum. Applies to PKM and non-PKM runs alike.
case "$(printf '%s' "${OPTIMIZER:-}" | tr '[:upper:]' '[:lower:]')" in
  ""|adam) : ;;
  sgd)
    EXTRA_OVERRIDES+=("optim.optimizer._target_=torch.optim.SGD")
    if [ -n "${OPTIMIZER_MOMENTUM:-}" ]; then
      EXTRA_OVERRIDES+=("+optim.optimizer.momentum=${OPTIMIZER_MOMENTUM}")
    fi
    echo "Optimizer: SGD (lr/weight_decay from optim.optimizer; momentum=${OPTIMIZER_MOMENTUM:-0})"
    ;;
  adamw)
    # AdamW decouples weight decay from the gradient update; optim.optimizer
    # already carries weight_decay, so only the class changes. Matches the
    # adam|adamw|sgd set the unlearning side accepts, so one OPTIMIZER value
    # means the same thing for training, retraining and unlearning.
    EXTRA_OVERRIDES+=("optim.optimizer._target_=torch.optim.AdamW")
    echo "Optimizer: AdamW (lr/weight_decay from optim.optimizer)"
    ;;
  *) echo "Unknown OPTIMIZER='${OPTIMIZER}' (expected adam|adamw|sgd)"; exit 1 ;;
esac

# PKM-only optimizer param group: give the PKM params their own lr/weight_decay
# (same optimizer class). e.g. PKM_PARAM_GROUP='{lr:0.05,weight_decay:0.0}'.
if [ -n "${PKM_PARAM_GROUP:-}" ]; then
  EXTRA_OVERRIDES+=("model.pkm_param_group=${PKM_PARAM_GROUP}")
  echo "PKM param group override: model.pkm_param_group=${PKM_PARAM_GROUP}"
fi

# Semantic-ID length (num_hierarchies). Default 4 (3 RKMeans codebooks + 1 dedup
# digit). "Longer IDs for finer-grained neighborhoods" (Jul 3): set NUM_HIER=8
# or 16 for longer RQ IDs (codebook size stays fixed via vocab_size=256). The
# semantic_id_path tensor MUST have this many hierarchies.
NUM_HIER="${NUM_HIER:-4}"

# Resume training from a checkpoint (multi-sbatch-job chains around the 2-day
# wall limit). RESUME_FROM accepts:
#   job:<slurm_jobid>  -> that training job's run dir, resolved AT JOB START
#                         (so segments can be chained with --dependency=afterany
#                         on a predecessor that is still queued/running now)
#   <run dir>          -> its checkpoints/
#   <file.ckpt>        -> used as-is
# Directory forms prefer 'last.ckpt' (the true latest state; save_last=true in
# the config) and fall back to the newest *.ckpt. Lightning restores optimizer/
# lr-scheduler/loop state from the ckpt; the RngStateCallback restores RNG
# states (post-callback ckpts), so a resumed run is reproducible (rerun the
# same segment -> same trajectory). NOTE: the streaming TFRecord dataloader
# cannot seek, so the input stream restarts at resume — the chain is NOT
# sample-identical to an uninterrupted run.
if [ -n "${RESUME_FROM:-}" ]; then
  _SRC="${RESUME_FROM}"
  if [[ "${_SRC}" == job:* ]]; then
    _JID="${_SRC#job:}"
    _SRC="$(ls -1d logs/train/runs/*/*job${_JID}_* 2>/dev/null | head -1 || true)"
    if [ -z "${_SRC}" ]; then
      echo "RESUME_FROM=${RESUME_FROM}: no training run dir found for job ${_JID}"; exit 1
    fi
  fi
  if [ -d "${_SRC}" ]; then
    _CKDIR="${_SRC%/}"
    [ -d "${_CKDIR}/checkpoints" ] && _CKDIR="${_CKDIR}/checkpoints"
    if [ -f "${_CKDIR}/last.ckpt" ]; then
      RESUME_CKPT="${_CKDIR}/last.ckpt"
    else
      RESUME_CKPT="$(ls -t "${_CKDIR}"/*.ckpt 2>/dev/null | head -1 || true)"
    fi
  else
    RESUME_CKPT="${_SRC}"
  fi
  if [ -z "${RESUME_CKPT:-}" ] || [ ! -f "${RESUME_CKPT}" ]; then
    echo "RESUME_FROM=${RESUME_FROM}: no checkpoint found (looked at '${_SRC}')"; exit 1
  fi
  EXTRA_OVERRIDES+=("ckpt_path='${RESUME_CKPT}'")
  echo "Resuming from checkpoint: ${RESUME_CKPT}"
fi

# History length in ITEMS. The config's sequence_length is a raw-code (token)
# budget that must be a multiple of NUM_HIER (each item = NUM_HIER codes; the
# item-block reshape in the encoder fails otherwise — this is what broke L=16
# with the default sequence_length=120, 120/16=7.5). HISTORY_ITEMS pins the
# number of history items independently of L via
#   sequence_length = HISTORY_ITEMS x NUM_HIER
# so different-L runs are comparable. Unset -> config default (120 codes, i.e.
# 30 items at L=4, 15 at L=8; INVALID at L=16 — set HISTORY_ITEMS for L=16).
if [ -n "${HISTORY_ITEMS:-}" ]; then
  SEQ_LEN=$(( HISTORY_ITEMS * NUM_HIER ))
  EXTRA_OVERRIDES+=("sequence_length=${SEQ_LEN}")
  echo "History budget: ${HISTORY_ITEMS} items x L=${NUM_HIER} -> sequence_length=${SEQ_LEN}"
fi

# Input-side item-token aggregation ("Longer IDs" options 1 & 2). Collapses each
# history item's num_hierarchies token embeddings into one encoder input vector
# so the encoder sequence stays short for long IDs.
#   unset / none / off / null / ""  -> OFF (per-token + separator layout)
#   mean                            -> option 1: mean pooling (1 vector/item)
#   attentive                       -> option 2: ACERec Attentive Token Merger
#                                      (k latents/item, k=4; intent token off)
#   {type:attentive,num_query_tokens:4,num_heads:8}  -> attentive w/ params
# Tune the attentive merger further via ITEM_TOKEN_AGG='{type:attentive,...}'
# (num_query_tokens, num_heads, dropout, mlp_ratio, positional_embedding,
#  intent_token, content_adaptive_queries).
ITEM_TOKEN_AGG_LC="$(printf '%s' "${ITEM_TOKEN_AGG:-}" | tr '[:upper:]' '[:lower:]')"
case "${ITEM_TOKEN_AGG_LC}" in
  ""|none|null|off|false)
    : ;;  # aggregation off — config default (null) applies
  *)
    EXTRA_OVERRIDES+=("model.item_token_aggregation=${ITEM_TOKEN_AGG}")
    echo "Item-token aggregation: model.item_token_aggregation=${ITEM_TOKEN_AGG}"
    ;;
esac

echo "[$(date -Is)] Starting tiger train (tiger_train_flat) dataset=${DATASET} variant=${VARIANT}"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Run dir: ${HYDRA_RUN_DIR}"
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ] || [ "$#" -ge 6 ]; then
  echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]:-} ${*:6}"
fi

# DIGER needs the DENSE content embeddings its tokenizer indexes (the same
# features the quantizer was fit on). Default to the dataset's canonical dense
# tensor; override with ITEM_CONTENT_EMBEDDINGS.
MODEL_OVERRIDES=()
if [ "${GRID_MODEL_NAME}" = "diger" ]; then
  ICE="${ITEM_CONTENT_EMBEDDINGS:-embeddings/${DATASET}_merged_predictions_tensor_latest.pt}"
  if [ ! -f "${ICE}" ]; then
    echo "MODEL=diger needs item content embeddings but '${ICE}' does not exist." >&2
    echo "Set ITEM_CONTENT_EMBEDDINGS=<path to the dense [N, D] tensor>." >&2
    exit 1
  fi
  echo "Using item_content_embeddings_path=${ICE}"
  MODEL_OVERRIDES+=("item_content_embeddings_path='${ICE}'")
fi

python -u -m src.train \
  experiment="${GRID_TRAIN_EXPERIMENT}" \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  num_hierarchies="${NUM_HIER}" \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  ${MODEL_OVERRIDES[@]+"${MODEL_OVERRIDES[@]}"} \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  "${@:6}"

LATEST_CKPT="$(ls -t logs/train/runs/*/*/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -n "${LATEST_CKPT}" ]; then
  echo "[$(date -Is)] tiger train finished"
  echo "Latest checkpoint (for tiger inference): ${LATEST_CKPT}"
else
  echo "[$(date -Is)] tiger train finished (no .ckpt under logs/train/runs/*/*/checkpoints/)"
fi
