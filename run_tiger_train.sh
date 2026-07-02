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
RUN_LABEL="${DATASET}_${VARIANT}"
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
# Optional free-form label suffix (e.g. "_det_seed7") so multi-seed / variant
# sweeps produce self-describing, non-colliding run dirs.
RUN_LABEL="${RUN_LABEL}${RUN_LABEL_SUFFIX:-}"
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
        "")          echo "null" ;;
        all|null)    echo "$1" ;;
        *)           echo "[$1]" ;;
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
case "$(printf '%s' "${DETERMINISTIC:-}" | tr '[:upper:]' '[:lower:]')" in
  1|true|yes|on)
    export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:4096:8}"
    export PYTHONHASHSEED="${PYTHONHASHSEED:-2}"
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
  *) echo "Unknown OPTIMIZER='${OPTIMIZER}' (expected adam|sgd)"; exit 1 ;;
esac

# PKM-only optimizer param group: give the PKM params their own lr/weight_decay
# (same optimizer class). e.g. PKM_PARAM_GROUP='{lr:0.05,weight_decay:0.0}'.
if [ -n "${PKM_PARAM_GROUP:-}" ]; then
  EXTRA_OVERRIDES+=("model.pkm_param_group=${PKM_PARAM_GROUP}")
  echo "PKM param group override: model.pkm_param_group=${PKM_PARAM_GROUP}"
fi

echo "[$(date -Is)] Starting tiger train (tiger_train_flat) dataset=${DATASET} variant=${VARIANT}"
echo "Using data_dir=${DATA_DIR}"
echo "Using semantic_id_path=${SEMANTIC_ID_PATH}"
echo "Run dir: ${HYDRA_RUN_DIR}"
if [ "${#EXTRA_OVERRIDES[@]}" -gt 0 ] || [ "$#" -ge 6 ]; then
  echo "Extra Hydra overrides: ${EXTRA_OVERRIDES[*]:-} ${*:6}"
fi

python -u -m src.train \
  experiment=tiger_train_flat \
  data_dir="${DATA_DIR}" \
  "semantic_id_path='${SEMANTIC_ID_PATH}'" \
  num_hierarchies=4 \
  hydra.run.dir="${HYDRA_RUN_DIR}" \
  ${EXTRA_OVERRIDES[@]+"${EXTRA_OVERRIDES[@]}"} \
  "${@:6}"

LATEST_CKPT="$(ls -t logs/train/runs/*/*/checkpoints/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -n "${LATEST_CKPT}" ]; then
  echo "[$(date -Is)] tiger train finished"
  echo "Latest checkpoint (for tiger inference): ${LATEST_CKPT}"
else
  echo "[$(date -Is)] tiger train finished (no .ckpt under logs/train/runs/*/*/checkpoints/)"
fi
