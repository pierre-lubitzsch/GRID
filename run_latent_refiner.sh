#!/usr/bin/env bash
#SBATCH --job-name=latent_refiner
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --partition=gpu
set -euo pipefail
# Step 2 of the minimal validation: train the latent refiner and export z.
#
#   z_i = MLP([SIDEmb(i), x_i])
#
#   sbatch run_latent_refiner.sh [strategy]        # one stratum
#   ./run_latent_refiner.sh                        # DRYRUN-style listing
#   STRATS="unpopular mid popular" ./submit ...    # see the loop at the bottom
#
# ONE REFINER PER MODEL, not one per dataset. SIDEmb(i) is read out of a
# specific checkpoint's semantic-ID table, so z describes the geometry of THAT
# recommender. The three bandwagon strata are three separately trained poisoned
# models with three different SID tables, so sharing one z across them would
# select neighbours in a space the model being unlearned never learned.
#
# Env knobs (defaults = the operating point the sweep assumes):
#   DATASET=beauty  STRATS="unpopular mid popular"
#   NEIGHBOR_K=8         x-space neighbours used as positives (match
#                        unlearning.neighborhood_count in the sweep)
#   LATENT_DIM=128  EPOCHS=30  LR=1e-3  SEED=2
#   W_REC=1.0  W_NBR=1.0  W_SID=0.1    objective weights (all three ACTIVE)
#   OUT_ROOT=embeddings/latent          where z tensors are written
#   DRYRUN=1                            print the commands only
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
STRATS="${STRATS:-${1:-unpopular mid popular}}"
POISON_SEED="${POISON_SEED:-2}"
POISONING_RATIO="${POISONING_RATIO:-0.01}"
N_TARGET="${N_TARGET:-1}"
NUM_HIER="${NUM_HIER:-4}"

SID="${SID:-embeddings/beauty/merged_predictions_tensor.pt}"
EMB="${EMB:-embeddings/beauty_merged_predictions_tensor_latest.pt}"
OUT_ROOT="${OUT_ROOT:-embeddings/latent}"

LATENT_DIM="${LATENT_DIM:-128}"
HIDDEN_DIM="${HIDDEN_DIM:-512}"
NEIGHBOR_K="${NEIGHBOR_K:-8}"
TEMPERATURE="${TEMPERATURE:-0.07}"
PREFIX_LENGTH="${PREFIX_LENGTH:-2}"
EPOCHS="${EPOCHS:-30}"
BATCH_SIZE="${BATCH_SIZE:-512}"
LR="${LR:-1e-3}"
SEED="${SEED:-2}"
W_REC="${W_REC:-1.0}"
W_NBR="${W_NBR:-1.0}"
W_SID="${W_SID:-0.1}"
DIAG_K="${DIAG_K:-8}"
DRYRUN="${DRYRUN:-0}"

# The beauty width-256 bandwagon n=1 poisoned models — the same train jobs the
# lambda_n sweep unlearns from (run_unlearn_sweep_lambda_n.sh, space w256), so
# the refined space is built on exactly the checkpoints the sweep will start at.
MODELS_MAP="${MODELS_MAP:-unpopular:9872116 mid:9872117 popular:9872118}"
jobid_for() { for kv in ${MODELS_MAP}; do [ "${kv%%:*}" = "$1" ] && { echo "${kv##*:}"; return; }; done; }

[ -f "${SID}" ] || { echo "SID tensor missing: ${SID}" >&2; exit 1; }
[ -f "${EMB}" ] || { echo "dense embeddings missing: ${EMB}" >&2; exit 1; }
mkdir -p "${OUT_ROOT}"

echo "#### latent refiner: dataset=${DATASET} H=${NUM_HIER} d_z=${LATENT_DIM}"
echo "     SID=${SID}"
echo "     x  =${EMB}"
echo "     weights: rec=${W_REC} nbr=${W_NBR} sid=${W_SID} | k=${NEIGHBOR_K} tau=${TEMPERATURE}"

for strat in ${STRATS}; do
  jid="$(jobid_for "${strat}")"
  [ -n "${jid}" ] || { echo "No job id for strategy '${strat}' in '${MODELS_MAP}'" >&2; exit 1; }
  rundir="$(ls -1d logs/train/runs/*/*job${jid}_* 2>/dev/null | head -1 || true)"
  [ -n "${rundir}" ] || { echo "No training run dir for job ${jid} (${strat})" >&2; exit 1; }
  # Same rule run_tiger_unlearn_sequential.sh uses to resolve a run dir to a
  # checkpoint: latest *.ckpt by mtime.
  ckpt="$(ls -t "${rundir}"/checkpoints/*.ckpt 2>/dev/null | head -1 || true)"
  [ -n "${ckpt}" ] || { echo "No .ckpt under ${rundir}/checkpoints" >&2; exit 1; }

  out="${OUT_ROOT}/${DATASET}_bw_tgt${strat}_n${N_TARGET}_seed${POISON_SEED}_dz${LATENT_DIM}.pt"
  echo "## ${strat}: train job ${jid}"
  echo "   ckpt=${ckpt}"
  echo "   out =${out}"
  if [ "${DRYRUN}" = "1" ]; then continue; fi

  python -u -m scripts.train_latent_refiner \
    --ckpt "${ckpt}" \
    --embedding_path "${EMB}" \
    --semantic_id_path "${SID}" \
    --num_hierarchies "${NUM_HIER}" \
    --out "${out}" \
    --latent_dim "${LATENT_DIM}" --hidden_dim "${HIDDEN_DIM}" \
    --w_reconstruction "${W_REC}" --w_neighborhood "${W_NBR}" \
    --w_sid_consistency "${W_SID}" \
    --neighbor_k "${NEIGHBOR_K}" --temperature "${TEMPERATURE}" \
    --sid_prefix_length "${PREFIX_LENGTH}" \
    --epochs "${EPOCHS}" --batch_size "${BATCH_SIZE}" --lr "${LR}" \
    --seed "${SEED}" --diag_k "${DIAG_K}"
done

echo "[$(date -Is)] latent refiner done for: ${STRATS}"
