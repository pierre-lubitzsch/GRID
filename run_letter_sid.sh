#!/usr/bin/env bash
#SBATCH --job-name=letter_sid
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:1

set -euo pipefail

# Build a LETTER identifier space, end to end, in one job.
#
#   A. CF embeddings   scripts/train_cf_embeddings.py  (SASRec over the
#                      dataset's own training sequences)
#   B. tokenizer       experiment=letter_rqvae_train_flat (RQ-VAE + LETTER's
#                      collaborative alignment + diversity + Sinkhorn)
#   C. assignment      scripts.assign_sids_exact (float64 on CPU, then the
#                      dedup digit and the transpose)
#
# Output: embeddings/<dataset>_letter/merged_predictions_tensor.pt -- an (L, N)
# tensor with L = LETTER_HIER + 1, the SAME layout as every TIGER identifier
# space. Pass it to run_tiger_train.sh as the semantic_id_path with MODEL=letter.
#
#   sbatch run_letter_sid.sh beauty
#   LETTER_ALPHA=0.05 LETTER_BETA=1e-3 sbatch run_letter_sid.sh beauty
#   LETTER_TOKENIZE=dedup sbatch run_letter_sid.sh beauty      # GRID convention
#
# TOKENIZATION MODE -- this changes the ID GRAMMAR, not just its quality
# ---------------------------------------------------------------------
# LETTER_TOKENIZE=paper (DEFAULT) reproduces the reference
# RQ-VAE/generate_indices.py: LETTER_HIER=4 SEMANTIC levels, no dedup digit, and
# collisions broken by re-assigning each collided group with Sinkhorn on the
# final level for up to 20 rounds. The reference drives the collision rate to
# ~0.12% this way (their checkpoint is named epoch_9999_collision_0.0012).
#
# LETTER_TOKENIZE=dedup is GRID's convention: LETTER_HIER=3 semantic levels plus
# an appended dedup digit, which keeps prefix depth meaning exactly what it means
# on a TIGER / RQ-VAE tensor.
#
# THEY ARE NOT INTERCHANGEABLE. Under `paper` the final position is a semantic
# code, so code sharing, neighbourhood sampling, collateral drift by depth and
# the RQ position analysis all measure something different than they do on a
# TIGER tensor of the same length. Re-baseline those; never compare across the
# two grammars. The two modes write to DIFFERENT directories for this reason.
#
# TRAINING LENGTH. The reference trains the tokenizer for ~10,000 epochs
# (main.py defaults to 20,000; their released checkpoint is epoch_9999).
# rqvae_train_flat.yaml's max_steps=3000 is ~500 epochs on beauty, i.e. 20x
# short, which measurably starves the codebook: a 3000-step beauty tokenizer used
# 52 of 256 level-0 codes. LETTER_EPOCHS therefore sets max_steps from the
# catalog size, per dataset.
#
# WHICH INTERACTIONS THE CF SIGNAL SEES -- READ THIS BEFORE CHANGING IT
# ---------------------------------------------------------------------
# LETTER's collaborative term reads the INTERACTIONS, so training it on a
# poisoned dataset produces a poisoned identifier space. This script therefore
# defaults to the CLEAN dataset (LETTER_CF_VARIANT=clean), which keeps ONE
# identifier space shared by the clean, poisoned and unlearned arms -- exactly
# the invariant TIGER gets for free, and the one every spam metric depends on:
# SH/ASI/TPM map their target items through semantic_id_path, so two arms with
# different codes for the same item are not comparable at all.
#
# LETTER_CF_VARIANT=poison builds the attacker-visible space instead. That is a
# legitimate and interesting experiment -- it is the one thing LETTER can do
# that TIGER cannot -- but it produces a DIFFERENT tensor, and every arm it is
# compared against must be retrained on that same tensor. It is opt-in for that
# reason, not because it is wrong.

DATASET="${1:?dataset name required (e.g. beauty)}"
EMBEDDING_PATH="${2:-}"

GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"
mkdir -p logs embeddings

# shellcheck source=scripts/resolve_grid_dataset.sh
source "${GRID_DIR}/scripts/resolve_grid_dataset.sh"
resolve_grid_dataset "${DATASET}" || exit 1

LETTER_TOKENIZE="${LETTER_TOKENIZE:-paper}"   # paper | dedup (see header)
case "${LETTER_TOKENIZE}" in
  paper) _DEFAULT_HIER=4 ;;   # 4 semantic levels, no dedup digit
  dedup) _DEFAULT_HIER=3 ;;   # 3 semantic levels + appended dedup digit -> L=4
  *) echo "LETTER_TOKENIZE must be paper|dedup, got '${LETTER_TOKENIZE}'" >&2; exit 1 ;;
esac
LETTER_HIER="${LETTER_HIER:-${_DEFAULT_HIER}}"
LETTER_EPOCHS="${LETTER_EPOCHS:-10000}"  # reference length; see header
LETTER_BATCH="${LETTER_BATCH:-2048}"     # must match the config's batch_size_per_device
CODEBOOK_WIDTH="${CODEBOOK_WIDTH:-256}"
LETTER_ALPHA="${LETTER_ALPHA:-0.01}"     # collaborative regularisation
LETTER_BETA="${LETTER_BETA:-0.0001}"     # diversity regularisation
LETTER_SK_EPS="${LETTER_SK_EPS:-0.003}"  # Sinkhorn epsilon, final level
CF_DIM="${CF_DIM:-64}"
CF_EPOCHS="${CF_EPOCHS:-200}"
CF_SEED="${CF_SEED:-42}"
LETTER_CF_VARIANT="${LETTER_CF_VARIANT:-clean}"

case "${LETTER_CF_VARIANT}" in
  clean)  CF_DATA_DIR="${GRID_DATA_DIR}"; CF_TAG="${DATASET}" ;;
  poison)
    CF_DATA_DIR="${GRID_POISON_DATA_DIR}"
    # The CELL must be in the name. The poison dir encodes method/stratum/seed/
    # ratio/n (beauty_spam_seed2_pct1_n1 vs beauty_spam_tgtmid_seed2_pct1_n1),
    # and WITHOUT this token two different cells would write the same CF table
    # and the same identifier space -- silently overwriting each other, so the
    # second run would be attributed to the first cell's attack.
    _CELL="$(basename "${GRID_POISON_DATA_DIR}")"
    _CELL="${_CELL#${DATASET}_spam_}"
    CF_TAG="${DATASET}_poison_${_CELL}"
    ;;
  *) echo "LETTER_CF_VARIANT must be clean|poison, got '${LETTER_CF_VARIANT}'" >&2; exit 1 ;;
esac
if [ ! -d "${CF_DATA_DIR}/training" ]; then
  echo "No training shards under ${CF_DATA_DIR} (LETTER_CF_VARIANT=${LETTER_CF_VARIANT})" >&2
  exit 1
fi

# Dense content embeddings: the SAME input the RQ-VAE / RQ-KMeans quantizers get.
if [ -z "${EMBEDDING_PATH}" ]; then
  EMBEDDING_PATH="embeddings/${DATASET}_merged_predictions_tensor_latest.pt"
  if [ ! -f "${EMBEDDING_PATH}" ]; then
    EMBEDDING_PATH="$(ls -t logs/inference/runs/embeddings/${DATASET}_*/pickle/merged_predictions_tensor.pt 2>/dev/null | head -n1 || true)"
  fi
fi
if [ ! -f "${EMBEDDING_PATH}" ]; then
  echo "Content-embedding file not found: ${EMBEDDING_PATH:-<empty>}; pass it as arg 2." >&2
  exit 1
fi

# Catalog size N, read off the content embeddings. The CF table MUST have one
# row per catalog item, not one per item that happens to appear in training:
# a short table would make the tokenizer's CF lookup an out-of-range index for
# the tail of the catalog (it raises), and a silently reordered one would align
# every item with another item's behaviour.
N_ITEMS="$(python3 - "${EMBEDDING_PATH}" <<'PY'
import sys, torch
t = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
if isinstance(t, dict):
    t = t["embeddings"]
print(int(torch.as_tensor(t).shape[0]))
PY
)"

# steps = ceil(N / batch) * epochs, so every dataset gets the SAME number of
# passes over its catalog rather than the same number of steps.
LETTER_STEPS="${LETTER_STEPS:-$(( ( (N_ITEMS + LETTER_BATCH - 1) / LETTER_BATCH ) * LETTER_EPOCHS ))}"

JOB_ID="${SLURM_JOB_ID:-local$$}"
CF_PATH="${CF_PATH:-embeddings/${CF_TAG}_cf${CF_DIM}.pt}"
RUN_TAG="${DATASET}_letter_${JOB_ID}"
RUN_DIR="${GRID_DIR}/logs/train/runs/codebook/${RUN_TAG}"
LOCAL_CKPT_DIR="${TMPDIR:-/tmp}/letter_ckpts_${JOB_ID}"
OUT_DIR="embeddings/${DATASET}_letter"
[ "${LETTER_CF_VARIANT}" = "poison" ] && OUT_DIR="embeddings/${DATASET}_letter_poisoncf_${_CELL}"
# The GRID-convention space gets its own directory: the two grammars are not
# comparable, so they must never share a path.
[ "${LETTER_TOKENIZE}" = "dedup" ] && OUT_DIR="${OUT_DIR}_dedup"
[ "${CODEBOOK_WIDTH}" != "256" ] && OUT_DIR="${OUT_DIR}w${CODEBOOK_WIDTH}"
[ "${LETTER_HIER}" != "${_DEFAULT_HIER}" ] && OUT_DIR="${OUT_DIR}_h${LETTER_HIER}"
OUT_PATH="${OUT_DIR}/merged_predictions_tensor.pt"

echo "[$(date -Is)] LETTER identifier space for dataset=${DATASET}"
echo "  content embeddings : ${EMBEDDING_PATH}  (N=${N_ITEMS})"
echo "  CF variant         : ${LETTER_CF_VARIANT}  <- ${CF_DATA_DIR}"
echo "  CF embeddings      : ${CF_PATH}  (d=${CF_DIM})"
if [ "${LETTER_TOKENIZE}" = "paper" ]; then
  echo "  tokenization       : paper (no dedup digit, collision resampling)"
  echo "  codebooks          : ${LETTER_HIER} x ${CODEBOOK_WIDTH}  (-> L=${LETTER_HIER})"
else
  echo "  tokenization       : dedup (GRID convention)"
  echo "  codebooks          : ${LETTER_HIER} x ${CODEBOOK_WIDTH}  (+1 dedup -> L=$((LETTER_HIER + 1)))"
fi
echo "  training length    : ${LETTER_STEPS} steps (~${LETTER_EPOCHS} epochs at batch ${LETTER_BATCH})"
echo "  alpha/beta/sk_eps  : ${LETTER_ALPHA} / ${LETTER_BETA} / ${LETTER_SK_EPS}"
echo "  run dir            : ${RUN_DIR}"
echo "  output             : ${OUT_PATH}"

# --- Step A: collaborative embeddings --------------------------------------- #
if [ -f "${CF_PATH}" ] && [ "${CF_FORCE:-0}" != "1" ]; then
  echo "[$(date -Is)] Step A: reusing ${CF_PATH} (CF_FORCE=1 to rebuild)"
else
  echo "[$(date -Is)] Step A: training CF embeddings"
  python -u -m scripts.train_cf_embeddings \
    --data-dir "${CF_DATA_DIR}" \
    --out "${CF_PATH}" \
    --dim "${CF_DIM}" \
    --epochs "${CF_EPOCHS}" \
    --seed "${CF_SEED}" \
    --num-items "${N_ITEMS}"
fi

# The alignment term is a dot product between the quantised latent and the CF
# vector, so a dim mismatch is a hard error inside the module. Catch it here,
# before a multi-hour tokenizer job, and say which knob to change.
python3 - "${CF_PATH}" "${CF_DIM}" "${N_ITEMS}" <<'PY' || exit 1
import sys, torch
path, dim, n = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
t = torch.load(path, map_location="cpu", weights_only=False)
t = torch.as_tensor(t["embeddings"] if isinstance(t, dict) else t)
if t.dim() != 2 or t.shape[1] != dim:
    raise SystemExit(f"{path} is {tuple(t.shape)}, expected [N, {dim}]")
if t.shape[0] != n:
    raise SystemExit(
        f"{path} has {t.shape[0]} rows but the catalog has {n} items. Rebuild "
        f"with CF_FORCE=1, or point CF_PATH at the table built for this catalog."
    )
print(f"  CF table OK: {tuple(t.shape)}")
PY

# --- Step B: train the LETTER tokenizer -------------------------------------- #
mkdir -p "${RUN_DIR}" "${LOCAL_CKPT_DIR}"
echo "[$(date -Is)] Step B: training the LETTER tokenizer"
python -u -m src.train \
  experiment=letter_rqvae_train_flat \
  data_dir="${GRID_DATA_DIR}" \
  hydra.run.dir="${RUN_DIR}" \
  "embedding_path='${EMBEDDING_PATH}'" \
  "cf_embedding_path='${CF_PATH}'" \
  embedding_dim=2048 \
  num_hierarchies="${LETTER_HIER}" \
  codebook_width="${CODEBOOK_WIDTH}" \
  letter_alpha="${LETTER_ALPHA}" \
  letter_beta="${LETTER_BETA}" \
  letter_sinkhorn_epsilon="${LETTER_SK_EPS}" \
  trainer.max_steps="${LETTER_STEPS}" \
  "callbacks.model_checkpoint.dirpath=${LOCAL_CKPT_DIR}" \
  "${@:3}"

CKPT="$(ls -t "${LOCAL_CKPT_DIR}"/*.ckpt 2>/dev/null | head -n1 || true)"
if [ -z "${CKPT}" ]; then
  echo "[$(date -Is)] ERROR: no tokenizer checkpoint in ${LOCAL_CKPT_DIR}" >&2
  exit 1
fi
mkdir -p "${RUN_DIR}/checkpoints"
cp -f "${CKPT}" "${RUN_DIR}/checkpoints/"
CKPT="${RUN_DIR}/checkpoints/$(basename "${CKPT}")"
echo "[$(date -Is)] Step B done: ${CKPT}"

# --- Step C: assignment ------------------------------------------------------ #
# Both modes assign in float64 on CPU from the checkpoint's own encoder front
# end, and both take the deterministic argmin for the semantic levels -- that is
# LETTER's own use_sk=False at tokenization time, so none of the training-time
# terms take part here.
#
#   paper: LETTER_HIER semantic levels, then collisions broken by re-assigning
#          each collided group with Sinkhorn on the FINAL level, up to 20 rounds.
#   dedup: LETTER_HIER semantic levels, then GRID's appended dedup digit.
echo "[$(date -Is)] Step C: assigning semantic ids (${LETTER_TOKENIZE})"
mkdir -p "${OUT_DIR}"
if [ "${LETTER_TOKENIZE}" = "paper" ]; then
  python -u -m scripts.assign_sids_letter \
    --codebook_ckpt "${CKPT}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --out "${OUT_PATH}" \
    --sk_epsilon "${LETTER_SK_EPS}" \
    --max_rounds "${LETTER_COLLISION_ROUNDS:-20}" \
    --max_collision_rate "${LETTER_MAX_COLLISION_RATE:-0.01}" \
    --overwrite
else
  python -u -m scripts.assign_sids_exact \
    --codebook_ckpt "${CKPT}" \
    --embedding_path "${EMBEDDING_PATH}" \
    --out "${OUT_PATH}" \
    --overwrite
fi

python3 - "${OUT_PATH}" "${CODEBOOK_WIDTH}" "${LETTER_TOKENIZE}" <<'PY'
import sys, torch
t = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
width, mode = int(sys.argv[2]), sys.argv[3]
print(f"  semantic ids: shape={tuple(t.shape)} dtype={t.dtype} max_code={int(t.max())}")
# Under `dedup` the last row is a COUNTER sharing the 256-wide vocabulary, so a
# large collision bucket overflows it and the model would index past its
# embedding table (food hit 295 > 256). Under `paper` every position is a real
# code and this cannot happen -- kept for both because it costs nothing.
if int(t.max()) >= width:
    extra = (" the dedup digit has overflowed the vocabulary."
             if mode == "dedup" else
             " this should be impossible in paper mode -- investigate.")
    raise SystemExit(f"max code {int(t.max())} >= codebook width {width}:{extra}")
PY

echo "[$(date -Is)] Done. Train LETTER with:"
echo "  MODEL=letter sbatch run_tiger_train.sh ${DATASET} clean ${OUT_PATH}"
