#!/usr/bin/env bash
#SBATCH --job-name=clean_bl_bw
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --partition=pgpu
#SBATCH --time=04:00:00

# Clean-model baseline SH@k / ASI@k / TPM@k against each bandwagon target set I_t
# (unpopular / mid / popular). These are the SH_clean / U_clean references needed
# to compute PGR@k for the bandwagon unlearning sweep — and they matter most for
# the 'popular' target, whose item may be recommended by the clean model anyway
# (so its raw poison/unlearn SH is only interpretable relative to clean).
#
# One clean checkpoint, evaluated test-only on the CLEAN beauty test set, once
# per strategy manifest. Deterministic. Resumable (skips a strategy already done).
#
# Usage:  sbatch run_clean_baseline_bandwagon.sh                    # non-PKM clean model
#         PKM_ARCH=allv2  sbatch run_clean_baseline_bandwagon.sh   # clean ALLv2  (job 9985374)
#         PKM_ARCH=e23d01 sbatch run_clean_baseline_bandwagon.sh   # clean E23D01 (job 9985378)
#         IDSPACE=w16     sbatch run_clean_baseline_bandwagon.sh   # clean w16/L4  (job 10269231)
#         IDSPACE=w8l6    sbatch run_clean_baseline_bandwagon.sh   # clean w8/L6   (job 10269232)
#         DRYRUN=1 ./run_clean_baseline_bandwagon.sh
#
# PKM_ARCH picks a PKM-trained clean checkpoint and passes the matching
# model.pkm_layers / model.pkm_mode overrides (without them the ckpt will not
# load). Needed because the PKM unlearning tables' PGR@k all assume a clean SH
# floor of 0, which was never measured for these architectures.
#
# IDSPACE picks a NARROW-CODEBOOK clean checkpoint (the 2026-08-04 runs) and its
# matching SID tensor / num_hierarchies / sequence_length. UR and PGR are ratios
# against the clean model *in the same identifier space*, so the width-256
# denominators (clean NDCG@10 0.03762655) must NOT be reused for these — hence
# this knob. IDSPACE and PKM_ARCH are mutually exclusive.

set -euo pipefail
GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"; mkdir -p logs

DRYRUN="${DRYRUN:-0}"
EVAL_SEED="${EVAL_SEED:-2}"
export PYTHONHASHSEED="${EVAL_SEED}"
export CUBLAS_WORKSPACE_CONFIG="${CUBLAS_WORKSPACE_CONFIG:-:16:8}"

CLEAN_DATA_DIR="src/data/amazon_data/beauty"
SID="embeddings/beauty/merged_predictions_tensor.pt"
CLEAN_CKPT="logs/train/runs/2026-05-29/14-06-14_job9096928_beauty_clean/checkpoints/checkpoint_epoch=000_step=004600.ckpt"

# PKM-trained clean checkpoints (2026-07-14 fixed-encoder-detection reruns) +
# their required arch overrides. Keep the layer selection IDENTICAL to training.
PKM_ARCH="${PKM_ARCH:-none}"
PKM_OVR=()
ARCH_TOK=""
case "${PKM_ARCH}" in
  none) : ;;
  allv2)
    CLEAN_CKPT="logs/train/runs/2026-07-14/10-23-47_job9985374_beauty_clean_pkmreplace_ALLv2/checkpoints/checkpoint_epoch=000_step=004100.ckpt"
    PKM_OVR=("model.pkm_layers={encoder:all,decoder:all}" "model.pkm_mode=replace")
    ARCH_TOK="_pkmreplaceallv2" ;;
  e23d01)
    CLEAN_CKPT="logs/train/runs/2026-07-14/10-23-47_job9985378_beauty_clean_pkmreplace_E23D01v2/checkpoints/checkpoint_epoch=000_step=004500.ckpt"
    PKM_OVR=("model.pkm_layers={encoder:[2,3],decoder:[0,1]}" "model.pkm_mode=replace")
    ARCH_TOK="_pkmreplacee23d01" ;;
  *) echo "PKM_ARCH must be none|allv2|e23d01, got '${PKM_ARCH}'" >&2; exit 1 ;;
esac

# Narrow-codebook identifier spaces (2026-08-04 clean trains). The last RQ digit
# is a uniqueness/dedup counter in every space, so the effective semantic depth
# is H-1: w256 and w16 both have 3 semantic levels, w8/L6 has 5.
NUM_HIER=4
SEQ_LEN=""
IDSPACE="${IDSPACE:-none}"
case "${IDSPACE}" in
  none) : ;;
  w16)  # L=4, per-level width 16 (mean prefix-3 neighbourhood 21.7 items)
    CLEAN_CKPT="logs/train/runs/2026-08-04/18-53-53_job10269231_beauty_clean_h30_w16/checkpoints/checkpoint_epoch=000_step=003200.ckpt"
    SID="embeddings/beauty_L4w16/merged_predictions_tensor.pt"
    NUM_HIER=4; SEQ_LEN=120; ARCH_TOK="_idw16" ;;
  w8l6) # L=6, per-level width 8 (mean prefix-4 neighbourhood 27.7 items)
    CLEAN_CKPT="logs/train/runs/2026-08-04/18-53-53_job10269232_beauty_clean_L6_h30_w8/checkpoints/checkpoint_epoch=000_step=002600.ckpt"
    SID="embeddings/beauty_L6w8/merged_predictions_tensor.pt"
    NUM_HIER=6; SEQ_LEN=180; ARCH_TOK="_idw8l6" ;;
  tracerv3) # REBUILT width-256 RQ-KMeans codebook for beauty, so TRACER can run
    # there at all: the original is lost and a rebuild reproduces only
    # 0.01/0.38/0.35% of the stored ids, while TRACER refuses unless phi=0 is
    # exact. Same L=4/width=256 shape as the original, but a DIFFERENT
    # identifier space, so it needs its own clean denominators.
    CLEAN_CKPT="logs/train/runs/2026-09-08/03-32-17_job10652133_beauty_clean_tracerv3/checkpoints/checkpoint_epoch=000_step=004500.ckpt"
    SID="embeddings/beauty_tracer_v3/merged_predictions_tensor.pt"
    NUM_HIER=4; ARCH_TOK="_tracerv3" ;;
  *) echo "IDSPACE must be none|w16|w8l6|tracerv3, got '${IDSPACE}'" >&2; exit 1 ;;
esac
if [ "${IDSPACE}" != "none" ] && [ "${PKM_ARCH}" != "none" ]; then
  echo "IDSPACE and PKM_ARCH are mutually exclusive (no PKM narrow-codebook models exist)" >&2
  exit 1
fi
SEQ_OVR=()
[ -n "${SEQ_LEN}" ] && SEQ_OVR=("sequence_length=${SEQ_LEN}")

# DATASET generalises this script beyond beauty. The clean SH floor is a
# PER-DATASET quantity and PGR divides by (SH_poison - SH_clean), so a new
# dataset cannot quote PGR until this has been run for it. On beauty the floor
# was exactly 0 for unpopular/mid but 0.00125 for popular, where ignoring it
# turns PGR 3.29 into 1.00 -- a 3x error, not a rounding one.
# For a non-beauty dataset, pass CLEAN_CKPT explicitly (the beauty checkpoints
# above are hardcoded) and the data dir / SID resolve from the dataset name.
DATASET="${DATASET:-beauty}"
if [ "${DATASET}" != "beauty" ]; then
  # shellcheck source=scripts/resolve_grid_dataset.sh
  source "${GRID_DIR:-$PWD}/scripts/resolve_grid_dataset.sh"
  resolve_grid_dataset "${DATASET}" || exit 1
  CLEAN_DATA_DIR="${GRID_DATA_DIR}"
  SID="${SID_OVERRIDE:-${GRID_SEMANTIC_ID_PATH}}"
  if [ -z "${CLEAN_CKPT_OVERRIDE:-}" ]; then
    echo "DATASET=${DATASET} needs CLEAN_CKPT_OVERRIDE=<clean checkpoint path>" >&2
    exit 1
  fi
  CLEAN_CKPT="${CLEAN_CKPT_OVERRIDE}"
  ARCH_TOK="${ARCH_TOK:-}_${DATASET}"
fi

# beauty is the historical default and had its checkpoint and SID hardcoded
# above, which made two cells of the grid unreachable:
#   * the RQ-VAE track needs embeddings/beauty_rqvae, and
#   * three run dirs are labelled `beauty_clean` (jobs 9096928 / 9465564 /
#     9642098) with test nDCG@10 spanning 7.5%, so WHICH one is the denominator
#     has to be stated, not left to a hardcoded path or to mtime.
# CLEAN_CKPT_OVERRIDE and SID_OVERRIDE now apply to every dataset.
[ -n "${CLEAN_CKPT_OVERRIDE:-}" ] && CLEAN_CKPT="${CLEAN_CKPT_OVERRIDE}"

# Track tokens. UR and PGR are ratios against the clean model IN THE SAME
# identifier space, so a reference dir has to name its track and seed: the three
# grid tracks (tiger+rkmeans, tiger+rqvae, diger+rqvae) are different identifier
# spaces whose numbers are not comparable, and the timestamp alone does not say
# which one produced a directory.
# POISON_SEED picks WHICH I_t the clean model is scored against. It used to be
# hardcoded to seed 2, which is wrong for every other seed of the grid: each seed
# has its own poison generation and therefore its own target set, and PGR divides
# by (poison_SH - clean_SH), so a seed-2 floor under a seed-7 numerator is a
# silent error in the denominator rather than a missing number.
POISON_SEED="${POISON_SEED:-2}"

MODEL="${MODEL:-tiger}"
QUANTIZER="${QUANTIZER:-rkmeans}"
[ "${MODEL}" != "tiger" ] && ARCH_TOK="${ARCH_TOK}_${MODEL}"
[ "${QUANTIZER}" != "rkmeans" ] && ARCH_TOK="${ARCH_TOK}_${QUANTIZER}"
# The identifier space has to match the checkpoint's own. An explicit
# SID_OVERRIDE always wins; otherwise a non-rkmeans track resolves to that
# quantizer's installed tensor.
if [ -n "${SID_OVERRIDE:-}" ]; then
  SID="${SID_OVERRIDE}"
elif [ "${QUANTIZER}" != "rkmeans" ]; then
  SID="embeddings/${DATASET}_${QUANTIZER}/merged_predictions_tensor.pt"
fi
[ -f "${SID}" ] || { echo "SID tensor missing: ${SID}" >&2; exit 1; }
[ "${POISON_SEED}" != "2" ] && ARCH_TOK="${ARCH_TOK}_seed${POISON_SEED}"

man_for() {  # strategy -> bandwagon manifest for POISON_SEED
  case "$1" in
    unpopular) echo "src/data/amazon_data/${DATASET}_spam_seed${POISON_SEED}_pct1_n1/forget_manifest.json" ;;
    *)         echo "src/data/amazon_data/${DATASET}_spam_tgt${1}_seed${POISON_SEED}_pct1_n1/forget_manifest.json" ;;
  esac
}

STAMP="${STAMP:-$(date +%Y-%m-%d_%H-%M-%S)}"
OUT="logs/eval/clean_baseline_bandwagon${ARCH_TOK}/${STAMP}"
[ "${DRYRUN}" = "1" ] || mkdir -p "${OUT}"
echo "[$(date -Is)] clean baseline vs bandwagon I_t -> ${OUT}"
[ -f "${CLEAN_CKPT}" ] || { echo "clean ckpt missing: ${CLEAN_CKPT}"; exit 1; }

for s in unpopular mid popular; do
  man="$(man_for "$s")"
  rundir="${OUT}/clean_vs_${s}"
  echo "== clean vs ${s}: I_t manifest ${man}"
  [ -f "${man}" ] || { echo "  manifest missing, skip"; continue; }
  [ -f "${rundir}/csv/version_0/metrics.csv" ] && { echo "  already done, skip"; continue; }
  [ "${DRYRUN}" = "1" ] && continue
  python -u -m scripts.eval_ckpt_on_test \
    experiment=tiger_train_flat \
    data_dir="${CLEAN_DATA_DIR}" \
    "semantic_id_path='${SID}'" \
    "ckpt_path='${CLEAN_CKPT}'" \
    "spam_forget_manifest='${man}'" \
    ${PKM_OVR[@]+"${PKM_OVR[@]}"} \
    ${SEQ_OVR[@]+"${SEQ_OVR[@]}"} \
    num_hierarchies="${NUM_HIER}" seed="${EVAL_SEED}" \
    train=False test=True \
    trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false \
    trainer.num_nodes=1 trainer.deterministic=true \
    callbacks.model_checkpoint=null callbacks.early_stopping=null \
    hydra.run.dir="${rundir}" || echo "  FAILED: clean_vs_${s}"
done

[ "${DRYRUN}" = "1" ] && { echo "DRYRUN done"; exit 0; }
echo "=== clean baseline summary (SH@10 / ASI@10 / TPM@10 per I_t) ==="
python - "${OUT}" <<'PY'
import csv,glob,math,os,sys
out=sys.argv[1]
def last(p):
    m={}
    with open(p,newline="") as f:
        r=csv.DictReader(f); cols=[c for c in (r.fieldnames or []) if c.startswith("test/")]
        m={c:math.nan for c in cols}
        for row in r:
            for c in cols:
                v=row.get(c,"")
                if v not in ("",None):
                    try:
                        fv=float(v)
                        if not math.isnan(fv): m[c]=fv
                    except ValueError: pass
    return {k:v for k,v in m.items() if not math.isnan(v)}
rows=[]
for mc in sorted(glob.glob(os.path.join(out,"clean_vs_*","csv","version_0","metrics.csv"))):
    s=mc.split(os.sep)[-4].replace("clean_vs_","")
    m=last(mc)
    rows.append([s, m.get("test/SH@10"), m.get("test/ASI@10"), m.get("test/TPM@10"), m.get("test/ndcg@10")])
with open(os.path.join(out,"clean_baseline_summary.csv"),"w",newline="") as f:
    w=csv.writer(f); w.writerow(["strategy","SH@10","ASI@10","TPM@10","ndcg@10"]); w.writerows(rows)
for r in rows: print("  ", r[0], "SH@10="+str(r[1]), "TPM@10="+str(r[2]))
print("Wrote", os.path.join(out,"clean_baseline_summary.csv"))
PY
cat > "${OUT}/coordinates.json" <<JSON
{
  "dataset": "${DATASET}",
  "model": "${MODEL}",
  "quantizer": "${QUANTIZER}",
  "poison_seed": ${POISON_SEED},
  "eval_seed": ${EVAL_SEED},
  "clean_ckpt": "${CLEAN_CKPT}",
  "semantic_id_path": "${SID}",
  "clean_data_dir": "${CLEAN_DATA_DIR}"
}
JSON
echo "Wrote ${OUT}/coordinates.json"
echo "[$(date -Is)] clean baseline done -> ${OUT}"
