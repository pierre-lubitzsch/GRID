#!/usr/bin/env bash
#SBATCH --job-name=latent_wsweep
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=03:00:00

# Can the refined ID space be made USEFUL for TIGER unlearning?
#
# The 3-seed result said no: N_z neither beat nor lost to N_emb on `mid`, the only
# stratum with headroom. The Step-2 diagnostic explains why -- at the default
# weights (rec 1.0, nbr 1.0, sid 0.1) the two neighbourhoods overlap at 0.778 at
# k=8, i.e. N_z is ~78% the SAME items as N_emb. Only ~1.8 of 8 neighbours differ,
# which bounds any downstream effect to roughly nothing.
#
# So the question is not "does N_z help" but "can N_z be made DIFFERENT while
# staying sensible". This sweep answers that on CPU in minutes, BEFORE any GPU
# unlearning run. It is the gate the earlier work should have passed first.
#
# The two objectives pull in opposite directions:
#   w_nbr  aligns z with the x-space top-k  -> drives overlap with N_emb TOWARD 1
#   w_sid  aligns z with the SID prefix     -> drives it AWAY, toward the decoder's
#                                              own partition
# w_rec (reconstruct x from z) anchors both; at 0 the latent is free to collapse.
#
# WHAT COUNTS AS SUCCESS: overlap well below 0.778 AND prefix recovery that does
# not collapse. A configuration with overlap ~0 and prefix recovery ~0 is not a
# refined space, it is noise -- and noise is guaranteed not to help unlearning.
# The useful band is a neighbourhood that is materially different from N_emb while
# still tracking SID structure at least as well as N_emb does (0.594 baseline).
#
#   sbatch run_latent_weight_sweep.sh
set -uo pipefail
_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" 2>/dev/null && pwd)"
[ -n "${_ROOT}" ] && [ -f "${_ROOT}/.project-root" ] || _ROOT="${SLURM_SUBMIT_DIR:-$(pwd)}"
cd "${_ROOT}"
echo "## repo root: ${_ROOT}"

DATASET="${DATASET:-beauty}"
SID="${SID:-embeddings/beauty/merged_predictions_tensor.pt}"
EMB="${EMB:-embeddings/beauty_merged_predictions_tensor_latest.pt}"
JID="${JID:-9872117}"                     # mid: the only stratum with headroom
OUT_ROOT="${OUT_ROOT:-embeddings/latent_sweep}"
EPOCHS="${EPOCHS:-30}"
DIAG_K="${DIAG_K:-8}"
mkdir -p "${OUT_ROOT}"

rundir="$(ls -1d logs/train/runs/*/*job${JID}_* 2>/dev/null | head -1)"
ckpt="$(ls -t "${rundir}"/checkpoints/*.ckpt 2>/dev/null | head -1)"
[ -n "${ckpt}" ] || { echo "no ckpt for job ${JID}" >&2; exit 1; }
echo "## ckpt=${ckpt}"

# name:w_rec:w_nbr:w_sid
CONFIGS="${CONFIGS:-\
baseline:1.0:1.0:0.1 \
sid_up:1.0:1.0:1.0 \
nbr_down:1.0:0.2:1.0 \
nbr_off:1.0:0.0:1.0 \
sid_dom:1.0:0.0:3.0 \
rec_light:0.2:0.0:2.0 \
sid_off:1.0:1.0:0.0}"

for cfg in ${CONFIGS}; do
  name="${cfg%%:*}"; rest="${cfg#*:}"
  wr="${rest%%:*}"; rest="${rest#*:}"
  wn="${rest%%:*}"; ws="${rest##*:}"
  out="${OUT_ROOT}/${DATASET}_bw_tgtmid_${name}_dz128.pt"
  echo
  echo "### ${name}: w_rec=${wr} w_nbr=${wn} w_sid=${ws}"
  python -u -m scripts.train_latent_refiner \
    --ckpt "${ckpt}" --embedding_path "${EMB}" --semantic_id_path "${SID}" \
    --num_hierarchies 4 --out "${out}" \
    --latent_dim 128 --hidden_dim 512 \
    --w_reconstruction "${wr}" --w_neighborhood "${wn}" --w_sid_consistency "${ws}" \
    --neighbor_k 8 --temperature 0.07 --sid_prefix_length 2 \
    --epochs "${EPOCHS}" --batch_size 512 --lr 1e-3 --seed 2 --diag_k "${DIAG_K}" \
    2>&1 | grep -E "\[diag\]|wrote|Error|Traceback" | head -6
done

echo
echo "=============================================================="
python - "${OUT_ROOT}" <<'PY'
import glob, json, os, sys, re
rows = []
for p in sorted(glob.glob(os.path.join(sys.argv[1], "*.stats.json"))):
    d = json.load(open(p))
    c, g = d["config"], d["diagnostics"]
    lve, pfx = g["latent_vs_embedding"], g["prefix_sample"]
    m = re.search(r"tgtmid_(\w+?)_dz", os.path.basename(p))
    rows.append((m.group(1) if m else "?", c["w_reconstruction"], c["w_neighborhood"],
                 c["w_sid_consistency"], lve["mean_overlap_at_k"], lve["frac_identical"],
                 pfx["mean_prefix_recovered_by_latent"],
                 pfx["mean_prefix_recovered_by_embedding"],
                 d["training"]["final"].get("rec", float("nan"))))
rows.sort(key=lambda r: r[4])
print(f"{'config':11s} {'w_rec':>5s} {'w_nbr':>5s} {'w_sid':>5s} | "
      f"{'overlap':>7s} {'ident%':>6s} | {'pfx_lat':>7s} {'pfx_emb':>7s} | {'L_rec':>6s}  verdict")
print("-" * 96)
for n, wr, wn, ws, ov, ident, pl, pe, rec in rows:
    if ov > 0.7:   v = "too similar to N_emb"
    elif pl < 0.3: v = "NOISE - prefix structure lost"
    elif ov < 0.6 and pl >= pe: v = "*** USEFUL: different AND structured"
    else:          v = "different, weaker structure"
    print(f"{n:11s} {wr:5.1f} {wn:5.1f} {ws:5.1f} | {ov:7.3f} {100*ident:5.1f}% | "
          f"{pl:7.3f} {pe:7.3f} | {rec:6.3f}  {v}")
PY
echo "[$(date -Is)] done -> ${OUT_ROOT}"
