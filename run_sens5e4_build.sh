#!/usr/bin/env bash
#SBATCH --job-name=sens5e4_build
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --partition=compute
#SBATCH --time=12:00:00
# -----------------------------------------------------------------------------
# Sensitive item deletion, LARGER-REQUEST variant (rebuttal; budget token 5e-4 = 0.05% of all
# training interactions instead of the paper's 1e-4). CPU only. For each (dataset, seed):
#   1. scripts/build_sensitive_setup.sh with FORGET_RATIO=5e-4 (manifest in the clean dir,
#      <ds>_sens_<cat>_5e-4_seed<N> with forget/retain split, <ds>_retrain_<cat>_5e-4_seed<N>);
#   2. scripts/build_scrubbed_eval_view.py --ratio 5e-4 (cleaned test/val views, protocol A).
# Disk: the sens and retrain dirs (~2 GB per cell) live on /sc-projects and are SYMLINKED into
# src/data/amazon_data (scratch had ~66 GB free); the small scrub views stay on scratch.
# Idempotent: every step skips what exists.
#   sbatch --nice=500 run_sens5e4_build.sh
# -----------------------------------------------------------------------------
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
RATIO="${RATIO:-5e-4}"
DATASETS="${DATASETS:-beauty sports toys}"
SEEDS="${SEEDS:-2 3 5 7 11}"
STORE="${STORE:-/sc-projects/sc-proj-ikim-erase/GRID/sensitive_${RATIO}_data}"
ROOT=src/data/amazon_data
export TF_CPP_MIN_LOG_LEVEL=2
declare -A CAT=([beauty]=hairloss [sports]=guns [toys]=toyweapons)
mkdir -p "${STORE}" logs/sens5e4
for ds in ${DATASETS}; do
  for sd in ${SEEDS}; do
    for kind in sens retrain; do
      name="${ds}_${kind}_${CAT[$ds]}_${RATIO}_seed${sd}"
      mkdir -p "${STORE}/${name}"
      [ -e "${ROOT}/${name}" ] || ln -s "${STORE}/${name}" "${ROOT}/${name}"
    done
    FORGET_RATIO="${RATIO}" bash scripts/build_sensitive_setup.sh "${ds}" "${sd}" 2>&1 \
      | tee "logs/sens5e4/build_${ds}_s${sd}.log"
  done
  todo=()
  for sd in ${SEEDS}; do
    [ -d "${ROOT}/${ds}_sens_${CAT[$ds]}_${RATIO}_seed${sd}_scrubtestview" ] || todo+=("${sd}")
  done
  if [ ${#todo[@]} -gt 0 ]; then
    python -u -m scripts.build_scrubbed_eval_view --ratio "${RATIO}" --datasets "${ds}" --seeds "${todo[@]}" 2>&1 \
      | tee "logs/sens5e4/scrub_${ds}.log"
  fi
done
# summary of the realized request sizes
python - "${RATIO}" <<'EOF' | tee logs/sens5e4/manifest_summary.tsv
import json, sys
r = sys.argv[1]
cat = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
print("dataset\tseed\tn_requesting_users\tn_forget_interactions\trealized_ratio\tn_users_touching_category\tnon_requesting_category_users")
for ds, c in cat.items():
    for sd in (2, 3, 5, 7, 11):
        try:
            m = json.load(open(f"src/data/amazon_data/{ds}/forget_manifest_{c}_{r}_seed{sd}.json"))
        except FileNotFoundError:
            continue
        print(f"{ds}\t{sd}\t{m['n_spam_users']}\t{m['n_forget_interactions']}\t{m['realized_forget_ratio']}\t"
              f"{m['n_users_touching_category']}\t{m['n_users_touching_category'] - m['n_spam_users']}")
EOF
echo "[$(date -Is)] done"
