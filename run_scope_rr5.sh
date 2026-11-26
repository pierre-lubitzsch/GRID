#!/usr/bin/env bash
#SBATCH --job-name=scope_rr5
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#
# Retrained-vs-retrained per-query reference for ALL 45 spam cells (reviewer QzxM follow-up 1:
# per-user target exposure when retrained exposure is nonzero). The five clean (retrained) TIGER
# models of a dataset are trained on identical data with different training seeds
# (run_retrain_floor.sh). For cell (dataset, stratum, seed s) this scores the four OTHER clean
# models (arms clean_s<s'>) with the Figure 1/2/3 scope evaluator on the identical test queries of
# that cell, next to the cell's canonical clean and poisoned models (re-scored in every JSON).
# Output: logs/diagnose/fig1_scope_rr5/<ds>[_<stratum>]_seed<s>.json
#
#   sbatch run_scope_rr5.sh "beauty sports toys"      # datasets handled by this job
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
STEM=fig1_scope_rr5
DATASETS="${1:-beauty sports toys}"
mkdir -p "logs/diagnose/${STEM}" logs/scope_rr5
ck() {  # canonical clean checkpoint of (dataset, seed), as run_retrain_floor.sh resolves it
  python3 -c "
import json, yaml
ds, sd = '$1', int('$2')
p = 'logs/eval/spam_references/' + ds + ('' if sd == 2 else f'_seed{sd}') + '.json'
prov = json.load(open(p))['provenance']['clean_vs_mid']
ov = yaml.safe_load(open(prov + '/.hydra/overrides.yaml'))
print([o.split('=', 1)[1].strip(\"'\") for o in ov if o.startswith('ckpt_path=')][0])"
}
for ds in ${DATASETS}; do
  declare -A CK=()
  for s in 2 3 5 7 11; do CK[$s]="$(ck "${ds}" "${s}")"; done
  for st in popular mid unpopular; do
    for s in 2 3 5 7 11; do
      name="${ds}_${st}_seed${s}.json"; [ "${st}" = mid ] && name="${ds}_seed${s}.json"
      if [ -s "logs/diagnose/${STEM}/${name}" ]; then echo "exists: ${name}"; continue; fi
      args=()
      for o in 2 3 5 7 11; do [ "${o}" = "${s}" ] || args+=(--ckpt "clean_s${o}=${CK[$o]}"); done
      echo "[$(date -Is)] ${ds} ${st} seed ${s}"
      python -u scripts/run_fig1_js_eval.py --dataset "${ds}" --seed "${s}" --stratum "${st}" \
        --metric scope --all-unrelated --out-stem "${STEM}" "${args[@]}" \
        > "logs/scope_rr5/${SLURM_JOB_ID:-local}_${ds}_${st}_s${s}.out" 2>&1 \
        && echo "  ok" || echo "  FAILED (logs/scope_rr5/${SLURM_JOB_ID:-local}_${ds}_${st}_s${s}.out)"
    done
  done
  unset CK
done
echo "[$(date -Is)] done"
