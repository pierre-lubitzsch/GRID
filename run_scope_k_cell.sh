#!/usr/bin/env bash
#SBATCH --job-name=scope_k
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=02:00:00
#
# Neighborhood (Figure 2) evaluation of NAU with different neighborhood sizes k, one Beauty
# cell (STRATUM, SEED). Arms: nau_k<K> for K in 2 4 16 32 64 (logs/nau_k_keepck_plan.tsv runs,
# checkpoints kept), nau_k8 (the Figure 2 NAU checkpoint), and Forget+Repair (Figure 2).
# Output: logs/diagnose/<OUT_STEM>/beauty[_<stratum>]_seed<seed>.json (default fig1_scope_k, test split).
# DATA_VIEW=_valview OUT_STEM=fig1_scope_k_val scores the same checkpoints on the validation split.
#
#   STRATUM=mid SEED=2 sbatch run_scope_k_cell.sh
#   DATA_VIEW=_valview OUT_STEM=fig1_scope_k_val STRATUM=mid SEED=2 sbatch run_scope_k_cell.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ST="${STRATUM:?}"; SD="${SEED:?}"
VIEW="${DATA_VIEW:-}"; STEM="${OUT_STEM:-fig1_scope_k}"
mapfile -t ARGS < <(python3 - "${ST}" "${SD}" <<'EOF'
import glob, os, re, sys
sys.path.insert(0, "scripts")
import importlib.util
spec = importlib.util.spec_from_file_location("f", "scripts/run_fig1_js_eval.py"); f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
st, sd = sys.argv[1], int(sys.argv[2])
led = f.unlearned_checkpoints("beauty", sd, st)
out = []
fr = led.get("forget_repair") or led.get("nosep_ln0")
if led.get("nau_s0p01_n0p01"):
    out.append(f"nau_k8={led['nau_s0p01_n0p01']}")
if fr:
    out.append(f"forget_repair={fr}")
sfx = "" if sd == 2 else f"_seed{sd}"
for k in (2, 4, 16, 32, 64):
    runs = glob.glob(f"logs/unlearn/runs/*_bs1_bw_tgt{st}_nau_s0p01_n0p01_k{k}_keepck{sfx}")
    runs = [r for r in runs if os.path.exists(r + "/checkpoints/unlearned.ckpt")]
    if runs:
        best = max(runs, key=lambda r: int(re.search(r"job(\d+)_", r).group(1)))
        out.append(f"nau_k{k}={os.path.realpath(best + '/checkpoints/unlearned.ckpt')}")
    else:
        print(f"missing k={k} for {st} seed {sd}", file=sys.stderr)
for a in out:
    print("--ckpt"); print(a)
EOF
)
echo "arms: ${ARGS[*]}"
python -u scripts/run_fig1_js_eval.py --dataset beauty --seed "${SD}" --stratum "${ST}" \
  --metric scope --all-unrelated --out-stem "${STEM}" ${VIEW:+--data-view "${VIEW}"} "${ARGS[@]}"
