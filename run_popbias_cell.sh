#!/usr/bin/env bash
#SBATCH --job-name=popbias
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#
# Does unlearning shift recommendations toward popular items? One Beauty spam cell: top-10
# recommendations (tiger_inference_flat, top_k_for_generation=10) on the clean test users for the
# clean (retrained) model, the poisoned model the runs start from, NAU and Forget+Repair (the
# Figure 2 checkpoints). Analysis: paper/plotting/popularity_bias.py.
# Output: logs/diagnose/popbias/beauty_<stratum>_s<seed>/<model>/pickle/
#
#   STRATUM=mid SEED=2 sbatch run_popbias_cell.sh
#
# LEDGER=paper/plotting/canonical_runs.tsv: NAU and Forget+Repair checkpoints from the ONE canonical
# execution of Table 1's configuration (configs nau_ls0.01_ln0.01 and forget_repair_lf0.1, column
# ckpt_resolved), output under logs/diagnose/popbias_ledger/. A model whose checkpoint (realpath) equals
# the one already scored under logs/diagnose/popbias/ is symlinked, not re-scored; the old outputs are
# never touched. Analysis: popularity_bias.py --root logs/diagnose/popbias_ledger.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
ST="${STRATUM:?}"; SD="${SEED:?}"
SID="embeddings/beauty/merged_predictions_tensor.pt"
OLD="logs/diagnose/popbias/beauty_${ST}_s${SD}"
OUT="${OLD}"
[ -n "${LEDGER:-}" ] && OUT="logs/diagnose/popbias_ledger/beauty_${ST}_s${SD}"
mapfile -t PAIRS < <(LEDGER="${LEDGER:-}" python3 - "${ST}" "${SD}" <<'EOF'
import glob, os, re, sys
import importlib.util
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("f", "scripts/run_fig1_js_eval.py"); f = importlib.util.module_from_spec(spec); spec.loader.exec_module(f)
from plan_reference_evals import ckpt_of
st, sd = sys.argv[1], sys.argv[2]
lab = "beauty_clean" + ("" if sd == "2" else f"_seed{sd}")
c = [p for p in glob.glob(f"logs/train/runs/*/*_job*_{lab}") if re.match(rf"^\d{{2}}-\d{{2}}-\d{{2}}_job\d+_{lab}$", os.path.basename(p))]
if sd == "2":
    c = glob.glob("logs/train/runs/*/*_job9096928_beauty_clean")
assert len(c) == 1, c
print(f"clean={ckpt_of(c[0])}")
for l in open("logs/poison_ref_start_plan.tsv"):
    ds, s, d, ck = l.split("\t")[:4]
    if ds == "beauty" and s == st and d == sd:
        print(f"poisoned={ck}")
if os.environ.get("LEDGER"):
    import csv
    L = {r["config"]: r["ckpt_resolved"] for r in csv.DictReader(open(os.environ["LEDGER"]), delimiter="\t")
         if r["setting"] == "spam" and r["dataset"] == "beauty" and r["track"] == "tiger+rkmeans"
         and r["stratum"] == st and r["seed"] == sd and r["config"] in ("nau_ls0.01_ln0.01", "forget_repair_lf0.1")}
    assert len(L) == 2 and all(os.path.isfile(v) for v in L.values()), L
    print(f"nau={L['nau_ls0.01_ln0.01']}")
    print(f"forget_repair={L['forget_repair_lf0.1']}")
else:
    led = f.unlearned_checkpoints("beauty", int(sd), st)
    print(f"nau={led['nau_s0p01_n0p01']}")
    print(f"forget_repair={led.get('forget_repair') or led.get('nosep_ln0')}")
EOF
)
same_ckpt() {  # $1 model dir already scored, $2 checkpoint: same file (realpath)?
  local ov="$1/.hydra/overrides.yaml"
  [ -e "$1/pickle/done" ] && [ -f "$ov" ] || return 1
  local old; old=$(grep -oP "ckpt_path='?\K[^'\n]+" "$ov" | head -1)
  [ "$(realpath "$old")" = "$(realpath "$2")" ]
}
for pair in "${PAIRS[@]}"; do
  name="${pair%%=*}"; ck="${pair#*=}"
  [ -s "${OUT}/${name}/pickle/done" ] && { echo "skip ${name}"; continue; }
  [ "${OUT}" != "${OLD}" ] && [ -e "${OUT}/${name}/pickle/done" ] && { echo "skip ${name}"; continue; }
  if [ "${OUT}" != "${OLD}" ] && same_ckpt "${OLD}/${name}" "${ck}"; then
    mkdir -p "${OUT}"; ln -sfn "$(realpath "${OLD}/${name}")" "${OUT}/${name}"; echo "link ${name} (same checkpoint)"; continue
  fi
  echo "== ${name}: ${ck}"
  python -u -m src.inference experiment=tiger_inference_flat data_dir=src/data/amazon_data/beauty \
    "semantic_id_path='${SID}'" "ckpt_path='${ck}'" num_hierarchies=4 seed="${SD}" model.pkm_layers=null \
    paths.output_dir="${OUT}/${name}" hydra.run.dir="${OUT}/${name}" \
    && touch "${OUT}/${name}/pickle/done" || echo "FAILED ${name}"
done
