#!/usr/bin/env bash
#SBATCH --job-name=valsel_refs
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=05:00:00
#
# Validation-split references for the hyperparameter selection: the CLEAN (retrained)
# and POISONED models of every (stratum, seed) cell of one dataset, evaluated on
# src/data/amazon_data/<ds>_valview (testing/ -> evaluation/), with the cell's spam
# manifest so SH@10 is scored. For beauty also the TRACER (tracerv3) references.
# Clean checkpoints use the same selection as the existing test references
# (scripts/plan_reference_evals.ckpt_of); poisoned checkpoints are the exact files the
# candidate runs start from (logs/valsel_plan.tsv).
# Output: logs/diagnose/valsel_refs/<track>_<ds>_<stratum>_s<seed>_<clean|poison>/
#
#   DATASET=beauty sbatch run_valsel_refs.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DATASET="${DATASET:?}"
VALVIEW="src/data/amazon_data/${DATASET}_valview"
python3 - "${DATASET}" <<'EOF' > "logs/valsel_refs_${DATASET}_${SLURM_JOB_ID:-local}.tsv"
import glob, os, re, sys
sys.path.insert(0, "scripts")
from plan_reference_evals import ckpt_of
ds = sys.argv[1]
plan = [l.rstrip("\n").split("\t") for l in open("logs/valsel_plan.tsv")]
seen = set()
for (d, st, sd, useed, tag, ckpt, algo, sid, na, rbs, rate, data_dir, tmpl, extra) in plan:
    if d != ds: continue
    track = "tracerv3" if algo == "tracer" and ds == "beauty" else "rkmeans"
    if (track, st, sd) in seen: continue
    seen.add((track, st, sd))
    tok = "" if sd == "2" else f"_seed{sd}"
    lab = f"{ds}_clean{tok}" + ("_tracerv3" if track == "tracerv3" else "")
    cands = [p for p in glob.glob(f"logs/train/runs/*/*_job*_{lab}")
             if re.match(rf"^\d{{2}}-\d{{2}}-\d{{2}}_job\d+_{re.escape(lab)}$", os.path.basename(p))]
    if ds == "beauty" and sd == "2" and track == "rkmeans":
        cands = glob.glob("logs/train/runs/*/*_job9096928_beauty_clean")
    assert len(cands) == 1, (lab, cands)
    clean = ckpt_of(cands[0]); assert clean and os.path.isfile(clean), (lab, clean)
    for kind, ck in (("clean", clean), ("poison", ckpt)):
        print("\t".join([track, st, sd, kind, ck, sid, data_dir]))
EOF
while IFS=$'\t' read -r track st sd kind ck sid data_dir; do
  out="logs/diagnose/valsel_refs/${track}_${DATASET}_${st}_s${sd}_${kind}"
  if [ -s "${out}/csv/version_0/metrics.csv" ]; then echo "skip ${out}"; continue; fi
  echo "== ${track} ${st} s${sd} ${kind}: ${ck}"
  python -u -m scripts.eval_ckpt_on_test \
    experiment="tiger_train_flat" data_dir="${VALVIEW}" \
    "semantic_id_path='${sid}'" "ckpt_path='${ck}'" \
    "spam_forget_manifest='${data_dir}/forget_manifest.json'" \
    num_hierarchies=4 seed="${sd}" train=False test=True \
    trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
    trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
    hydra.run.dir="${out}" || echo "FAILED ${out}"
done < "logs/valsel_refs_${DATASET}_${SLURM_JOB_ID:-local}.tsv"
