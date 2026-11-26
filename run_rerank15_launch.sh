#!/usr/bin/env bash
# Decode-time re-ranking baseline on all 15 Beauty cells (3 targets x 5 seeds); extends the testing
# tree's C8 (Beauty mid only). Per cell, beam-50 test evals (scripts/eval_ckpt_rerank.py) of
#   poison : the poisoned checkpoint the canonical runs start from (ckpt_path of the ledger
#            forget_repair_lf0.1 run), none + 6 re-rankers
#   nau    : canonical NAU (ledger nau_ls0.01_ln0.01), none + 6 re-rankers
#   clean  : the seed's clean (retrained) model (logs/valsel_refs_beauty_10904023.tsv), none
# Re-rankers: clean CF / poisoned CF (SASRec d64, run_rerank15_cf.sh) / content embedding
# (embeddings/beauty_merged_predictions_tensor_latest.pt), alpha 3 and 10.
# Outputs logs/rerank15/beauty_<st>_seed<sd>/<model>/beam50_<spec>/. Table: paper/plotting/rerank15_table.py.
#   bash run_rerank15_launch.sh ; DRYRUN=0 bash run_rerank15_launch.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
mkdir -p logs/rerank15
python3 - <<'EOF'
import csv, os, yaml
L = list(csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t"))
refs = {}
for line in open("logs/valsel_refs_beauty_10904023.tsv"):
    f = line.rstrip("\n").split("\t")
    if f[0] == "rkmeans" and f[3] == "clean":
        refs[(f[1], int(f[2]))] = f[4]
X = "embeddings/beauty_merged_predictions_tensor_latest.pt"
CF = "embeddings/cf/rerank15"
rows, cells = [], []
for st in ("unpopular", "mid", "popular"):
    for sd in (2, 3, 5, 7, 11):
        get = lambda cfg: next(r for r in L if r["setting"] == "spam" and r["dataset"] == "beauty"
                               and r["config"] == cfg and r["stratum"] == st and int(r["seed"]) == sd)
        fr, nau = get("forget_repair_lf0.1"), get("nau_ls0.01_ln0.01")
        kv = dict(o.split("=", 1) for o in yaml.safe_load(open(fr["run_dir"] + "/.hydra/overrides.yaml")))
        kv2 = dict(o.split("=", 1) for o in yaml.safe_load(open(nau["run_dir"] + "/.hydra/overrides.yaml")))
        pck = kv["ckpt_path"].strip("'\""); assert pck == kv2["ckpt_path"].strip("'\""), (pck, kv2["ckpt_path"])
        pdir = kv["data_dir"]; assert pdir == kv2["data_dir"]
        base = os.path.basename(pck.split("/checkpoints/")[0])
        assert base.endswith(f"_tgt{st}") and ((sd == 2 and "_seed" not in base) or f"_seed{sd}_" in base), pck
        cck = refs[(st, sd)]; assert os.path.isfile(cck) and os.path.isfile(pck)
        nck = nau["run_dir"] + "/checkpoints/unlearned.ckpt"
        cfr, cfp = f"{CF}/beauty_clean_cf64.pt", f"{CF}/beauty_poison_tgt{st}_seed{sd}_cf64.pt"
        specs = [("none", "-", "0")] + [(f"{n}_a{a}", p, a) for n, p in (("cfr", cfr), ("cfp", cfp), ("x", X)) for a in ("3", "10")]
        o = f"logs/rerank15/beauty_{st}_seed{sd}"
        for name, p, a in specs:
            rows.append((f"{o}/poison/beam50_{name}", pck, pdir, str(sd), p, a))
            rows.append((f"{o}/nau/beam50_{name}", nck, pdir, str(sd), p, a))
        rows.append((f"{o}/clean/beam50_none", cck, pdir, str(sd), "-", "0"))
        cells.append((st, sd, pck, nau["run_dir"], cck, pdir))
with open("logs/rerank15/cells.tsv", "w") as f:
    f.write("stratum\tseed\tpoison_ckpt\tnau_run\tclean_ckpt\tpoison_data_dir\n")
    for c in cells: f.write("\t".join(map(str, c)) + "\n")
# references first (clean + poison none) so landed status is known early; 4 packed jobs
rows.sort(key=lambda r: (0 if r[0].endswith(("clean/beam50_none", "poison/beam50_none")) else 1, r[0]))
for k in range(4):
    with open(f"logs/rerank15/plan_{k}.tsv", "w") as f:
        for r in rows[k::4]: f.write("\t".join(r) + "\n")
print(len(rows), "evals in 4 plans;", len(cells), "cells")
EOF
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^rr15_' <<<"${QUEUED}"; then echo "Refusing: rr15_ jobs queued"; exit 1; fi
run() { if [ "${DRYRUN}" = "1" ]; then printf '%q ' "$@" >&2; echo >&2; echo "<jid>"; else "$@"; fi; }
CF=$(run sbatch --parsable run_rerank15_cf.sh); echo "cf ${CF}"
for k in 0 1 2 3; do
  j=$(run sbatch --parsable --job-name=rr15_eval_${k} --dependency=afterok:${CF} run_rerank15_batch.sh logs/rerank15/plan_${k}.tsv)
  echo "eval ${k} ${j}"
done
