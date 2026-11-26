#!/usr/bin/env bash
# Validation-split hyperparameter selection: every eligible candidate configuration,
# re-run on every (stratum, seed) cell with a TEST and a VALIDATION evaluation of the
# same unlearned model (run_valsel_one.sh).
#
# Candidates (RQ-KMeans; TRACER on Beauty uses its own tracerv3 tokenizer, as in Table 1):
#   baselines with more than one eligible configuration (eligible = recorded on all 15
#   cells): Forget only, Forget+Repair, SEIF, Kookmin, Fanchuan, TRACER, per dataset;
#   NAU: lambda_s, lambda_n in {0, 0.01, 0.1} at lambda_f = 0.1 on Beauty (9 configs),
#   templated on the lambda_s = 0.01 NAU run of each cell.
# Finetune and SCIF have one eligible configuration each, so there is nothing to select.
#
# Each job replays the recorded template run of its cell (same poisoned ckpt FILE, SID,
# seed, data dir, unlearning.* overrides from .hydra/overrides.yaml); only the NAU grid
# changes lambda_s / lambda_n. Tags get a _valsel suffix; fresh run dirs only; skips
# existing or queued tags; refuses on data_dir mismatch.
#
#   bash run_valsel_launch.sh            # DRY RUN (default)
#   DRYRUN=0 bash run_valsel_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
PLAN="logs/valsel_plan.tsv"
mkdir -p logs

python3 - <<'EOF' > "${PLAN}"
import csv, glob, itertools, os, sys, yaml
sys.path.insert(0, "scripts")
import run_tags
SEEDS = ("2", "3", "5", "7", "11"); STRATA = ("unpopular", "mid", "popular")
FAMS = {"ga_abl", "negtrain_abl", "seif", "kookmin", "fanchuan", "tracer"}
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
cells = {}
for r in csv.DictReader(open("paper/data/runs.csv")):
    if r["setting"] != "spam" or r["done"] != "True" or r["pkm"] or "pkm" in r["tag"]:
        continue
    tr = r["track"]; ds = r["dataset"]
    fam = run_tags.family_of(r["arm"])
    if fam not in FAMS: continue
    if not (tr == "tiger+rkmeans" and not (fam == "tracer" and ds == "beauty")) and not (fam == "tracer" and ds == "beauty" and tr == "tiger+tracerv3"):
        continue
    k = (ds, tr, r["arm"], r["stratum"], r["seed"])
    # prefer the primary run over unlearning replicates (_useed<N>); among equals, the latest
    rank = (0 if not r.get("replicate") else 1, -int(r["jid"]))
    if k not in cells or rank < cells[k][0]: cells[k] = (rank, r)
cells = {k: v[1] for k, v in cells.items()}
by_arm = {}
for (ds, tr, arm, st, sd), r in cells.items():
    by_arm.setdefault((ds, tr, arm), {})[(st, sd)] = r
jobs = []
def emit(ds, st, sd, tag, path, lam=None):
    ov = yaml.safe_load(open(os.path.join(path, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    useed = str(int(kv["seed"]))   # the template's own unlearning seed (differs from sd for _useed replicates)
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
    algo = kv["unlearning.algorithm"]
    na = kv.get("unlearning.neighborhood_aware", "false"); rbs = kv.get("unlearning.request_batch_size", "1")
    rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
    extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
    if lam:
        new = []
        for o in extra:
            if o.startswith("unlearning.lambda_s="): o = f"unlearning.lambda_s={lam[0]}"
            if o.startswith("unlearning.lambda_n="): o = f"unlearning.lambda_n={lam[1]}"
            new.append(o)
        assert sum(o.startswith("unlearning.lambda_s=") for o in new) == 1 and sum(o.startswith("unlearning.lambda_n=") for o in new) == 1, path
        extra = new
    jobs.append([ds, st, sd, useed, tag, ckpt, algo, sid, na, rbs, rate, kv["data_dir"], path, " ".join(extra)])
# baselines: every arm of a family recorded on all 15 cells
for (ds, tr, arm), cc in sorted(by_arm.items()):
    if len(cc) < 15: continue
    for (st, sd), r in sorted(cc.items()):
        tag = r["tag"].replace(arm, arm + "_valsel")
        assert tag != r["tag"], r["tag"]
        emit(ds, st, sd, tag, r["path"])
# NAU grid on Beauty, templated on the lambda_s=0.01 NAU run of each cell
for st in STRATA:
    for sd in SEEDS:
        sfx = "" if sd == "2" else f"_seed{sd}"
        t = glob.glob(f"logs/unlearn/runs/*_bs1_bw_tgt{st}_nau_s0p01_n0p01_keepck{sfx}")
        assert len(t) == 1, (st, sd, t)
        for ls, ln in itertools.product(("0.0", "0.01", "0.1"), repeat=2):
            tag = f"bw_tgt{st}_valsel_nau_ls{ls.replace('.', 'p')}_ln{ln.replace('.', 'p')}{sfx}"
            emit("beauty", st, sd, tag, t[0], lam=(ls, ln))
for j in jobs: print("\t".join(j))
EOF

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
n=0; skip=0; bad=0
while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  skip ${tag}: run dir exists"; skip=$((skip+1)); continue; fi
  if grep -qxF "vs_${tag}" <<<"${QUEUED}"; then echo "  skip ${tag}: queued"; skip=$((skip+1)); continue; fi
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  if [ "${resolved}" != "${data_dir}" ]; then echo "  MISMATCH ${tag}: ${resolved} vs ${data_dir}"; bad=$((bad+1)); continue; fi
  # shellcheck disable=SC2206
  ov=(${extra})
  cmd=(sbatch --job-name="vs_${tag}" run_valsel_one.sh "${ckpt}" "${ds}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${ov[@]}")
  if [ "${DRYRUN}" = "1" ]; then
    [ "${n}" -lt 3 ] && { echo "== ${tag} (template $(basename "${tmpl}"))"; printf '   %q' "${cmd[@]}"; printf '\n'; }
  else
    VALSEL_DATASET="${ds}" POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" \
    TARGET_STRATEGY="${st}" UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED="${useed}" "${cmd[@]}"
  fi
  n=$((n+1))
done < "${PLAN}"
echo "${n} job(s) $([ "${DRYRUN}" = "1" ] && echo 'would be submitted' || echo submitted); ${skip} skipped; ${bad} mismatched"
[ "${bad}" = "0" ] || { echo "Refusing: data_dir mismatches above." >&2; exit 1; }
