#!/usr/bin/env bash
# Canonical re-runs WITH checkpoint of the NON-SELECTED Table 8 candidates (unwanted-item validation
# selection, TIGER + RQ-KMeans): SCIF / SEIF / Kookmin / Fanchuan / TRACER alternatives and NAU
# lambda_n=-0.1, 33 configs x 5 seeds = 165 rows (user decision 2026-11-07, protocol A: Table 8's
# validation AND test columns come from one checkpointed execution per configuration; the test
# columns on the cleaned test histories). Candidate list = paper/plotting/valsens_candidates_ledger.csv
# rows with source=valsens, minus NAU lambda_n=0 / -10 (run_canonck_sens_launch.sh).
#
# Each row replays the cell's `_valsens` run EXACTLY (start ckpt, SID, unlearning seed, data dir and
# every override after unlearning_run_tag in its .hydra/overrides.yaml) through
# run_valsel_batch_sens.sh -> run_valsel_one.sh with VALSEL_KEEP_CKPT=1 (unlearn + test eval +
# validation eval). Tag: <valsens tag minus _valsens>_canonck[_seedN]. Plan: logs/canonck_sens/plan_t8.tsv
# (same 15 columns as plan.tsv). scripts/build_canonical_runs.py lists them as setting=unwanted_canonck.
# Then the cleaned-view eval (scripts/plan_sensitive_scrub_eval_canonck.py reads both plans), afterany.
#
#   bash run_canonck_sens_t8_launch.sh            # DRY RUN
#   DRYRUN=0 bash run_canonck_sens_t8_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
NJOBS="${NJOBS:-6}"
D=logs/canonck_sens
mkdir -p "${D}"
PLAN="${D}/plan_t8.tsv"

if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import csv, glob, os, re, yaml
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
start = {}
for r in csv.reader(open("logs/eval/sensitive_scrub/plan.tsv"), delimiter="\t"):
    if r[3] == "original" and r[6] == "normal":
        start[(r[1], int(r[2]))] = r[7]

def latest(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
         if os.path.isfile(os.path.join(d, "eval/csv/version_0/metrics.csv"))]
    assert c, f"no finished run for {tag}"
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))

cands = [r for r in csv.DictReader(open("paper/plotting/valsens_candidates_ledger.csv"))
         if r["source"] == "valsens" and r["config"] not in ("lambda_n=0.0", "lambda_n=-10.0")]
assert len(cands) == 33, len(cands)
rows = []
for c in cands:
    ds, cfg = c["dataset"], c["config"]
    if cfg.startswith("lambda_n="):
        v = float(cfg.split("=")[1]); assert v == -0.1, cfg
        arm = "unified_hp2_sepgen_lf0.1_ls0.0_lnneg0.1"
    else:
        arm = cfg
    for sd in SEEDS:
        sfx = "" if sd == 2 else f"_seed{sd}"
        tmpl = latest(f"sens_{ds}_{CAT[ds]}_{arm}_valsens{sfx}")
        tag = f"sens_{ds}_{CAT[ds]}_{arm}_canonck{sfx}"
        ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
        kv = dict(o.split("=", 1) for o in ov)
        assert tuple(o.split("=", 1)[0] for o in ov[:len(FIXED)]) == FIXED, tmpl
        extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
        sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
        assert int(kv["seed"]) == sd and kv["data_dir"] == sens, tmpl
        ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
        assert ckpt == start[(ds, sd)], (tmpl, ckpt, start[(ds, sd)])     # same start as `_mia` / original row
        assert sid == f"embeddings/{ds}/merged_predictions_tensor.pt" and os.path.isfile(ckpt), (sid, ckpt)
        algo = kv["unlearning.algorithm"]
        assert arm.startswith(algo if algo != "unified" else "unified_hp2"), (arm, algo)
        assert kv["experiment"] in ("tiger_unlearn_scif_sequential", "tiger_unlearn_unified_sequential"), kv["experiment"]
        assert f"spam_forget_manifest=src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json" in extra
        rows.append(["tiger", ds, "sens", str(sd), kv["seed"], tag, ckpt, algo, sid,
                     kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
                     kv["unlearning.neighborhood_aware_sample_rate"], sens, tmpl, " ".join(extra)])
assert len(rows) == 165, len(rows)
for r in rows:
    print("\t".join(r))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_canonck_t8|scrub_canonck_t8)' <<<"${QUEUED}"; then
  echo "Refusing: already queued/running:"; grep -E '^(vsb_canonck_t8|scrub_canonck_t8)' <<<"${QUEUED}"; exit 1
fi
rm -f "${D}"/todo_t8*.tsv
: > "${D}/todo_t8.tsv"
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${D}/todo_t8.tsv"
done < "${PLAN}"
n=$(wc -l < "${D}/todo_t8.tsv")
deps=()
if [ "${n}" -gt 0 ]; then
  # round-robin split so every job gets a mix of algorithms (similar run time per job)
  for ((k=0; k<NJOBS; k++)); do awk -v k="${k}" -v m="${NJOBS}" '(NR-1)%m==k' "${D}/todo_t8.tsv" > "${D}/todo_t8_${k}.tsv"; done
  for ((k=0; k<NJOBS; k++)); do
    p="${D}/todo_t8_${k}.tsv"; c=$(wc -l < "${p}"); [ "${c}" -gt 0 ] || continue
    cmd=(sbatch --parsable --job-name="vsb_canonck_t8_${k}" --partition=gpu --gres=gpu:1
         --cpus-per-task=8 --mem=64G --time=16:00:00 run_valsel_batch_sens.sh "${p}")
    if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=tiger VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${c}"
    else j=$(MODEL=tiger VALSEL_KEEP_CKPT=1 "${cmd[@]}"); deps+=("${j}"); echo "submitted vsb_canonck_t8_${k} (${c} rows): ${j}"; fi
  done
fi
# also after the first scrub job (scrub_canonck), so two scrub evals never write the same dir at once
for j in $(squeue -h -u "$USER" -n scrub_canonck -o '%i' 2>/dev/null); do deps+=("${j}"); done
dep=""; [ "${#deps[@]}" -gt 0 ] && dep="--dependency=afterany:$(IFS=:; echo "${deps[*]}")"
scmd=(sbatch --parsable --job-name=scrub_canonck_t8 ${dep} --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
      --time=10:00:00 --output=logs/%j.out --wrap
      "python scripts/plan_sensitive_scrub_eval_canonck.py && CELL=all PLAN=logs/eval/sensitive_scrub/plan_canonck.tsv PAR=3 bash run_sensitive_scrub_eval.sh")
if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${scmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else echo "submitted scrub_canonck_t8: $("${scmd[@]}")"; fi
