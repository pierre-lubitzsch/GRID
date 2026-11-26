#!/usr/bin/env bash
# Canonical re-runs WITH checkpoint for the unwanted-item ("sensitive") Table 4 / Table 8 rows that
# had no checkpointed execution (protocol A of paper/revision/canonical_runs_plan.md 4.3: every
# reported model is evaluated on the cleaned test histories, so every reported row needs a ckpt).
#
#   RQ-KMeans (TIGER) NAU lambda_n=0 and lambda_n=-10 (sepgen, lambda_f=0.1, lambda_s=0):
#       3 datasets x 5 seeds = 30 rows. Each row replays that cell's `_valsens` run EXACTLY
#       (start ckpt, SID, unlearning seed, data dir and every override after unlearning_run_tag
#       in its .hydra/overrides.yaml), so Table 8 and Table 4 read one execution per config.
#   LETTER NAU lambda_n in {0, -1, -10}: 45 rows. lambda_n=0 / -1 replay the recorded LETTER run
#       of the cell; lambda_n=-10 is templated from the lambda_n=-1 run with only lambda_n changed
#       (exactly how the RQ-KMeans `_valsens` lambda_n=-10 arms and the one LETTER Beauty seed-2
#       lambda_n=-10 run were built).
#
# Mechanics as run_canonck_sep001_launch.sh: run_valsel_batch_sens.sh -> run_valsel_one.sh
# (unlearn + test eval on the clean test split + validation eval on <ds>_valview) with
# VALSEL_KEEP_CKPT=1. LETTER batches run with MODEL=letter (letter_unlearn_unified_sequential,
# letter_train_flat for both evals). Tags: <recorded tag minus _valsens>_canonck[_seedN]
# (no `_valsens/_valsel/_mia`, so those globs ignore them; scripts/build_canonical_runs.py lists
# them as setting=unwanted_canonck). gpu partition, one 80 GB A100 per job (one lane).
# Then the cleaned-view evaluation (run_sensitive_scrub_eval.sh with PLAN=plan_canonck.tsv,
# afterok), see scripts/plan_sensitive_scrub_eval_canonck.py.
#
#   bash run_canonck_sens_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_canonck_sens_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/canonck_sens
mkdir -p "${D}"
PLAN="${D}/plan.tsv"

if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import glob, os, re, yaml
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
NAU = "unified_hp2_sepgen_lf0.1_ls0.0_ln{}"
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")

def latest(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
         if os.path.isfile(os.path.join(d, "eval/csv/version_0/metrics.csv"))]
    assert c, f"no finished run for {tag}"
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))

rows = []
for model, lns in (("tiger", ("0.0", "neg10.0")), ("letter", ("0.0", "neg1.0", "neg10.0"))):
    for ds, cat in CAT.items():
        for ln in lns:
            for sd in SEEDS:
                sfx = "" if sd == 2 else f"_seed{sd}"
                arm = NAU.format(ln)
                if model == "tiger":
                    tmpl = latest(f"sens_{ds}_{cat}_{arm}_valsens{sfx}")
                    tag = f"sens_{ds}_{cat}_{arm}_canonck{sfx}"
                else:
                    src = NAU.format("neg1.0") if ln == "neg10.0" else arm
                    tmpl = latest(f"letter_letter_sens_{ds}_{cat}_{src}{sfx}")
                    tag = f"letter_letter_sens_{ds}_{cat}_{arm}_canonck{sfx}"
                ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
                kv = dict(o.split("=", 1) for o in ov)
                head = [o.split("=", 1)[0] for o in ov[:len(FIXED)]]
                assert tuple(head) == FIXED, (tmpl, head)
                extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
                assert int(kv["seed"]) == sd, (tmpl, kv["seed"])
                sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
                assert kv["data_dir"] == sens, (tmpl, kv["data_dir"])
                ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
                assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
                base = ckpt.split("/checkpoints/")[0]
                want = (f"_{ds}_clean" if model == "tiger" else f"_letter_{ds}_clean")
                assert want in base and "retrain" not in base, ckpt                      # start = clean model
                assert (sd == 2 and "_seed" not in os.path.basename(base)) or base.endswith(f"_seed{sd}"), ckpt
                assert sid == (f"embeddings/{ds}/merged_predictions_tensor.pt" if model == "tiger"
                               else f"embeddings/{ds}_letter/merged_predictions_tensor.pt"), sid
                assert kv["experiment"] == f"{model}_unlearn_unified_sequential", kv["experiment"]
                assert kv["unlearning.algorithm"] == "unified" and kv.get("num_hierarchies") == "4"
                lam = {o.split("=", 1)[0]: o.split("=", 1)[1] for o in extra}
                want_ln = "-10.0" if ln == "neg10.0" else ("-1.0" if ln == "neg1.0" else "0.0")
                if model == "letter" and ln == "neg10.0":
                    assert lam["unlearning.lambda_n"] == "-1.0", tmpl
                    extra = [("unlearning.lambda_n=-10.0" if o == "unlearning.lambda_n=-1.0" else o) for o in extra]
                    lam["unlearning.lambda_n"] = "-10.0"
                assert (lam["unlearning.lambda_f"], lam["unlearning.lambda_s"], lam["unlearning.lambda_n"],
                        lam["unlearning.sep_loss_type"]) == ("0.1", "0.0", want_ln, "generative"), (tmpl, lam)
                assert lam["spam_forget_manifest"] == f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
                rows.append([model, ds, "sens", str(sd), kv["seed"], tag, ckpt, "unified", sid,
                             kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
                             kv["unlearning.neighborhood_aware_sample_rate"], sens, tmpl, " ".join(extra)])
assert len(rows) == 75, len(rows)
for r in rows:
    print("\t".join(r))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_canonck_sens|scrub_canonck)' <<<"${QUEUED}"; then
  echo "Refusing: already queued/running:"; grep -E '^(vsb_canonck_sens|scrub_canonck)' <<<"${QUEUED}"; exit 1
fi
rm -f "${D}"/todo_*.tsv
for model in tiger letter; do : > "${D}/todo_${model}.tsv"; done
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${D}/todo_${model}.tsv"
done < "${PLAN}"
deps=()
for model in tiger letter; do
  f="${D}/todo_${model}.tsv"; n=$(wc -l < "${f}"); [ "${n}" -gt 0 ] || continue
  parts=$(( (n + 14) / 15 ))                       # ~15 rows (about 4 h) per single-GPU job
  split -n "l/${parts}" -d --additional-suffix=.tsv "${f}" "${D}/todo_${model}_"
  for p in "${D}/todo_${model}"_[0-9]*.tsv; do
    k=$(wc -l < "${p}"); [ "${k}" -gt 0 ] || continue
    name="vsb_canonck_sens_${model}_$(basename "${p}" .tsv | sed "s/todo_${model}_//")"
    cmd=(sbatch --parsable --job-name="${name}" --partition=gpu --gres=gpu:nvidia_a100_80gb_pcie:1
         --cpus-per-task=8 --mem=64G --time=08:00:00 run_valsel_batch_sens.sh "${p}")
    if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=%s VALSEL_KEEP_CKPT=1' "${model}"; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${k}"
    else j=$(MODEL="${model}" VALSEL_KEEP_CKPT=1 "${cmd[@]}"); deps+=("${j}"); echo "submitted ${name} (${k} rows): ${j}"; fi
  done
done
dep=""; [ "${#deps[@]}" -gt 0 ] && dep="--dependency=afterany:$(IFS=:; echo "${deps[*]}")"
# afterany: the scrub-eval job only evaluates runs whose checkpoint exists (a failed row is skipped
# and reported, not silently dropped); it plans at start time, so it sees every finished run.
scmd=(sbatch --parsable --job-name=scrub_canonck ${dep} --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
      --time=08:00:00 --output=logs/%j.out --wrap
      "python scripts/plan_sensitive_scrub_eval_canonck.py && CELL=all PLAN=logs/eval/sensitive_scrub/plan_canonck.tsv PAR=3 bash run_sensitive_scrub_eval.sh")
if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${scmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else echo "submitted scrub_canonck: $("${scmd[@]}")"; fi
