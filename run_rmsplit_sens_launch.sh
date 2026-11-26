#!/usr/bin/env bash
# Robustness check for sensitive item deletion: the RETRAINING-MATCHED forget/retain split
# (unlearning.split_mode=retrain_matched, see configs/experiment/tiger_unlearn_scif_flat.yaml).
#
# Current rule (every recorded run, kept for the paper): forget set = the deleted interactions plus
# every later prefix of the requesting user's sequence (include_context_rows=true), and the
# requesting users' whole sequences leave the retain set. Retraining-matched: forget set = only the
# rows whose label is a deleted interaction (original histories); retain set = every user, the
# requesting users with exactly the deleted items removed = the retrained reference's training data
# (src/data/amazon_data/<ds>_retrain_<cat>_1e-4_seed<N>/training, scripts/build_retrain_dataset.py).
#
# 30 runs: NAU (lambda_f=0.1, lambda_s=0, lambda_n=-1) and Forget+Repair (lambda_f=0.1, lambda_s=0,
# lambda_n=0) x Beauty/Sports/Toys x 5 seeds. Each row replays the cell's canonical `_mia` run
# (paper/plotting/canonical_runs.tsv) EXACTLY (start ckpt, SID, seed, data dir, every override) except
#   unlearning.include_context_rows=true -> false
#   + unlearning.split_mode=retrain_matched + unlearning.split_retain_dir=<retrain dir>/training
# Plus 1 CONTROL row: Beauty seed 3 NAU replayed with the current split (no change at all) on the
# same hardware class, to check the replay reproduces the `_mia` run (so differences are the split).
#
# Tags: sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN] (control: ..._userrepl80_canonck_seed3).
# The category token `<cat>rm` != `<cat>` keeps them out of every category-filtered reader
# (run_tags / extract_sensitive.collect, scrub-view collects); no `_mia` / `_valsens`; not in
# logs/canonck_sens/plan*.tsv, so build_canonical_runs.py and the scrub canonck plan ignore them.
#
# Mechanics as run_canonck_sens_launch.sh: run_valsel_batch_sens.sh -> run_valsel_one.sh (unlearn +
# test eval + validation eval), VALSEL_KEEP_CKPT=1. gpu partition, 1 A100 per job (the `_mia` runs
# ran on A100s; A100 executions reproduce each other). Then (afterok) run_rmsplit_eval.sh:
# cleaned-view (protocol A) evals + membership inference.
#
#   bash run_rmsplit_sens_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_rmsplit_sens_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
NJOBS="${NJOBS:-3}"
D=logs/rmsplit_sens
mkdir -p "${D}"
PLAN="${D}/plan.tsv"

if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import csv, json, os, yaml
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
CFG = {"unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0": ("NAU", "-1.0"),
       "unified_abl_lf0.1_ls0.0_ln0.0": ("Forget+Repair", "0.0")}
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
led = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
       if r["setting"] == "unwanted" and r["track"] == "tiger+rkmeans" and r["config"] in CFG]
assert len(led) == 30, len(led)
rows = []
def row(r, tag, rm):
    ds, sd, cat = r["dataset"], int(r["seed"]), CAT[r["dataset"]]
    tmpl = r["run_dir"]
    assert r["status"] == "ok" and r["source"] == "mia" and os.path.isdir(tmpl), r
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    head = [o.split("=", 1)[0] for o in ov[:len(FIXED)]]
    assert tuple(head) == FIXED, (tmpl, head)
    assert kv["unlearning_run_tag"] == r["tag"], (tmpl, kv["unlearning_run_tag"])
    extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
    lam = dict(o.split("=", 1) for o in extra)
    meth, ln = CFG[r["config"]]
    assert r["method"] == meth and lam["unlearning.lambda_f"] == "0.1" and lam["unlearning.lambda_s"] == "0.0" \
        and lam["unlearning.lambda_n"] == ln, (tmpl, lam)
    assert int(kv["seed"]) == sd and kv["experiment"] == "tiger_unlearn_unified_sequential"
    sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
    man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
    assert kv["data_dir"] == sens and lam["spam_forget_manifest"] == man, tmpl
    assert lam["unlearning.deletion_spec"] == "item_pairs" and lam["unlearning.include_context_rows"] == "true", tmpl
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
    if rm:
        retr = f"src/data/amazon_data/{ds}_retrain_{cat}_1e-4_seed{sd}"
        meta = json.load(open(f"{retr}/retrain_meta.json"))
        assert os.path.realpath(meta["manifest"]) == os.path.realpath(man), (retr, meta["manifest"])
        assert os.path.isdir(f"{retr}/training")
        extra = [("unlearning.include_context_rows=false" if o == "unlearning.include_context_rows=true" else o)
                 for o in extra]
        extra += ["unlearning.split_mode=retrain_matched", f"unlearning.split_retain_dir={retr}/training"]
    return ["tiger", ds, "sens", str(sd), kv["seed"], tag, ckpt, kv["unlearning.algorithm"], sid,
            kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
            kv["unlearning.neighborhood_aware_sample_rate"], sens, tmpl, " ".join(extra)]
for r in sorted(led, key=lambda r: (r["dataset"], r["config"], int(r["seed"]))):
    ds, cat = r["dataset"], CAT[r["dataset"]]
    base, sfx = r["tag"].rsplit("_mia", 1)
    assert base.startswith(f"sens_{ds}_{cat}_")
    arm = base[len(f"sens_{ds}_{cat}_"):]
    rows.append(row(r, f"sens_{ds}_{cat}rm_{arm}_rmsplit_canonck{sfx}", True))
ctl = [r for r in led if r["dataset"] == "beauty" and r["seed"] == "3" and r["method"] == "NAU"][0]
rows.append(row(ctl, "sens_beauty_hairlossrm_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_userrepl80_canonck_seed3", False))
assert len(rows) == 31 and len({r[5] for r in rows}) == 31
for r in rows:
    print("\t".join(r))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^(vsb_rmsplit_[0-9]|rmsplit_eval)' <<<"${QUEUED}"; then
  echo "Refusing: already queued/running:"; grep -E '^(vsb_rmsplit_[0-9]|rmsplit_eval)' <<<"${QUEUED}"; exit 1
fi
rm -f "${D}"/todo*.tsv
: > "${D}/todo.tsv"
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  # rows of the smoke job (logs/rmsplit_sens/smoke.tsv, job 11476682) are run there
  if [ -s "${D}/smoke.tsv" ] && cut -f5 "${D}/smoke.tsv" | grep -qxF "${tag}"; then echo "  in smoke job: ${tag}"; continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${D}/todo.tsv"
done < "${PLAN}"
n=$(wc -l < "${D}/todo.tsv")
deps=()
if [ "${n}" -gt 0 ]; then
  # round-robin so every job mixes datasets and both methods
  for ((j=0; j<NJOBS; j++)); do awk -v j="${j}" -v k="${NJOBS}" '(NR-1)%k==j' "${D}/todo.tsv" > "${D}/todo_${j}.tsv"; done
  for ((j=0; j<NJOBS; j++)); do
    p="${D}/todo_${j}.tsv"; k=$(wc -l < "${p}"); [ "${k}" -gt 0 ] || continue
    name="vsb_rmsplit_${j}"
    cmd=(sbatch --parsable --job-name="${name}" --partition=gpu --gres=gpu:nvidia_a100-pcie-40gb:1
         --cpus-per-task=8 --mem=64G --time=08:00:00 run_valsel_batch_sens.sh "${p}")
    if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=tiger VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${k}"
    else jid=$(MODEL=tiger VALSEL_KEEP_CKPT=1 "${cmd[@]}"); deps+=("${jid}"); echo "submitted ${name} (${k} rows): ${jid}"; fi
  done
fi
# the eval also waits for the smoke job (its two rows), if given
[ -n "${SMOKE_JOB:-}" ] && deps+=("${SMOKE_JOB}")
dep=""; [ "${#deps[@]}" -gt 0 ] && dep="--dependency=afterok:$(IFS=:; echo "${deps[*]}")"
ecmd=(sbatch --parsable --job-name=rmsplit_eval ${dep} --partition=gpu --gres=gpu:nvidia_a100-pcie-40gb:1
      --cpus-per-task=8 --mem=64G --time=06:00:00 --output=logs/%j.out run_rmsplit_eval.sh)
if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${ecmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else echo "submitted rmsplit_eval: $("${ecmd[@]}")"; fi
