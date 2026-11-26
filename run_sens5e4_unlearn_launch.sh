#!/usr/bin/env bash
# Sensitive item deletion, larger-request variant (budget 5e-4): every canonical unlearning
# configuration replayed on the 5e-4 data, checkpoints kept.
#   Templates (paper/plotting/canonical_runs.tsv, track tiger+rkmeans):
#     setting=unwanted  (`_mia` rows: Finetune, Forget only, Forget+Repair, SCIF, SEIF, Kookmin,
#                        Fanchuan, TRACER [Sports/Toys], NAU lambda_n=-1)           130 rows
#     setting=unwanted_canonck, config ..._ln0.0 (NAU lambda_n=0, Table 4 contrast)  15 rows
#   Each row replays the template run's .hydra/overrides.yaml (same start ckpt, SID, unlearning
#   seed, every override) with ONLY data_dir -> <ds>_sens_<cat>_5e-4_seed<N> and
#   spam_forget_manifest -> the 5e-4 manifest changed (checkpoint_fractions dropped, as in the
#   canonck launchers). Tag: sens_<ds>_<cat>5e-4_<arm>[_seed<N>] (category token `<cat>5e-4`, so
#   extract_sensitive.collect(category=<cat>), the `_mia` / `_canonck` / `_valsens` globs and
#   build_canonical_runs.py never read these runs).
# Mechanics: run_valsel_batch_sens.sh -> run_valsel_one.sh (unlearn + test eval on the clean test
# split + validation eval on <ds>_valview), VALSEL_KEEP_CKPT=1. pgpu, 2 GPUs (= 2 lanes), the
# hardware the `_mia` runs used (A100 40 GB OOMs for TRACER / NAU, seen in vsb_canonck_t8).
# 10 rows per job, at most 4 jobs at once (afterany chain), all afterok on the data build.
#   BUILD_JID=<jid> bash run_sens5e4_unlearn_launch.sh             # dry run
#   BUILD_JID=<jid> DRYRUN=0 bash run_sens5e4_unlearn_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/sens5e4
mkdir -p "${D}"
PLAN="${D}/unlearn_plan.tsv"
if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import csv, os, yaml
R = "5e-4"
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
rows = []
for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t"):
    if r["track"] != "tiger+rkmeans":
        continue
    if not (r["setting"] == "unwanted" or
            (r["setting"] == "unwanted_canonck" and r["config"] == "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0")):
        continue
    ds, cat, sd, arm, tmpl = r["dataset"], r["stratum"], int(r["seed"]), r["config"], r["run_dir"]
    assert cat == CAT[ds] and r["status"] == "ok", r
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    assert tuple(o.split("=", 1)[0] for o in ov[:len(FIXED)]) == FIXED, tmpl
    assert int(kv["seed"]) == sd, tmpl
    assert kv["data_dir"] == f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}", tmpl
    old_man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
    new_man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_{R}_seed{sd}.json"
    extra = []
    for o in ov[len(FIXED):]:
        k, v = o.split("=", 1)
        if k == "unlearning.checkpoint_fractions":
            continue
        if k == "spam_forget_manifest":
            assert v == old_man, (tmpl, v)
            o = f"spam_forget_manifest={new_man}"
        assert "1e-4" not in o, (tmpl, o)
        extra.append(o)
    assert any(o.startswith("spam_forget_manifest=") for o in extra), tmpl
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
    assert "retrain" not in ckpt and f"_{ds}_clean" in ckpt, ckpt
    sfx = "" if sd == 2 else f"_seed{sd}"
    tag = f"sens_{ds}_{cat}{R}_{arm}{sfx}"
    rows.append([ds, "sens", str(sd), kv["seed"], tag, ckpt, kv["unlearning.algorithm"], sid,
                 kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
                 kv["unlearning.neighborhood_aware_sample_rate"],
                 f"src/data/amazon_data/{ds}_sens_{cat}_{R}_seed{sd}", tmpl, " ".join(extra)])
assert len(rows) == 145, len(rows)
assert len({r[4] for r in rows}) == 145
# interleave algorithms so each batch mixes slow (SCIF) and fast rows
rows.sort(key=lambda r: (r[4].split(f"{R}_", 1)[1], r[0], int(r[2])))
for r in rows:
    print("\t".join(r))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^s5e4_unl' <<<"${QUEUED}"; then echo "Refusing: s5e4_unl jobs already queued"; exit 1; fi
rm -f "${D}"/unl_todo*.tsv
: > "${D}/unl_todo.tsv"
while IFS=$'\t' read -r ds st sd useed tag rest; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" "${rest}" >> "${D}/unl_todo.tsv"
done < "${PLAN}"
n=$(wc -l < "${D}/unl_todo.tsv")
[ "${n}" -gt 0 ] || { echo "nothing to do"; exit 0; }
# round-robin split so every batch mixes algorithms
parts=$(( (n + 9) / 10 ))
split -n "r/${parts}" -d -a 2 --additional-suffix=.tsv "${D}/unl_todo.tsv" "${D}/unl_todo_"
jids=(); i=0
for p in "${D}"/unl_todo_[0-9]*.tsv; do
  k=$(wc -l < "${p}"); [ "${k}" -gt 0 ] || continue
  dep=(); [ -n "${BUILD_JID:-}" ] && dep+=("afterok:${BUILD_JID}")
  [ "${i}" -ge 4 ] && dep+=("afterany:${jids[$((i-4))]}")
  depopt=""; [ ${#dep[@]} -gt 0 ] && depopt="--dependency=$(IFS=,; echo "${dep[*]}")"
  name="s5e4_unl_$(basename "${p}" .tsv | sed 's/unl_todo_//')"
  cmd=(sbatch --parsable --nice=500 --job-name="${name}" --partition=pgpu --gres=gpu:2
       --exclude=s-sc-pgpu01,s-sc-pgpu02 --cpus-per-task=16 --mem=96G --time=12:00:00 ${depopt}
       run_valsel_batch_sens.sh "${p}")
  if [ "${DRYRUN}" = "1" ]; then printf 'VALSEL_KEEP_CKPT=1 '; printf '%q ' "${cmd[@]}"; echo "  # ${k} rows"; jids+=("DRY${i}")
  else j=$(VALSEL_KEEP_CKPT=1 "${cmd[@]}"); jids+=("${j}"); echo "submitted ${name} (${k} rows): ${j}"
       echo -e "${name}\t${j}\t${p}" >> "${D}/unlearn_jobs.tsv"; fi
  i=$((i+1))
done
echo "jobs: ${jids[*]}"
