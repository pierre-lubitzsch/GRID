#!/usr/bin/env bash
# Canonical re-runs WITH checkpoint for the spam cells whose canonical execution had neither a
# validation eval nor a checkpoint (paper/revision/canonical_runs_plan.md 3.2, decision (b) read
# strictly): Finetune (unified_abl_lf0.0_ls0.0_ln0.0) Sports/Toys unpopular+popular (20) and SCIF
# outside the mid target (26: Beauty/Toys seeds 3-11, Sports all seeds; Beauty/Toys seed 2 already
# have a checkpoint and get a validation re-eval instead).
#
# Each row replays the RECORDED run of its cell exactly (same poisoned ckpt file, SID, unlearning
# seed, data dir and unlearning.* overrides from .hydra/overrides.yaml), like run_fig1_fix_launch.sh,
# through run_valsel_batch.sh -> run_valsel_one.sh (unlearn + test eval + validation eval) with
# VALSEL_KEEP_CKPT=1. Tag suffix _canonck (not _valsel*, so valsel_table.py's glob ignores them).
# The template list is read from the ledger (rows with source=recorded and no val/ckpt), so it must
# be run BEFORE scripts/build_canonical_runs.py switches those cells to _canonck.
#
#   bash run_canonck_missing_launch.sh             # DRY RUN (default)
#   DRYRUN=0 bash run_canonck_missing_launch.sh    # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
mkdir -p logs/canonck_missing
PLAN="logs/canonck_missing/plan.tsv"

if [ ! -s "${PLAN}" ]; then
python3 - <<'EOF' > "${PLAN}.tmp"
import csv, os, re, yaml
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
EXPECT_EXP = {"scif": "tiger_unlearn_scif_sequential", "unified": "tiger_unlearn_unified_sequential"}
rows = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
        if r["setting"] == "spam" and r["config"] in ("finetune", "scif") and r["source"] == "recorded"
        and r["status"] == "ok" and r["has_val"] == "0" and r["has_ckpt"] == "0"]
assert len(rows) == 46, len(rows)
for r in sorted(rows, key=lambda r: (r["config"], r["dataset"], r["stratum"], int(r["seed"]))):
    ds, st, sd, tmpl, rtag = r["dataset"], r["stratum"], int(r["seed"]), r["run_dir"], r["tag"]
    assert st != "mid"
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    assert kv["unlearning_run_tag"] == rtag, (kv["unlearning_run_tag"], rtag)
    useed = str(int(kv["seed"])); assert int(useed) == sd, (tmpl, useed)
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt), ckpt; assert os.path.isfile(sid), sid
    assert sid == f"embeddings/{ds}/merged_predictions_tensor.pt", sid
    assert kv.get("num_hierarchies", "4") == "4"
    # wrong-seed guard: the poisoned checkpoint must be this cell's (dataset, stratum, seed)
    base = os.path.basename(ckpt.split("/checkpoints/")[0])
    assert f"_{ds}_poison_pct1_n1" in base, ckpt
    assert base.endswith(f"_tgt{st}"), ckpt
    assert (sd == 2 and "_seed" not in base) or f"_seed{sd}_" in base, ckpt
    algo = kv["unlearning.algorithm"]
    assert kv["experiment"] == EXPECT_EXP[algo], kv["experiment"]
    na = kv.get("unlearning.neighborhood_aware", "false"); rbs = kv.get("unlearning.request_batch_size", "1")
    rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
    extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
    tag = re.sub(r"(_seed\d+)?$", lambda m: "_canonck" + (m.group(1) or ""), rtag, count=1)
    print("\t".join([ds, st, str(sd), useed, tag, ckpt, algo, sid, na, rbs, rate,
                     kv["data_dir"], tmpl, " ".join(extra)]))
EOF
mv "${PLAN}.tmp" "${PLAN}"
fi
echo "plan: ${PLAN} ($(wc -l < "${PLAN}") rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^vsb_canonck_' <<<"${QUEUED}"; then
  echo "Refusing: a vsb_canonck_ job is already queued/running:"; grep -E '^vsb_canonck_' <<<"${QUEUED}"; exit 1
fi
: > logs/canonck_missing/todo_finetune.tsv; : > logs/canonck_missing/todo_scif.tsv
while IFS=$'\t' read -r ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "  exists: ${tag}"; continue; fi
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  [ "${resolved}" = "${data_dir}" ] || { echo "MISMATCH ${tag}: ${resolved} vs ${data_dir}" >&2; exit 1; }
  echo "  todo:   ${tag}  <- $(basename "${tmpl}")  [${algo} ${extra}]"
  if [ "${algo}" = "scif" ]; then f=scif; else f=finetune; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "logs/canonck_missing/todo_${f}.tsv"
done < "${PLAN}"

# Finetune: 1 job (2 lanes, ~13 min/row). SCIF: split in 2 jobs (2 lanes each, ~20 min/row).
split -n l/2 -d --additional-suffix=.tsv logs/canonck_missing/todo_scif.tsv logs/canonck_missing/todo_scif_
submit() {  # name time file
  local n; n=$(wc -l < "$3"); [ "${n}" -gt 0 ] || { echo "  ${1}: nothing to do"; return; }
  local cmd=(sbatch --parsable --job-name="$1" --time="$2" run_valsel_batch.sh "$3")
  if [ "${DRYRUN}" = "1" ]; then printf 'VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${n}"
  else echo "submitted $1 ($n rows): $(VALSEL_KEEP_CKPT=1 "${cmd[@]}")"; fi
}
submit vsb_canonck_finetune 04:00:00 logs/canonck_missing/todo_finetune.tsv
submit vsb_canonck_scif_0 05:00:00 logs/canonck_missing/todo_scif_00.tsv
submit vsb_canonck_scif_1 05:00:00 logs/canonck_missing/todo_scif_01.tsv
[ "${DRYRUN}" = "1" ] && echo "DRY RUN (DRYRUN=0 to submit)"
exit 0
