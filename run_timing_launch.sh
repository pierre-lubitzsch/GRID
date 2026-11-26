#!/usr/bin/env bash
# Efficiency (E) timing campaign launcher (see scripts/plan_timing_campaign.py, run_timing_batch.sh).
# Splits logs/timing_e/plan.tsv into 4 single-lane H200 jobs:
#   timing_e_0..2: Beauty spam (Table 9 methods, 150 rows, interleaved so SCIF is spread) + Toys unwanted;
#   timing_e_3:    Beauty unwanted first, then Sports unwanted.
# Spam rows: the dataset resolution (POISON_* env) must reproduce each template's data_dir.
#
#   bash run_timing_launch.sh                 # DRY RUN
#   SMOKE=1 DRYRUN=0 bash run_timing_launch.sh  # submit the one smoke row (Beauty mid seed 2 Forget+Repair)
#   DRYRUN=0 [AFTER=<jobid>] bash run_timing_launch.sh   # submit the 4 jobs
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/timing_e
PLAN="${D}/plan.tsv"
[ -s "${PLAN}" ] || python3 scripts/plan_timing_campaign.py > "${PLAN}"
echo "plan: ${PLAN} ($(($(wc -l < "${PLAN}") - 1)) rows)"

QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^timing_e_[0-9]' <<<"${QUEUED}"; then
  echo "Refusing: a timing_e job is already queued/running:"; grep -E '^timing_e_[0-9]' <<<"${QUEUED}"; exit 1
fi

# data_dir check for spam rows (cheap: sources the resolver only)
tail -n +2 "${PLAN}" | while IFS=$'\t' read -r setting ds st sd useed tag ckpt algo sid na rbs rate data_dir rest; do
  [ "${setting}" = "spam" ] || continue
  resolved="$(POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" TARGET_STRATEGY="${st}" \
    bash -c 'source scripts/resolve_unlearn_dataset.sh >/dev/null 2>&1; resolve_unlearn_dataset "$0" >/dev/null 2>&1; echo "${UNLEARN_DATA_DIR}"' "${ds}")"
  [ "${resolved}" = "${data_dir}" ] || { echo "MISMATCH ${tag}: ${resolved} vs ${data_dir}" >&2; exit 1; }
done
echo "spam data_dir resolution: OK"

submit() {  # name time file [dependency]
  local n; n=$(wc -l < "$3"); [ "${n}" -gt 0 ] || { echo "  ${1}: nothing to do"; return; }
  local cmd=(sbatch --parsable --job-name="$1" --time="$2")
  if [ -n "${4:-}" ]; then
    cmd+=(--dependency="afterok:$4" --export="ALL,SMOKE_CHECK_TAG=$(cut -f6 "${D}/smoke.tsv")")
  fi
  cmd+=(run_timing_batch.sh "$3")
  if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${cmd[@]}"; printf '  # %s rows\n' "${n}"
  else echo "submitted $1 ($n rows): $("${cmd[@]}")"; fi
}

if [ "${SMOKE:-0}" = "1" ]; then
  awk -F'\t' '$1=="spam" && $3=="mid" && $4=="2" && $15=="Forget+Repair"' "${PLAN}" > "${D}/smoke.tsv"
  [ "$(wc -l < "${D}/smoke.tsv")" = 1 ] || { echo "smoke row not unique" >&2; exit 1; }
  submit timing_e_smoke 00:40:00 "${D}/smoke.tsv"
  exit 0
fi

python3 - "${PLAN}" "${D}" <<'EOF'
import csv, sys
plan, d = sys.argv[1], sys.argv[2]
rows = list(csv.reader(open(plan), delimiter="\t"))[1:]
spam = [r for r in rows if r[0] == "spam"]
spam.sort(key=lambda r: (r[14], r[2], int(r[3])))
toys = [r for r in rows if r[0] == "unwanted" and r[1] == "toys"]
chunks = [[], [], [], []]
for i, r in enumerate(spam):
    chunks[i % 3].append(r)
for i, r in enumerate(toys):
    chunks[i % 3].append(r)
chunks[3] = ([r for r in rows if r[0] == "unwanted" and r[1] == "beauty"]
             + [r for r in rows if r[0] == "unwanted" and r[1] == "sports"])
assert sum(map(len, chunks)) == len(rows)
for k, c in enumerate(chunks):
    with open(f"{d}/chunk_{k}.tsv", "w") as f:
        for r in c:
            f.write("\t".join(r) + "\n")
    print(f"chunk_{k}: {len(c)} rows")
EOF
for k in 0 1 2 3; do submit "timing_e_${k}" 10:00:00 "${D}/chunk_${k}.tsv" "${AFTER:-}"; done
[ "${DRYRUN}" = "1" ] && echo "DRY RUN (DRYRUN=0 to submit)"
exit 0
