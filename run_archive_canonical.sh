#!/usr/bin/env bash
#SBATCH --job-name=archive_canon
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --time=12:00:00
#
# Archive the checkpoints of the canonical runs (user decision 2026-11-07: keep them) with
# archive_unlearn_ckpts.sh (payload -> /sc-projects/.../unlearn_checkpoints/<bucket>/<run>/,
# absolute symlink left in the scratch run dir), then verify every listed run: each
# checkpoints/unlearned_*.ckpt is a symlink that resolves to a file under the archive root whose
# size equals the bytes recorded in MANIFEST.tsv.
#
#   PASS=1 sbatch run_archive_canonical.sh   # canonical runs, minus those queued readers will load
#   PASS=2 sbatch --dependency=afterany:<readers> run_archive_canonical.sh   # the rest (rebuilds the ledger first)
# Run list: logs/archive_canonical/pass<PASS>_runs.txt (scripts/list_canonical_ckpt_runs.py).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PASS="${PASS:-1}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/sc-projects/sc-proj-ikim-erase/GRID/unlearn_checkpoints}"
D=logs/archive_canonical; mkdir -p "${D}"
LIST="${D}/pass${PASS}_runs.txt"
LEDGER=paper/plotting/canonical_runs.tsv
if [ "${PASS}" = 1 ]; then
  python3 scripts/list_canonical_ckpt_runs.py --ledger "${LEDGER}" --defer-readers --sizes > "${LIST}"
else
  LEDGER="${D}/ledger_pass${PASS}.tsv"     # fresh: picks up finished SCIF canonck and F+R tie-break runs
  python3 scripts/build_canonical_runs.py --out "${LEDGER}" > "${D}/ledger_pass${PASS}.log" 2>&1
  python3 scripts/list_canonical_ckpt_runs.py --ledger "${LEDGER}" --sizes > "${LIST}"
fi
echo "[$(date -Is)] pass ${PASS}: $(wc -l < "${LIST}") runs listed in ${LIST}"
df -h /sc-scratch/sc-scratch-ikim-erase | tail -1
RUN_LIST="${LIST}" ARCHIVE_ROOT="${ARCHIVE_ROOT}" bash archive_unlearn_ckpts.sh
rc=$?
echo "[$(date -Is)] archive script exit ${rc}; verifying"
python3 - "${LIST}" "${ARCHIVE_ROOT}" > "${D}/pass${PASS}_verify.txt" <<'EOF'
import csv, glob, os, sys
lst, root = sys.argv[1], os.path.realpath(sys.argv[2])
man = {}
for r in csv.DictReader(open(os.path.join(root, "MANIFEST.tsv")), delimiter="\t"):
    man[(r["run_dir"], r["file"])] = int(r["bytes"])
ok = bad = still = 0; byt = 0
for run in [l.strip() for l in open(lst) if l.strip()]:
    for f in sorted(glob.glob(f"logs/unlearn/runs/{run}/checkpoints/unlearned_*.ckpt")):
        name = os.path.basename(f)
        if not os.path.islink(f):
            print(f"NOT MOVED {run}/{name}"); still += 1; continue
        tgt = os.path.realpath(f)
        if not os.path.isfile(tgt) or not tgt.startswith(root + "/"):
            print(f"BROKEN {run}/{name} -> {tgt}"); bad += 1; continue
        size, want = os.path.getsize(tgt), man.get((run, name))
        if want is None or size != want:
            print(f"SIZE {run}/{name}: {size} vs manifest {want}"); bad += 1; continue
        ok += 1; byt += size
print(f"verified {ok} symlinked checkpoints ({byt / 1024 ** 3:.1f} GiB), {bad} broken/size mismatch, {still} not moved")
EOF
tail -1 "${D}/pass${PASS}_verify.txt"; grep -c -E "^(BROKEN|SIZE)" "${D}/pass${PASS}_verify.txt" || true
df -h /sc-scratch/sc-scratch-ikim-erase | tail -1
exit "${rc}"
