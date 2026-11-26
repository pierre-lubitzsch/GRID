#!/usr/bin/env bash
#SBATCH --job-name=archive_ckpts
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --partition=compute
#SBATCH --time=12:00:00

# Move unlearned-model checkpoints of INACTIVE runs off scratch onto the project
# filesystem, leaving an absolute symlink behind so every recorded path still
# resolves:
#
#   logs/unlearn/runs/<run>/checkpoints/unlearned_9.ckpt
#     -> /sc-projects/sc-proj-ikim-erase/GRID/unlearn_checkpoints/<bucket>/<run>/unlearned_9.ckpt
#
# Scratch is 1.0T and unlearn sweeps are disk-bound: ~1.9G per non-PKM run,
# ~7.6G per PKM-all run. Only `unlearned_*.ckpt` payloads move; metrics.csv,
# unlearn.log, requests/ and every other small artifact stay in place, so result
# extraction needs no path changes.
#
# INACTIVE means: no job of that run id is in squeue, and the run directory has
# not been written to within HOLD_MIN minutes. Runs of queued/running jobs are
# skipped, so this is safe to run while a sweep is in flight.
#
# Idempotent and resumable: rsync verifies the transfer before the source is
# unlinked (--remove-source-files), and an already-symlinked file is skipped.
#
# Usage:
#   DRYRUN=1 ./archive_unlearn_ckpts.sh                 # report what would move
#   sbatch archive_unlearn_ckpts.sh                     # do it
#   BUCKETS="beauty_pkm beauty_bandwagon" sbatch archive_unlearn_ckpts.sh
#   MAX_GB=200 sbatch archive_unlearn_ckpts.sh          # stop after ~200 GB
#
# Env knobs (defaults):
#   DRYRUN 0   HOLD_MIN 60   MAX_GB 0 (=no limit)   BUCKETS "" (=all)
#   RUN_LIST "" (=all runs): a file of run dir basenames (one per line); only those runs'
#            unlearned_*.ckpt files are considered (e.g. scripts/list_canonical_ckpt_runs.py)
#   ARCHIVE_ROOT /sc-projects/sc-proj-ikim-erase/GRID/unlearn_checkpoints
#
# Caveat: the symlink target only resolves where /sc-projects is mounted. Jobs
# that must LOAD an archived checkpoint have to run on a node with that mount.

set -euo pipefail
GRID_DIR="${SLURM_SUBMIT_DIR:-$PWD}"
cd "${GRID_DIR}"; mkdir -p logs

DRYRUN="${DRYRUN:-0}"
HOLD_MIN="${HOLD_MIN:-60}"
MAX_GB="${MAX_GB:-0}"
BUCKETS="${BUCKETS:-}"
RUN_LIST="${RUN_LIST:-}"
ARCHIVE_ROOT="${ARCHIVE_ROOT:-/sc-projects/sc-proj-ikim-erase/GRID/unlearn_checkpoints}"
MANIFEST="${ARCHIVE_ROOT}/MANIFEST.tsv"
RUNS_ROOT="logs/unlearn/runs"

if [ ! -d "${ARCHIVE_ROOT}" ]; then
  echo "Archive root not reachable: ${ARCHIVE_ROOT} (is /sc-projects mounted here?)" >&2
  exit 1
fi

# --- which runs are off limits ------------------------------------------------
# Every job id currently in the queue, in any state.
ACTIVE_JOBS="$(squeue -u "${USER}" -h -o '%i' 2>/dev/null | sed 's/_.*//' | sort -u || true)"
echo "[$(date -Is)] active job ids: $(echo ${ACTIVE_JOBS} | tr '\n' ' ')"

is_active() {                       # $1 = run dir basename (contains job<id>)
  local jid="${1#job}"; jid="${jid%%_*}"
  [ -n "${ACTIVE_JOBS}" ] && printf '%s\n' "${ACTIVE_JOBS}" | grep -qx "${jid}"
}

# --- campaign bucket ----------------------------------------------------------
# Matches the layout already on the archive: beauty gets its study-specific
# buckets, every other dataset gets <ds>_<campaign>.
bucket_of() {
  local run="$1"
  case "${run}" in
    *_sens_*)                       echo "sensitive" ;;
    *_toys_*)                       echo "toys_bandwagon" ;;
    *_sports_*)                     echo "sports_bandwagon" ;;
    *_food_*)                       echo "food_bandwagon" ;;
    *rsc15_smoke*)                  echo "rsc15_smoke" ;;
    *rsc15*)                        echo "rsc15_bandwagon" ;;
    *pkmreplaceallv2*|*pkmaddallv2*) echo "beauty_bandwagon_pkm_allv2" ;;
    *pkmreplacee23d01*|*pkmadde23d01*) echo "beauty_bandwagon_pkm_e23d01" ;;
    *pkm*)                          echo "beauty_bandwagon_pkm_d01" ;;
    *_idw*|*_agg*|*_L8_*|*_L16_*)   echo "beauty_bandwagon_longid" ;;
    *clone_*)                       echo "beauty_clone_flood_inject" ;;
    *_bw_*)                         echo "beauty_bandwagon" ;;
    *)                              echo "other" ;;
  esac
}

want_bucket() {
  [ -z "${BUCKETS}" ] && return 0
  printf '%s\n' ${BUCKETS} | grep -qx "$1"
}

[ -f "${MANIFEST}" ] || printf 'run_dir\tbucket\tfile\tbytes\n' > "${MANIFEST}"

moved=0 moved_bytes=0 skipped_active=0 skipped_recent=0 skipped_bucket=0 failed=0
limit_bytes=$(( MAX_GB * 1024 * 1024 * 1024 ))

# Real checkpoint files only: unlearned.ckpt is a symlink to unlearned_9.ckpt,
# so a size scan that follows it reports ~16 bytes and hides a terabyte.
while IFS= read -r src; do
  run="$(basename "$(dirname "$(dirname "${src}")")")"
  file="$(basename "${src}")"
  bucket="$(bucket_of "${run}")"

  want_bucket "${bucket}" || { skipped_bucket=$((skipped_bucket+1)); continue; }
  if is_active "${run}"; then
    echo "  SKIP (job active)  ${run}/${file}"; skipped_active=$((skipped_active+1)); continue
  fi
  # Recently touched run dir: a job may be mid-write without being in squeue yet.
  if [ -n "$(find "$(dirname "${src}")" -maxdepth 1 -newermt "-${HOLD_MIN} minutes" -print -quit)" ]; then
    skipped_recent=$((skipped_recent+1)); continue
  fi

  bytes=$(stat -c %s "${src}")
  dst_dir="${ARCHIVE_ROOT}/${bucket}/${run}"
  dst="${dst_dir}/${file}"

  if [ "${DRYRUN}" = "1" ]; then
    printf '  WOULD MOVE %8.2f GB  %-28s %s/%s\n' \
      "$(python3 -c "print(${bytes}/1024**3)")" "${bucket}" "${run}" "${file}"
    moved=$((moved+1)); moved_bytes=$((moved_bytes+bytes))
    if [ "${limit_bytes}" -gt 0 ] && [ "${moved_bytes}" -ge "${limit_bytes}" ]; then
      echo "  reached MAX_GB=${MAX_GB}"; break
    fi
    continue
  fi

  mkdir -p "${dst_dir}"
  # --remove-source-files unlinks the source only after a verified transfer.
  if rsync -a --remove-source-files --partial "${src}" "${dst}"; then
    if [ "$(stat -c %s "${dst}")" != "${bytes}" ]; then
      echo "  SIZE MISMATCH after copy: ${dst}" >&2; failed=$((failed+1)); continue
    fi
    ln -sfn "${dst}" "${src}"
    printf '%s\t%s\t%s\t%s\n' "${run}" "${bucket}" "${file}" "${bytes}" >> "${MANIFEST}"
    moved=$((moved+1)); moved_bytes=$((moved_bytes+bytes))
    printf '  moved %8.2f GB  %s/%s\n' "$(python3 -c "print(${bytes}/1024**3)")" "${run}" "${file}"
  else
    echo "  RSYNC FAILED: ${src}" >&2; failed=$((failed+1))
  fi

  if [ "${limit_bytes}" -gt 0 ] && [ "${moved_bytes}" -ge "${limit_bytes}" ]; then
    echo "  reached MAX_GB=${MAX_GB}, stopping"; break
  fi
done < <(if [ -n "${RUN_LIST}" ]; then
            while IFS= read -r r; do
              [ -n "${r}" ] && [ -d "${RUNS_ROOT}/${r}/checkpoints" ] || continue
              find "${RUNS_ROOT}/${r}/checkpoints" -maxdepth 1 -type f -name 'unlearned_*.ckpt'
            done < "${RUN_LIST}" | sort
          else
            find "${RUNS_ROOT}" -type f -name 'unlearned_*.ckpt' | sort
          fi)

printf '[%s] %s: %d files, %.1f GB | skipped: %d active, %d recent(<%smin), %d other-bucket | failed: %d\n' \
  "$(date -Is)" "$([ "${DRYRUN}" = 1 ] && echo DRYRUN || echo moved)" \
  "${moved}" "$(python3 -c "print(${moved_bytes}/1024**3)")" \
  "${skipped_active}" "${skipped_recent}" "${HOLD_MIN}" "${skipped_bucket}" "${failed}"
df -h . | tail -1
