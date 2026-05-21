#!/usr/bin/env bash
# Shared helpers for unlearning SLURM wrappers.
# Source from run_tiger_unlearn*.sh — do not execute directly.
#
# Builds a unique Hydra output directory so concurrent or back-to-back runs
# never share logs/unlearn/runs/<same-second-timestamp>/.

unlearn_build_output_dir() {
  local grid_dir="${1:?grid_dir required}"
  local request_batch_size="${2:-}"

  local job_id="${SLURM_JOB_ID:-local$$}"
  local ts
  ts="$(date +%Y-%m-%d_%H-%M-%S)"
  local nano=""
  if command -v date >/dev/null 2>&1 && date +%N 2>/dev/null | grep -qv '^0*$'; then
    nano="_$(date +%N)"
  else
    nano="_${RANDOM}"
  fi

  local tag_parts=("job${job_id}" "${ts}${nano}")
  if [ -n "${request_batch_size}" ]; then
    tag_parts+=("bs${request_batch_size}")
  fi
  if [ -n "${UNLEARN_RUN_TAG:-}" ]; then
    tag_parts+=("${UNLEARN_RUN_TAG}")
  fi

  local run_leaf
  run_leaf="$(IFS=_; echo "${tag_parts[*]}")"
  printf '%s/logs/unlearn/runs/%s\n' "${grid_dir}" "${run_leaf}"
}

# Atomically create the run directory before Python/Hydra starts.
unlearn_allocate_output_dir() {
  local out_dir="${1:?output dir required}"
  local parent
  parent="$(dirname "${out_dir}")"
  mkdir -p "${parent}"
  if ! mkdir "${out_dir}" 2>/dev/null; then
    echo "Refusing to run: output directory already exists (another job may own it):"
    echo "  ${out_dir}"
    return 1
  fi
  {
    echo "created_at=$(date -Is)"
    echo "slurm_job_id=${SLURM_JOB_ID:-}"
    echo "pid=$$"
    echo "hostname=$(hostname -s 2>/dev/null || hostname)"
  } > "${out_dir}/.run_allocated"
  return 0
}
