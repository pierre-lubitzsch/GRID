#!/usr/bin/env bash
# Launch run_paper_methods_ds.sh only once the needed training chains have CONVERGED.
# A chain counts as converged when one of its segments (CKJOB in run_paper_methods_ds.sh,
# read at run time) ended COMPLETED, i.e. early stopping fired; TIMEOUT segments do not
# count. Meant to run as a compute job with --dependency=afterany on the last segments,
# so a timed-out chain neither launches unlearning on an unconverged model nor leaves a
# never-satisfiable dependency behind: it prints which chain needs more segments and exits 1.
#
#   DS=food_k5 MODEL=tiger NT=1 bash pm_launch_if_converged.sh
set -uo pipefail
cd "$(dirname "$(readlink -f "$0")")"
DS="${DS:?}"; MODEL="${MODEL:?}"; NT="${NT:?}"
ok=1
for key in "${MODEL}_${DS}_clean" "${MODEL}_${DS}_${NT}"; do
  ids=$(grep -oE "\[${key}\]=\"[0-9 ]+\"" run_paper_methods_ds.sh | grep -oE "[0-9]+( [0-9]+)*" | tail -1)
  [ -n "$ids" ] || { echo "no CKJOB entry for ${key}"; ok=0; continue; }
  done_j=""
  for j in $ids; do
    [ "$(sacct -j "$j" -X -n -o State 2>/dev/null | tr -d ' ')" = COMPLETED ] && { done_j=$j; break; }
  done
  if [ -n "$done_j" ]; then echo "${key}: converged (segment ${done_j})"
  else echo "${key}: NOT converged (segments: ${ids}); add FORCE=1 RESUME_FROM segments"; ok=0; fi
done
[ "$ok" = 1 ] || exit 1
DS="$DS" MODEL="$MODEL" NT="$NT" DRYRUN=0 bash run_paper_methods_ds.sh
