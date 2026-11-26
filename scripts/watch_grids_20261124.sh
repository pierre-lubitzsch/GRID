#!/usr/bin/env bash
# Watcher for the 2026-11-24 grids (Toys, MUNKEY, clone-append, references): exits on the first
# NEW failed job (recorded in logs/watch_grids_seen_fail.txt) or when none of them is left.
cd "$(dirname "$0")/.."
SEEN=logs/watch_grids_seen_fail.txt
touch $SEEN
PAT='ITEMMEM_toys|_ca_|_ca$|MUNKEY_mk11_(up|pop|d)|ref_spam|chain_(toys|mk|ca|ref)|munkey_keys'
ids=$(cat logs/toys_grid_submit_20261124.txt logs/munkey_ca_refs_submit_20261124.txt | grep -oE "[0-9]{8}" | sort -u | paste -sd,)
while true; do
  bad=$( { sacct -X -n -P -j "$ids" -o JobID,JobName,State; sacct -X -n -P -S 2026-11-24T12:00 -o JobID,JobName,State | grep -E "$PAT"; } \
        | grep -E "FAILED|TIMEOUT|OUT_OF|NODE_FAIL|CANCELLED" | sort -u | grep -vFf $SEEN || true)
  if [ -n "$bad" ]; then echo "$bad" | tee -a $SEEN; echo "FAILURE above"; exit 0; fi
  n=$( { squeue -u $USER -h -o "%i %j" | grep -E "$(echo $ids | tr , '|')"; squeue -u $USER -h -o "%i %j" | grep -E "$PAT"; } | sort -u | wc -l)
  [ "$n" = 0 ] && { echo "ALL JOBS DONE"; exit 0; }
  sleep 120
done
