#!/usr/bin/env bash
# Pack the remaining validation-selection candidates of logs/valsel_plan.tsv into
# run_valsel_batch.sh jobs of BATCH rows each. NAU-grid rows (tag contains
# _valsel_nau_) come first at normal priority; baseline rows get --nice=200.
# Skips rows whose run dir exists or whose single job (vs_<tag>) is queued or running.
#
#   bash run_valsel_batch_launch.sh            # DRY RUN
#   DRYRUN=0 bash run_valsel_batch_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"; BATCH="${BATCH:-6}"; BASE_NICE="${BASE_NICE:-200}"
PLAN="${PLAN:-logs/valsel_plan.tsv}"; OUT="logs/valsel_batches"; mkdir -p "${OUT}"
Q="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
# rows already inside a queued/running batch job (job name -> batch file)
INB="$(for n in $(grep -E '^vsb_' <<<"${Q}"); do
  if [[ "$n" =~ ^vsb_(nau|base)_([0-9]{3})$ ]]; then f="${OUT}/20260925_020656_${BASH_REMATCH[1]}_${BASH_REMATCH[2]}.tsv"
  elif [[ "$n" =~ ^vsb_([0-9]{8}_[0-9]{6})_([a-z0-9]+)_([0-9]{3})$ ]]; then f="${OUT}/${BASH_REMATCH[1]}_${BASH_REMATCH[2]}_${BASH_REMATCH[3]}.tsv"
  else echo "unknown batch job name $n" >&2; exit 1; fi
  [ -f "$f" ] || { echo "missing batch file $f" >&2; exit 1; }; cut -f5 "$f"; done)"
nau=(); base=(); skip=0
while IFS= read -r line; do
  tag="$(cut -f5 <<<"${line}")"
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null || grep -qxF "vs_${tag}" <<<"${Q}" || grep -qxF "${tag}" <<<"${INB}"; then skip=$((skip+1)); continue; fi
  if [[ "${tag}" == *_valsel_nau_* || "${tag}" == *_valselck_nau_* ]]; then nau+=("${line}"); else base+=("${line}"); fi
done < "${PLAN}"
echo "remaining: ${#nau[@]} NAU-grid rows, ${#base[@]} baseline rows; ${skip} skipped (done, queued or running)"
stamp="$(date +%Y%m%d_%H%M%S)"; nj=0
submit() {  # kind nice rows...
  local kind="$1" nice="$2"; shift 2; local rows=("$@") k=0
  for ((s=0; s<${#rows[@]}; s+=BATCH)); do
    f="${OUT}/${stamp}_${kind}_$(printf %03d $k).tsv"; printf '%s\n' "${rows[@]:s:BATCH}" > "${f}"
    if [ "${DRYRUN}" = "1" ]; then [ $k -lt 2 ] && echo "  would submit ${f} ($(wc -l < "$f") rows, nice ${nice})"
    else sbatch --job-name="vsb_${stamp}_${kind}_$(printf %03d $k)" --nice="${nice}" "${BATCH_SCRIPT:-run_valsel_batch.sh}" "${f}" | sed 's/^/  /'; fi
    k=$((k+1)); nj=$((nj+1))
  done
}
[ ${#nau[@]} -gt 0 ] && submit "${NAU_KIND:-nau}" 0 "${nau[@]}"
[ ${#base[@]} -gt 0 ] && submit base "${BASE_NICE}" "${base[@]}"
echo "${nj} batch job(s) $([ "${DRYRUN}" = "1" ] && echo 'would be submitted' || echo submitted)"
[ "${DRYRUN}" = "1" ] && rm -f "${OUT}/${stamp}_"*.tsv
