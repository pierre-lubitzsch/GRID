#!/usr/bin/env bash
# Sensitive item deletion, retraining-matched split, for the OTHER SID tokenizers (RQ-VAE, LETTER,
# tracerv3) at the configurations pinned for TIGER + RQ-KMeans (2026-11-17; not for the current paper).
# New launcher modelled on run_rmsplit_grid_launch.sh (left unchanged). Differences:
#   * MODEL comes from the plan's first column (one model per plan; letter -> MODEL=letter),
#   * todo files live in logs/rmsplit_sens/othertok/ so nothing of the TIGER grid is touched,
#   * after the unlearning jobs (afterany): a CPU planning job (scripts/plan_rmsplit_othertok_eval.py),
#     then two 2-GPU eval jobs on pgpu (cleaned test view, cleaned validation view; afterok
#     on the planning job), then an archive job (afterany on both evals) that moves the unlearned
#     checkpoints of runs with BOTH evals finished to /sc-projects (archive_unlearn_ckpts.sh RUN_LIST).
# Unlearning rows go through run_valsel_batch_sens.sh -> run_valsel_one.sh with VALSEL_KEEP_CKPT=1,
# exactly as the TIGER grid; pgpu, 2 GPUs (two lanes), 40 GB pgpu nodes excluded.
#
#   TRACK=rqvae bash run_rmsplit_othertok_launch.sh                     # DRY RUN
#   TRACK=rqvae DRYRUN=0 bash run_rmsplit_othertok_launch.sh
#   TRACK=letter DEP=afterany:<archive job> DRYRUN=0 bash run_rmsplit_othertok_launch.sh
# Last line printed: ARCHIVE_JOB=<id> (the job a following wave can depend on).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
TRACK="${TRACK:?TRACK=rqvae|letter|tracerv3}"
DRYRUN="${DRYRUN:-1}"
ROWS_PER_JOB="${ROWS_PER_JOB:-36}"
PLAN="${PLAN:-logs/rmsplit_sens/plan_othertok_${TRACK}.tsv}"
NAME="${NAME:-otk_${TRACK}}"
NOEVAL="${NOEVAL:-0}"
UNL_TIME="${UNL_TIME:-10:00:00}"
D=logs/rmsplit_sens/othertok
mkdir -p "${D}"
[ -s "${PLAN}" ] || { echo "missing plan ${PLAN} (python scripts/plan_rmsplit_othertok.py)" >&2; exit 1; }
MODELS="$(cut -f1 "${PLAN}" | sort -u)"
[ "$(wc -l <<<"${MODELS}")" = "1" ] || { echo "plan mixes models: ${MODELS}" >&2; exit 1; }
MODEL_="${MODELS}"
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE "^(vsb_${NAME}_|${NAME}_(plan|eval|archive))" <<<"${QUEUED}"; then echo "Refusing: ${NAME} jobs already queued"; exit 1; fi
rm -f "${D}"/todo_${NAME}_*.tsv
: > "${D}/todo_${NAME}_big.tsv"; : > "${D}/todo_${NAME}_any.tsv"
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then continue; fi
  case "${algo}" in tracer|unified) f="${D}/todo_${NAME}_big.tsv" ;; *) f="${D}/todo_${NAME}_any.tsv" ;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${f}"
done < "${PLAN}"
dep=""; [ -n "${DEP:-}" ] && dep="--dependency=${DEP}"
LEDGER="${D}/jobs.tsv"
log_job() { printf '%s\t%s\t%s\t%s\n' "$(date -Is)" "${TRACK}" "$1" "$2" >> "${LEDGER}"; }
jobs_=()
for cls in big any; do
  f="${D}/todo_${NAME}_${cls}.tsv"; n=$(wc -l < "${f}"); echo "${cls}: ${n} rows"; [ "${n}" -gt 0 ] || continue
  k=$(( (n + ROWS_PER_JOB - 1) / ROWS_PER_JOB ))
  for ((j=0; j<k; j++)); do awk -v j="${j}" -v k="${k}" '(NR-1)%k==j' "${f}" > "${D}/todo_${NAME}_${cls}_${j}.tsv"; done
  part=(--partition=pgpu --gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02 --cpus-per-task=16 --mem=128G)
  for ((j=0; j<k; j++)); do
    p="${D}/todo_${NAME}_${cls}_${j}.tsv"
    cmd=(sbatch --parsable --job-name="vsb_${NAME}_${cls}_${j}" ${dep} "${part[@]}"
         --time="${UNL_TIME}" run_valsel_batch_sens.sh "${p}")
    if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=%s VALSEL_KEEP_CKPT=1' "${MODEL_}"; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "$(wc -l < "${p}")"
    else jid=$(MODEL="${MODEL_}" VALSEL_KEEP_CKPT=1 "${cmd[@]}"); jobs_+=("${jid}"); log_job "vsb_${NAME}_${cls}_${j}" "${jid}"; echo "submitted vsb_${NAME}_${cls}_${j}: ${jid}"; fi
  done
done
[ "${NOEVAL}" = "1" ] && { echo "NOEVAL=1: no eval/archive jobs"; exit 0; }

sub() {  # name, dependency (may be empty), sbatch args...
  local name="$1" d="$2"; shift 2
  local c=(sbatch --parsable --job-name="${name}" --output=logs/%j.out ${d:+--dependency=${d}} "$@")
  if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${c[@]}" >&2; printf '\n' >&2; echo "DRY_${name}"
  else local id; id=$("${c[@]}"); log_job "${name}" "${id}"; echo "submitted ${name}: ${id}" >&2; echo "${id}"; fi
}
# no unlearning job submitted (all rows done or dry run): the plan job inherits DEP instead
pdep="${DEP:-}"; [ "${#jobs_[@]}" -gt 0 ] && pdep="afterany:$(IFS=:; echo "${jobs_[*]}")"
PJ=$(sub "${NAME}_plan" "${pdep}" --partition=compute --cpus-per-task=2 --mem=8G --time=00:30:00 \
       --wrap "python scripts/plan_rmsplit_othertok_eval.py --track ${TRACK}")
EVAL_PLAN="logs/eval/sensitive_scrub/plan_othertok_${TRACK}.tsv"
EJ=()
for v in scrub scrubval; do
  # pgpu, 2 GPUs (user instruction 2026-11-17); the runner spreads PAR evals over both GPUs
  EJ+=("$(sub "${NAME}_eval_${v}" "afterok:${PJ}" --partition=pgpu --gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02 \
          --cpus-per-task=16 --mem=128G --time=12:00:00 \
          --wrap "CELL=all PLAN=${EVAL_PLAN} VIEWS=${v} PAR=4 bash run_sensitive_scrub_eval_othertok.sh")")
done
AJ=$(sub "${NAME}_archive" "afterany:${EJ[0]}:${EJ[1]}" --partition=compute --cpus-per-task=4 --mem=8G --time=12:00:00 \
       --wrap "python scripts/othertok_archive_list.py --track ${TRACK} --out ${D}/archive_${TRACK}.txt && RUN_LIST=${D}/archive_${TRACK}.txt HOLD_MIN=10 bash archive_unlearn_ckpts.sh")
[ "${DRYRUN}" = "1" ] && echo "DRY RUN (DRYRUN=0 to submit)"
echo "ARCHIVE_JOB=${AJ}"
