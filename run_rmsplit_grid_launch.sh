#!/usr/bin/env bash
# Full sensitive-deletion hyperparameter grid with the RETRAINING-MATCHED split (decision 2026-11-14,
# reviewer QzxM follow-up 2). Plan: scripts/plan_rmsplit_grid.py -> logs/rmsplit_sens/plan_grid.tsv
# (385 rows = 77 configs x 5 seeds; rows whose run dir exists are skipped, e.g. the 30 earlier
# NAU lambda_n=-1 / Forget+Repair rmsplit runs). Mechanics as run_rmsplit_sens_launch.sh:
# run_valsel_batch_sens.sh -> run_valsel_one.sh with VALSEL_KEEP_CKPT=1.
# GPU class: TRACER and NAU/ablation (unified) rows on 80 GB A100s (lambda_n=-0.1 and TRACER hit
# CUDA OOM on 40 GB on 2026-11-08), SCIF/SEIF/Kookmin/Fanchuan on any gpu-partition GPU.
# Then (afterany) the cleaned test + cleaned validation evals: scripts/plan_rmsplit_grid_eval.py.
#
#   bash run_rmsplit_grid_launch.sh                 # DRY RUN
#   DEP=afterok:<archive job> DRYRUN=0 bash run_rmsplit_grid_launch.sh
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
ROWS_PER_JOB="${ROWS_PER_JOB:-30}"   # 2 lanes per job
D=logs/rmsplit_sens
PLAN="${PLAN:-${D}/plan_grid_wave1.tsv}"
NAME="${NAME:-rmgrid}"
NOEVAL="${NOEVAL:-0}"
[ -s "${PLAN}" ] || python3 scripts/plan_rmsplit_grid.py
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE "^(vsb_${NAME}_|${NAME}_eval)" <<<"${QUEUED}"; then echo "Refusing: rmgrid jobs already queued"; exit 1; fi
rm -f "${D}"/todo_${NAME}_*.tsv
: > "${D}/todo_${NAME}_big.tsv"; : > "${D}/todo_${NAME}_any.tsv"
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then continue; fi
  case "${algo}" in tracer|unified) f="${D}/todo_${NAME}_big.tsv" ;; *) f="${D}/todo_${NAME}_any.tsv" ;; esac
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${tag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${f}"
done < "${PLAN}"
dep=""; [ -n "${DEP:-}" ] && dep="--dependency=${DEP}"
jobs_=()
for cls in big any; do
  f="${D}/todo_${NAME}_${cls}.tsv"; n=$(wc -l < "${f}"); echo "${cls}: ${n} rows"; [ "${n}" -gt 0 ] || continue
  k=$(( (n + ROWS_PER_JOB - 1) / ROWS_PER_JOB ))
  for ((j=0; j<k; j++)); do awk -v j="${j}" -v k="${k}" '(NR-1)%k==j' "${f}" > "${D}/todo_${NAME}_${cls}_${j}.tsv"; done
  # pgpu, 2 GPUs per job (two lanes in run_valsel_batch_sens.sh); the 40 GB pgpu nodes are excluded
  part=(--partition=pgpu --gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02 --cpus-per-task=16 --mem=128G)
  for ((j=0; j<k; j++)); do
    p="${D}/todo_${NAME}_${cls}_${j}.tsv"
    cmd=(sbatch --parsable --job-name="vsb_${NAME}_${cls}_${j}" ${dep} "${part[@]}"
         --time=10:00:00 run_valsel_batch_sens.sh "${p}")
    if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=tiger VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "$(wc -l < "${p}")"
    else jid=$(MODEL=tiger VALSEL_KEEP_CKPT=1 "${cmd[@]}"); jobs_+=("${jid}"); echo "submitted vsb_${NAME}_${cls}_${j}: ${jid}"; fi
  done
done
[ "${NOEVAL}" = "1" ] && { echo "NOEVAL=1: no eval job"; exit 0; }
edep=""; [ "${#jobs_[@]}" -gt 0 ] && edep="--dependency=afterany:$(IFS=:; echo "${jobs_[*]}")"
ecmd=(sbatch --parsable --job-name=${NAME}_eval ${edep} --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G
      --time=24:00:00 --output=logs/%j.out --wrap
      "python scripts/plan_rmsplit_grid_eval.py && CELL=all PLAN=logs/eval/sensitive_scrub/plan_rmgrid.tsv VIEWS='scrub scrubval' PAR=3 bash run_sensitive_scrub_eval.sh")
if [ "${DRYRUN}" = "1" ]; then printf '%q ' "${ecmd[@]}"; printf '\nDRY RUN (DRYRUN=0 to submit)\n'
else echo "submitted rmgrid_eval: $("${ecmd[@]}")"; fi
