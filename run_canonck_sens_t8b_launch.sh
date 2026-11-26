#!/usr/bin/env bash
# Re-run, WITH checkpoint, the sensitive-deletion selection candidates whose `_canonck` execution
# (run_canonck_sens_t8_launch.sh, 2026-11-08) died with CUDA OOM on 40 GB A100s and left no
# checkpoint: NAU lambda_n=-0.1 (3 datasets) and six TRACER configurations (Sports, Toys),
# 9 configs x 5 seeds = 45 rows. Rows are copied from logs/canonck_sens/plan_t8.tsv unchanged
# (same start ckpt, SID, unlearning seed, data dir and overrides as the recorded `_valsens` run);
# only the tag changes (_canonck -> _canonckb), so the crashed run dirs stay untouched.
# Needed for the reviewer QzxM follow-up: model selection on the CLEANED validation histories
# (scripts/plan_sensitive_scrubval_eval.py), which needs a checkpoint per candidate and seed.
# 80 GB GPUs only (gpu partition, nvidia_a100_80gb_pcie).
#
#   bash run_canonck_sens_t8b_launch.sh            # DRY RUN
#   DRYRUN=0 bash run_canonck_sens_t8b_launch.sh   # submit
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"
D=logs/canonck_sens
SRC="${D}/plan_t8.tsv"
TODO="${D}/todo_t8b.tsv"
QUEUED="$(squeue -h -u "$USER" -o '%j' 2>/dev/null || true)"
if grep -qE '^vsb_canonck_t8b' <<<"${QUEUED}"; then echo "Refusing: vsb_canonck_t8b already queued"; exit 1; fi
: > "${TODO}"
while IFS=$'\t' read -r model ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl extra; do
  # only rows whose _canonck run exists but has no checkpoint (the OOM casualties)
  run=$(compgen -G "logs/unlearn/runs/*_bs1_${tag}" | head -1 || true)
  [ -n "${run}" ] || continue
  compgen -G "${run}/checkpoints/*.ckpt" >/dev/null && continue
  ntag="${tag/_canonck/_canonckb}"
  if compgen -G "logs/unlearn/runs/*_bs1_${ntag}" >/dev/null; then echo "  exists: ${ntag}"; continue; fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${ds}" "${st}" "${sd}" "${useed}" "${ntag}" \
    "${ckpt}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${data_dir}" "${tmpl}" "${extra}" >> "${TODO}"
done < "${SRC}"
n=$(wc -l < "${TODO}")
echo "rows to run: ${n}"; cut -f5 "${TODO}" | sed 's/_seed[0-9]*$//' | sort | uniq -c
[ "${n}" -gt 0 ] || exit 0
rm -f "${D}"/todo_t8b_[0-9]*.tsv
parts=$(( (n + 14) / 15 ))
split -n "l/${parts}" -d --additional-suffix=.tsv "${TODO}" "${D}/todo_t8b_"
for p in "${D}"/todo_t8b_[0-9]*.tsv; do
  k=$(basename "${p}" .tsv | sed 's/todo_t8b_//'); c=$(wc -l < "${p}")
  cmd=(sbatch --parsable --job-name="vsb_canonck_t8b_${k}" --partition=gpu --gres=gpu:nvidia_a100_80gb_pcie:1
       --cpus-per-task=8 --mem=64G --time=10:00:00 run_valsel_batch_sens.sh "${p}")
  if [ "${DRYRUN}" = "1" ]; then printf 'MODEL=tiger VALSEL_KEEP_CKPT=1'; printf ' %q' "${cmd[@]}"; printf '   # %s rows\n' "${c}"
  else echo "submitted vsb_canonck_t8b_${k} (${c} rows): $(MODEL=tiger VALSEL_KEEP_CKPT=1 "${cmd[@]}")"; fi
done
[ "${DRYRUN}" = "1" ] && echo "DRY RUN (DRYRUN=0 to submit)"
