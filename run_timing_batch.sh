#!/usr/bin/env bash
#SBATCH --job-name=timing_e
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:nvidia_h200:2
#SBATCH --time=10:00:00
#
# Efficiency (E) timing campaign: rows of logs/timing_e/plan.tsv (scripts/plan_timing_campaign.py),
# run strictly ONE AT A TIME on the FIRST allocated GPU (the second stays idle), so the
# per-request wall_seconds in <run>/*_info.json has no lane contention. Each row is the
# run_valsel_batch.sh (spam) / run_valsel_batch_sens.sh (unwanted) invocation of the canonical
# run, but calls run_tiger_unlearn_sequential.sh directly with UNLEARN_RUN_POST_EVAL=false
# (no test/val eval: they do not touch wall_seconds) and UNLEARN_KEEP_CKPT=0 (checkpoint deleted).
# After each row, host + GPU model are written to <run>/timing_host.txt and the manifest
# logs/timing_e/manifest.tsv. A row whose run dir already exists is skipped.
#
#   sbatch run_timing_batch.sh logs/timing_e/<chunk>.tsv     (via run_timing_launch.sh)
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
BATCH="${1:?batch tsv required}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
GPU="${GPUS[0]}"
HOST="$(hostname -s)"
GPUNAME="$(nvidia-smi --query-gpu=name --format=csv,noheader -i "${GPU}" 2>/dev/null | head -1)"
echo "[$(date -Is)] batch ${BATCH}: $(wc -l < "${BATCH}") rows, ONE lane on GPU ${GPU} (${GPUNAME}) of ${CUDA_VISIBLE_DEVICES:-?} on ${HOST}"
case "${GPUNAME}" in *H200*) ;; *) echo "ABORT: not an H200 (${GPUNAME})"; exit 1 ;; esac
MAN="logs/timing_e/manifest.tsv"

# Gate (chained after the smoke job): the smoke row (Beauty mid seed 2 Forget+Repair) must show
# sub-second-range wall_seconds per request, i.e. the unified timer fix is in effect.
if [ -n "${SMOKE_CHECK_TAG:-}" ]; then
  python3 - "${SMOKE_CHECK_TAG}" "${SMOKE_MAX_S:-3.0}" <<'EOF' || { echo "ABORT: smoke timing check failed"; exit 1; }
import glob, json, statistics, sys
tag, mx = sys.argv[1], float(sys.argv[2])
d = [p for p in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}") if p.endswith(tag)]
assert d, f"no smoke run dir for {tag}"
js = json.load(open(glob.glob(d[0] + "/*_info.json")[0]))
w = [v["wall_seconds"] for r in js["requests"] for k, v in r.items() if k.endswith("_info") and isinstance(v, dict) and "wall_seconds" in v]
m = statistics.median(w)
print(f"smoke {tag}: {len(w)} requests, median wall_seconds {m:.3f} (limit {mx})")
sys.exit(0 if w and m <= mx else 1)
EOF
fi

i=0
while IFS=$'\t' read -r setting ds st sd useed tag ckpt algo sid na rbs rate data_dir tmpl method config extra; do
  i=$((i+1))
  if compgen -G "logs/unlearn/runs/*_bs1_${tag}" >/dev/null; then echo "skip ${tag}: run dir exists"; continue; fi
  echo "[$(date -Is)] start ${i} ${tag}"
  LOG="logs/timing_e/${SLURM_JOB_ID:-local}_${i}.out"
  # shellcheck disable=SC2206
  ov=(${extra})
  if [ "${setting}" = "spam" ]; then
    CUDA_VISIBLE_DEVICES="${GPU}" UNLEARN_RUN_POST_EVAL=false UNLEARN_KEEP_CKPT=0 \
      POISON_METHOD=bandwagon POISONING_RATIO=0.01 N_TARGET_ITEMS=1 POISON_SEED="${sd}" \
      TARGET_STRATEGY="${st}" UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED="${useed}" \
      bash run_tiger_unlearn_sequential.sh "${ckpt}" "${ds}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${ov[@]}" \
      > "${LOG}" 2>&1 < /dev/null
  else
    CUDA_VISIBLE_DEVICES="${GPU}" UNLEARN_RUN_POST_EVAL=false UNLEARN_KEEP_CKPT=0 \
      UNLEARN_EVAL_DATA_DIR="src/data/amazon_data/${ds}" UNLEARN_RUN_TAG="${tag}" UNLEARN_SEED="${useed}" \
      bash run_tiger_unlearn_sequential.sh "${ckpt}" "${data_dir}" "${algo}" "${sid}" "${na}" "${rbs}" "${rate}" "${ov[@]}" \
      > "${LOG}" 2>&1 < /dev/null
  fi
  rc=$?
  OUT="$(grep -m1 '^Using output_dir=' "${LOG}" | sed 's/^Using output_dir=//')"
  DD="$(grep -m1 '^Using data_dir=' "${LOG}" | sed 's/^Using data_dir=//')"
  if [ "${DD}" != "${data_dir}" ]; then echo "  WARNING ${tag}: data_dir ${DD} != template ${data_dir}"; fi
  if [ -n "${OUT}" ] && [ -d "${OUT}" ]; then
    printf 'host=%s\ngpu=%s\ngpu_index=%s\njob=%s\nrc=%s\ntemplate=%s\n' "${HOST}" "${GPUNAME}" "${GPU}" \
      "${SLURM_JOB_ID:-local}" "${rc}" "${tmpl}" > "${OUT}/timing_host.txt"
  fi
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' "${setting}" "${ds}" "${method}" "${config}" "${st}" "${sd}" \
    "${tag}" "${OUT}" "${HOST}" "${GPUNAME}" "${SLURM_JOB_ID:-local}" "${rc}" >> "${MAN}"
  if [ "${rc}" -eq 0 ]; then echo "[$(date -Is)] done ${tag}"; else echo "[$(date -Is)] FAILED ${tag} (rc ${rc}, see ${LOG})"; fi
done < "${BATCH}"
echo "[$(date -Is)] batch finished"
