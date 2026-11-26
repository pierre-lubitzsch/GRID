#!/usr/bin/env bash
#SBATCH --job-name=otk_scrub_eval
#SBATCH --output=logs/%A_%a.out
#SBATCH --error=logs/%A_%a.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --time=06:00:00
#SBATCH --array=0-14%8
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
# -----------------------------------------------------------------------------
# OTHERTOK COPY (2026-11-17) of run_sensitive_scrub_eval_mgpu.sh: identical except the optional
# 15th plan column `quantizer` (scripts/plan_rmsplit_othertok_eval.py) and the job name.
#
# Unwanted-item track, scrubbed-history re-evaluation. One array task = one
# (dataset, seed) cell of logs/eval/sensitive_scrub/plan.tsv (written by
# scripts/plan_sensitive_scrub_eval.py): original, retrained and every `_mia`
# unlearned model, each on the NORMAL test view (same-code baseline) and on the
# SCRUBBED test view (scripts/build_scrubbed_eval_view.py). 20 to 22 evals per
# task, PAR of them at a time on the task's single GPU.
#
# The eval command is the one run_tiger_unlearn_sequential.sh's post-eval used for
# the paper's sensitive numbers (eval_ckpt_on_test, experiment=tiger_train_flat,
# spam_forget_manifest = the sensitive manifest so SHF/SHR are registered); only
# data_dir and the output dir differ. Finished evals (metrics.csv present) are
# skipped, so a resubmit only fills gaps.
#
#   sbatch run_sensitive_scrub_eval.sh                       # all 15 cells
#   sbatch --array=0 run_sensitive_scrub_eval.sh             # one cell (beauty_s2)
#   CELL=beauty_s2 DRYRUN=1 bash run_sensitive_scrub_eval.sh # print commands only
#   CELL=all PLAN=logs/eval/sensitive_scrub/plan_canonck.tsv bash run_sensitive_scrub_eval.sh
#       (every row of a plan in one allocation; an optional 14th plan column `experiment`,
#        default tiger_train_flat, selects the model's eval config, e.g. letter_train_flat)
# -----------------------------------------------------------------------------
PLAN="${PLAN:-logs/eval/sensitive_scrub/plan.tsv}"
CELLS_FILE="${CELLS_FILE:-logs/eval/sensitive_scrub/cells.txt}"
PAR="${PAR:-2}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
DRYRUN="${DRYRUN:-0}"
VIEWS="${VIEWS:-normal scrub}"
if [ -z "${CELL:-}" ]; then
  CELL="$(sed -n "$(( ${SLURM_ARRAY_TASK_ID:-0} + 1 ))p" "${CELLS_FILE}")"
fi
[ -n "${CELL}" ] || { echo "no cell for task ${SLURM_ARRAY_TASK_ID:-?}" >&2; exit 1; }
echo "[$(date -Is)] cell ${CELL} on $(hostname), PAR=${PAR}, GPUs ${GPUS[*]}"

run_one() {  # one plan row (tab-separated fields as args)
  local cell="$1" ds="$2" sd="$3" role="$4" method="$5" arm="$6" view="$7" ck="$8" dd="$9"
  local man="${10}" sid="${11}" out="${12}" src="${13}" exp="${14:-tiger_train_flat}" qcol="${15:-}"
  local model=tiger quant=rkmeans
  case "${exp}" in letter_*) model=letter; quant=letter ;; tiger_*) ;; *) echo "   unknown experiment ${exp}" >&2; return 1 ;; esac
  # othertok copy: the 15th plan column names the identifier space (rqvae | letter | tracerv3), so a
  # reference's coordinates.json never claims rkmeans for another tokenizer
  [ -n "${qcol}" ] && quant="${qcol}"
  if [ -f "${out}/csv/version_0/metrics.csv" ]; then echo "   done already: ${out}"; return 0; fi
  [ -d "${dd}/testing" ] || { echo "   MISSING view ${dd} (build it first)" >&2; return 1; }
  [ -e "${ck}" ] || { echo "   MISSING ckpt ${ck}" >&2; return 1; }
  echo "   ${view} ${role} ${method} ${arm} -> ${out}"
  [ "${DRYRUN}" = "1" ] && return 0
  mkdir -p "${out}"
  if [ "${role}" != "unlearned" ]; then
    cat > "${out}/coordinates.json" <<JSON
{
  "dataset": "${ds}",
  "role": "${arm}",
  "category": "$(python -c "import json;print({'beauty':'hairloss','sports':'guns','toys':'toyweapons'}['${ds}'])")",
  "model": "${model}",
  "quantizer": "${quant}",
  "seed": ${sd},
  "forget_ratio": "${FORGET_RATIO_TOK:-1e-4}",
  "ckpt": "${ck}",
  "manifest": "${man}",
  "semantic_id_path": "${sid}",
  "data_dir": "${dd}",
  "view": "${view}",
  "ckpt_source": "${src}"
}
JSON
  fi
  python -u -m scripts.eval_ckpt_on_test \
    experiment="${exp}" \
    data_dir="${dd}" \
    "semantic_id_path='${sid}'" \
    "ckpt_path='${ck}'" \
    "spam_forget_manifest='${man}'" \
    num_hierarchies=4 seed="${sd}" \
    train=False test=True \
    trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false \
    trainer.num_nodes=1 trainer.deterministic=true \
    callbacks.model_checkpoint=null callbacks.early_stopping=null \
    hydra.run.dir="${out}" > "${out}.log" 2>&1 \
    && echo "   ok ${out}" || echo "   FAILED ${out} (see ${out}.log)" >&2
}

n=0
while IFS=$'\t' read -r -a f; do
  f[${#f[@]}-1]="${f[${#f[@]}-1]%$'\r'}"   # plan.tsv was written with CRLF line ends
  [ "${CELL}" = "all" ] || [ "${f[0]}" = "${CELL}" ] || continue
  case " ${VIEWS} " in *" ${f[6]} "*) ;; *) continue ;; esac
  # multi-GPU copy (2026-11-15): spread the PAR concurrent evals round-robin over the job's GPUs
  CUDA_VISIBLE_DEVICES="${GPUS[$(( n % ${#GPUS[@]} ))]}" run_one "${f[@]}" &
  n=$((n + 1))
  while [ "$(jobs -rp | wc -l)" -ge "${PAR}" ]; do wait -n || true; done
done < <(tail -n +2 "${PLAN}")
wait || true
echo "[$(date -Is)] cell ${CELL}: ${n} evals dispatched"
