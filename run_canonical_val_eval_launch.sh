#!/usr/bin/env bash
# Plan and submit the validation re-evaluations of canonical checkpoints
# (paper/revision/canonical_runs_plan.md step 2): every row of paper/plotting/canonical_runs.tsv
# with a checkpoint and no <run>/eval_val, split into N_JOBS chunks of run_canonical_val_eval.sh.
#
# Double-submission guard: rows already in a chunk of a queued/running canon_val job
# (job name cve_<stamp>_<k>, chunk logs/canon_val/<stamp>_<k>.tsv) are left out; the job body
# also skips rows with output or a held lock.
#
#   bash run_canonical_val_eval_launch.sh             # DRY RUN (default): plan + print commands
#   DRYRUN=0 bash run_canonical_val_eval_launch.sh    # submit
#   N_JOBS=4 LANES_PER_GPU=1 AFTER=<jobid> ...        # optional knobs (AFTER = afterok dependency)
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DRYRUN="${DRYRUN:-1}"; N_JOBS="${N_JOBS:-4}"
LEDGER="${LEDGER:-paper/plotting/canonical_runs.tsv}"
STAMP="$(date +%Y%m%d_%H%M%S)"
mkdir -p logs/canon_val
PLAN="logs/canon_val/plan_${STAMP}.tsv"
python3 scripts/plan_canonical_val_eval.py --ledger "${LEDGER}" --out "${PLAN}" || {
  echo "planner refused some rows (see above); fix or inspect before submitting" >&2; exit 1; }

# run dirs already covered by queued/running chunks
QUEUED_CHUNKS="$(squeue -h -u "$USER" -o '%j' 2>/dev/null | sed -n 's/^cve_\(.*\)$/logs\/canon_val\/\1.tsv/p' || true)"
COVERED="logs/canon_val/covered_${STAMP}.txt"; : > "${COVERED}"
for c in ${QUEUED_CHUNKS}; do [ -f "$c" ] && cut -f1 "$c" >> "${COVERED}"; done
TODO="logs/canon_val/todo_${STAMP}.tsv"
awk -F'\t' 'FILENAME==ARGV[1] {c[$1]=1; next} !($1 in c)' "${COVERED}" "${PLAN}" > "${TODO}"
n=$(wc -l < "${TODO}"); echo "${n} row(s) to submit ($(wc -l < "${COVERED}") already in queued chunks)"
[ "${n}" -gt 0 ] || exit 0
# round-robin split so every chunk mixes datasets/configs
for ((k=0; k<N_JOBS; k++)); do
  awk -v k="${k}" -v n="${N_JOBS}" '(NR-1) % n == k' "${TODO}" > "logs/canon_val/${STAMP}_${k}.tsv"
done
for ((k=0; k<N_JOBS; k++)); do
  chunk="logs/canon_val/${STAMP}_${k}.tsv"
  [ -s "${chunk}" ] || continue
  cmd=(sbatch --parsable --job-name="cve_${STAMP}_${k}" ${AFTER:+--dependency=afterok:${AFTER}} run_canonical_val_eval.sh "${chunk}")
  if [ "${DRYRUN}" = "1" ]; then
    printf '  %q' "${cmd[@]}"; printf '   # %s rows\n' "$(wc -l < "${chunk}")"
  else
    jid="$(LANES_PER_GPU="${LANES_PER_GPU:-1}" "${cmd[@]}")"; echo "submitted ${jid}: ${chunk} ($(wc -l < "${chunk}") rows)"
  fi
done
if [ "${DRYRUN}" = "1" ]; then
  echo "DRY RUN. First row as the job would run it:"
  IFS=$'\t' read -r OUT ds VALVIEW SID DATA_DIR SEED cfg src < "${TODO}"
  echo "  python -u -m scripts.eval_ckpt_on_test experiment=tiger_train_flat data_dir=${VALVIEW} \"semantic_id_path='${SID}'\" \"ckpt_path='${OUT}/checkpoints/unlearned.ckpt'\" \"spam_forget_manifest='${DATA_DIR}/forget_manifest.json'\" num_hierarchies=4 seed=${SEED} train=False test=True trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir=${OUT}/eval_val"
  rm -f logs/canon_val/"${STAMP}"_*.tsv
fi
