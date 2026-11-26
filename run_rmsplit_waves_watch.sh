#!/usr/bin/env bash
# Watcher for the retraining-matched sensitive grid (2026-11-14). Runs on the login node (polling only):
#   1. wait until both smoke jobs are gone from the queue:
#        SMOKE_SEP  generative lambda_s=0.1 (Beauty + Sports seed 2, lambda_n=-1)
#        SMOKE_BASE SCIF / SEIF / Kookmin / Fanchuan / TRACER with the new split (Sports seed 2)
#   2. lambda_s=0.1 score: generative if both generative smoke runs finished with metrics, else cosine;
#      regenerate the plans (scripts/plan_rmsplit_grid.py, SEP_LS01=...);
#   3. submit wave 1b (NAU lambda_s=0.1 x 4 lambda_n), then wave 2 (Forget+Repair lambda_f 0.01 / 1.0 and
#      the baselines; an algorithm whose smoke run failed is left out and reported);
#   4. submit the cleaned test + cleaned validation evals, afterany on every grid job (waves 1a, 1b, 2).
# Status: logs/rmsplit_sens/watch_status.txt
#
#   nohup bash run_rmsplit_waves_watch.sh > logs/rmsplit_sens/watch.log 2>&1 &
set -uo pipefail
cd "$(dirname "$0")"
D=logs/rmsplit_sens
ST="${D}/watch_status.txt"
SMOKE_SEP="${SMOKE_SEP:-11540419}"
SMOKE_BASE="${SMOKE_BASE:-11540408}"
say() { echo "[$(date -Is)] $*" | tee -a "${ST}"; }
done_ok() { local d; d=$(compgen -G "logs/unlearn/runs/*_bs1_$1" | tail -1); [ -n "${d}" ] && [ -s "${d}/eval/csv/version_0/metrics.csv" ] && [ -e "${d}/checkpoints/unlearned.ckpt" ]; }

say "watcher start; waiting for smoke jobs ${SMOKE_SEP} ${SMOKE_BASE}"
while squeue -h -j "${SMOKE_SEP},${SMOKE_BASE}" 2>/dev/null | grep -q .; do sleep 300; done
say "smoke jobs finished"

SEP=generative   # with unlearning.sep_checkpoint=true (identical loss, ~40x less memory); no cosine fallback
say "lambda_s=0.1 separation score: ${SEP}"
SEP_LS01="${SEP}" python3 scripts/plan_rmsplit_grid.py | tee -a "${ST}"

FAILED=()
for a in scif_d0.01 seif_std0.06 kookmin_ir0.001 fanchuan_t0.07 tracer_lf1.0_lc0.1_ep4; do
  done_ok "sens_sports_gunsrm_${a}_rmsplit_canonck" || FAILED+=("${a%%_*}")
done
say "baseline smoke failures: ${FAILED[*]:-none}"

grep -E '_ls0\.1_' "${D}/plan_grid_wave1.tsv" > "${D}/plan_grid_wave1b.tsv"
grep -v "_hp2_fr_" "${D}/plan_grid_wave2.tsv" | grep -v -P "\t(scif|seif)\t" > "${D}/plan_grid_wave2_run.tsv"   # F+R variants (rmg2fr) and SCIF/SEIF (rmg2a) submitted separately   # Forget+Repair lambda_f 0.01/1.0 submitted separately (rmg2fr)
for a in "${FAILED[@]}"; do grep -v -P "\t${a}\t" "${D}/plan_grid_wave2_run.tsv" > "${D}/tmp.tsv"; mv "${D}/tmp.tsv" "${D}/plan_grid_wave2_run.tsv"; done
say "wave 1b: $(wc -l < "${D}/plan_grid_wave1b.tsv") rows; wave 2: $(wc -l < "${D}/plan_grid_wave2_run.tsv") rows"

# wave 1b already submitted manually (11541287-88)
PLAN="${D}/plan_grid_wave2_run.tsv" NAME=rmg2 NOEVAL=1 DRYRUN=0 bash run_rmsplit_grid_launch.sh 2>&1 | tee -a "${ST}"

ids=$(squeue -h -u "$USER" -o '%i %j' | awk '$2 ~ /^vsb_(rmg1a|rmg1b|rmg2|rmg2a|rmg2fr|rmanc)_/ {print $1}' | paste -sd: -)
dep=""; [ -n "${ids}" ] && dep="--dependency=afterany:${ids}"
eid=$(sbatch --parsable --job-name=rmgrid_eval ${dep} --partition=gpu --gres=gpu:1 --cpus-per-task=8 --mem=64G \
      --time=24:00:00 --output=logs/%j.out --wrap \
      "python scripts/plan_rmsplit_grid_eval.py && CELL=all PLAN=logs/eval/sensitive_scrub/plan_rmgrid.tsv VIEWS='scrub scrubval' PAR=3 bash run_sensitive_scrub_eval.sh")
say "eval job ${eid} (afterany ${ids})"
say "watcher done"
