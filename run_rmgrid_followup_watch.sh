#!/usr/bin/env bash
# Watcher for the remaining sensitive-deletion follow-ups (2026-11-15). Login node, polling only.
#  A. after eval array 11543116 (all cells): selection -> MIA for every selected method except
#     Kookmin/Fanchuan (scripts/launch_mia_rmgrid.py, pgpu 2 GPUs)
#  B. after the takedown follow-up runs (vsb_rmft_*): cleaned-history takedown eval of those rows
#  C. after array rmgrid_eval2 (Kookmin/Fanchuan): selection -> Kookmin/Fanchuan takedown rows + MIA pass _kf
# Status: logs/rmsplit_sens/followup_watch_status.txt
set -uo pipefail
cd "$(dirname "$0")"
ST=logs/rmsplit_sens/followup_watch_status.txt
say() { echo "[$(date -Is)] $*" | tee -a "${ST}"; }
waitname() { while squeue -h -u "$USER" -o '%j' | grep -qE "$1"; do sleep 300; done; }
PG=(--partition=pgpu --gres=gpu:2 --exclude=s-sc-pgpu01,s-sc-pgpu02 --cpus-per-task=16 --mem=128G)

say "start"
while squeue -h -j 11543116 2>/dev/null | grep -q .; do sleep 300; done
say "A: eval array 11543116 finished; selection"
PYTHONPATH=. python paper/plotting/critique_followup/rmsplit_selection.py 2>&1 | tail -25 | tee -a "${ST}"
M=$(python3 - <<'EOF'
import csv
ms=sorted({r["method"] for r in csv.DictReader(open("paper/revision/rmsplit_selected.csv"))} - {"Kookmin","Fanchuan"})
print(" ".join(ms))
EOF
)
say "A: MIA for: ${M}"
python scripts/launch_mia_rmgrid.py --methods ${M} --submit 2>&1 | tee -a "${ST}"

waitname '^vsb_rmft_'
say "B: takedown follow-up runs finished; cleaned-history eval"
python scripts/plan_takedown_scrub_eval.py --plans logs/rmsplit_sens/plan_followup_takedown.tsv \
  --out logs/eval/sensitive_scrub/plan_takedown_scrub_followup.tsv 2>&1 | tee -a "${ST}"
j=$(sbatch --parsable --job-name=scrub_takedown_fu "${PG[@]}" --time=06:00:00 --output=logs/%j.out \
    --export=ALL,CELL=all,PLAN=logs/eval/sensitive_scrub/plan_takedown_scrub_followup.tsv,VIEWS=scrub,PAR=4 \
    run_sensitive_scrub_eval_mgpu.sh)
say "B: takedown follow-up eval job ${j}"

while squeue -h -j 11543117 2>/dev/null | grep -q .; do sleep 300; done
sleep 60
waitname '^rmgrid_eval2$'
say "C: rmgrid_eval2 finished; selection incl. Kookmin/Fanchuan"
PYTHONPATH=. python paper/plotting/critique_followup/rmsplit_selection.py 2>&1 | tail -25 | tee -a "${ST}"
SFX=_kf ONLY=Kookmin,Fanchuan python scripts/plan_rmgrid_followup.py 2>&1 | tail -5 | tee -a "${ST}"
if [ -s logs/rmsplit_sens/plan_followup_takedown_kf.tsv ]; then
  PLAN=logs/rmsplit_sens/plan_followup_takedown_kf.tsv NAME=rmftkf NOEVAL=1 DRYRUN=0 bash run_rmsplit_grid_launch.sh 2>&1 | tee -a "${ST}"
fi
python scripts/launch_mia_rmgrid.py --methods Kookmin Fanchuan --suffix _kf --submit 2>&1 | tee -a "${ST}"
say "done"
