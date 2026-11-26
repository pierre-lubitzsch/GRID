#!/usr/bin/env bash
# After the NAU lambda_s=0.01 runs (11544494-95) and their eval (planner 11544496 -> array rmgrid_eval3):
# redo the selection; if NAU's pick changed from lambda_s=0.1/lambda_n=-1, plan + launch NAU follow-ups
# (anchor, takedown row) with suffix _ls001 and run NAU's membership test (pass _nau2).
set -uo pipefail
cd "$(dirname "$0")"
ST=logs/rmsplit_sens/ls001_watch_status.txt
say() { echo "[$(date -Is)] $*" | tee -a "${ST}"; }
say start
while squeue -h -j 11544496 2>/dev/null | grep -q .; do sleep 300; done
sleep 60
while squeue -h -u "$USER" -o '%j' | grep -qx rmgrid_eval3; do sleep 300; done
PYTHONPATH=. python paper/plotting/critique_followup/rmsplit_selection.py 2>&1 | grep -E 'NAU|margin|pending' | tee -a "${ST}"
pick=$(python3 -c "import csv;print([r['new_config'] for r in csv.DictReader(open('paper/revision/rmsplit_selected.csv')) if r['method']=='NAU' and r['dataset']=='beauty'][:1])")
say "NAU pick now: ${pick}"
if echo "${pick}" | grep -q 'ls0.1_lnneg1.0'; then say "NAU pick unchanged; follow-ups already launched"; else
  SFX=_ls001 ONLY=NAU python scripts/plan_rmgrid_followup.py 2>&1 | tail -3 | tee -a "${ST}"
  for k in anchor takedown; do f=logs/rmsplit_sens/plan_followup_${k}_ls001.tsv; [ -s "$f" ] && PLAN=$f NAME=rmf${k:0:1}2 NOEVAL=1 DRYRUN=0 bash run_rmsplit_grid_launch.sh 2>&1 | tee -a "${ST}"; done
fi
python scripts/launch_mia_rmgrid.py --methods NAU --suffix _nau2 --submit 2>&1 | tee -a "${ST}"
say done
