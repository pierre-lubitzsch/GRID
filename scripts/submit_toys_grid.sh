#!/usr/bin/env bash
# Poisoned Toys architecture ablation (second dataset of the PKM paper): plain, im, imc, ims,
# imsc x 3 strata x 5 data seeds, 4000 steps; each training is followed by a CPU chain job
# (afterok) that submits pre (0-step eval) and full F+R / NAU unlearning with the memory-off
# evals (OFF_EVALS=1). Same setup as the beauty grid (run_item_memory_beauty.sh, DS=toys).
# Training is pinned to one GPU type for the whole dataset (H200; A100-80GB was full).
# Usage (from GRID/): bash scripts/submit_toys_grid.sh  [ARCHS="plain im" STRATA=mid SEEDS=2]
set -euo pipefail
unset ${!SLURM_*}
cd "$(dirname "$0")/.."
LOG=logs/toys_grid_submit_$(date +%Y%m%d).txt
for st in ${STRATA:-unpopular mid popular}; do
  case $st in unpopular) tok=_up ;; popular) tok=_pop ;; *) tok="" ;; esac
  for d in ${SEEDS:-2 3 5 7 11}; do
    dt=""; [ "$d" != 2 ] && dt="_d$d"
    for a in ${ARCHS:-plain im imc ims imsc}; do
      m="$a$tok$dt"
      tid=$(TRAIN_GRES=${TRAIN_GRES:-gpu:nvidia_h200:2} DS=toys STAGE=train ARMS=$a STRATUM=$st DSEED=$d ./run_item_memory_beauty.sh \
            | grep -oE "Submitted batch job [0-9]+" | awk '{print $4}')
      cid=$(sbatch --parsable --partition=compute --gres=none -c 1 --mem=2G --time=00:20:00 \
            --dependency=afterok:$tid --job-name="chain_toys_$m" --output=logs/%j.out \
            --wrap "cd $PWD && unset \${!SLURM_*} && DS=toys STAGE=unlearn SPECS='$m:pre $m:full' ./run_item_memory_beauty.sh")
      echo "$st $d $m train=$tid chain=$cid" | tee -a "$LOG"
    done
  done
done
