#!/usr/bin/env bash
# 2026-11-25, clean data, `_es` recipe (run_itemmem_grid.sh CLEAN=1):
#   wide   parameter-matched TIGER (plainwide, d_ff 1792): beauty toys sports x seeds 2 3 5 7 11
#   sweep  memory size, imc_r32 imc_r64 (row size) and imc_h10 imc_h25 imc_h50 (head-only rows):
#          beauty toys x seeds 2 3 5, each followed by its memory-off eval (cleaneval)
#   quant  post-hoc int8 / fp16 rows on the existing imc _es models (eval only, 1 GPU each)
# Usage (from GRID/): bash scripts/submit_size_controls.sh [wide] [sweep] [quant]
set -euo pipefail
unset ${!SLURM_*}
cd "$(dirname "$0")/.."
TSV=logs/size_controls_submit_20261125.tsv
touch $TSV
WHAT="${*:-wide sweep quant}"
train() {  # ds arch seed -> train job (+ memory-off eval chain for memory archs)
  grep -qP "^$1\t$2\t$3\t" $TSV && return 0
  local j c=""
  j=$(STAGE=train DATASET=$1 ARCH=$2 DSEED=$3 CLEAN=1 ./run_itemmem_grid.sh)
  if [[ "$2" == imc_* ]]; then
    c=$(sbatch --parsable --partition=compute --gres=none -c 1 --mem=2G --time=00:20:00 \
        --dependency=afterok:$j --job-name="chain_sz_$1_$2_s$3" --output=logs/%j.out \
        --wrap "cd $PWD && unset \${!SLURM_*} && STAGE=cleaneval DATASET=$1 ARCH=$2 DSEED=$3 CLEAN=1 ./run_itemmem_grid.sh")
  fi
  printf "%s\t%s\t%s\t%s\t%s\n" $1 $2 $3 $j "$c" | tee -a $TSV
}
if [[ " $WHAT " == *" wide "* ]]; then
  for ds in beauty toys sports; do for s in 2 3 5 7 11; do train $ds plainwide $s; done; done
fi
if [[ " $WHAT " == *" sweep "* ]]; then
  for ds in beauty toys; do for s in 2 3 5; do
    for a in imc_r32 imc_r64 imc_h10 imc_h25 imc_h50; do train $ds $a $s; done
  done; done
fi
if [[ " $WHAT " == *" quant "* ]]; then
  for ds in beauty toys sports; do for s in 2 3 5 7 11; do
    STAGE=cleaneval DATASET=$ds ARCH=imc DSEED=$s CLEAN=1 VARIANTS="int8 fp16" ./run_itemmem_grid.sh \
      | sed "s/^/quant $ds imc s$s job=/" | tee -a $TSV
  done; done
fi
