#!/usr/bin/env bash
# LETTER (MODEL=letter, LETTER identifier space) plain vs imc on clean Beauty / Toys / Sports, the
# clean-study `_es` recipe, training seeds 2 3 5 7 11 (label suffix _eslt). Each imc training is
# followed by a memory-off eval (STAGE=cleaneval: best-by-val checkpoint, eval_pathway nulltoken).
# Usage (from GRID/): bash scripts/submit_letter_clean.sh [DATASETS="beauty"] [SEEDS="2"] [ARCHS="plain imc"]
# Skips (dataset, arch, seed) rows already in the TSV.
set -euo pipefail
unset ${!SLURM_*}
cd "$(dirname "$0")/.."
TSV=logs/itemmem_letter_submit_20261125.tsv
touch $TSV
for ds in ${DATASETS:-beauty toys sports}; do
  for s in ${SEEDS:-2 3 5 7 11}; do
    for a in ${ARCHS:-plain imc}; do
      grep -qP "^$ds\t$a\t$s\t" $TSV && continue
      j=$(STAGE=train DATASET=$ds ARCH=$a DSEED=$s CLEAN=1 TOKENIZER=letter ./run_itemmem_grid.sh)
      c=""
      if [ $a != plain ]; then
        c=$(sbatch --parsable --partition=compute --gres=none -c 1 --mem=2G --time=00:20:00 \
            --dependency=afterok:$j --job-name="chain_lt_${ds}_${a}_s$s" --output=logs/%j.out \
            --wrap "cd $PWD && unset \${!SLURM_*} && STAGE=cleaneval DATASET=$ds ARCH=$a DSEED=$s CLEAN=1 TOKENIZER=letter ./run_itemmem_grid.sh")
      fi
      printf "%s\t%s\t%s\t%s\tletter_es\t%s\n" $ds $a $s $j "$c" | tee -a $TSV
    done
  done
done
