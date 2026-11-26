#!/usr/bin/env bash
#SBATCH --job-name=popbias_ledger
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gpus-per-node=1
#SBATCH --partition=gpu
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#
# Popularity analysis on Table 1's canonical checkpoints: all 15 Beauty cells through
# run_popbias_cell.sh with LEDGER set. Checkpoints already scored under logs/diagnose/popbias/ are
# symlinked; only the 5 mid Forget+Repair cells (old run used fig1js_forget_repair, the ledger has
# the sepgen_keepck execution) are re-scored. Then:
#   PYTHONPATH=. python paper/plotting/popularity_bias.py --root logs/diagnose/popbias_ledger
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
for st in unpopular mid popular; do
  for sd in 2 3 5 7 11; do
    echo "#### ${st} s${sd}"
    STRATUM=$st SEED=$sd LEDGER=paper/plotting/canonical_runs.tsv bash run_popbias_cell.sh
  done
done
PYTHONPATH=. python paper/plotting/popularity_bias.py --root logs/diagnose/popbias_ledger
