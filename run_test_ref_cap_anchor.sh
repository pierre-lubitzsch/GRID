#!/usr/bin/env bash
#SBATCH --job-name=rbvtest_unit
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --partition=compute
#SBATCH --time=00:45:00
# CPU unit test of the reference cap / retain-side anchor (mock model, no checkpoint).
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
CUDA_VISIBLE_DEVICES="" python -u test_ref_cap_anchor.py
