#!/bin/bash
#SBATCH --job-name=sasrec_v2
#SBATCH --output=logs/sasrec_nau_v2/slurm/%x_%j.out
#SBATCH --error=logs/sasrec_nau_v2/slurm/%x_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=128G
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=08:00:00
# SASRec vs TIGER under the paper's protocol (rebuttal; scripts/sasrec_nau_v2.py).
# One job per DATASET: five lanes (one per data seed), alternating over the two GPUs.
# Each lane: clean SASRec (retrained reference) for the seed, then for each stratum the
# poisoned SASRec and the six unlearning arms + clean/poison evaluation of that cell.
# Idempotent: existing checkpoints and arm JSONs are skipped.
#
#   DATASET=beauty sbatch --job-name=sasrec_v2_beauty run_sasrec_nau_v2.sh
#
# Beauty seed 2 reuses the testing tree's SASRec checkpoints (symlinked, same code/protocol):
#   /sc-projects/sc-proj-ikim-erase/workspace/GRID_testing/logs/sasrec_nau/ckpt/sasrec_beauty_{clean,bw_tgt*_n1}.pt
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
DS="${DATASET:?DATASET=beauty|sports|toys}"
SEEDS="${SEEDS:-2 3 5 7 11}"
STRATA="${STRATA:-unpopular mid popular}"
OUT=logs/sasrec_nau_v2
CK="${OUT}/ckpt"
mkdir -p "${CK}" "${OUT}/slurm" "${OUT}/lanes"
TEST_CK=/sc-projects/sc-proj-ikim-erase/workspace/GRID_testing/logs/sasrec_nau/ckpt
if [ "${DS}" = beauty ]; then
  [ -e "${CK}/beauty_clean_seed2.pt" ] || ln -s "${TEST_CK}/sasrec_beauty_clean.pt" "${CK}/beauty_clean_seed2.pt"
  for st in unpopular mid popular; do
    [ -e "${CK}/beauty_poison_${st}_seed2.pt" ] || ln -s "${TEST_CK}/sasrec_beauty_bw_tgt${st}_n1.pt" "${CK}/beauty_poison_${st}_seed2.pt"
  done
fi
nvidia-smi -L || true

lane() {  # $1 seed, $2 gpu
  local sd=$1
  export CUDA_VISIBLE_DEVICES=$2
  local clean="${CK}/${DS}_clean_seed${sd}.pt"
  python -u -m scripts.sasrec_nau_v2 train --dataset "${DS}" --train_dir "src/data/amazon_data/${DS}/training" \
      --seed "${sd}" --out "${clean}" || { echo "FAIL clean ${DS} ${sd}"; return 1; }
  for st in ${STRATA}; do
    local pd
    if [ "${st}" = unpopular ]; then pd="src/data/amazon_data/${DS}_spam_seed${sd}_pct1_n1"
    else pd="src/data/amazon_data/${DS}_spam_tgt${st}_seed${sd}_pct1_n1"; fi
    local poison="${CK}/${DS}_poison_${st}_seed${sd}.pt"
    python -u -m scripts.sasrec_nau_v2 train --dataset "${DS}" --train_dir "${pd}/training" \
        --seed "${sd}" --out "${poison}" || { echo "FAIL poison ${DS} ${st} ${sd}"; continue; }
    python -u -m scripts.sasrec_nau_v2 cell --dataset "${DS}" --seed "${sd}" --stratum "${st}" \
        --clean_ckpt "${clean}" --poison_ckpt "${poison}" \
        --out_dir "${OUT}/${DS}/${st}_seed${sd}" || echo "FAIL cell ${DS} ${st} ${sd}"
  done
}

i=0
for sd in ${SEEDS}; do
  lane "${sd}" $((i % 2)) > "${OUT}/lanes/${DS}_seed${sd}_${SLURM_JOB_ID:-local}.log" 2>&1 &
  i=$((i + 1))
done
wait
echo "done $(date -Is)"
grep -h FAIL "${OUT}"/lanes/${DS}_seed*_"${SLURM_JOB_ID:-local}".log || echo "no failures"
