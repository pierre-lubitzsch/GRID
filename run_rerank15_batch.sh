#!/usr/bin/env bash
#SBATCH --job-name=rr15_eval
#SBATCH --output=logs/rerank15/eval_%j.out
#SBATCH --error=logs/rerank15/eval_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=12:00:00
#
# Beam-50 test evaluations with/without decode-time re-ranking (scripts/eval_ckpt_rerank.py), one
# row per eval of logs/rerank15/plan_*.tsv:  out_dir  ckpt  poison_data_dir  seed  latent(-)  alpha
# 2 lanes (one per GPU); a row whose metrics.csv exists is skipped. Checkpoints are resolved with
# readlink -f at run time (archive symlinks).
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
PLAN="${1:?plan tsv}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
lane() {
  local l="$1" i=0 rc=0
  while IFS=$'\t' read -r out ck pdir sd lat alpha; do
    if [ $(( i % ${#GPUS[@]} )) -ne "${l}" ]; then i=$((i+1)); continue; fi
    i=$((i+1))
    [ -s "${out}/csv/version_0/metrics.csv" ] && { echo "[skip] ${out}"; continue; }
    ckr="$(readlink -f "${ck}")"; [ -f "${ckr}" ] || { echo "MISSING ckpt ${ck}"; rc=1; continue; }
    echo "[$(date -Is)] lane ${l}: ${out}"
    mkdir -p "${out}"
    if [ "${lat}" = "-" ]; then RL=""; else RL="${lat}"; fi
    CUDA_VISIBLE_DEVICES="${GPUS[$l]}" RERANK_LATENT_PATH="${RL}" RERANK_ALPHA="${alpha}" \
    python -u -m scripts.eval_ckpt_rerank experiment=tiger_train_flat \
      data_dir=src/data/amazon_data/beauty \
      "semantic_id_path='embeddings/beauty/merged_predictions_tensor.pt'" \
      "ckpt_path='${ckr}'" "spam_forget_manifest='${pdir}/forget_manifest.json'" \
      ++model.top_k_for_generation=50 num_hierarchies=4 seed="${sd}" train=False test=True \
      trainer.devices=1 trainer.strategy=auto trainer.sync_batchnorm=false trainer.num_nodes=1 \
      trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
      hydra.run.dir="${out}" > "${out}.log" 2>&1 \
      && [ -s "${out}/csv/version_0/metrics.csv" ] || { echo "FAILED ${out} (see ${out}.log)"; rc=1; }
  done < "${PLAN}"
  return "${rc}"
}
pids=(); for ((l=0; l<${#GPUS[@]}; l++)); do lane "${l}" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
echo "[$(date -Is)] finished ${PLAN} rc=${rc}"; exit "${rc}"
