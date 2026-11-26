#!/usr/bin/env bash
#SBATCH --job-name=pathway_eval
#SBATCH --output=logs/%j.out
#SBATCH --error=logs/%j.out
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --partition=pgpu
#SBATCH --time=02:00:00
# MUNKEY pathway sensitivity for a user-memory TIGER: test metrics with the
# history nulled (token only), the token nulled (history only) and both kept.
#   P_s = |N(no_token) - N(no_hist)| / N(both)   (MUNKEY: |A_img - A_tok| / A_both)
# Usage: sbatch run_pathway_eval.sh <train_run_dir> <neighbor_file> <p_hist> [data_dir] [manifest]
set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
RUN="${1:?train run dir}"; NB="${2:?neighbor file}"; PH="${3:-0}"
DATA="${4:-src/data/amazon_data/beauty}"
MAN="${5:-src/data/amazon_data/beauty_sens_hairloss_1e-3_seed2/forget_manifest.json}"
CKPT="$RUN/checkpoints/last.ckpt"
for mode in none no_hist no_token; do
  OUT="logs/pathway_eval/$(basename "$RUN")/$mode"
  echo "[$(date -Is)] $mode -> $OUT"
  python -u -m scripts.eval_ckpt_on_test experiment=tiger_train_flat data_dir="$DATA" \
    "semantic_id_path='embeddings/beauty_rqvae/merged_predictions_tensor.pt'" \
    "ckpt_path='$CKPT'" "spam_forget_manifest=$MAN" num_hierarchies=4 seed=2 \
    train=False test=True trainer.devices=1 trainer.strategy=auto trainer.num_nodes=1 \
    trainer.deterministic=true callbacks.model_checkpoint=null callbacks.early_stopping=null \
    "model.user_memory={neighbor_file:$NB,p_hist:$PH,eval_pathway:$mode}" \
    hydra.run.dir="$OUT"
done
echo "[$(date -Is)] done"
