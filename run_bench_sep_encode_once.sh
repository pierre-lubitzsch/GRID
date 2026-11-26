#!/usr/bin/env bash
#SBATCH --job-name=bench_sep
#SBATCH --output=logs/bench_sep/%j.out
#SBATCH --error=logs/bench_sep/%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=01:00:00
# Read-only timing benchmark of generative L_sep: current per-item scoring vs encode-once (scripts/bench_sep_encode_once.py).
# Beauty seed 2 (106 negatives) on GPU 0, Sports seed 2 (202 negatives) on GPU 1. Writes only logs/bench_sep/.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
nvidia-smi --query-gpu=name --format=csv,noheader
bench() {  # gpu ds cat ckpt sid info
  CUDA_VISIBLE_DEVICES=$1 python -u -m scripts.bench_sep_encode_once experiment=tiger_unlearn_scif_flat \
    data_dir="src/data/amazon_data/$2_retrain_$3_1e-4_seed2" "semantic_id_path='$5'" "ckpt_path='$4'" \
    num_hierarchies=4 seed=2 +unlearning.bench_neg_json="$6" +unlearning.bench_reps=3 \
    hydra.run.dir="logs/bench_sep/${SLURM_JOB_ID:-local}_$2" > "logs/bench_sep/${SLURM_JOB_ID:-local}_$2.txt" 2>&1
}
B=$(ls -d logs/unlearn/runs/job*_bs1_sens_beauty_hairlossrm_unified_hp2_sepgen_lf0.1_ls0.01_lnneg10.0_sepck_rmsplit_canonck | tail -1)
S=$(ls -d logs/unlearn/runs/job*_bs1_sens_sports_gunsrm_unified_hp2_sepgen_lf0.1_ls0.01_lnneg10.0_sepck_rmsplit_canonck | tail -1)
bench 0 beauty hairloss "logs/train/runs/2026-05-29/14-06-14_job9096928_beauty_clean/checkpoints/checkpoint_epoch=000_step=004600.ckpt" embeddings/beauty/merged_predictions_tensor.pt "$B/scif_info.json" &
bench 1 sports guns "logs/train/runs/2026-08-21/02-44-32_job10464996_sports_clean/checkpoints/last.ckpt" embeddings/sports/merged_predictions_tensor.pt "$S/scif_info.json" &
wait
grep -h "GPU\|\[eval\]\|\[train\]\|Error" logs/bench_sep/${SLURM_JOB_ID:-local}_*.txt
