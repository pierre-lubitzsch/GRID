CKPT="logs/train/runs/2026-05-13/13-01-47/checkpoints/checkpoint_epoch=000_step=004400.ckpt"
DATASET="beauty"
SID=""
NEIGHBORHOOD_AWARE_SAMPLE_RATE=1.0

for bs in 1 2 4 8 16 32 64 128 256; do
  sbatch -p gpu --gpus-per-node=1 --cpus-per-task=8 \
    --job-name="unlearn_${DATASET}_bs${bs}" \
    run_tiger_unlearn_sequential.sh \
      "${CKPT}" "${DATASET}" "${SID}" \
      false \
      "${bs}" \
      "${NEIGHBORHOOD_AWARE_SAMPLE_RATE}" \
      unlearning.target_params=all
  done
