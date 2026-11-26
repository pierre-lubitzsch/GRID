#!/usr/bin/env bash
#SBATCH --job-name=rr15_cf
#SBATCH --output=logs/rerank15/cf_%j.out
#SBATCH --error=logs/rerank15/cf_%j.out
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --partition=pgpu
#SBATCH --gres=gpu:2
#SBATCH --exclude=s-sc-pgpu01,s-sc-pgpu02
#SBATCH --time=04:00:00
#
# Collaborative (SASRec d64) item tables for the 15-cell re-ranking baseline (run_rerank15_launch.sh):
#   clean    : src/data/amazon_data/beauty, training split only (no spam users; asserted below)
#   poisoned : each cell's poisoned data dir, training split (includes that cell's spam users)
# Same settings as the testing tree's C8 (scripts/train_cf_embeddings.py --dim 64 --epochs 200
# --seed 42 --num-items 12101). Fitted fresh here; existing tables are skipped.
set -uo pipefail
cd "${SLURM_SUBMIT_DIR:-$PWD}"
IFS=',' read -r -a GPUS <<< "${CUDA_VISIBLE_DEVICES:-0}"
OUTD=embeddings/cf/rerank15
python3 - <<'EOF' || exit 1
# clean-only check: the clean training split holds no user id of any poisoned manifest's spam users
import glob, json, os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import tensorflow as tf
tf.config.set_visible_devices([], "GPU")
shards = sorted(glob.glob("src/data/amazon_data/beauty/training/*.tfrecord.gz"))
assert shards and not any("spam" in os.path.basename(s) for s in shards), shards
uids = set()
for r in tf.data.TFRecordDataset(shards, compression_type="GZIP"):
    ex = tf.train.Example(); ex.ParseFromString(r.numpy())
    uids.update(ex.features.feature["user_id"].int64_list.value)
spam = set()
for m in glob.glob("src/data/amazon_data/beauty_spam_*seed*_pct1_n1/forget_manifest.json"):
    spam |= set(json.load(open(m))["spam_user_ids"])
print(f"clean training split: {len(shards)} shards, {len(uids)} users, max id {max(uids)}; "
      f"spam ids over all manifests: {len(spam)}, overlap {len(uids & spam)}")
assert not (uids & spam), "clean CF data contains spam users"
EOF
JOBS=("src/data/amazon_data/beauty ${OUTD}/beauty_clean_cf64.pt")
for st in unpopular mid popular; do for sd in 2 3 5 7 11; do
  if [ "${st}" = unpopular ]; then d="src/data/amazon_data/beauty_spam_seed${sd}_pct1_n1"; else d="src/data/amazon_data/beauty_spam_tgt${st}_seed${sd}_pct1_n1"; fi
  JOBS+=("${d} ${OUTD}/beauty_poison_tgt${st}_seed${sd}_cf64.pt")
done; done
lane() {
  local l="$1" i=0 rc=0 j d o
  for j in "${JOBS[@]}"; do
    if [ $(( i % ${#GPUS[@]} )) -eq "${l}" ]; then
      read -r d o <<< "${j}"
      if [ -s "${o}" ]; then echo "[skip] ${o}"; else
        echo "[$(date -Is)] lane ${l}: ${d} -> ${o}"
        CUDA_VISIBLE_DEVICES="${GPUS[$l]}" python -u -m scripts.train_cf_embeddings --data-dir "${d}" --out "${o}" \
          --dim 64 --epochs 200 --seed 42 --num-items 12101 > "${o%.pt}.log" 2>&1 || { echo "FAILED ${o}"; rc=1; }
      fi
    fi
    i=$((i+1))
  done
  return "${rc}"
}
pids=(); for ((l=0; l<${#GPUS[@]}; l++)); do lane "${l}" & pids+=($!); done
rc=0; for p in "${pids[@]}"; do wait "${p}" || rc=1; done
ls -la "${OUTD}"; exit "${rc}"
