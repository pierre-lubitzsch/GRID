#!/usr/bin/env bash
# Whole-user deletion data (MUNKEY's setting): 2% of beauty users, seed 2.
set -euo pipefail
cd /sc-projects/sc-proj-ikim-erase/workspace/PKM_GRID/GRID
D=src/data/amazon_data
OUT=$D/beauty_userdel_2pct_seed2
EMB=embeddings/beauty_merged_predictions_tensor_latest.pt
PYTHONPATH=. python -m scripts.build_user_deletion_split --clean $D/beauty --out $OUT --frac 0.02 --seed 2
PYTHONPATH=. python -m src.data.unlearning.split_forget_retain --data_dir $OUT --forget_manifest $OUT/forget_manifest.json
# neighbours over the clean user rows, but deleted users are never a neighbour:
# used to train the retrain reference and to reset deleted users' tokens
PYTHONPATH=. python -m scripts.build_user_neighbors --history $D/beauty/training --universe $D/beauty/training \
  --emb $EMB --k 4 --exclude_manifest $OUT/forget_manifest.json --out embeddings/user_neighbors/beauty_userdel_2pct_seed2_excl.pt
ls $OUT $OUT"_retrain" $OUT"_evalforget" $OUT"_evalretain"
