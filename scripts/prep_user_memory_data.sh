#!/usr/bin/env bash
set -euo pipefail
cd /sc-projects/sc-proj-ikim-erase/workspace/PKM_GRID/GRID
D=src/data/amazon_data
EMB=embeddings/beauty_merged_predictions_tensor_latest.pt
RET=$D/beauty_retrain_hairloss_1e-3_seed2
if [ ! -d "$RET/training" ]; then
  PYTHONPATH=. python -m scripts.build_retrain_dataset --data_dir $D/beauty \
    --manifest $D/beauty_sens_hairloss_1e-3_seed2/forget_manifest.json --out_dir $RET
fi
ls $RET
PYTHONPATH=. python -m scripts.build_user_neighbors --history $D/beauty/training --universe $D/beauty/training \
  --emb $EMB --k 4 --out embeddings/user_neighbors/beauty_clean.pt
PYTHONPATH=. python -m scripts.build_user_neighbors --history $RET/training --universe $D/beauty/training \
  --emb $EMB --k 4 --out embeddings/user_neighbors/beauty_retrain_hairloss_1e-3_seed2.pt
PYTHONPATH=. python -m scripts.build_user_neighbors --history $D/beauty_spam_tgtmid_seed2_pct1_n1/training \
  --universe $D/beauty_spam_tgtmid_seed2_pct1_n1/training --emb $EMB --k 4 \
  --out embeddings/user_neighbors/beauty_spam_tgtmid_seed2_pct1_n1.pt
