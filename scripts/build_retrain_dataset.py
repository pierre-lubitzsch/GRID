"""Build the RETRAIN reference dataset for sensitive-category unlearning.

The retrain reference is the model you would get if the deletion request had been
honoured before training: same data, minus exactly the interactions in the forget
set. Training on it from scratch gives the gold standard that approximate
unlearning is measured against.

WHY THIS SCRIPT EXISTS. For the SPAM scenario the `clean` model already is the
retrain reference, because every attack only *appends* fake users, so removing
them recovers the clean dataset byte for byte. That does not hold for
SENSITIVE-category deletion: the forget interactions sit inside REAL users'
sessions, so `clean` still contains them and is NOT a retrain reference. This
builds the missing one.

WHAT IS REMOVED. Only the sensitive items belonging to the users named in the
manifest (`spam_user_ids`, the key reused by the sensitive manifest builder).
Other users keep their sensitive interactions: ERASE's request is per user, not
a global ban, so removing the category everywhere would answer a different
question.

WHAT IS NOT TOUCHED. `evaluation/`, `testing/` and `items/` are copied verbatim.
The evaluation protocol must stay fixed across the retrained, unlearned and
poisoned models, or the metrics are not comparable.

    python -m scripts.build_retrain_dataset \
        --data_dir src/data/amazon_data/beauty \
        --manifest src/data/amazon_data/beauty/forget_manifest_hairloss_1e-4_seed2.json \
        --out_dir  src/data/amazon_data/beauty_retrain_hairloss_1e-4_seed2
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
from typing import List, Set

import tensorflow as tf

USER_ID_FIELD = "user_id"
SEQUENCE_FIELD = "sequence_data"


def _shards(d: str) -> List[str]:
    return [
        os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".tfrecord.gz")
    ]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data_dir", required=True, help="the CLEAN dataset")
    p.add_argument("--manifest", required=True, help="sensitive forget manifest")
    p.add_argument("--out_dir", required=True)
    p.add_argument("--min_seq_len", type=int, default=2,
                   help="drop training rows shorter than this after removal")
    p.add_argument("--rows_per_shard", type=int, default=4096)
    args = p.parse_args()

    man = json.load(open(args.manifest))
    targets: Set[int] = {int(i) for i in man["target_items"]}
    users: Set[int] = {int(u) for u in man.get("spam_user_ids", [])}
    if not targets or not users:
        raise SystemExit("manifest must carry target_items and spam_user_ids")
    print(f"[manifest] {len(targets)} sensitive items, {len(users)} forget users, "
          f"{man.get('n_forget_interactions')} interactions to remove")

    os.makedirs(args.out_dir, exist_ok=True)
    # Splits that define the evaluation protocol are copied verbatim.
    for sub in ("evaluation", "testing", "items"):
        src = os.path.join(args.data_dir, sub)
        if os.path.isdir(src):
            dst = os.path.join(args.out_dir, sub)
            if os.path.exists(dst):
                shutil.rmtree(dst)
            shutil.copytree(src, dst)
            print(f"[copy] {sub}")

    src_train = os.path.join(args.data_dir, "training")
    dst_train = os.path.join(args.out_dir, "training")
    if os.path.exists(dst_train):
        shutil.rmtree(dst_train)
    os.makedirs(dst_train, exist_ok=True)

    n_rows = n_touched = n_removed = n_dropped = 0
    shard_i = written = 0
    writer = None
    opts = tf.io.TFRecordOptions(compression_type="GZIP")

    for shard in _shards(src_train):
        for rec in tf.data.TFRecordDataset(shard, compression_type="GZIP"):
            ex = tf.train.Example.FromString(rec.numpy())
            feat = ex.features.feature
            uid = int(feat[USER_ID_FIELD].int64_list.value[0])
            seq = list(feat[SEQUENCE_FIELD].int64_list.value)
            n_rows += 1
            if uid in users:
                kept = [i for i in seq if int(i) not in targets]
                if len(kept) != len(seq):
                    n_touched += 1
                    n_removed += len(seq) - len(kept)
                    if len(kept) < args.min_seq_len:
                        n_dropped += 1
                        continue
                    del feat[SEQUENCE_FIELD].int64_list.value[:]
                    feat[SEQUENCE_FIELD].int64_list.value.extend(int(x) for x in kept)
            if writer is None or written % args.rows_per_shard == 0:
                if writer is not None:
                    writer.close()
                writer = tf.io.TFRecordWriter(
                    os.path.join(dst_train, f"data_{shard_i}.tfrecord.gz"), options=opts
                )
                shard_i += 1
            writer.write(ex.SerializeToString())
            written += 1
    if writer is not None:
        writer.close()

    meta = {
        "built_from": os.path.abspath(args.data_dir),
        "manifest": os.path.abspath(args.manifest),
        "category": man.get("category"),
        "forget_ratio": man.get("forget_ratio"),
        "n_target_items": len(targets),
        "n_forget_users": len(users),
        "rows_in": n_rows,
        "rows_out": written,
        "rows_modified": n_touched,
        "interactions_removed": n_removed,
        "rows_dropped_too_short": n_dropped,
        "note": "evaluation/testing/items copied verbatim; only training/ differs",
    }
    with open(os.path.join(args.out_dir, "retrain_meta.json"), "w") as fh:
        json.dump(meta, fh, indent=2)
    print(f"[training] {n_rows} rows in -> {written} out; {n_touched} modified, "
          f"{n_removed} interactions removed, {n_dropped} dropped as too short")
    print(f"[write] {args.out_dir}")
    if n_removed != man.get("n_forget_interactions"):
        print(f"[warn] removed {n_removed} but the manifest lists "
              f"{man.get('n_forget_interactions')}; the manifest counts occurrences "
              "in the full sessions while training/ holds the leave-one-out prefix "
              "seq[:-2], so a sensitive item in the last two positions is not present "
              "here. Expected, but check it is not larger than that.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
