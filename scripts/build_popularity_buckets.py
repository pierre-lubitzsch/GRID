"""Split a dataset's TEST set into 3 equal-size buckets by the target item's training count
(tail / mid / head), for the cold-start analysis of the item memory.

Writes <data>_popb{1,2,3}/testing/*.tfrecord.gz (records copied unchanged; the target is the last
item of sequence_data) and symlinks training / evaluation / items to the original. Bucket 1 = the
rarest targets. Prints the count range per bucket.
Usage (from GRID/): python -m scripts.build_popularity_buckets --data src/data/amazon_data/beauty \
    --counts embeddings/item_counts/beauty.pt
"""
import argparse
import os

import tensorflow as tf
import torch


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--counts", required=True)
    a = p.parse_args()
    counts = torch.load(a.counts, weights_only=False)
    tdir = os.path.join(a.data, "testing")
    recs = []
    for f in sorted(os.listdir(tdir)):
        if f.endswith(".tfrecord.gz"):
            for r in tf.data.TFRecordDataset(os.path.join(tdir, f), compression_type="GZIP"):
                b = r.numpy()
                tgt = int(tf.train.Example.FromString(b).features.feature["sequence_data"].int64_list.value[-1])
                recs.append((int(counts[tgt]), b))
    recs.sort(key=lambda x: x[0])
    n = len(recs)
    for k in range(3):
        part = recs[k * n // 3:(k + 1) * n // 3]
        out = f"{a.data.rstrip('/')}_popb{k + 1}"
        os.makedirs(os.path.join(out, "testing"), exist_ok=True)
        for sub in ("training", "evaluation", "items"):
            link = os.path.join(out, sub)
            if not os.path.lexists(link):
                os.symlink(os.path.abspath(os.path.join(a.data, sub)), link)
        with tf.io.TFRecordWriter(os.path.join(out, "testing", "partition_0.tfrecord.gz"),
                                  options="GZIP") as w:
            for _, b in part:
                w.write(b)
        print(f"{out}: {len(part)} test users, target train count {part[0][0]}..{part[-1][0]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
