"""Member and non-member sets for membership inference in the unwanted-item scenario.

Members: the deleted interactions. For every requesting user (the users in
<sens_dir>/training_forget) and every training interaction with an unwanted-category item
that has at least one earlier item, a record holding the history up to and including that
item, so the item is the record's last element.
Non-members: test-split records whose last item is in the unwanted category; no model
trains on the test split.

Per-item features (sequence_data, text, and the 768-dimensional embedding) are cut to the
prefix consistently. Output: <out_dir>/members/ and <out_dir>/nonmembers/ (TFRecord shards).

    python -m scripts.make_mia_unwanted_sets --sens_dir src/data/amazon_data/beauty_sens_hairloss_1e-4_seed5 \
        --out_dir logs/diagnose/mia_unwanted/sets/beauty_s5
"""
import argparse
import glob
import json
import os

import tensorflow as tf


def _records(folder):
    for f in sorted(glob.glob(os.path.join(folder, "*.gz"))):
        for r in tf.data.TFRecordDataset(f, compression_type="GZIP"):
            ex = tf.train.Example()
            ex.ParseFromString(r.numpy())
            yield ex


def _prefix(ex, length):
    """Copy of `ex` with every per-item feature cut to the first `length` items."""
    out = tf.train.Example()
    out.CopyFrom(ex)
    f = out.features.feature
    n = len(ex.features.feature["sequence_data"].int64_list.value)
    seq = list(f["sequence_data"].int64_list.value)[:length]
    del f["sequence_data"].int64_list.value[:]
    f["sequence_data"].int64_list.value.extend(seq)
    if "text" in f:
        txt = list(f["text"].bytes_list.value)[:length]
        del f["text"].bytes_list.value[:]
        f["text"].bytes_list.value.extend(txt)
    if "embedding" in f:
        emb = list(f["embedding"].float_list.value)
        dim = len(emb) // n
        del f["embedding"].float_list.value[:]
        f["embedding"].float_list.value.extend(emb[: dim * length])
    return out


def _write(examples, folder):
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, "data_0.tfrecord.gz")
    with tf.io.TFRecordWriter(path, options=tf.io.TFRecordOptions(compression_type="GZIP")) as w:
        for ex in examples:
            w.write(ex.SerializeToString())
    return len(examples)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sens_dir", required=True)
    ap.add_argument("--out_dir", required=True)
    a = ap.parse_args()
    man = json.load(open(os.path.join(a.sens_dir, "forget_manifest.json")))
    targets = set(int(t) for t in man["target_items"])
    users = set(int(u) for u in man["spam_user_ids"])

    members = []
    for ex in _records(os.path.join(a.sens_dir, "training_forget")):
        if int(ex.features.feature["user_id"].int64_list.value[0]) not in users:
            continue
        seq = list(ex.features.feature["sequence_data"].int64_list.value)
        members += [_prefix(ex, j + 1) for j in range(1, len(seq)) if seq[j] in targets]
    nonmembers = [ex for ex in _records(os.path.join(a.sens_dir, "testing"))
                  if ex.features.feature["sequence_data"].int64_list.value[-1] in targets]
    nm = _write(members, os.path.join(a.out_dir, "members"))
    nn = _write(nonmembers, os.path.join(a.out_dir, "nonmembers"))
    print(f"members {nm} (deleted interactions with history; manifest lists {man['n_forget_interactions']}), "
          f"non-members {nn} -> {a.out_dir}")


if __name__ == "__main__":
    main()
