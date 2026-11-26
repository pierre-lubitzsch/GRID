"""Whole-user deletion split (MUNKEY's setting, users as instances).

Samples a fraction of the CLEAN dataset's training users (seeded) and writes:
  <out>/                 training/items/evaluation/testing symlinked to the clean
                         data + forget_manifest.json (deletion_spec=session,
                         spam_user_ids = the deleted users, no target items);
                         run src.data.unlearning.split_forget_retain on it next.
  <out>_retrain/         training WITHOUT the deleted users (retrain reference);
                         items/evaluation/testing symlinked.
  <out>_evalforget/      evaluation/testing rows of the deleted users only.
  <out>_evalretain/      evaluation/testing rows of all other users.
The scoped eval dirs symlink training/items, so they can be passed as data_dir.
"""

import argparse
import json
import os
import random

import tensorflow as tf


def shards(d):
    return [os.path.join(d, f) for f in sorted(os.listdir(d)) if f.endswith(".tfrecord.gz")]


def users_of(d):
    out = set()
    for sh in shards(d):
        for rec in tf.data.TFRecordDataset(sh, compression_type="GZIP"):
            out.add(int(tf.train.Example.FromString(rec.numpy()).features.feature["user_id"].int64_list.value[0]))
    return out


def filter_dir(src, dst, keep, rows_per_shard=4096):
    os.makedirs(dst, exist_ok=True)
    opts = tf.io.TFRecordOptions(compression_type="GZIP")
    w, n, k, shard_i = None, 0, 0, 0
    for sh in shards(src):
        for rec in tf.data.TFRecordDataset(sh, compression_type="GZIP"):
            uid = int(tf.train.Example.FromString(rec.numpy()).features.feature["user_id"].int64_list.value[0])
            n += 1
            if not keep(uid):
                continue
            if w is None or k % rows_per_shard == 0:
                if w is not None:
                    w.close()
                w = tf.io.TFRecordWriter(os.path.join(dst, f"data_{shard_i}.tfrecord.gz"), options=opts)
                shard_i += 1
            w.write(rec.numpy())
            k += 1
    if w is not None:
        w.close()
    return n, k


def link(src, dst):
    if not os.path.exists(dst):
        os.symlink(os.path.realpath(src), dst)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--clean", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--frac", type=float, default=0.02)
    p.add_argument("--seed", type=int, default=2)
    a = p.parse_args()

    users = sorted(users_of(os.path.join(a.clean, "training")))
    rng = random.Random(a.seed)
    forget = sorted(rng.sample(users, max(1, round(a.frac * len(users)))))
    fs = set(forget)
    man = {"scenario": "user_deletion", "deletion_spec": "session", "spam_user_ids": forget,
           "n_spam_users": len(forget), "target_items": [], "n_target_items": 0,
           "forget_fraction_users": a.frac, "n_users_total": len(users), "seed": a.seed,
           "source_dataset": os.path.realpath(a.clean)}

    os.makedirs(a.out, exist_ok=True)
    for sub in ("training", "items", "evaluation", "testing"):
        link(os.path.join(a.clean, sub), os.path.join(a.out, sub))
    json.dump(man, open(os.path.join(a.out, "forget_manifest.json"), "w"), indent=1)

    ret = a.out + "_retrain"
    os.makedirs(ret, exist_ok=True)
    n, k = filter_dir(os.path.join(a.clean, "training"), os.path.join(ret, "training"), lambda u: u not in fs)
    for sub in ("items", "evaluation", "testing"):
        link(os.path.join(a.clean, sub), os.path.join(ret, sub))
    json.dump({**man, "rows_in": n, "rows_kept": k}, open(os.path.join(ret, "retrain_meta.json"), "w"), indent=1)

    for name, keep in (("_evalforget", lambda u: u in fs), ("_evalretain", lambda u: u not in fs)):
        d = a.out + name
        os.makedirs(d, exist_ok=True)
        for sub in ("evaluation", "testing"):
            filter_dir(os.path.join(a.clean, sub), os.path.join(d, sub), keep)
        for sub in ("training", "items"):
            link(os.path.join(a.clean, sub), os.path.join(d, sub))
    print(f"users={len(users)} forget={len(forget)} retrain rows kept {k}/{n} -> {a.out}*")


if __name__ == "__main__":
    main()
