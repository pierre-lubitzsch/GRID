"""Precompute user neighbours for the user-keyed exemplar memory.

Key of user u = mean of the L2-normalised FROZEN item embeddings over u's training
history (MUNKEY's frozen-encoder key, with the item text embedding as encoder).
Neighbours = top-K users by cosine similarity of those keys, self excluded.

The ROW ORDER is fixed by --universe (a training dir whose user ids define the
rows), so that files built from different datasets (the base data and the
retrain data, whose histories differ only for the requesting users) share row
indices and can be swapped at unlearning time. Users in the universe without a
history in --history keep their key from the universe data.

Output (torch.save): {"user_ids": LongTensor (U,), "neighbor_idx": (U, K) rows,
"neighbor_sim": (U, K)}, plus "keys": (U, D) float16 with --save_keys (the MUNKEY
baseline, model.munkey_memory.key_file, retrieves over the keys themselves).

Usage:
  python -m scripts.build_user_neighbors --history <data>/training \
      --universe <base>/training --emb embeddings/beauty_merged_predictions_tensor_latest.pt \
      --k 4 --out embeddings/user_neighbors/<name>.pt
"""

import argparse
import os
from collections import defaultdict
from typing import Dict, List

import tensorflow as tf
import torch
import torch.nn.functional as F


def read_histories(train_dir: str) -> Dict[int, List[int]]:
    hist: Dict[int, List[int]] = defaultdict(list)
    for f in sorted(os.listdir(train_dir)):
        if not f.endswith(".tfrecord.gz"):
            continue
        for rec in tf.data.TFRecordDataset(os.path.join(train_dir, f), compression_type="GZIP"):
            feat = tf.train.Example.FromString(rec.numpy()).features.feature
            uid = int(feat["user_id"].int64_list.value[0])
            hist[uid].extend(int(i) for i in feat["sequence_data"].int64_list.value)
    return hist


def user_keys(hist: Dict[int, List[int]], uids: List[int], emb: torch.Tensor) -> torch.Tensor:
    e = F.normalize(emb.float(), dim=-1)
    keys = torch.zeros(len(uids), e.size(1))
    for r, u in enumerate(uids):
        items = [i for i in set(hist.get(u, [])) if 0 <= i < e.size(0)]
        if items:
            keys[r] = e[items].mean(0)
    return keys


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history", required=True, help="training dir whose histories define the keys")
    p.add_argument("--universe", required=True, help="training dir whose user ids define the rows")
    p.add_argument("--emb", required=True)
    p.add_argument("--k", type=int, default=4)
    p.add_argument("--out", required=True)
    p.add_argument("--chunk", type=int, default=2048)
    p.add_argument("--exclude_manifest", default=None,
                   help="forget manifest; its spam_user_ids are removed from the neighbour "
                        "CANDIDATES (rows stay), so deleted users are never anyone's neighbour")
    p.add_argument("--save_keys", action="store_true",
                   help="also store the (U, D) user keys (model.munkey_memory.key_file)")
    a = p.parse_args()

    emb = torch.load(a.emb, map_location="cpu", weights_only=False)
    uni = read_histories(a.universe)
    uids = sorted(uni)
    hist = uni if os.path.abspath(a.history) == os.path.abspath(a.universe) else read_histories(a.history)
    merged = {u: (hist[u] if u in hist else uni[u]) for u in uids}
    n_changed = sum(1 for u in uids if u in hist and set(hist[u]) != set(uni[u]))
    keys = F.normalize(user_keys(merged, uids, emb), dim=-1)

    excl = torch.zeros(len(uids), dtype=torch.bool)
    if a.exclude_manifest:
        import json
        bad = {int(u) for u in json.load(open(a.exclude_manifest))["spam_user_ids"]}
        excl = torch.tensor([u in bad for u in uids])
        print(f"excluding {int(excl.sum())} users from the neighbour candidates")
    idx_out, sim_out = [], []
    for s in range(0, len(uids), a.chunk):
        sims = keys[s : s + a.chunk] @ keys.T
        sims[:, excl] = -float("inf")
        rows = torch.arange(s, min(s + a.chunk, len(uids)))
        sims[torch.arange(len(rows)), rows] = -float("inf")
        sv, si = sims.topk(a.k, dim=-1)
        idx_out.append(si)
        sim_out.append(sv)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    out = {"user_ids": torch.tensor(uids, dtype=torch.long), "neighbor_idx": torch.cat(idx_out),
           "neighbor_sim": torch.cat(sim_out)}
    if a.save_keys:
        out["keys"] = keys.half()
    torch.save(out, a.out)
    print(f"users={len(uids)} max_uid={max(uids)} k={a.k} users_with_changed_history={n_changed} -> {a.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
