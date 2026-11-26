"""Random-SID control: permute items across the existing RQ-KMeans SID tree.

Each item receives the SID tuple of another item, drawn as a uniformly random
bijection (fixed seed). The SET of tuples is unchanged, so every prefix bucket
(first token, first two tokens, first three tokens, full tuple) keeps exactly its
size, the dedup digit stays valid, and every code stays < 256. Only WHICH items
share a prefix changes: semantic grouping is destroyed, code statistics are not.

    python scripts/make_random_sids.py                  # writes embeddings/beauty_randsid/
    python scripts/make_random_sids.py --check-only     # re-verify an existing file
    python scripts/make_random_sids.py --dataset sports # embeddings/sports_randsid/ (also toys)

Light (a 4 x N tensor, N <= 18357); safe on the login node.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# content embeddings the RQ-KMeans SIDs (and NAU's k=8 neighbors) were built from
EMBEDDINGS = {"beauty": "embeddings/beauty_merged_predictions_tensor_latest.pt",
              "sports": "logs/inference/runs/embeddings/sports_10321537/pickle/merged_predictions_tensor.pt",
              "toys": "logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"}


def load_embeddings(path: str) -> torch.Tensor:
    """N x D content embeddings; the Sports/Toys files are {embeddings, item_ids} dicts."""
    obj = torch.load(path, map_location="cpu")
    if isinstance(obj, dict):
        ids = obj["item_ids"]
        assert torch.equal(ids, torch.arange(len(ids))), "item_ids are not 0..N-1"
        obj = obj["embeddings"]
    return obj.float()


def bucket_sizes(codes: np.ndarray, depth: int) -> list[int]:
    """Sorted multiset of prefix-bucket sizes at the given depth (codes: N x L)."""
    return sorted(collections.Counter(map(tuple, codes[:, :depth].tolist())).values())


def within_bucket_cosine(codes: np.ndarray, emb: np.ndarray) -> float:
    """Mean pairwise content-embedding cosine between distinct items sharing a first token."""
    total, pairs = 0.0, 0
    for tok in np.unique(codes[:, 0]):
        idx = np.where(codes[:, 0] == tok)[0]
        n = len(idx)
        if n < 2:
            continue
        s = emb[idx].sum(0)
        total += float(s @ s) - n          # sum over ordered pairs i != j of cos(i, j)
        pairs += n * (n - 1)
    return total / pairs


def check(orig: torch.Tensor, rand: torch.Tensor, perm: np.ndarray, emb_path: str) -> dict:
    o, r = orig.numpy().T, rand.numpy().T            # N x L
    n, depth = o.shape
    assert rand.shape == orig.shape and rand.dtype == orig.dtype, (rand.shape, rand.dtype)
    assert int(rand.min()) >= 0 and int(rand.max()) < 256, (int(rand.min()), int(rand.max()))
    assert sorted(perm.tolist()) == list(range(n)), "perm is not a permutation"
    assert np.array_equal(r, o[perm]), "rand != orig[perm]"
    assert len(set(map(tuple, r.tolist()))) == n, "random SIDs are not unique (not a bijection)"
    assert set(map(tuple, r.tolist())) == set(map(tuple, o.tolist())), "tuple set changed"
    for d in range(1, depth + 1):
        assert bucket_sizes(o, d) == bucket_sizes(r, d), f"bucket sizes differ at depth {d}"
    emb = load_embeddings(emb_path)
    assert emb.shape[0] == n, (emb.shape, n)
    emb = torch.nn.functional.normalize(emb, dim=1).numpy()
    same_first = float((o[:, 0] == r[:, 0]).mean())
    # expected share of items whose random first token equals their own under a random bijection
    sizes = np.array(bucket_sizes(o, 1), dtype=float)
    return {
        "n_items": n, "levels": depth, "max_code": int(rand.max()), "min_code": int(rand.min()),
        "bijection": True, "tuple_set_identical": True,
        "bucket_counts_by_depth": {d: len(bucket_sizes(o, d)) for d in range(1, depth + 1)},
        "largest_bucket_by_depth": {d: max(bucket_sizes(o, d)) for d in range(1, depth + 1)},
        "bucket_size_multisets_identical": True,
        "items_keeping_first_token": same_first,
        "expected_keep_first_token_under_random_bijection": float((sizes ** 2).sum() / n ** 2),
        "items_keeping_full_sid": float((perm == np.arange(n)).mean()),
        "within_first_token_cosine_rqkmeans": within_bucket_cosine(o, emb),
        "within_first_token_cosine_random": within_bucket_cosine(r, emb),
        "mean_cosine_all_pairs": float(((emb.sum(0) @ emb.sum(0)) - n) / (n * (n - 1))),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="beauty", choices=sorted(EMBEDDINGS))
    ap.add_argument("--src", default="", help="default embeddings/<dataset>/merged_predictions_tensor.pt")
    ap.add_argument("--out-dir", default="", help="default embeddings/<dataset>_randsid")
    ap.add_argument("--embeddings", default="",
                    help="content embeddings (only for the semantic-coherence check); default per dataset")
    ap.add_argument("--seed", type=int, default=20261107)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    a.src = a.src or f"embeddings/{a.dataset}/merged_predictions_tensor.pt"
    a.out_dir = a.out_dir or f"embeddings/{a.dataset}_randsid"
    a.embeddings = a.embeddings or EMBEDDINGS[a.dataset]
    os.chdir(ROOT)
    orig = torch.load(a.src, map_location="cpu")
    out = os.path.join(a.out_dir, "merged_predictions_tensor.pt")
    perm_path = os.path.join(a.out_dir, "permutation.npy")
    if a.check_only:
        perm, rand = np.load(perm_path), torch.load(out, map_location="cpu")
    else:
        if os.path.exists(out):
            raise SystemExit(f"{out} exists; refusing to overwrite (use --check-only)")
        perm = np.random.default_rng(a.seed).permutation(orig.shape[1])
        # item i takes the SID that item perm[i] has under RQ-KMeans
        rand = orig[:, torch.from_numpy(perm)].clone().contiguous()
        os.makedirs(a.out_dir, exist_ok=True)
        torch.save(rand, out)
        np.save(perm_path, perm)
    stats = check(orig, rand, perm, a.embeddings)
    stats.update({"dataset": a.dataset, "embeddings": a.embeddings, "source": a.src, "seed": a.seed, "rule": "rand[:, i] = orig[:, perm[i]]",
                  "permutation": perm_path})
    with open(os.path.join(a.out_dir, "checks.json"), "w") as f:
        json.dump(stats, f, indent=2)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
