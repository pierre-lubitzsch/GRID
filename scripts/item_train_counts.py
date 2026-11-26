"""Item training-frequency counts (occurrences in the training sequences), indexed by item id.

Used by the head-only item memory (model.item_memory.head_frac / counts_path) and the
cold-start popularity-bucket analysis.
Usage (from GRID/): python -m scripts.item_train_counts --data src/data/amazon_data/beauty \
    --num_items 12101 --out embeddings/item_counts/beauty.pt
"""
import argparse
import os

import torch

from scripts.build_user_neighbors import read_histories


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True)
    p.add_argument("--num_items", type=int, required=True)
    p.add_argument("--out", required=True)
    a = p.parse_args()
    c = torch.zeros(a.num_items, dtype=torch.long)
    for seq in read_histories(os.path.join(a.data, "training")).values():
        t = torch.tensor([i for i in seq if 0 <= i < a.num_items], dtype=torch.long)
        c.index_add_(0, t, torch.ones_like(t))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    torch.save(c, a.out)
    nz = int((c > 0).sum())
    print(f"{a.out}: {a.num_items} items, {nz} seen in training, total {int(c.sum())}, max {int(c.max())}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
