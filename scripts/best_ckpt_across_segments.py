"""Best checkpoint of a model trained in chained resume segments.

A resumed segment writes to a new run dir, and Lightning's ModelCheckpoint resets its best score
when the dirpath changes, so each segment's last.ckpt only knows that segment's best. This reads
every segment's last.ckpt and prints the best_model_path with the best monitored score overall.

Usage (from GRID/): python -m scripts.best_ckpt_across_segments <run_dir> [<run_dir> ...]
"""

import os
import sys

import torch

sys.path.insert(0, ".")
import src.utils  # noqa: F401,E402  (classes pickled in the checkpoints)


def main() -> int:
    best = None
    for d in sys.argv[1:]:
        f = os.path.join(d, "checkpoints", "last.ckpt")
        if not os.path.isfile(f):
            continue
        cbs = torch.load(f, map_location="cpu", weights_only=False)["callbacks"]
        for k, v in cbs.items():
            if not k.startswith("ModelCheckpoint") or v.get("best_model_score") is None:
                continue
            s, p = float(v["best_model_score"]), v["best_model_path"]
            sign = -1.0 if "'mode': 'min'" in k else 1.0
            if os.path.isfile(p) and (best is None or sign * s > best[0]):
                best = (sign * s, p)
    if best is None:
        return 1
    print(best[1])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
