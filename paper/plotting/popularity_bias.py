"""Does unlearning shift recommendations toward popular items?

Reads the top-10 recommendations written by run_popbias_cell.sh
(logs/diagnose/popbias/beauty_<stratum>_s<seed>/<model>/pickle/merged_predictions.pkl) for the
clean (retrained) model, the poisoned model the runs start from, NAU and Forget+Repair, and
compares the popularity of what they recommend. Popularity = interaction count in the clean
training split; head / tail = the top / bottom 20% of items by that count (the bins of
src/data/poisoning/bandwagon.py). The cell's spam target is excluded from the popularity metrics
and reported separately, so a shift is not just the target being removed.

Per model: mean popularity (ARP, average count of the recommended items), mean log10(1+count),
head and tail share of the recommendations, catalog coverage (distinct items recommended / items),
Gini of how often each item is recommended, invalid SIDs, target share. Cells are averaged; the
paired difference to the clean model is reported per cell and averaged.

    PYTHONPATH=. python paper/plotting/popularity_bias.py [--out logs/diagnose/popbias/summary.json]

--root logs/diagnose/popbias_ledger (run_popbias_cell.sh with LEDGER=...): read the cells scored with
Table 1's canonical NAU / Forget+Repair checkpoints; the default --out is then <root>/summary.json.
Also prints the paired differences against the poisoned model and NAU vs Forget+Repair (mean over
cells, cells where the value goes down).
"""
import argparse
import glob
import json
import math
import os
import pickle
import statistics as S
from collections import Counter

import numpy as np
import torch

MODELS = ("clean", "poisoned", "nau", "forget_repair")
DATA = "src/data/amazon_data/beauty"
SID = "embeddings/beauty/merged_predictions_tensor.pt"
COUNTS_CACHE = "logs/diagnose/popbias/beauty_item_counts.json"


def item_counts():
    if os.path.exists(COUNTS_CACHE):
        return {int(k): v for k, v in json.load(open(COUNTS_CACHE)).items()}
    from src.data.poisoning.bandwagon import _scan_clean_training
    c = _scan_clean_training(f"{DATA}/training")[2]
    c = {int(k): int(v) for k, v in c.items()}
    os.makedirs(os.path.dirname(COUNTS_CACHE), exist_ok=True)
    json.dump(c, open(COUNTS_CACHE, "w"))
    return c


def gini(x):
    x = np.sort(np.asarray(x, dtype=float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return float("nan")
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load_recs(path, code2item):
    rows = pickle.load(open(path, "rb"))
    recs, invalid = [], 0
    for r in rows:
        sids = r["semantic_ids"]
        sids = sids.tolist() if hasattr(sids, "tolist") else sids
        items = []
        for s in sids:
            it = code2item.get(tuple(int(v) for v in s))
            if it is None:
                invalid += 1
            else:
                items.append(it)
        recs.append(items)
    return recs, invalid


def metrics(recs, invalid, counts, n_items, head, tail, target):
    flat = [i for items in recs for i in items]
    tgt = sum(i == target for i in flat)
    pop = [i for i in flat if i != target]
    freq = Counter(pop)
    return {
        "n_users": len(recs),
        "ARP": S.mean(counts.get(i, 0) for i in pop),
        "log_pop": S.mean(math.log10(1 + counts.get(i, 0)) for i in pop),
        "head_share": sum(i in head for i in pop) / len(pop),
        "tail_share": sum(i in tail for i in pop) / len(pop),
        "coverage": len(freq) / n_items,
        "gini": gini([freq.get(i, 0) for i in range(n_items)]),
        "invalid_share": invalid / max(1, invalid + len(flat)),
        "target_share": tgt / max(1, len(flat)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="logs/diagnose/popbias")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = os.path.join(a.root, "summary.json")
    counts = item_counts()
    sid = torch.load(SID, map_location="cpu")
    sid = sid if sid.shape[0] != 4 else sid.T          # (items, 4)
    code2item = {tuple(int(v) for v in row): i for i, row in enumerate(sid.tolist())}
    n_items = len(code2item)
    by_pop = sorted(range(n_items), key=lambda i: -counts.get(i, 0))
    head, tail = set(by_pop[: int(0.2 * n_items)]), set(by_pop[-int(0.2 * n_items):])
    res = {}
    for cell in sorted(glob.glob(os.path.join(a.root, "beauty_*_s*"))):
        name = os.path.basename(cell)
        st, sd = name.split("_")[1], name.split("_s")[-1]
        tok = "" if st == "unpopular" else f"_tgt{st}"
        man = json.load(open(f"{DATA}_spam{tok}_seed{sd}_pct1_n1/forget_manifest.json"))
        target = int(man["target_items"][0])
        row = {}
        for m in MODELS:
            p = f"{cell}/{m}/pickle/merged_predictions.pkl"
            if os.path.exists(p):
                row[m] = metrics(*load_recs(p, code2item), counts, n_items, head, tail, target)
        if row:
            res[name] = row
    keys = ("ARP", "log_pop", "head_share", "tail_share", "coverage", "gini", "invalid_share", "target_share")
    print(f"cells: {list(res)}  (items {n_items}, head/tail = top/bottom 20%)")
    print(f"{'model':14s} " + " ".join(f"{k:>13s}" for k in keys))
    for m in MODELS:
        v = [r[m] for r in res.values() if m in r]
        if v:
            print(f"{m:14s} " + " ".join(f"{S.mean(x[k] for x in v):13.4f}" for k in keys) + f"   cells {len(v)}")
    print("paired difference to the clean model (mean over cells):")
    for m in MODELS[1:]:
        d = [{k: r[m][k] - r["clean"][k] for k in keys} for r in res.values() if m in r and "clean" in r]
        if d:
            print(f"{m:14s} " + " ".join(f"{S.mean(x[k] for x in d):+13.4f}" for k in keys) + f"   cells {len(d)}")
    if a.root != "logs/diagnose/popbias":
        for m, base in (("nau", "poisoned"), ("forget_repair", "poisoned"), ("nau", "forget_repair")):
            d = [{k: r[m][k] - r[base][k] for k in keys} for r in res.values() if m in r and base in r]
            if d:
                print(f"{m} vs {base} (cells {len(d)}): " + "  ".join(
                    f"{k} {S.mean(x[k] for x in d):+.4f} ({sum(x[k] < 0 for x in d)} down)" for k in ("ARP", "head_share", "coverage", "gini")))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(res, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
