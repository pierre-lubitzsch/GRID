"""Clean-data item memory vs plain TIGER: test N@10 per (dataset, arch, training seed).

Reads the end-of-training test of train.py (best-by-val checkpoint) from each clean training run
(label _ITEMMEM_<arch>clean<recipe>[_d<seed>]) and, if present, the memory-off eval
(logs/diagnose/itemmem_clean/<ds>_<arch><recipe>_s<seed>_null/). Prints one table per dataset with
the per-seed values, the mean and the paired gain over plain.

Usage (from GRID/): python -m scripts.itemmem_clean_summary [--recipe es|tigr] [--csv out.csv]
"""

import argparse
import csv
import glob
import statistics as st

DATASETS = ("beauty", "sports", "toys")
ARCHS = ("plain", "im", "imc", "imfree")
SEEDS = (2, 3, 5, 7, 11)


def last_test(path_glob: str, key: str = "test/ndcg@10"):
    for f in sorted(glob.glob(path_glob), reverse=True):
        v = None
        for r in csv.DictReader(open(f)):
            if r.get(key) not in (None, ""):
                v = float(r[key])
        if v is not None:
            return v
    return None


def train_ndcg(ds: str, arch: str, seed: int, recipe: str):
    # chained TIGER runs: the cross-segment best checkpoint's test (train.py tests only the last
    # segment's own best)
    v = last_test(f"logs/diagnose/itemmem_clean/{ds}_{arch}_{recipe}_s{seed}_full_xseg/csv/version_*/metrics.csv")
    if v is not None:
        return v
    lab = f"ITEMMEM_{arch}clean_{recipe}" + ("" if seed == 2 else f"_d{seed}")
    runs = [d for d in sorted(glob.glob(f"logs/train/runs/*/*_{ds}_*{lab}"), reverse=True)
            if d.endswith(lab)]
    for d in runs:  # newest run (or resume segment) that logged a test
        v = last_test(f"{d}/csv/version_*/metrics.csv")
        if v is not None:
            return v
    return None


def null_ndcg(ds: str, arch: str, seed: int, recipe: str):
    v = last_test(f"logs/diagnose/itemmem_clean/{ds}_{arch}_{recipe}_s{seed}_null_xseg/csv/version_*/metrics.csv")
    if v is not None:
        return v
    return last_test(f"logs/diagnose/itemmem_clean/{ds}_{arch}_{recipe}_s{seed}_null/csv/version_*/metrics.csv")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--recipe", default="es", choices=("es", "esrk", "gprk", "gp40rk", "tigr", "tigraf", "tigrtok"))
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    f = lambda x: "  -   " if x is None else f"{x:.4f}"
    rows = []
    for ds in DATASETS:
        print(f"\n{ds} ({a.recipe}) test N@10, best-val ckpt   " + "  ".join(f"s{s:<5d}" for s in SEEDS)
              + "  mean    vs plain (paired)   memory off (null)")
        plain = {s: train_ndcg(ds, "plain", s, a.recipe) for s in SEEDS}
        for arch in ARCHS:
            v = {s: train_ndcg(ds, arch, s, a.recipe) for s in SEEDS}
            nl = {s: null_ndcg(ds, arch, s, a.recipe) for s in SEEDS} if arch != "plain" else {}
            have = [x for x in v.values() if x is not None]
            mean = st.mean(have) if have else None
            gains = [100 * (v[s] / plain[s] - 1) for s in SEEDS if v[s] is not None and plain[s]]
            g = "" if arch == "plain" or not gains else f"{st.mean(gains):+5.1f}% (n={len(gains)}, min {min(gains):+.1f})"
            n = "" if not nl else " ".join(f(nl[s]) for s in SEEDS)
            print(f"  {arch:7s}" + " ".join(f(v[s]) for s in SEEDS) + f"  {f(mean)}  {g:24s} {n}")
            for s in SEEDS:
                rows.append({"dataset": ds, "arch": arch, "seed": s, "recipe": a.recipe,
                             "ndcg10": v[s], "ndcg10_memory_off": nl.get(s)})
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
