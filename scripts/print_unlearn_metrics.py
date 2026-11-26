"""Final test metrics of unlearning runs, by SLURM job id or run-tag substring.

Usage (from GRID/): python -m scripts.print_unlearn_metrics 11268351 ref_spam_d3 ...
A run-tag substring prints every matching run (newest last).
"""
import csv
import glob
import sys

COLS = ["test/ndcg@10", "test/SH@10", "test/SHF@10", "test/SHR@10"]


def main() -> int:
    for key in sys.argv[1:]:
        runs = sorted(glob.glob(f"logs/unlearn/runs/job{key}_*")) or sorted(
            glob.glob(f"logs/unlearn/runs/*{key}*"))
        if not runs:
            print(f"{key}: no run dir")
            continue
        for run in runs:
            tag = run.split("_bs1_")[-1]
            files = sorted(glob.glob(f"{run}/eval/csv/version_*/metrics.csv"))
            if not files:
                print(f"{key:10s} {tag:52s} no eval")
                continue
            vals = {}
            for r in csv.DictReader(open(files[-1])):
                for c in COLS:
                    if r.get(c) not in (None, ""):
                        vals[c] = float(r[c])
            print(f"{key:10s} {tag:52s} " + "  ".join(
                f"{c.split('/')[1]}={vals[c]:.4f}" for c in COLS if c in vals))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
