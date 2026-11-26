"""List the run dirs of an other-tokenizer sensitive track whose cleaned TEST and cleaned VALIDATION
evals are both finished, i.e. whose unlearned checkpoint is no longer needed on scratch.

    python scripts/othertok_archive_list.py --track rqvae --out logs/rmsplit_sens/othertok/archive_rqvae.txt
The list is the RUN_LIST of archive_unlearn_ckpts.sh (which moves the payload to /sc-projects and
leaves a symlink). Runs whose evals are not both done are reported and left in place.
"""
import argparse
import glob
import os

S = "logs/eval/sensitive_scrub"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    keep, left = [], []
    for line in open(f"logs/rmsplit_sens/plan_othertok_{a.track}.tsv"):
        tag = line.split("\t")[5]
        for d in sorted(glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")):
            run = os.path.basename(d)
            if not glob.glob(f"{d}/checkpoints/unlearned_*.ckpt"):
                continue
            ok = all(os.path.isfile(f"{S}/{v}/runs/{run}/eval/csv/version_0/metrics.csv") for v in ("scrub", "scrubval"))
            (keep if ok else left).append(run)
    with open(a.out, "w") as fh:
        fh.write("".join(r + "\n" for r in keep))
    print(f"{a.out}: {len(keep)} runs to archive; {len(left)} left in place (evals not both done): {left}")


if __name__ == "__main__":
    main()
