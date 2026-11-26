"""Free disk after the per-tokenizer grid of sensitive item deletion (2026-11-23, user decision: delete what we do not need).

For one tokenizer track:
  1. per-tokenizer selection on cleaned validation (othertok_tables.select, TUNED=1);
  2. KEEP the unlearned checkpoints of every selected configuration (all seeds): written to a RUN_LIST for
     archive_unlearn_ckpts.sh, so a later membership test can still score them;
  3. DELETE the unlearned checkpoints (checkpoints/unlearned*.ckpt, including an archived payload a symlink points to)
     of every other run of logs/rmsplit_sens/plan_othertok_grid_<track>.tsv, but only if all four evals exist:
     eval/ (clean test), eval_val/ (validation), cleaned test (scrub) and cleaned validation (scrubval).
     Run logs, configs, requests/ and every metrics file stay.
A dataset whose grid is not completely evaluated on validation is left untouched (no selection possible yet).
Runs of the pinned campaign (plan_othertok_<track>.tsv) are never deleted here.

    python scripts/othertok_grid_cleanup.py --track rqvae --dry-run
    python scripts/othertok_grid_cleanup.py --track rqvae      # deletes, writes the keep list and a deletion log
"""
import argparse
import glob
import importlib.util
import os
import re
import sys

os.environ["TUNED"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "T", os.path.join(HERE, "..", "paper", "plotting", "critique_followup", "othertok_tables.py"))
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)
D = "logs/rmsplit_sens/othertok"


def has_metrics(d):
    return bool(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))


def all_evals(run):
    b = os.path.basename(run)
    return (has_metrics(f"{run}/eval") and has_metrics(f"{run}/eval_val")
            and has_metrics(f"{T.R.ROOT}/scrub/runs/{b}/eval") and has_metrics(f"{T.R.ROOT}/scrubval/runs/{b}/eval"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(T.TRACKS))
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    track = a.track
    prefix, _, _, _, dss, _ = T.TRACKS[track]
    picks = T.select(track)                          # {(dataset, method): config}
    # datasets whose selection is complete for every method of the grid
    complete = {ds for ds in dss if all((ds, m) in picks for m in T.grid_methods(track, ds))}
    keep_arms = {ds: {c for (d, m), c in picks.items() if d == ds} for ds in complete}
    plan = f"logs/rmsplit_sens/plan_othertok_grid_{track}.tsv"
    keep_runs, delete, skipped = [], [], []
    for line in open(plan):
        f = line.split("\t")
        ds, tg = f[1], f[5]
        arm = re.match(rf"\w*?sens_{ds}_\w+?rm_(.*)_rmsplit_canonck(_seed\d+)?$", tg).group(1)
        for run in glob.glob(f"logs/unlearn/runs/job*_bs1_{tg}"):
            cks = glob.glob(f"{run}/checkpoints/unlearned*.ckpt")
            if not cks:
                continue
            if ds not in complete:
                skipped.append((run, "dataset selection incomplete")); continue
            if arm in keep_arms[ds]:
                keep_runs.append(os.path.basename(run)); continue
            if not all_evals(run):
                skipped.append((run, "evals incomplete")); continue
            delete.append((run, cks))
    # the selected configurations of the pinned campaign are already archived; list them too (idempotent)
    keep_file = f"{D}/keep_selected_{track}.txt"
    log_file = f"{D}/deleted_ckpts_{track}.log"
    print(f"{track}: complete datasets {sorted(complete)}; keep {len(keep_runs)} runs, delete {len(delete)} runs, "
          f"skipped {len(skipped)}")
    for r, why in skipped[:10]:
        print("  skip", os.path.basename(r), why)
    if a.dry_run:
        for r, cks in delete[:5]:
            print("  would delete", [os.path.basename(c) for c in cks], "in", os.path.basename(r))
        return
    with open(keep_file, "w") as fh:
        fh.write("\n".join(sorted(set(keep_runs))) + ("\n" if keep_runs else ""))
    freed = 0
    with open(log_file, "a") as log:
        for run, cks in delete:
            for c in cks:
                targets = [c]
                if os.path.islink(c):
                    t = os.path.realpath(c)
                    if t.startswith("/sc-projects/") and os.path.isfile(t):
                        targets.append(t)        # archived payload of this run
                for t in targets:
                    if os.path.isfile(t) and not os.path.islink(t):
                        freed += os.path.getsize(t)
                    os.remove(t)
                    log.write(f"{t}\n")
    print(f"deleted {len(delete)} runs' checkpoints, freed {freed / 2**30:.1f} GiB; keep list {keep_file}; log {log_file}")


if __name__ == "__main__":
    main()
