"""Cleaned-view (protocol A) evaluation plan for the retraining-matched split runs.

Rows: every finished run of logs/rmsplit_sens/plan.tsv (run_rmsplit_sens_launch.sh: NAU and
Forget+Repair with unlearning.split_mode=retrain_matched, 30 runs, plus the current-split control
replay), SCRUB view only; their normal-view number is the run's own test eval (<run>/eval, same
command). Same columns and output layout as scripts/plan_sensitive_scrub_eval_canonck.py, read by
run_sensitive_scrub_eval.sh (CELL=all PLAN=logs/eval/sensitive_scrub/plan_rmsplit.tsv).
Original / retrained / `_mia` rows on the cleaned view already exist (plan.tsv).

    python scripts/plan_rmsplit_scrub_eval.py   # -> logs/eval/sensitive_scrub/plan_rmsplit.tsv
"""
import csv
import glob
import os
import re
import sys

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
OUT = "logs/eval/sensitive_scrub"
COLS = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source", "experiment"]


def run_dir(tag):
    runs = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
            if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
    return max(runs, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))) if runs else None


def main():
    rows, missing = [], []
    for model, ds, _st, sd, _us, tag, _ck, algo, sid, *_r in csv.reader(open("logs/rmsplit_sens/plan.tsv"),
                                                                          delimiter="\t"):
        sd = int(sd)
        sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
        r = run_dir(tag)
        if not r:
            missing.append(tag)
            continue
        arm = re.sub(r"_seed\d+$", "", re.sub(r"^sens_[a-z]+_[a-z]+_", "", tag))
        rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=sd, role="unlearned", method=f"{algo}_{model}",
                         arm=arm, view="scrub", ckpt=os.path.join(r, "checkpoints", "unlearned.ckpt"),
                         data_dir=f"{sens}_scrubtestview",
                         manifest=f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json",
                         semantic_id_path=sid, out_dir=f"{OUT}/scrub/runs/{os.path.basename(r)}/eval", source=r,
                         experiment=f"{model}_train_flat"))
    for r in rows:
        assert os.path.isdir(os.path.join(r["data_dir"], "testing")), r["data_dir"]
    with open(f"{OUT}/plan_rmsplit.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} evals -> {OUT}/plan_rmsplit.tsv; {len(missing)} runs without checkpoint")
    for t in missing:
        print("  MISSING", t, file=sys.stderr)


if __name__ == "__main__":
    main()
