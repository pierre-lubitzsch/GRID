"""Cleaned-view (protocol A) evaluation plan for the 30 retain-side anchor runs.

Companion of scripts/plan_sensitive_scrub_eval.py and plan_sensitive_scrub_eval_canonck.py, same
output layout and columns, read by run_sensitive_scrub_eval.sh (CELL=all PLAN=...):
  unlearned  every run of logs/rebuttal_variants/plan_anchor.tsv
             (sens_<ds>_<cat>_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_anchor{0p1,1p0}_canonck[_seedN],
             TIGER + RQ-KMeans), on the SCRUB view and, as a same-code reproduction check of the
             run's own <run>/eval, on the NORMAL view.
Original, retrained and NAU (`_mia`) on both views already exist from plan.tsv.

    python scripts/plan_anchor_scrub_eval.py   # -> logs/eval/sensitive_scrub/plan_anchor.tsv
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


def main():
    rows, missing = [], []
    for ds, _st, sd, _us, tag, start, algo, sid, *_r in csv.reader(open("logs/rebuttal_variants/plan_anchor.tsv"),
                                                                     delimiter="\t"):
        sd = int(sd)
        runs = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
                if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
        if not runs:
            missing.append(tag)
            continue
        r = max(runs, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))
        arm = re.sub(r"_seed\d+$", "", re.sub(r"^sens_[a-z]+_[a-z]+_", "", tag))
        manifest = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"
        for view in ("normal", "scrub"):
            dd = (f"src/data/amazon_data/{ds}" if view == "normal"
                  else f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}_scrubtestview")
            rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=sd, role="unlearned", method="nau_anchor",
                             arm=arm, view=view, ckpt=os.path.join(r, "checkpoints", "unlearned.ckpt"),
                             data_dir=dd, manifest=manifest, semantic_id_path=sid,
                             out_dir=f"{OUT}/{view}/runs/{os.path.basename(r)}/eval", source=r,
                             experiment="tiger_train_flat"))
    for r in rows:
        assert os.path.isdir(os.path.join(r["data_dir"], "testing")), r["data_dir"]
        assert os.path.isfile(r["manifest"]), r["manifest"]
    with open(f"{OUT}/plan_anchor.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} evals -> {OUT}/plan_anchor.tsv; {len(missing)} anchor runs without checkpoint")
    for t in missing:
        print("  MISSING", t, file=sys.stderr)


if __name__ == "__main__":
    main()
