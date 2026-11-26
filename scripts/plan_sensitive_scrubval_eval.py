"""Plan: every sensitive-deletion selection candidate on the CLEANED validation histories.

Reviewer QzxM follow-up: model selection for sensitive item deletion used the original validation
histories, while the reported test results use cleaned ones. This plans the evaluation needed to
redo the selection on cleaned validation histories: for each (dataset, seed),
  - the original and the retrained model (reference for tau_P on the same view),
  - every candidate configuration of paper/plotting/valsens_candidates_ledger.csv that has a
    checkpointed execution: the selected `_mia` runs (plan.tsv), the `_canonck` re-runs
    (plan_canonck.tsv, TIGER rows only) and the `_canonckb` re-runs found on disk at planning time.
Same rows as the cleaned TEST plans, with data_dir *_scrubtestview -> *_scrubvalview (whose
`testing/` is the cleaned validation split, so trainer.test scores validation), view=scrubval and
out_dir under logs/eval/sensitive_scrub/scrubval/.

    python scripts/plan_sensitive_scrubval_eval.py
    CELL=all PLAN=logs/eval/sensitive_scrub/plan_scrubval.tsv VIEWS=scrubval PAR=3 bash run_sensitive_scrub_eval.sh
"""
import csv
import glob
import os
import re

S = "logs/eval/sensitive_scrub"
OUT = f"{S}/plan_scrubval.tsv"
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SKIP_METHODS = {"finetune", "forget_only", "forget_repair"}   # fixed configurations, not selected


def rows_of(path):
    with open(path, newline="") as fh:
        rd = csv.reader(fh, delimiter="\t")
        head = [h.strip() for h in next(rd)]
        for r in rd:
            r = [x.strip() for x in r]
            yield dict(zip(head, r + [""] * (len(head) - len(r))))


def convert(r):
    r = dict(r)
    assert r["data_dir"].endswith("_scrubtestview"), r["data_dir"]
    r["data_dir"] = r["data_dir"][: -len("_scrubtestview")] + "_scrubvalview"
    assert "/scrub/" in r["out_dir"], r["out_dir"]
    r["out_dir"] = r["out_dir"].replace(f"{S}/scrub/", f"{S}/scrubval/", 1)
    r["view"] = "scrubval"
    r["experiment"] = r.get("experiment") or "tiger_train_flat"
    return r


def main():
    head = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
            "semantic_id_path", "out_dir", "source", "experiment"]
    out, seen = [], set()
    for plan in ("plan.tsv", "plan_canonck.tsv"):
        for r in rows_of(f"{S}/{plan}"):
            if r["view"] != "scrub" or (r.get("experiment") or "tiger_train_flat") != "tiger_train_flat":
                continue
            if r["role"] == "unlearned" and r["method"] in SKIP_METHODS:
                continue
            key = (r["cell"], r["role"], r["arm"])
            if key in seen:
                continue
            seen.add(key)
            out.append(convert(r))
    # _canonckb re-runs (run_canonck_sens_t8b_launch.sh): template a row from a canonck row of the cell
    tmpl = {r["cell"]: r for r in out if r["role"] == "unlearned"}
    for d in sorted(glob.glob("logs/unlearn/runs/job*_bs1_sens_*_canonckb*")):
        ck = os.path.join(d, "checkpoints", "unlearned.ckpt")
        if not os.path.exists(ck):
            print("no checkpoint (yet):", os.path.basename(d))
            continue
        tag = os.path.basename(d).split("_bs1_", 1)[1]
        m = re.match(r"sens_(\w+?)_(hairloss|guns|toyweapons)_(.+?)(?:_seed(\d+))?$", tag)
        ds, arm, sd = m.group(1), m.group(3), int(m.group(4) or 2)
        cell = f"{ds}_s{sd}"
        t = dict(tmpl[cell])
        t.update(role="unlearned", method=("tracer" if arm.startswith("tracer") else "unified_tiger"),
                 arm=arm, ckpt=ck, source=d,
                 out_dir=f"{S}/scrubval/runs/{os.path.basename(d)}/eval")
        if (cell, "unlearned", arm) not in seen:
            seen.add((cell, "unlearned", arm))
            out.append(t)
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(head)
        for r in out:
            w.writerow([r.get(h, "") for h in head])
    from collections import Counter
    print(OUT, len(out), "rows")
    for k, v in sorted(Counter((r["dataset"], r["role"], r["method"]) for r in out).items()):
        print(" ", k, v)


if __name__ == "__main__":
    main()
