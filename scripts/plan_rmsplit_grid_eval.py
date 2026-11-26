"""Plan: cleaned TEST and cleaned VALIDATION evals of every retraining-matched grid run.

For each finished run of logs/rmsplit_sens/plan_grid.tsv with a checkpoint, two rows:
  view=scrub     data_dir <sens>_scrubtestview  -> logs/eval/sensitive_scrub/scrub/runs/<run>/eval
  view=scrubval  data_dir <sens>_scrubvalview   -> logs/eval/sensitive_scrub/scrubval/runs/<run>/eval
plus the original and retrained models of every cell on both views (out dirs shared with the earlier
plans, so finished references are skipped by run_sensitive_scrub_eval.sh).

    python scripts/plan_rmsplit_grid_eval.py   -> logs/eval/sensitive_scrub/plan_rmgrid.tsv
"""
import csv
import glob
import os

S = "logs/eval/sensitive_scrub"
OUT = os.environ.get("OUT", f"{S}/plan_rmgrid.tsv")   # env overrides (2026-11-18): OUT, PLANS (comma list)
HEAD = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source", "experiment"]
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}


def refs():
    out = {}
    # TIGER references only (plan.tsv holds the TIGER refs for the cleaned test view; plan_canonck.tsv
    # also lists LETTER refs, which must not be used -- fixed 2026-11-15)
    for plan, view in ((f"{S}/plan.tsv", "scrub"), (f"{S}/plan_scrubval.tsv", "scrubval")):
        with open(plan, newline="") as fh:
            rd = csv.reader(fh, delimiter="\t"); head = [h.strip() for h in next(rd)]
            for r in rd:
                r = dict(zip(head, [x.strip() for x in r]))
                if (r["role"] in ("original", "retrained") and r["view"] == view
                        and (r.get("experiment") or "tiger_train_flat") == "tiger_train_flat"):
                    out.setdefault((r["cell"], r["role"], view), r)
    return out


def main():
    rows = list(refs().values())
    n_run = n_miss = 0
    plans = os.environ.get("PLANS", "plan_grid_wave1.tsv,plan_grid_wave2.tsv,plan_extra_anchor.tsv,plan_followup_anchor.tsv,plan_followup_anchor_rank.tsv").split(",")
    for line in (l for p in plans for l in open(f"logs/rmsplit_sens/{p}")):
        f = line.rstrip("\n").split("\t")
        ds, sd, tag, algo, sid = f[1], int(f[3]), f[5], f[7], f[8]
        runs = sorted(glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}"))
        ck = [os.path.join(d, "checkpoints", "unlearned.ckpt") for d in runs
              if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
        if not ck:
            n_miss += 1; print("no checkpoint:", tag); continue
        ck = ck[-1]; run = os.path.basename(os.path.dirname(os.path.dirname(ck)))
        arm = tag.split("rm_", 1)[1]
        sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
        man = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"
        for view, sub in (("scrub", "scrubtestview"), ("scrubval", "scrubvalview")):
            rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=str(sd), role="unlearned", method=f"{algo}_rm",
                             arm=arm, view=view, ckpt=ck, data_dir=f"{sens}_{sub}", manifest=man,
                             semantic_id_path=sid, out_dir=f"{S}/{view}/runs/{run}/eval",
                             source=os.path.dirname(os.path.dirname(ck)), experiment="tiger_train_flat"))
        n_run += 1
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(HEAD)
        for r in rows:
            w.writerow([r.get(h, "") for h in HEAD])
    print(f"{OUT}: {len(rows)} rows ({n_run} runs x 2 views + references); {n_miss} runs without checkpoint")


if __name__ == "__main__":
    main()
