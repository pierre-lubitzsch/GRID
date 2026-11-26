"""Plan: cleaned TEST (scrub) and cleaned VALIDATION (scrubval) evals for the other-tokenizer
retraining-matched sensitive runs (scripts/plan_rmsplit_othertok.py), plus the original and retrained
reference models of every (dataset, seed) cell of the track.

    python scripts/plan_rmsplit_othertok_eval.py --track rqvae|letter|tracerv3 [--views scrub scrubval]
      -> logs/eval/sensitive_scrub/plan_othertok_<track>.tsv
Run with run_sensitive_scrub_eval_othertok.sh (CELL=all PLAN=<that file>); 15 columns: the 14 of
plan_rmgrid.tsv plus `quantizer`, written into the references' coordinates.json.

Output dirs (the same layout as the TIGER grid; finished evals are skipped by the runner):
    unlearned  logs/eval/sensitive_scrub/<view>/runs/<run>/eval
    reference  logs/eval/sensitive_scrub/<view>/reference/v1/<ds>_{base,retrain}_<reftok>[_seedN]
               reftok: rqvae | letter_letter (the existing LETTER refs) | tracerv3
Original = the clean model's last.ckpt (the unlearning start point); retrained = its best-epoch checkpoint.
An existing reference dir must describe the same checkpoint, or the planner stops.
Reference rows are left out (and reported) while a retrained model is missing or unfinished.
"""
import argparse
import csv
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plan_rmsplit_othertok as P  # noqa: E402

S = "logs/eval/sensitive_scrub"
HEAD = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source", "experiment", "quantizer"]
SUB = {"scrub": "scrubtestview", "scrubval": "scrubvalview"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(P.TRACKS))
    ap.add_argument("--views", nargs="+", default=["scrub", "scrubval"])
    ap.add_argument("--plans", nargs="+", default=None, help="unlearning plans (default: plan_othertok_<track>.tsv)")
    ap.add_argument("--out", default=None, help="eval plan path (default: plan_othertok_<track>.tsv in the eval dir)")
    a = ap.parse_args()
    t = P.TRACKS[a.track]
    rows, missing_refs, n_run, n_miss = [], [], 0, 0
    for ds in t["datasets"]:
        cat = P.CAT[ds]
        sid = t["sid"].format(ds=ds)
        for sd in P.SEEDS:
            sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
            man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
            refs = (("original", "base", P.clean_ckpt(a.track, ds, sd)),
                    ("retrained", "retrain", P.retrain_ckpt(a.track, ds, sd, required=False)))
            for role, arm, ck in refs:
                if ck is None:
                    missing_refs.append(f"{ds} seed{sd} {arm}")
                    continue
                for view in a.views:
                    out = f"{S}/{view}/reference/v1/{ds}_{arm}_{t['reftok']}{P.sfx(sd)}"
                    if os.path.isfile(f"{out}/coordinates.json"):
                        c = json.load(open(f"{out}/coordinates.json"))
                        assert c["ckpt"] == ck and c["semantic_id_path"] == sid and c["view"] == view \
                            and int(c["seed"]) == sd and c["data_dir"] == f"{sens}_{SUB[view]}", (out, c)
                    rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=str(sd), role=role, method=role, arm=arm,
                                     view=view, ckpt=ck, data_dir=f"{sens}_{SUB[view]}", manifest=man,
                                     semantic_id_path=sid, out_dir=out,
                                     source=f"plan_rmsplit_othertok.py {a.track} ({'clean last.ckpt' if arm == 'base' else 'retrain best epoch'})",
                                     experiment=t["experiment"], quantizer=t["quantizer"]))
    plans = a.plans or [f"logs/rmsplit_sens/plan_othertok_{a.track}.tsv"]
    for line in (l for p in plans for l in open(p)):
        f = line.rstrip("\n").split("\t")
        ds, sd, tag, algo, sid = f[1], int(f[3]), f[5], f[7], f[8]
        runs = sorted(glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}"))
        ck = [os.path.join(d, "checkpoints", "unlearned.ckpt") for d in runs
              if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
        if not ck:
            n_miss += 1
            print("no checkpoint:", tag)
            continue
        if len(ck) > 1:
            print("WARNING several finished run dirs, using the latest:", tag)
        ck = ck[-1]
        run = os.path.basename(os.path.dirname(os.path.dirname(ck)))
        arm = tag.split("rm_", 1)[1]
        cat = P.CAT[ds]
        sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
        man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
        for view in a.views:
            rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=str(sd), role="unlearned", method=f"{algo}_rm",
                             arm=arm, view=view, ckpt=ck, data_dir=f"{sens}_{SUB[view]}", manifest=man,
                             semantic_id_path=sid, out_dir=f"{S}/{view}/runs/{run}/eval",
                             source=os.path.dirname(os.path.dirname(ck)), experiment=t["experiment"],
                             quantizer=t["quantizer"]))
        n_run += 1
    out = a.out or f"{S}/plan_othertok_{a.track}.tsv"
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(HEAD)
        for r in rows:
            w.writerow([r[h] for h in HEAD])
    print(f"{out}: {len(rows)} rows ({n_run} runs x {len(a.views)} views + references); "
          f"{n_miss} planned runs without checkpoint; references missing: {missing_refs or 'none'}")


if __name__ == "__main__":
    main()
