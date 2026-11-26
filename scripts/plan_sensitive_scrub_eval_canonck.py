"""Cleaned-view (scrubbed test histories) evaluation plan for the unwanted-item `_canonck` re-runs.

Companion of scripts/plan_sensitive_scrub_eval.py (which covers original, retrained and the 130
`_mia` runs of TIGER + RQ-KMeans). This plan adds, in the same output layout:
  unlearned  every finished `_canonck` run of logs/canonck_sens/plan.tsv (run_canonck_sens_launch.sh:
             RQ-KMeans NAU lambda_n=0 / -10 and LETTER NAU lambda_n=0 / -1 / -10) and of
             logs/canonck_sens/plan_t8.tsv (run_canonck_sens_t8_launch.sh: the other non-selected
             Table 8 candidates), SCRUB view only
             (their normal-view number is the run's own test eval, <run>/eval, same command);
  original   LETTER, the exact ckpt_path the cell's LETTER `_canonck` runs start from (last.ckpt),
             normal and scrub view;
  retrained  LETTER, the retrain reference (newest logs/eval/sensitive_reference/*/<tok>/ with
             model=letter, quantizer=letter, matching category and seed), normal and scrub view.
Views are shared with TIGER: LETTER runs use the same data dirs (item indexing is shared; only
semantic_id_path and the experiment config differ), so <sens>_scrubtestview applies unchanged.

Rows whose unlearned checkpoint does not exist (a failed re-run) are skipped and printed.
Extra 14th column `experiment` (tiger_train_flat | letter_train_flat), read by
run_sensitive_scrub_eval.sh.

    python scripts/plan_sensitive_scrub_eval_canonck.py   # -> logs/eval/sensitive_scrub/plan_canonck.tsv
"""
import csv
import glob
import json
import os
import re
import sys

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
OUT = "logs/eval/sensitive_scrub"
COLS = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source", "experiment"]


def retrain_letter(ds, sd):
    tok = f"{ds}_retrain_letter_letter" + ("" if sd == 2 else f"_seed{sd}")
    for d in sorted(glob.glob(f"logs/eval/sensitive_reference/*/{tok}"), key=os.path.getmtime, reverse=True):
        c = os.path.join(d, "coordinates.json")
        if os.path.isfile(c):
            j = json.load(open(c))
            if j.get("category") == CAT[ds] and int(j.get("seed", 2)) == sd \
                    and j.get("model") == "letter" and j.get("quantizer") == "letter":
                return j["ckpt"], d
    raise SystemExit(f"no LETTER retrain reference for {ds} seed {sd}")


def main():
    plan = [r for f in ("logs/canonck_sens/plan.tsv", "logs/canonck_sens/plan_t8.tsv") if os.path.isfile(f)
            for r in csv.reader(open(f), delimiter="\t")]
    rows, missing, starts = [], [], {}
    for model, ds, _st, sd, _us, tag, ckpt, algo, sid, *_r in plan:
        sd = int(sd)
        sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
        manifest = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"
        exp = f"{model}_train_flat"
        if model == "letter":
            assert starts.setdefault((ds, sd), ckpt) == ckpt, (ds, sd, "two start checkpoints")
        runs = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
                if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
        if not runs:
            missing.append(tag)
            continue
        r = max(runs, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))
        arm = re.sub(r"^(letter_letter_)?sens_[a-z]+_[a-z]+_", "", tag)
        arm = re.sub(r"_seed\d+$", "", arm)
        rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=sd, role="unlearned", method=f"{algo}_{model}",
                         arm=arm, view="scrub", ckpt=os.path.join(r, "checkpoints", "unlearned.ckpt"),
                         data_dir=f"{sens}_scrubtestview", manifest=manifest, semantic_id_path=sid,
                         out_dir=f"{OUT}/scrub/runs/{os.path.basename(r)}/eval", source=r, experiment=exp))
    for (ds, sd), orig in sorted(starts.items()):
        sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
        manifest = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"
        sid = f"embeddings/{ds}_letter/merged_predictions_tensor.pt"
        rck, rsrc = retrain_letter(ds, sd)
        assert os.path.exists(orig) and os.path.exists(rck), (orig, rck)
        for view in ("normal", "scrub"):
            dd = f"src/data/amazon_data/{ds}" if view == "normal" else f"{sens}_scrubtestview"
            for role, arm, ck, src in (("original", "base", orig, "ckpt_path of the cell's LETTER _canonck runs"),
                                       ("retrained", "retrain", rck, rsrc)):
                tok = f"{ds}_{arm}_letter_letter" + ("" if sd == 2 else f"_seed{sd}")
                rows.append(dict(cell=f"{ds}_s{sd}", dataset=ds, seed=sd, role=role, method=role, arm=arm,
                                 view=view, ckpt=ck, data_dir=dd, manifest=manifest, semantic_id_path=sid,
                                 out_dir=f"{OUT}/{view}/reference/v1/{tok}", source=src,
                                 experiment="letter_train_flat"))
    for r in rows:
        assert os.path.isdir(os.path.join(r["data_dir"], "testing")), r["data_dir"]
    with open(f"{OUT}/plan_canonck.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} evals -> {OUT}/plan_canonck.tsv; {len(missing)} `_canonck` runs without checkpoint")
    for t in missing:
        print("  MISSING", t, file=sys.stderr)


if __name__ == "__main__":
    main()
