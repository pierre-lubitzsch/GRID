"""Eval plan for the larger-request sensitive variant (budget 5e-4; run_sens5e4_*.sh).

One row per (model, view), same columns as scripts/plan_sensitive_scrub_eval.py so that
run_sensitive_scrub_eval.sh (CELL=all PLAN=...) executes it unchanged. Models per (dataset, seed):
  original   the exact start ckpt of the cell's 5e-4 unlearning runs (= the `_mia` start ckpt)
  retrained  newest epoch checkpoint of the 5e-4 retrain training `<ds>_clean[_seedN]_retrain5e-4_<cat>`
             (the rule run_sensitive_reference_eval.sh used for the 1e-4 retrained references)
  unlearned  every run in logs/sens5e4/unlearn_plan.tsv with a checkpoints/unlearned.ckpt
Views: normal (src/data/amazon_data/<ds>) and scrub (<ds>_sens_<cat>_5e-4_seed<N>_scrubtestview).
The manifest is the 5e-4 one, so SHF/SHR are scored on the 5e-4 requesting users.
Output: logs/eval/sensitive_5e-4/<view>/{runs/<unlearn run>/eval, reference/v1/<ds>_{base,retrain}[_seedN]}
(a separate tree: no existing reader globs it). Missing models are reported, not silently dropped.

    python scripts/plan_sens5e4_eval.py [--datasets beauty] [--out logs/eval/sensitive_5e-4/plan_beauty.tsv]
"""
import argparse
import csv
import glob
import os
import sys

R = "5e-4"
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
OUT = f"logs/eval/sensitive_{R}"
COLS = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source"]
METHOD = [("nau_ln0", "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0"), ("nau", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"),
          ("finetune", "unified_abl_lf0.0_ls0.0_ln0.0"), ("forget_only", "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0"),
          ("forget_repair", "unified_abl_lf0.1_ls0.0_ln0.0"), ("scif", "scif_"), ("seif", "seif_"),
          ("kookmin", "kookmin_"), ("fanchuan", "fanchuan_"), ("tracer", "tracer_")]


def method_of(arm):
    for name, p in METHOD:
        if (p.endswith("_") and arm.startswith(p)) or arm == p:
            return name
    raise SystemExit(f"unknown arm {arm}")


def retrain_ckpt(ds, sd):
    lab = f"{ds}_clean" + ("" if sd == 2 else f"_seed{sd}") + f"_retrain{R}_{CAT[ds]}"
    runs = [d for d in glob.glob(f"logs/train/runs/*/*_{lab}") if os.path.basename(d).endswith("_" + lab)]
    cks = sorted((c for d in runs for c in glob.glob(f"{d}/checkpoints/checkpoint_epoch=*.ckpt")),
                 key=os.path.getmtime, reverse=True)
    return (cks[0], runs[0]) if cks else (None, lab)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=list(CAT))
    ap.add_argument("--out", default=f"{OUT}/plan.tsv")
    a = ap.parse_args()
    plan = [l.rstrip("\n").split("\t") for l in open("logs/sens5e4/unlearn_plan.tsv")]
    rows, missing = [], []
    for ds in a.datasets:
        for sd in SEEDS:
            cell, sfx = f"{ds}_s{sd}", ("" if sd == 2 else f"_seed{sd}")
            sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_{R}_seed{sd}"
            man = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_{R}_seed{sd}.json"
            sid = f"embeddings/{ds}/merged_predictions_tensor.pt"
            mine = [p for p in plan if p[0] == ds and int(p[2]) == sd]
            starts = {p[5] for p in mine}
            assert len(starts) == 1, (cell, starts)
            models = [("original", "original", "base", starts.pop(), None, "start ckpt of the 5e-4 unlearning runs")]
            rck, rsrc = retrain_ckpt(ds, sd)
            if rck:
                models.append(("retrained", "retrained", "retrain", rck, None, rsrc))
            else:
                missing.append(f"retrain {rsrc}")
            for p in mine:
                tag = p[4]
                runs = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
                        if os.path.exists(f"{d}/checkpoints/unlearned.ckpt")
                        and os.path.isfile(f"{d}/eval/csv/version_0/metrics.csv")]
                if not runs:
                    missing.append(f"unlearned {tag}")
                    continue
                d = max(runs, key=lambda x: int(os.path.basename(x).split("_")[0][3:]))
                arm = tag[len(f"sens_{ds}_{CAT[ds]}{R}_"):]
                arm = arm[: -len(sfx)] if sfx else arm
                models.append(("unlearned", method_of(arm), arm, f"{d}/checkpoints/unlearned.ckpt",
                               os.path.basename(d), d))
            for view in ("normal", "scrub"):
                dd = f"src/data/amazon_data/{ds}" if view == "normal" else f"{sens}_scrubtestview"
                for role, meth, arm, ck, rundir, src in models:
                    out = (f"{OUT}/{view}/runs/{rundir}/eval" if role == "unlearned"
                           else f"{OUT}/{view}/reference/v1/{ds}_{arm}{sfx}")
                    rows.append(dict(cell=cell, dataset=ds, seed=sd, role=role, method=meth, arm=arm, view=view,
                                     ckpt=ck, data_dir=dd, manifest=man, semantic_id_path=sid, out_dir=out,
                                     source=src))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"{len(rows)} evals -> {a.out}")
    for m in missing:
        print(f"  MISSING {m}", file=sys.stderr)


if __name__ == "__main__":
    main()
