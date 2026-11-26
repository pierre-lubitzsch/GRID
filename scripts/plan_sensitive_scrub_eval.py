"""Plan for the scrubbed-history re-evaluation of the unwanted-item ("sensitive") track.

One row per (model, view). Models per (dataset, seed) cell:
  original   the exact ckpt_path the cell's `_mia` unlearning runs started from
             (their .hydra/overrides.yaml; NOT the best-epoch checkpoint)
  retrained  the retrain reference, newest logs/eval/sensitive_reference/*/<tok>/
             coordinates.json whose category matches (as extract_sensitive.reference)
  unlearned  every `_mia` run (logs/mia_unwanted_plan.tsv), checkpoints/unlearned.ckpt
Views:
  normal     data_dir = src/data/amazon_data/<ds>         (what the paper evaluated)
  scrub      data_dir = <ds>_sens_<cat>_1e-4_seed<sd>_scrubtestview
             (scripts/build_scrubbed_eval_view.py)

Output layout is what scripts/extract_sensitive.py reads:
  logs/eval/sensitive_scrub/<view>/runs/<unlearn run basename>/eval/csv/version_0/metrics.csv
      -> run_tags.iter_runs(runs_root=logs/eval/sensitive_scrub/<view>/runs)
  logs/eval/sensitive_scrub/<view>/reference/v1/<tok>/{coordinates.json,csv/...}
      -> GRID_SENSITIVE_REF_ROOT=logs/eval/sensitive_scrub/<view>/reference

    python scripts/plan_sensitive_scrub_eval.py     # writes logs/eval/sensitive_scrub/plan.tsv
"""
import csv
import glob
import json
import os
import re

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
OUT = "logs/eval/sensitive_scrub"
# Same names / prefixes as scripts/launch_mia_unwanted.py (paper method -> arm prefix).
METHOD = [("nau", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"), ("finetune", "unified_abl_lf0.0_ls0.0_ln0.0"),
          ("forget_only", "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0"), ("forget_repair", "unified_abl_lf0.1_ls0.0_ln0.0"),
          ("scif", "scif_"), ("seif", "seif_"), ("kookmin", "kookmin_"), ("fanchuan", "fanchuan_"), ("tracer", "tracer_")]
COLS = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source"]


def method_of(arm):
    for name, prefix in METHOD:
        if prefix.endswith("_"):
            if arm.startswith(prefix):
                return name
        elif arm == f"{prefix}_mia" or arm.startswith(f"{prefix}_mia"):
            return name
    return None


def retrain_ckpt(ds, sd):
    tok = f"{ds}_retrain" + ("" if sd == 2 else f"_seed{sd}")
    for d in sorted(glob.glob(f"logs/eval/sensitive_reference/*/{tok}"), key=os.path.getmtime, reverse=True):
        c = os.path.join(d, "coordinates.json")
        if os.path.isfile(c):
            j = json.load(open(c))
            if j.get("category") == CAT[ds] and int(j.get("seed", 2)) == sd \
                    and j.get("model", "tiger") == "tiger" and j.get("quantizer", "rkmeans") == "rkmeans":
                return j["ckpt"], d
    raise SystemExit(f"no retrain reference for {ds} seed {sd}")


def overrides(run):
    o = {}
    for line in open(os.path.join(run, ".hydra", "overrides.yaml")):
        line = line.strip().lstrip("- ").strip()
        if "=" in line:
            k, v = line.split("=", 1)
            o[k] = v.strip("'\"")
    return o


def main():
    plan = [r.rstrip("\n").split("\t") for r in open("logs/mia_unwanted_plan.tsv")]
    runs = sorted(glob.glob("logs/unlearn/runs/*_bs1_sens_*_mia*"))
    rows = []
    for ds in CAT:
        for sd in SEEDS:
            sfx = "" if sd == 2 else f"_seed{sd}"
            sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
            manifest = f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"
            sid = f"embeddings/{ds}/merged_predictions_tensor.pt"
            cell = f"{ds}_s{sd}"
            mine = [r for r in runs if re.search(rf"_bs1_sens_{ds}_{CAT[ds]}_.*_mia{sfx}$", r)]
            tags = {p[4] for p in plan if p[0] == ds and int(p[2]) == sd}
            assert {re.sub(r"^.*_bs1_", "", os.path.basename(r)) for r in mine} == tags, (cell, len(mine), len(tags))
            starts = set()
            models = []
            for r in mine:
                o = overrides(r)
                assert o["data_dir"] == sens and int(o["seed"]) == sd, r
                starts.add(o["ckpt_path"])
                arm = re.sub(rf"^.*_bs1_sens_{ds}_{CAT[ds]}_", "", os.path.basename(r))
                arm = arm[: -len(sfx)] if sfx else arm
                ck = os.path.join(r, "checkpoints", "unlearned.ckpt")
                assert os.path.exists(ck), ck
                models.append(("unlearned", method_of(arm), arm, ck, os.path.basename(r), r))
            assert len(starts) == 1, (cell, starts)
            orig = starts.pop()
            rck, rsrc = retrain_ckpt(ds, sd)
            assert os.path.exists(orig) and os.path.exists(rck)
            models = [("original", "original", "base", orig, None, "overrides.yaml of the cell's _mia runs"),
                      ("retrained", "retrained", "retrain", rck, None, rsrc)] + sorted(models, key=lambda m: m[1])
            for view in ("normal", "scrub"):
                dd = f"src/data/amazon_data/{ds}" if view == "normal" else f"{sens}_scrubtestview"
                for role, meth, arm, ck, rundir, src in models:
                    if role == "unlearned":
                        out = f"{OUT}/{view}/runs/{rundir}/eval"
                    else:
                        tok = f"{ds}_{arm}" + ("" if sd == 2 else f"_seed{sd}")
                        out = f"{OUT}/{view}/reference/v1/{tok}"
                    rows.append(dict(cell=cell, dataset=ds, seed=sd, role=role, method=meth, arm=arm, view=view,
                                     ckpt=ck, data_dir=dd, manifest=manifest, semantic_id_path=sid,
                                     out_dir=out, source=src))
    os.makedirs(OUT, exist_ok=True)
    with open(f"{OUT}/plan.tsv", "w", newline="") as fh:
        w = csv.DictWriter(fh, COLS, delimiter="\t")
        w.writeheader()
        w.writerows(rows)
    cells = sorted({r["cell"] for r in rows}, key=lambda c: (c.split("_s")[0], int(c.split("_s")[1])))
    with open(f"{OUT}/cells.txt", "w") as fh:
        fh.write("\n".join(cells) + "\n")
    print(f"{len(rows)} evals in {len(cells)} cells -> {OUT}/plan.tsv, {OUT}/cells.txt")
    for c in cells:
        n = [r for r in rows if r["cell"] == c]
        print(f"  {c:10s} {len(n):3d} evals  methods={sorted({r['method'] for r in n})}")


if __name__ == "__main__":
    main()
