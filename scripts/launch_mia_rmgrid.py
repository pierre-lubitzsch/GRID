"""Membership inference on the deleted interactions for the configurations SELECTED on cleaned validation
with the retraining-matched split (paper/revision/rmsplit_selected.csv, written by
paper/plotting/critique_followup/rmsplit_selection.py). Same scorer and member / non-member sets as
scripts/launch_mia_unwanted.py (logs/diagnose/mia_unwanted/sets/<ds>_s<N>).

Models per (dataset, seed): original (the exact start checkpoint of the grid runs), retrained reference
(logs/eval/sensitive_reference/*/<ds>_retrain[_seedN]/coordinates.json, the same checkpoint the cleaned
views use), and the selected run of Finetune, Forget only, Forget+Repair, SCIF, SEIF, Kookmin, Fanchuan,
TRACER (Sports / Toys) and NAU (latest run dir of the tag with checkpoints/unlearned.ckpt).
Output: logs/diagnose/mia_unwanted_rmgrid/<ds>_seed<N>.json (new directory; existing JSONs are skipped).
Method keys in the JSON: original, retrained, finetune, forget_only, forget_repair, scif, seif, kookmin,
fanchuan, tracer, nau.

    python scripts/launch_mia_rmgrid.py            # dry run
    python scripts/launch_mia_rmgrid.py --submit   # one pgpu job (2 GPUs, mandatory there) per dataset
"""
import argparse
import csv
import glob
import json
import os
import re
import subprocess

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
KEY = {"Finetune": "finetune", "Forget only": "forget_only", "Forget+Repair": "forget_repair", "SCIF": "scif",
       "SEIF": "seif", "Kookmin": "kookmin", "Fanchuan": "fanchuan", "TRACER": "tracer", "NAU": "nau"}
OUTD = "logs/diagnose/mia_unwanted_rmgrid"


def run_ckpt(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}") if os.path.exists(f"{d}/checkpoints/unlearned.ckpt")]
    if not c:
        return None
    d = max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))
    return f"{d}/checkpoints/unlearned.ckpt"


def start_ckpt(ds, sd):
    for line in open("logs/rmsplit_sens/plan_grid_wave1.tsv"):
        f = line.rstrip("\n").split("\t")
        if f[1] == ds and int(f[3]) == sd:
            return f[6]


def retrained(ds, sd):
    tok = f"{ds}_retrain" + ("" if sd == 2 else f"_seed{sd}")
    c = sorted(glob.glob(f"logs/eval/sensitive_reference/*/{tok}/coordinates.json"), key=os.path.getmtime)
    return json.load(open(c[-1]))["ckpt"] if c else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selected", default="paper/revision/rmsplit_selected.csv")
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--methods", nargs="+", default=list(KEY), help="selected methods to score (with original + retrained)")
    ap.add_argument("--datasets", nargs="+", default=None, help="restrict to these datasets (default: all)")
    ap.add_argument("--local", action="store_true", help="run the cells sequentially in this shell instead of sbatch")
    ap.add_argument("--cells", nargs="+", default=None, help="with --local: only these seeds, e.g. 2 3")
    ap.add_argument("--partition", default="pgpu", choices=["pgpu", "gpu", "compute"],
                    help="pgpu: 2 GPUs (mandatory there); gpu: 1 GPU; compute: CPU only (the scorer falls back to CPU)")
    ap.add_argument("--suffix", default="", help="output <ds>_seed<N><suffix>.json (a second pass for late methods)")
    a = ap.parse_args()
    sel = {(r["dataset"], r["method"]): r["new_config"] for r in csv.DictReader(open(a.selected))}
    os.makedirs(OUTD, exist_ok=True)
    per_ds = {}
    for ds, cat in CAT.items():
        if a.datasets and ds not in a.datasets:
            continue
        if not any(k[0] == ds and k[1] in a.methods for k in sel):
            print(f"{ds}: no selected method yet, skip")
            continue
        for sd in SEEDS:
            out = f"{OUTD}/{ds}_seed{sd}{a.suffix}.json"
            sfx = "" if sd == 2 else f"_seed{sd}"
            ck = {"original": start_ckpt(ds, sd), "retrained": retrained(ds, sd)}
            for m, k in KEY.items():
                if (ds, m) in sel and m in a.methods:
                    ck[k] = run_ckpt(f"sens_{ds}_{cat}rm_{sel[(ds, m)]}_rmsplit_canonck{sfx}")
            missing = [k for k, v in ck.items() if not v]
            status = "done" if os.path.exists(out) else ("ready" if not missing else "waiting")
            print(f"{ds:7s} s{sd:<2d} {status:8s} models {sorted(ck)} missing {missing}")
            if status != "ready":
                continue
            sets = f"logs/diagnose/mia_unwanted/sets/{ds}_s{sd}"
            per_ds.setdefault(ds, []).append(
                ["python", "-u", "-m", "src.measure_membership", "experiment=tiger_unlearn_scif_flat",
                 f"data_dir=src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}",
                 f"semantic_id_path=embeddings/{ds}/merged_predictions_tensor.pt",
                 f"\"ckpt_path='{ck['original']}'\"", "num_hierarchies=4", f"seed={sd}",
                 f"+unlearning.js_dataset={ds}", f"+unlearning.mia_member_dir={sets}/members",
                 f"+unlearning.mia_nonmember_dir={sets}/nonmembers", f"+unlearning.scope_output={out}",
                 *[f"\"+unlearning.js_{k}_ckpt='{v}'\"" for k, v in ck.items()],
                 *(["trainer.accelerator=cpu", "trainer.devices=1", "trainer.strategy=auto"]
                   if (a.local or a.partition == "compute") else []),
                 f"hydra.run.dir={OUTD}/hydra/{ds}_s{sd}{a.suffix}", f"> {OUTD}/{ds}_s{sd}{a.suffix}.log 2>&1"])
    if a.local:
        import time
        for ds, cmds in per_ds.items():
            for c in cmds:
                sd = re.search(r"seed=(\d+)", " ".join(c)).group(1)
                if a.cells and sd not in a.cells:
                    continue
                t0 = time.time()
                r = subprocess.run("cd /sc-scratch/sc-scratch-ikim-erase/GRID; " + " ".join(c), shell=True)
                print(f"local {ds} s{sd}: exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)
        return
    for ds, cmds in per_ds.items():
        wrap = "cd /sc-scratch/sc-scratch-ikim-erase/GRID; " + " ; ".join(" ".join(c) for c in cmds)
        part = {"pgpu": ["--partition=pgpu", "--gres=gpu:2", "--exclude=s-sc-pgpu01,s-sc-pgpu02", "--time=04:00:00", "--cpus-per-task=8"],
                "gpu": ["--partition=gpu", "--gres=gpu:1", "--time=04:00:00", "--cpus-per-task=8"],
                "compute": ["--partition=compute", "--time=12:00:00", "--cpus-per-task=16"]}[a.partition]
        sb = ["sbatch", "--parsable", "--job-name", f"miarm_{ds}{a.suffix}", *part,
              "--mem=64G", "--output=logs/%j.out", "--wrap", wrap]
        if a.submit:
            jid = subprocess.run(sb, check=True, capture_output=True, text=True).stdout.strip()
            print(f"submitted miarm_{ds}: {jid} ({len(cmds)} cells)")
        else:
            print("DRY:", ds, len(cmds), "cells")


if __name__ == "__main__":
    main()
