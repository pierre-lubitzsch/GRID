"""Membership inference on the deleted interactions for the OTHER SID tokenizers (RQ-VAE, LETTER, tracerv3), 2026-11-25.

Same attack and member / non-member sets as scripts/launch_mia_rmgrid.py (logs/diagnose/mia_unwanted/sets/<ds>_s<N>,
tokenizer independent: interaction data, the tokenizer enters through semantic_id_path), scored with each tokenizer's
own models: the model before deletion and the retrained model (the checkpoints the cleaned-view references evaluated,
read from their coordinates.json), and the configuration SELECTED per tokenizer on cleaned validation
(othertok_tables.select, TUNED=1) for every method. LETTER uses experiment=letter_unlearn_scif_flat.

    python scripts/launch_mia_othertok.py --track rqvae                       # dry run: prints cells and models
    python scripts/launch_mia_othertok.py --track rqvae --submit              # one pgpu job (2 GPUs) per dataset
    python scripts/launch_mia_othertok.py --track rqvae --local --cells 2     # run in this shell (CPU fallback)
Output: logs/diagnose/mia_unwanted_othertok/<track>_<ds>_seed<N>.json
"""
import argparse
import importlib.util
import json
import os
import re
import subprocess
import time

os.environ["TUNED"] = "1"
HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location(
    "T", os.path.join(HERE, "..", "paper", "plotting", "critique_followup", "othertok_tables.py"))
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)

KEY = {"Finetune": "finetune", "Forget only": "forget_only", "Forget+Repair": "forget_repair", "SCIF": "scif",
       "SEIF": "seif", "Kookmin": "kookmin", "Fanchuan": "fanchuan", "TRACER": "tracer", "NAU": "nau"}
EXPERIMENT = {"rqvae": "tiger_unlearn_scif_flat", "letter": "letter_unlearn_scif_flat", "tracerv3": "tiger_unlearn_scif_flat"}
SID = {"rqvae": "embeddings/{ds}_rqvae/merged_predictions_tensor.pt", "letter": "embeddings/{ds}_letter/merged_predictions_tensor.pt",
       "tracerv3": "embeddings/beauty_tracer_v3/merged_predictions_tensor.pt"}
OUTD = "logs/diagnose/mia_unwanted_othertok"


def ref_ckpt(track, ds, role, sd):
    tok = T.TRACKS[track][1]
    d = f"{T.R.ROOT}/scrub/reference/v1/{ds}_{role}_{tok}" + ("" if sd == 2 else f"_seed{sd}")
    ck = json.load(open(f"{d}/coordinates.json"))["ckpt"]
    assert os.path.isfile(ck), ck
    return ck


def sel_ckpt(track, ds, arm, sd):
    import glob
    tg = T.tag(T.TRACKS[track][0], ds, arm, sd)
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tg}") if os.path.exists(f"{d}/checkpoints/unlearned.ckpt")]
    if not c:
        return None
    d = max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))
    return f"{d}/checkpoints/unlearned.ckpt"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track", required=True, choices=list(T.TRACKS))
    ap.add_argument("--submit", action="store_true")
    ap.add_argument("--local", action="store_true")
    ap.add_argument("--cells", nargs="+", default=None, help="seeds to run (default: all)")
    ap.add_argument("--datasets", nargs="+", default=None)
    a = ap.parse_args()
    track = a.track
    picks = T.select(track)
    os.makedirs(OUTD, exist_ok=True)
    per_ds = {}
    for ds in T.TRACKS[track][4]:
        if a.datasets and ds not in a.datasets:
            continue
        for sd in T.SEEDS:
            if a.cells and str(sd) not in a.cells:
                continue
            out = f"{OUTD}/{track}_{ds}_seed{sd}.json"
            if os.path.exists(out):
                print(f"{ds} s{sd}: done already"); continue
            ck = {"original": ref_ckpt(track, ds, "base", sd), "retrained": ref_ckpt(track, ds, "retrain", sd)}
            missing = []
            for m, k in KEY.items():
                if (ds, m) in picks:
                    c = sel_ckpt(track, ds, picks[(ds, m)], sd)
                    if c and os.path.isfile(os.path.realpath(c)):
                        ck[k] = c
                    else:
                        missing.append(f"{m}:{picks[(ds, m)]}")
            print(f"{ds} s{sd}: models {sorted(ck)} missing {missing}")
            sets = f"logs/diagnose/mia_unwanted/sets/{ds}_s{sd}"
            assert os.path.isdir(f"{sets}/members") and os.path.isdir(f"{sets}/nonmembers"), sets
            cmd = ["python", "-u", "-m", "src.measure_membership", f"experiment={EXPERIMENT[track]}",
                   f"data_dir=src/data/amazon_data/{ds}_sens_{T.CAT[ds]}_1e-4_seed{sd}",
                   f"semantic_id_path={SID[track].format(ds=ds)}", f"\"ckpt_path='{ck['original']}'\"",
                   "num_hierarchies=4", f"seed={sd}", f"+unlearning.js_dataset={ds}",
                   f"+unlearning.mia_member_dir={sets}/members", f"+unlearning.mia_nonmember_dir={sets}/nonmembers",
                   f"+unlearning.scope_output={out}",
                   *[f"\"+unlearning.js_{k}_ckpt='{v}'\"" for k, v in ck.items()],
                   *(["trainer.accelerator=cpu", "trainer.devices=1", "trainer.strategy=auto"] if a.local else []),
                   f"hydra.run.dir={OUTD}/hydra/{track}_{ds}_s{sd}", f"> {OUTD}/{track}_{ds}_s{sd}.log 2>&1"]
            per_ds.setdefault(ds, []).append((sd, cmd))
    for ds, cmds in per_ds.items():
        prefix = "cd /sc-scratch/sc-scratch-ikim-erase/GRID; " + ("export MODEL=letter; " if track == "letter" else "")
        if a.local:
            for sd, c in cmds:
                t0 = time.time()
                r = subprocess.run(prefix + " ".join(c), shell=True)
                print(f"local {track} {ds} s{sd}: exit {r.returncode}, {time.time() - t0:.0f} s", flush=True)
        elif a.submit:
            wrap = prefix.rstrip() + " " + " ; ".join(" ".join(c) for _, c in cmds)
            jid = subprocess.run(["sbatch", "--parsable", "--job-name", f"miaotk_{track}_{ds}", "--partition=pgpu",
                                  "--gres=gpu:2", "--exclude=s-sc-pgpu01,s-sc-pgpu02", "--time=06:00:00",
                                  "--cpus-per-task=8", "--mem=64G", "--output=logs/%j.out", "--wrap", wrap],
                                 check=True, capture_output=True, text=True).stdout.strip()
            print(f"submitted miaotk_{track}_{ds}: {jid} ({len(cmds)} cells)")
        else:
            print("DRY:", ds, len(cmds), "cells")


if __name__ == "__main__":
    main()
