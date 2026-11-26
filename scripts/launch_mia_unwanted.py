"""Submit the unwanted-item membership-inference scoring, one job per (dataset, seed).

Scores the original model, the retrained reference (both from the newest
logs/eval/sensitive_reference/*/<tok>/coordinates.json), and the unlearned models of the
`_mia` re-runs (logs/mia_unwanted_plan.tsv, checkpoints kept) on the sets from
scripts/make_mia_unwanted_sets.py. Output: logs/diagnose/mia_unwanted/<ds>_seed<seed>.json

    python scripts/launch_mia_unwanted.py            # dry run: list checkpoints per cell
    python scripts/launch_mia_unwanted.py --submit   # submit cells whose checkpoints are all present
    python scripts/launch_mia_unwanted.py --run      # score ready cells in this process (inside a job)

--original-lastckpt (with --submit/--run): score ONLY the original model, at the exact checkpoint the
`_mia` runs start from (logs/eval/sensitive_reference_lastckpt/v1/<ds>_base[_seedN]/coordinates.json,
i.e. last.ckpt; Beauty seed 2 keeps its epoch checkpoint, which is already the start), into
logs/diagnose/mia_unwanted_lastckpt/<ds>_seed<seed>.json. Read by mia_unwanted_table.py --original-root.
"""
import argparse
import glob
import json
import os
import re
import subprocess

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
METHOD = [("nau", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"), ("finetune", "unified_abl_lf0.0_ls0.0_ln0.0"),
          ("forget_only", "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0"), ("forget_repair", "unified_abl_lf0.1_ls0.0_ln0.0"),
          ("scif", "scif_"), ("seif", "seif_"), ("kookmin", "kookmin_"), ("fanchuan", "fanchuan_"), ("tracer", "tracer_")]


def reference(ds, role, sd):
    tok = f"{ds}_{role}" + ("" if sd == 2 else f"_seed{sd}")
    c = sorted(glob.glob(f"logs/eval/sensitive_reference/*/{tok}/coordinates.json"), key=os.path.getmtime)
    if not c:
        return None
    return json.load(open(c[-1]))["ckpt"]


def unlearned(ds, sd):
    """method -> unlearned checkpoint of the cell's `_mia` run (None while missing)."""
    out = {}
    sfx = "" if sd == 2 else f"_seed{sd}"
    for name, prefix in METHOD:
        runs = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_sens_{ds}_{CAT[ds]}_{prefix}*_mia{sfx}")
                if re.search(rf"_bs1_sens_{ds}_{CAT[ds]}_{re.escape(prefix)}[\w.]*_mia{sfx}$", d)]
        if prefix.endswith("_") is False:
            runs = [d for d in runs if d.endswith(f"_{prefix}_mia{sfx}")]
        if not runs:
            continue
        d = max(runs, key=lambda x: int(re.search(r"job(\d+)_", x).group(1)))
        ck = os.path.join(d, "checkpoints", "unlearned.ckpt")
        out[name] = ck if os.path.exists(ck) else None
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--submit", action="store_true", help="submit one job per ready cell")
    ap.add_argument("--run", action="store_true", help="score every ready cell here, one after another")
    ap.add_argument("--original-lastckpt", action="store_true", help="score only the original model at its start ckpt")
    ap.add_argument("--anchor", action="store_true",
                    help="score the retain-side anchor runs (lambda_a 0.1 and 1, `_canonck`) and the retrained "
                         "reference on the same sets, into logs/diagnose/mia_unwanted_anchor/")
    a = ap.parse_args()
    if a.original_lastckpt:
        return original_lastckpt(a)
    if a.anchor:
        return anchor(a)
    for ds in CAT:
        for sd in SEEDS:
            sets = f"logs/diagnose/mia_unwanted/sets/{ds}_s{sd}"
            out = f"logs/diagnose/mia_unwanted/{ds}_seed{sd}.json"
            ck = {"original": reference(ds, "base", sd), "retrained": reference(ds, "retrain", sd), **unlearned(ds, sd)}
            missing = [k for k, v in ck.items() if not v]
            expected = 9 if ds != "beauty" else 8   # no TRACER on Beauty
            status = "done" if os.path.exists(out) else ("ready" if not missing and len(ck) - 2 >= expected else "waiting")
            print(f"{ds:7s} s{sd:<2d} {status:8s} have {sorted(k for k, v in ck.items() if v)} missing {missing}")
            if not (a.submit or a.run) or status != "ready":
                continue
            sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
            cmd = ["python", "-u", "-m", "src.measure_membership", "experiment=tiger_unlearn_scif_flat",
                   f"data_dir={sens}", f"semantic_id_path=embeddings/{ds}/merged_predictions_tensor.pt",
                   f"ckpt_path='{ck['original']}'", "num_hierarchies=4", f"seed={sd}",
                   f"+unlearning.js_dataset={ds}", f"+unlearning.mia_member_dir={sets}/members",
                   f"+unlearning.mia_nonmember_dir={sets}/nonmembers", f"+unlearning.scope_output={out}",
                   *[f"+unlearning.js_{k}_ckpt='{v}'" for k, v in ck.items()],
                   f"hydra.run.dir=logs/diagnose/mia_unwanted/hydra/{ds}_s{sd}"]
            if a.run:
                subprocess.run(cmd, check=False)
                continue
            wrap = " ".join(cmd)
            subprocess.run(["sbatch", "--job-name", f"miau_{ds}_{sd}", "--partition=pgpu", "--gres=gpu:2",
                            "--exclude=s-sc-pgpu01,s-sc-pgpu02", "--time=00:45:00", "--cpus-per-task=8", "--mem=64G",
                            "--output=logs/%j.out", "--wrap", wrap], check=True)


def original_lastckpt(a):
    cmds = []
    for ds in CAT:
        for sd in SEEDS:
            sets = f"logs/diagnose/mia_unwanted/sets/{ds}_s{sd}"
            out = f"logs/diagnose/mia_unwanted_lastckpt/{ds}_seed{sd}.json"
            tok = f"{ds}_base" + ("" if sd == 2 else f"_seed{sd}")
            f = f"logs/eval/sensitive_reference_lastckpt/v1/{tok}/coordinates.json"
            ck = json.load(open(f))["ckpt"] if os.path.exists(f) else reference(ds, "base", sd)   # Beauty s2: start = old ref
            status = "done" if os.path.exists(out) else "ready"
            print(f"{ds:7s} s{sd:<2d} {status:6s} original={ck}")
            if status != "ready":
                continue
            sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
            cmds.append(["python", "-u", "-m", "src.measure_membership", "experiment=tiger_unlearn_scif_flat",
                         f"data_dir={sens}", f"semantic_id_path=embeddings/{ds}/merged_predictions_tensor.pt",
                         f"ckpt_path='{ck}'", "num_hierarchies=4", f"seed={sd}",
                         f"+unlearning.js_dataset={ds}", f"+unlearning.mia_member_dir={sets}/members",
                         f"+unlearning.mia_nonmember_dir={sets}/nonmembers", f"+unlearning.scope_output={out}",
                         f"+unlearning.js_original_ckpt='{ck}'",
                         f"hydra.run.dir=logs/diagnose/mia_unwanted_lastckpt/hydra/{ds}_s{sd}"])
    os.makedirs("logs/diagnose/mia_unwanted_lastckpt", exist_ok=True)
    if a.run:
        for c in cmds:
            subprocess.run(c, check=False)
    elif a.submit and cmds:
        wrap = " ; ".join(" ".join(c) for c in cmds)
        subprocess.run(["sbatch", "--job-name", "miau_origlast", "--partition=gpu", "--gpus-per-node=1",
                        "--time=03:00:00", "--cpus-per-task=8", "--mem=64G",
                        "--output=logs/%j.out", "--wrap", wrap], check=True)


def anchor(a):
    """Retain-side anchor runs (opt-in variant of NAU for sensitive item deletion), same member and
    non-member sets as the main table, so the AUCs pair with logs/diagnose/mia_unwanted/."""
    per_ds = {}
    for ds in CAT:
        for sd in SEEDS:
            sets = f"logs/diagnose/mia_unwanted/sets/{ds}_s{sd}"
            out = f"logs/diagnose/mia_unwanted_anchor/{ds}_seed{sd}.json"
            sfx = "" if sd == 2 else f"_seed{sd}"
            ck = {"retrained": reference(ds, "retrain", sd)}
            for lab, tag in (("anchor0p1", "anchor0p1"), ("anchor1p0", "anchor1p0")):
                runs = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_sens_{ds}_{CAT[ds]}_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_{tag}_canonck{sfx}")
                        if d.endswith(f"_{tag}_canonck{sfx}")]
                d = max(runs, key=lambda x: int(re.search(r"job(\d+)_", x).group(1))) if runs else None
                c = os.path.join(d, "checkpoints", "unlearned.ckpt") if d else None
                ck[lab] = c if c and os.path.exists(c) else None
            missing = [k for k, v in ck.items() if not v]
            status = "done" if os.path.exists(out) else ("ready" if not missing else "waiting")
            print(f"{ds:7s} s{sd:<2d} {status:8s} missing {missing}")
            if status != "ready":
                continue
            sens = f"src/data/amazon_data/{ds}_sens_{CAT[ds]}_1e-4_seed{sd}"
            per_ds.setdefault(ds, []).append(
                ["python", "-u", "-m", "src.measure_membership", "experiment=tiger_unlearn_scif_flat",
                 f"data_dir={sens}", f"semantic_id_path=embeddings/{ds}/merged_predictions_tensor.pt",
                 f"ckpt_path='{ck['retrained']}'", "num_hierarchies=4", f"seed={sd}",
                 f"+unlearning.js_dataset={ds}", f"+unlearning.mia_member_dir={sets}/members",
                 f"+unlearning.mia_nonmember_dir={sets}/nonmembers", f"+unlearning.scope_output={out}",
                 *[f"+unlearning.js_{k}_ckpt='{v}'" for k, v in ck.items()],
                 f"hydra.run.dir=logs/diagnose/mia_unwanted_anchor/hydra/{ds}_s{sd}"])
    os.makedirs("logs/diagnose/mia_unwanted_anchor", exist_ok=True)
    for ds, cmds in per_ds.items():
        if a.run:
            for c in cmds:
                subprocess.run(c, check=False)
        elif a.submit:
            wrap = " ; ".join(" ".join(c) for c in cmds)
            subprocess.run(["sbatch", "--job-name", f"miau_anchor_{ds}", "--partition=gpu", "--gpus-per-node=1",
                            "--time=03:00:00", "--cpus-per-task=8", "--mem=64G",
                            "--output=logs/%j.out", "--wrap", wrap], check=True)


if __name__ == "__main__":
    main()
