"""Plan: paper runs outside the selection grid, re-run with the RETRAINING-MATCHED split (2026-11-14).

  anchor    NAU lambda_n=-1 + retain-side anchor lambda_a in {0.1, 1} (Beauty/Sports/Toys x 5 seeds),
            templates: the `_canonck` anchor runs (sens_<ds>_<cat>_..._anchor{0p1,1p0}_canonck[_seedN])
  takedown  Beauty brand takedown (e.l.f.), every configuration of the brand table x 5 seeds,
            templates: the recorded takedown runs (sens_beauty_takedown_elf_<arm>[_seedN])
Each row replays its template EXACTLY (start ckpt, SID, seed, data dir, every override after
unlearning_run_tag) with only: unlearning.include_context_rows=true -> false, + split_mode=retrain_matched,
+ split_retain_dir=<the scenario's retrain dir>/training. New tags insert "rm" after the category token
(sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN]), so no existing run dir is touched.

    python scripts/plan_rmsplit_extra.py -> logs/rmsplit_sens/plan_extra_{anchor,takedown}.tsv
"""
import glob
import json
import os
import re

import yaml

FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
TAKEDOWN_ARMS = ("unified_frbase_lf0.1_ls0.01_ln0.01",   # the brand table's NAU row (F+R template + L_sep/L_nbr, sep_checkpoint)
                  "unified_abl_lf0.1_ls0.0_ln0.0",
                 "unified_abl_lf0.0_ls0.0_ln0.0", "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0",
                 "scif_d0.01", "seif_std0.6", "kookmin_ir0.01", "fanchuan_t0.007")


def finished(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
         if os.path.isfile(os.path.join(d, "eval/csv/version_0/metrics.csv"))]
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))) if c else None


def row(tmpl, tag, retr, sd):
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    head = [o.split("=", 1)[0] for o in ov[:len(FIXED)]]
    assert tuple(head) == FIXED, (tmpl, head)
    extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
    lam = dict(o.split("=", 1) for o in extra)
    assert int(kv["seed"]) == sd, tmpl
    assert lam.get("unlearning.deletion_spec") == "item_pairs" and lam.get("unlearning.include_context_rows") == "true", tmpl
    assert "unlearning.split_mode" not in lam, tmpl
    meta = json.load(open(f"{retr}/retrain_meta.json"))
    assert os.path.realpath(meta["manifest"]) == os.path.realpath(lam["spam_forget_manifest"]), (retr, tmpl)
    assert os.path.isdir(f"{retr}/training"), retr
    ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
    extra = [("unlearning.include_context_rows=false" if o == "unlearning.include_context_rows=true" else o) for o in extra]
    extra += ["unlearning.split_mode=retrain_matched", f"unlearning.split_retain_dir={retr}/training"]
    return ["tiger", kv["data_dir"].split("/")[-1].split("_")[0], "sens", str(sd), kv["seed"], tag, ckpt,
            kv["unlearning.algorithm"], sid, kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
            kv["unlearning.neighborhood_aware_sample_rate"], kv["data_dir"], tmpl, " ".join(extra)]


def main():
    plans = {"anchor": [], "takedown": []}
    miss = []
    for ds, cat in CAT.items():
        for a in ("0p1", "1p0"):
            arm = f"unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_anchor{a}"
            for sd in SEEDS:
                sfx = "" if sd == 2 else f"_seed{sd}"
                t = finished(f"sens_{ds}_{cat}_{arm}_canonck{sfx}")
                if not t:
                    miss.append(("anchor", ds, arm, sd)); continue
                plans["anchor"].append(row(t, f"sens_{ds}_{cat}rm_{arm}_rmsplit_canonck{sfx}",
                                           f"src/data/amazon_data/{ds}_retrain_{cat}_1e-4_seed{sd}", sd))
    for arm in TAKEDOWN_ARMS:
        for sd in SEEDS:
            sfx = "" if sd == 2 else f"_seed{sd}"
            t = finished(f"sens_beauty_takedown_elf_{arm}{sfx}")
            if not t:
                miss.append(("takedown", arm, sd)); continue
            plans["takedown"].append(row(t, f"sens_beauty_takedown_elfrm_{arm}_rmsplit_canonck{sfx}",
                                         f"src/data/amazon_data/beauty_retrain_takedown_elf_0_seed{sd}", sd))
    for k, rows in plans.items():
        p = f"logs/rmsplit_sens/plan_extra_{k}.tsv"
        with open(p, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(p, len(rows), "rows")
    if miss:
        print("MISSING TEMPLATES:", miss)


if __name__ == "__main__":
    main()
