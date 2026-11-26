"""Plan for the baseline tuning grids on Beauty validation (reviewer question: are the
baselines under-tuned relative to NAU's 9-setting grid?).

Every baseline gets a 3 x 3 grid over its two main knobs, centered on its current setting,
on all 15 Beauty cells (3 targets x 5 data seeds, TIGER + RQ-KMeans). Configurations that
already have _valsel runs are skipped. Each row replays a template run of its cell with only
the swept knobs changed:
  * families with _valsel runs: the cell's _valsel run of the current configuration;
  * Finetune and SCIF (no _valsel runs): the cell's Table 1 run from paper/data/runs.csv.
Every template must use the same poisoned checkpoint and seed as the cell's NAU _valsel run
(guards against the wrong-seed bug of logs/unlearn/runs_wrongseed/).

Output rows match logs/valsel_plan.tsv (run_valsel_batch_launch.sh / run_valsel_batch.sh).

--ext: the second round. Where a method's selected (or best-utility) setting sat on the edge
of its 3 x 3 grid, steps beyond that edge; for SCIF, the step multiplier scif_tau_scale,
because the original influence step moves the model by ~3e-5 in L2 so neither damping nor
the norm cap changed anything in round one.

    python scripts/make_valsel_basegrid_plan.py > logs/valsel_basegrid_plan.tsv
    python scripts/make_valsel_basegrid_plan.py --ext > logs/valsel_basegrid_ext_plan.tsv
    python scripts/make_valsel_basegrid_plan.py --ext2 > logs/valsel_basegrid_ext2_plan.tsv   (Fanchuan lr gap)
"""
import csv
import glob
import itertools
import os
import re
import sys

import yaml

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

SEEDS = ("2", "3", "5", "7", "11"); STRATA = ("unpopular", "mid", "popular")
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")
EP = ("2", "4", "8")


def tag_ab(prefix, lf, ep, suffix=""):
    return f"{prefix}_lf{lf}_ls0.0_ln0.0" + ("" if ep == "4" else f"_ep{ep}") + suffix


# family -> (template arm, [(arm, {override key: value})])
GRIDS = {
    "finetune": ("unified_abl_lf0.0_ls0.0_ln0.0",
                 [(f"unified_abl_lf0.0_ls0.0_ln0.0_ulr{lr}_ep{ep}", {"unified_lr": lr, "n_epochs": ep})
                  for lr, ep in itertools.product(("0.00001", "0.0001", "0.001"), EP)]),
    "negtrain": ("unified_abl_lf0.1_ls0.0_ln0.0_valsel",
                 [(tag_ab("unified_abl", lf, ep), {"lambda_f": lf, "n_epochs": ep})
                  for lf, ep in itertools.product(("0.01", "0.1", "1.0"), EP)]),
    "ga": ("unified_abl_lf1.0_ls0.0_ln0.0_lr0.0_valsel",
           [(tag_ab("unified_abl", lf, ep, "_lr0.0"), {"lambda_f": lf, "n_epochs": ep})
            for lf, ep in itertools.product(("0.01", "0.1", "1.0"), EP)]),
    "scif": ("scif",
             [(f"scif_d{d}_n{n}", {"damping": d, "update_max_norm": n})
              for d, n in itertools.product(("0.001", "0.01", "0.1"), ("0.5", "1.0", "5.0"))]),
    "seif": ("seif_std0.06_valsel",
             [(f"seif_std{s}" + ("" if r == "0.0007" else f"_rlr{r}"), {"seif_erase_std": s, "seif_repair_lr": r})
              for s, r in itertools.product(("0.006", "0.06", "0.6"), ("0.00007", "0.0007", "0.007"))]),
    "kookmin": ("kookmin_ir0.001_valsel",
                [(f"kookmin_ir{i}" + ("" if r == "0.001" else f"_rlr{r}"), {"kookmin_init_rate": i, "kookmin_retain_lr": r})
                 for i, r in itertools.product(("0.0001", "0.001", "0.01"), ("0.0001", "0.001", "0.01"))]),
    "fanchuan": ("fanchuan_t0.07_valsel",
                 [(f"fanchuan_t{t}" + ("" if lr == "0.001" else f"_lr{lr}"), {"fanchuan_contrastive_temperature": t, "fanchuan_lr": lr})
                  for t, lr in itertools.product(("0.007", "0.07", "1.15"), ("0.0001", "0.001", "0.01"))]),
}

GRIDS_EXT = {
    "finetune": (GRIDS["finetune"][0],
                 [(f"unified_abl_lf0.0_ls0.0_ln0.0_ulr0.0003_ep{ep}", {"unified_lr": "0.0003", "n_epochs": ep}) for ep in ("2", "4")]),
    "negtrain": (GRIDS["negtrain"][0],
                 [(tag_ab("unified_abl", lf, ep), {"lambda_f": lf, "n_epochs": ep})
                  for lf, ep in (("0.03", "4"), ("0.03", "8"), ("0.3", "2"), ("0.3", "4"))]),
    "ga": (GRIDS["ga"][0],
           [(tag_ab("unified_abl", lf, "1", "_lr0.0"), {"lambda_f": lf, "n_epochs": "1"}) for lf in ("0.01", "0.1", "1.0")]),
    "scif": (GRIDS["scif"][0],
             [(f"scif_d{d}_n1.0_ts{t}", {"damping": d, "update_max_norm": "1.0", "scif_tau_scale": t})
              for d, t in itertools.product(("0.001", "0.01", "0.1"), ("10", "100", "1000"))]),
    "seif": (GRIDS["seif"][0],
             [(f"seif_std{sd}_rlr{r}", {"seif_erase_std": sd, "seif_repair_lr": r})
              for sd, r in (("0.0006", "0.000007"), ("0.0006", "0.00002"), ("0.0006", "0.00007"),
                            ("0.006", "0.000007"), ("0.006", "0.00002"))]),
    "kookmin": (GRIDS["kookmin"][0],
                [(f"kookmin_ir{i}_rlr{r}", {"kookmin_init_rate": i, "kookmin_retain_lr": r})
                 for i, r in itertools.product(("0.0001", "0.001", "0.01"), ("0.00001", "0.00003"))]),
    "fanchuan": (GRIDS["fanchuan"][0],
                 [(f"fanchuan_t{t}_lr{lr}", {"fanchuan_contrastive_temperature": t, "fanchuan_lr": lr})
                  for t, lr in (("1.15", "0.00001"), ("1.15", "0.00003"), ("5.0", "0.00001"), ("5.0", "0.00003"), ("5.0", "0.0001"))]),
}

# Third round: Fanchuan between lr 3e-5 (meets the bar, Delta 0.19) and 1e-4 (tau 93, Delta 0.075).
GRIDS_EXT2 = {
    "fanchuan": (GRIDS["fanchuan"][0],
                 [(f"fanchuan_t{t}_lr{lr}", {"fanchuan_contrastive_temperature": t, "fanchuan_lr": lr})
                  for t, lr in itertools.product(("1.15", "5.0"), ("0.00005", "0.00007"))]),
}


def overrides(path):
    ov = yaml.safe_load(open(os.path.join(path, ".hydra", "overrides.yaml")))
    return ov, dict(o.split("=", 1) for o in ov)


def run_dir(arm, st, sd):
    sfx = "" if sd == "2" else f"_seed{sd}"
    c = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_bw_tgt{st}_{arm}{sfx}") if os.path.isdir(d)]
    return max(c, key=lambda d: int(re.search(r"job(\d+)_", d).group(1))) if c else None


def main():
    grids = GRIDS_EXT2 if "--ext2" in sys.argv[1:] else GRIDS_EXT if "--ext" in sys.argv[1:] else GRIDS
    table1 = {}
    for r in csv.DictReader(open("paper/data/runs.csv")):
        if r["setting"] == "spam" and r["dataset"] == "beauty" and r["track"] == "tiger+rkmeans" and r["done"] == "True" \
                and not r["pkm"] and not r.get("replicate") and os.path.isdir(r["path"]):
            k = (r["arm"], r["stratum"], str(r["seed"]))
            if k not in table1 or int(r["jid"]) > int(table1[k]["jid"]):
                table1[k] = r
    existing = {os.path.basename(d).split("_bs1_bw_tgt", 1)[1] for d in glob.glob("logs/unlearn/runs/*_bs1_bw_tgt*_valsel*")}
    rows, n_new = [], {}
    for fam, (tmpl_arm, grid) in grids.items():
        for st in STRATA:
            for sd in SEEDS:
                sfx = "" if sd == "2" else f"_seed{sd}"
                ref = run_dir("valsel_nau_ls0p01_ln0p01", st, sd)
                assert ref, ("no NAU valsel run", st, sd)
                _, rkv = overrides(ref)
                if tmpl_arm.endswith("_valsel"):
                    tmpl = run_dir(tmpl_arm, st, sd)
                else:
                    r = table1.get((tmpl_arm, st, sd))
                    tmpl = r["path"] if r else None
                assert tmpl, (fam, st, sd, "no template")
                ov, kv = overrides(tmpl)
                assert kv["ckpt_path"].strip("'\"") == rkv["ckpt_path"].strip("'\""), (fam, st, sd, kv["ckpt_path"], rkv["ckpt_path"])
                assert kv["data_dir"] == rkv["data_dir"] and kv["seed"] == rkv["seed"] == sd, (fam, st, sd)
                ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
                assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
                base = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)
                        and not o.startswith("unlearning_run_tag")]
                for arm, knobs in grid:
                    tag = f"bw_tgt{st}_{arm}_valsel{sfx}"
                    if f"{st}_{arm}_valsel{sfx}" in existing:
                        continue
                    extra = [o for o in base if o.split("=", 1)[0][len("unlearning."):] not in knobs]
                    extra += [f"unlearning.{k}={v}" for k, v in knobs.items()]
                    fam_chk = run_tags.family_of(arm)
                    assert fam_chk.startswith({"finetune": "finetune", "negtrain": "negtrain", "ga": "ga"}.get(fam, fam)), (arm, fam_chk)
                    rows.append(["beauty", st, sd, sd, tag, ckpt, kv["unlearning.algorithm"], sid,
                                 kv.get("unlearning.neighborhood_aware", "false"), kv.get("unlearning.request_batch_size", "1"),
                                 kv.get("unlearning.neighborhood_aware_sample_rate", "0.0"), kv["data_dir"], tmpl, " ".join(extra)])
                    n_new[fam] = n_new.get(fam, 0) + 1
    for r in rows:
        print("\t".join(r))
    print("new runs per family:", n_new, "total", len(rows), file=sys.stderr)


if __name__ == "__main__":
    main()
