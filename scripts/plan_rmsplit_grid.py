"""Plan: the sensitive-deletion hyperparameter grid with the RETRAINING-MATCHED split, in two waves.

Decisions 2026-11-14 (reviewer QzxM follow-up 2):
  * main protocol for sensitive item deletion = retraining-matched forget/retain split
    (unlearning.split_mode=retrain_matched): forget set = only the training rows whose label is a
    deleted interaction; retain set = every user, the requesting users with exactly the deleted
    items removed (= the retrained reference's training data); selection on CLEANED validation;
  * Finetune, Forget only and Forget+Repair are run as ablations of NAU's own configuration
    (NAU's recorded lambda_n=0 run as template, incl. its unlearning batch size 32), with the unused
    terms set to 0: Finetune lambda_f=0 (one config); Forget only lambda_r=0, lambda_f in
    {0.01, 0.1, 1.0}; Forget+Repair lambda_f in {0.01, 0.1, 1.0}, where lambda_f=0.1 IS NAU's
    lambda_n=0 run (not run twice);
  * NAU: lambda_n in {0, -0.1, -1, -10} x lambda_s in {0, 0.1}; lambda_s=0.1 uses the cosine
    separation score (the generative score runs out of memory with a whole category as targets,
    job 10649140 on a 140 GB H200).
Wave 1 (most important): NAU grid, Finetune, Forget only. Wave 2: Forget+Repair lambda_f 0.01 / 1.0
and the baselines with their own grids (SCIF, SEIF, Kookmin, Fanchuan, TRACER).

Each row replays the recorded execution of its template (config, cell) EXACTLY (start ckpt, SID,
unlearning seed, data dir, every override after unlearning_run_tag) -- the canonical `_mia` run if one
exists, else the latest finished `_valsens` run -- with
    unlearning.include_context_rows=true -> false
    + unlearning.split_mode=retrain_matched + unlearning.split_retain_dir=<retrain dir>/training
plus the variant's own override changes. Tag: sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN]
(rows whose run dir exists are skipped by the launcher, e.g. NAU lambda_n=-1, lambda_s=0).

    python scripts/plan_rmsplit_grid.py  -> logs/rmsplit_sens/plan_grid_wave{1,2}.tsv
"""
import csv
import glob
import json
import os
import re

import yaml

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
NAU = "unified_hp2_sepgen_lf0.1_ls0.0_ln{}"
LNS = ("0.0", "neg0.1", "neg1.0", "neg10.0")
LN_VAL = {"0.0": "0.0", "neg0.1": "-0.1", "neg1.0": "-1.0", "neg10.0": "-10.0"}

SEP_LS01 = os.environ.get("SEP_LS01", "generative")   # lambda_s=0.1 score: generative | cosine
# arm -> (template arm, {override: new value or None to drop}, wave)
VARIANTS = {}
for ln in LNS:
    VARIANTS[NAU.format(ln)] = (NAU.format(ln), {}, 1)
    if SEP_LS01 == "generative":   # same separation score as everywhere else in the paper
        # sep_checkpoint: identical loss and gradients, peak memory 43.7 -> 1.1 GiB at 95 negatives
        # (scripts/check_sep_checkpoint.py); needed with a whole category as separation negatives
        VARIANTS[f"unified_hp2_sepgen_lf0.1_ls0.1_ln{ln}_sepck"] = (
            NAU.format(ln), {"unlearning.lambda_s": "0.1", "unlearning.sep_checkpoint": "true"}, 1)
        # lambda_s=0.01 added 2026-11-15 (user request)
        VARIANTS[f"unified_hp2_sepgen_lf0.1_ls0.01_ln{ln}_sepck"] = (
            NAU.format(ln), {"unlearning.lambda_s": "0.01", "unlearning.sep_checkpoint": "true"}, 1)
    else:                          # fallback if the generative score does not fit in memory
        VARIANTS[f"unified_hp2_sepcos_lf0.1_ls0.1_ln{ln}"] = (
            NAU.format(ln), {"unlearning.lambda_s": "0.1", "unlearning.sep_loss_type": "cosine",
                             "unlearning.sep_gen_temperature": None}, 1)
VARIANTS["unified_hp2_ft_lf0.0_ls0.0_ln0.0"] = (NAU.format("0.0"), {"unlearning.lambda_f": "0.0"}, 1)
for lf in ("0.01", "0.1", "1.0"):
    VARIANTS[f"unified_hp2_fo_lf{lf}_ls0.0_ln0.0_lr0.0"] = (
        NAU.format("0.0"), {"unlearning.lambda_f": lf, "unlearning.lambda_r": "0.0"}, 1)
for lf in ("0.01", "1.0"):                      # lambda_f=0.1 is NAU's lambda_n=0 run itself
    VARIANTS[f"unified_hp2_fr_lf{lf}_ls0.0_ln0.0"] = (NAU.format("0.0"), {"unlearning.lambda_f": lf}, 2)


def finished(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
         if os.path.isfile(os.path.join(d, "eval/csv/version_0/metrics.csv"))]
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))) if c else None


def main():
    arms = {}
    for r in csv.DictReader(open("paper/plotting/valsens_candidates_ledger.csv")):
        if r["method"] != "NAU":                      # baselines with their own grids: wave 2
            arms[(r["dataset"], r["config"])] = (r["config"], {}, 2)
    for ds in CAT:
        for arm, v in VARIANTS.items():
            arms[(ds, arm)] = v
    out = {1: [], 2: []}
    miss = []
    for (ds, arm), (src, change, wave) in sorted(arms.items()):
        cat = CAT[ds]
        for sd in SEEDS:
            sfx = "" if sd == 2 else f"_seed{sd}"
            tmpl = finished(f"sens_{ds}_{cat}_{src}_mia{sfx}") or finished(f"sens_{ds}_{cat}_{src}_valsens{sfx}")
            if tmpl is None:
                miss.append((ds, arm, sd)); continue
            ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
            kv = dict(o.split("=", 1) for o in ov)
            head = [o.split("=", 1)[0] for o in ov[:len(FIXED)]]
            assert tuple(head) == FIXED, (tmpl, head)
            extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
            lam = dict(o.split("=", 1) for o in extra)
            sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
            man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
            assert int(kv["seed"]) == sd and kv["data_dir"] == sens and lam["spam_forget_manifest"] == man, tmpl
            assert lam["unlearning.deletion_spec"] == "item_pairs" and lam["unlearning.include_context_rows"] == "true", tmpl
            assert "unlearning.split_mode" not in lam, tmpl
            if src.startswith("unified_hp2_sepgen"):
                assert lam["unlearning.lambda_f"] == "0.1" and lam["unlearning.lambda_s"] == "0.0", tmpl
                assert lam["unlearning.lambda_n"] == LN_VAL[src.rsplit("_ln", 1)[1]], tmpl
                assert lam.get("unlearning.batch_size_per_device") == "32", tmpl
            ckpt = kv["ckpt_path"].strip("'\""); sid = kv["semantic_id_path"].strip("'\"")
            assert os.path.isfile(ckpt) and os.path.isfile(sid), (ckpt, sid)
            retr = f"src/data/amazon_data/{ds}_retrain_{cat}_1e-4_seed{sd}"
            meta = json.load(open(f"{retr}/retrain_meta.json"))
            assert os.path.realpath(meta["manifest"]) == os.path.realpath(man), retr
            assert os.path.isdir(f"{retr}/training"), retr
            new = []
            for o in extra:
                k, v = o.split("=", 1)
                if k == "unlearning.include_context_rows":
                    v = "false"
                if k in change:
                    if change[k] is None:
                        continue
                    v = change[k]
                new.append(f"{k}={v}")
            have = {o.split("=", 1)[0] for o in new}
            new += [f"{k}={v}" for k, v in change.items() if v is not None and k not in have]
            new += ["unlearning.split_mode=retrain_matched", f"unlearning.split_retain_dir={retr}/training"]
            tag = f"sens_{ds}_{cat}rm_{arm}_rmsplit_canonck{sfx}"
            out[wave].append(["tiger", ds, "sens", str(sd), kv["seed"], tag, ckpt, kv["unlearning.algorithm"], sid,
                              kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
                              kv["unlearning.neighborhood_aware_sample_rate"], sens, tmpl, " ".join(new)])
    os.makedirs("logs/rmsplit_sens", exist_ok=True)
    for w, rows in out.items():
        assert len({r[5] for r in rows}) == len(rows)
        p = f"logs/rmsplit_sens/plan_grid_wave{w}.tsv"
        with open(p, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(p, len(rows), "rows")
    if miss:
        print("NO TEMPLATE:", miss)


if __name__ == "__main__":
    main()
