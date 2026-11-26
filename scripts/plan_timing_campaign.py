"""Plan of the efficiency (E) timing campaign: replay canonical runs, time only.

E = median wall-clock seconds per unlearning request (`*_info.json` ->
requests[i].<algo>_info.wall_seconds). The unified-family timer had been moved before
the request data preparation by the val_selection request-cache refactor (fixed
2026-11-07 in tiger_unlearning_module._run_unified), so E is re-measured for EVERY
method with the fixed code, on H200 only, one run at a time (run_timing_batch.sh).

Rows (one per canonical cell of paper/plotting/canonical_runs.tsv):
  * spam, Beauty, Table 1 / Table 9 configurations (TRACER on tiger+tracerv3), 3 targets x 5 seeds;
  * unwanted items, the `_mia` configuration of every Table 3/22/23 method, 5 seeds, all datasets.
Each row replays its template run exactly: poisoned/clean ckpt, SID, seed, data dir and
every pass-through Hydra override from .hydra/overrides.yaml. Pending `_canonck` cells use
the recorded run they themselves replay (logs/canonck_missing/plan.tsv,
logs/fig1_fix/canonck_plan.tsv). Tag = template tag minus `_valsel/_keepck/_canonck/_mia`,
plus `_timing` (before `_seed<N>`); run_tags.iter_runs skips `_timing` tags.

    python scripts/plan_timing_campaign.py > logs/timing_e/plan.tsv
"""
import csv
import os
import re
import sys

import yaml

LEDGER = "paper/plotting/canonical_runs.tsv"
SPAM_T1_BEAUTY = {
    "finetune": "Finetune", "forget_only_lf1.0": "Forget only", "forget_repair_lf0.1": "Forget+Repair",
    "scif": "SCIF", "seif_std0.06": "SEIF", "kookmin_ir0.001": "Kookmin", "fanchuan_t0.07": "Fanchuan",
    "tracer_lf1.0_lc0.1_plr0.001_ep4": "TRACER", "nau_ls0.01_ln0.01": "NAU",
    "nau_ls0.1_ln0.01": "NAU (ls0.1)",
}
# Hydra keys the runner sets itself from its positional args / env.
RUNNER_KEYS = {"experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
               "unlearning.algorithm", "unlearning.neighborhood_aware",
               "unlearning.neighborhood_aware_sample_rate", "unlearning.request_batch_size",
               "unlearning_run_tag"}
EXPECT_EXP = {"unified": "tiger_unlearn_unified_sequential"}
COLS = ["setting", "dataset", "stratum", "seed", "useed", "tag", "ckpt", "algo", "sid", "na", "rbs",
        "rate", "data_dir", "template", "method", "config", "extra"]


def canonck_templates():
    out = {}
    for p in ("logs/canonck_missing/plan.tsv", "logs/fig1_fix/canonck_plan.tsv"):
        for ln in open(p):
            f = ln.rstrip("\n").split("\t")
            out[f[4]] = f[12]
    return out


def timing_tag(tag):
    m = re.match(r"^(.*?)(_seed\d+)?$", tag)
    base, sfx = m.group(1), m.group(2) or ""
    base = re.sub(r"_(valsel|keepck|canonck|mia)$", "", base)
    assert not re.search(r"_(valsel|valsens|keepck|canonck|mia)(_|$)", base), tag
    return f"{base}_timing{sfx}"


def main():
    rows = list(csv.DictReader(open(LEDGER), delimiter="\t"))
    ck = canonck_templates()
    sel = []
    for r in rows:
        if r["setting"] == "spam":
            if r["dataset"] != "beauty" or r["config"] not in SPAM_T1_BEAUTY:
                continue
            if r["config"].startswith("tracer") != (r["track"] == "tiger+tracerv3"):
                continue
            sel.append((r, SPAM_T1_BEAUTY[r["config"]]))
        elif r["setting"] == "unwanted":
            sel.append((r, r["method"]))
    order = {"spam": 0, "unwanted": 1}
    sel.sort(key=lambda x: (order[x[0]["setting"]], x[0]["dataset"], x[0]["config"], x[0]["stratum"],
                            int(x[0]["seed"])))
    print("\t".join(COLS))
    tags = set()
    for r, method in sel:
        ds, st, sd = r["dataset"], r["stratum"], int(r["seed"])
        if r["status"] == "ok" and r["run_dir"] and os.path.isdir(r["run_dir"]):
            tmpl = r["run_dir"]
        else:
            assert r["source"] == "canonck", r
            tmpl = ck[r["tag"]]
        ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
        kv = {}
        for o in ov:
            k, v = o.split("=", 1)
            kv.setdefault(k, v)
        rtag = kv["unlearning_run_tag"]
        useed = int(kv["seed"])
        assert useed == sd, (tmpl, useed, sd)
        assert kv.get("num_hierarchies", "4") == "4", tmpl
        assert not any(o.startswith(("sequence_length=", "model.")) for o in ov), tmpl
        ckpt = kv["ckpt_path"].strip("'\"")
        sid = kv["semantic_id_path"].strip("'\"")
        assert os.path.isfile(ckpt), ckpt
        assert os.path.isfile(sid), sid
        algo = kv["unlearning.algorithm"]
        assert kv["experiment"] == EXPECT_EXP.get(algo, "tiger_unlearn_scif_sequential"), (tmpl, kv["experiment"])
        extra = [o for o in ov if o.split("=", 1)[0] not in RUNNER_KEYS]
        other = {o.split("=", 1)[0] for o in extra if not o.startswith("unlearning.")}
        assert other <= {"spam_forget_manifest"}, (tmpl, other)
        assert not any(" " in o for o in extra), (tmpl, extra)
        base = os.path.basename(ckpt.split("/checkpoints/")[0])
        if r["setting"] == "spam":   # wrong-seed guard: poisoned ckpt of this (dataset, stratum, seed)
            assert f"_{ds}_poison_pct1_n1" in base, ckpt
            # unpopular is the default target strategy: some poisoned ckpts carry no _tgt token
            assert f"_tgt{st}" in base or (st == "unpopular" and "_tgt" not in base), ckpt
            assert (sd == 2 and "_seed" not in base) or f"_seed{sd}_" in base, ckpt
        else:
            assert f"_{ds}_clean" in base, ckpt
            assert (sd == 2 and "_seed" not in base) or base.endswith(f"_seed{sd}"), ckpt
            assert kv["data_dir"].endswith(f"_seed{sd}"), kv["data_dir"]
        tag = timing_tag(rtag)
        assert tag not in tags, tag
        tags.add(tag)
        print("\t".join([r["setting"], ds, st, str(sd), str(useed), tag, ckpt, algo, sid,
                         kv.get("unlearning.neighborhood_aware", "false"),
                         kv.get("unlearning.request_batch_size", "1"),
                         kv.get("unlearning.neighborhood_aware_sample_rate", "0.0"),
                         kv["data_dir"], tmpl, method, r["config"], " ".join(extra)]))
    print(f"{len(tags)} rows", file=sys.stderr)


if __name__ == "__main__":
    main()
