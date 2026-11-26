#!/usr/bin/env python3
"""Plans for the two rebuttal variants of NAU (both opt-in config keys, default off).

VARIANT 1, reference-capped forgetting (unlearning.ref_cap_forget / ref_cap_sep),
Beauty spam, TIGER + RQ-KMeans, 5 seeds:
  capref   NAU (lf0.1/ls0.01/ln0.01, k=8 embedding, generative sep), both caps on,
           all three strata (15 rows)
  caprefF  NAU, forget cap only, popular stratum (5 rows; attributes the effect)
  capref   Forget+Repair (lf0.1), forget cap on, popular stratum (5 rows)
VARIANT 2, retain-side anchor (unlearning.lambda_a), unwanted items, NAU
  (lf0.1/ls0/ln-1), Beauty/Sports/Toys x 5 seeds x lambda_a in {0.1, 1.0} (30 rows).

Every row replays the CANONICAL run of its cell (paper/plotting/canonical_runs.tsv)
exactly: same starting checkpoint file, SID tensor, unlearning seed, data dir and every
non-positional override from the run's .hydra/overrides.yaml; only the new keys are
appended. Rows run through run_valsel_batch.sh / run_valsel_batch_sens.sh ->
run_valsel_one.sh (unlearn + test eval + validation eval) with VALSEL_KEEP_CKPT=1.

Tags: the template tag with `_keepck` (spam) / `_mia` (unwanted) replaced by
`_capref_canonck`, `_caprefF_canonck`, `_anchor0p1_canonck`, `_anchor1p0_canonck`
(seed suffix kept). None of them contains `_valsel`, `_valsens`, `_mia`, `_keepck`, and
scripts/run_tags.iter_runs skips `_capref` / `_anchor` tags, so no table reader takes
them as existing configurations.

    python scripts/plan_rebuttal_variants.py      # writes logs/rebuttal_variants/plan_*.tsv
"""
from __future__ import annotations

import csv
import os
import re
import sys

import yaml

LEDGER = "paper/plotting/canonical_runs.tsv"
OUT = "logs/rebuttal_variants"
# Set by run_tiger_unlearn_sequential.sh itself (positional arguments / fixed overrides).
RUNNER = ("experiment=", "data_dir=", "semantic_id_path=", "ckpt_path=", "num_hierarchies=",
          "seed=", "unlearning_run_tag=", "unlearning.algorithm=", "unlearning.neighborhood_aware=",
          "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=",
          "hydra.")
NEW_KEYS = ("unlearning.ref_cap_", "unlearning.lambda_a=", "unlearning.anchor_",
            "unlearning.ref_weights=")


def ledger():
    return list(csv.DictReader(open(LEDGER), delimiter="\t"))


def row_for(r, suffix_from, suffix_to, new_overrides):
    tmpl, rtag = r["run_dir"], r["tag"]
    ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
    kv = dict(o.split("=", 1) for o in ov)
    assert kv["unlearning_run_tag"] == rtag, (kv["unlearning_run_tag"], rtag)
    assert kv["experiment"] == "tiger_unlearn_unified_sequential", kv["experiment"]
    assert kv["unlearning.algorithm"] == "unified"
    assert not any(o.startswith(NEW_KEYS) for o in ov), tmpl
    useed = str(int(kv["seed"]))
    assert int(useed) == int(r["seed"]), (tmpl, useed)
    ckpt = kv["ckpt_path"].strip("'\"")
    sid = kv["semantic_id_path"].strip("'\"")
    assert os.path.isfile(ckpt), ckpt
    assert os.path.isfile(sid), sid
    assert kv.get("num_hierarchies", "4") == "4"
    na = kv.get("unlearning.neighborhood_aware", "false")
    rbs = kv.get("unlearning.request_batch_size", "1")
    rate = kv.get("unlearning.neighborhood_aware_sample_rate", "0.0")
    extra = [o for o in ov if not o.startswith(RUNNER)] + list(new_overrides)
    m = re.match(rf"^(.*){re.escape(suffix_from)}(_seed\d+)?$", rtag)
    assert m, (rtag, suffix_from)
    tag = m.group(1) + suffix_to + (m.group(2) or "")
    for bad in ("_valsel", "_valsens", "_mia", "_keepck", "_timing"):
        assert bad not in tag, tag
    ds = r["dataset"]
    st = r["stratum"] if r["setting"] == "spam" else "sens"
    return [ds, st, str(int(r["seed"])), useed, tag, ckpt, "unified", sid, na, rbs, rate,
            kv["data_dir"], tmpl, " ".join(extra)]


def main():
    os.makedirs(OUT, exist_ok=True)
    L = ledger()
    spam, sens = [], []
    seeds = (2, 3, 5, 7, 11)
    for r in L:
        if not (r["setting"] == "spam" and r["dataset"] == "beauty" and r["track"] == "tiger+rkmeans"):
            continue
        if r["config"] not in ("nau_ls0.01_ln0.01", "forget_repair_lf0.1"):
            continue
        assert r["status"] == "ok" and r["has_ckpt"] == "1", r
        if r["config"] == "nau_ls0.01_ln0.01":
            spam.append(row_for(r, "_keepck", "_capref_canonck",
                                ["unlearning.ref_cap_forget=true", "unlearning.ref_cap_sep=true"]))
            if r["stratum"] == "popular":
                spam.append(row_for(r, "_keepck", "_caprefF_canonck",
                                    ["unlearning.ref_cap_forget=true"]))
        elif r["config"] == "forget_repair_lf0.1" and r["stratum"] == "popular":
            spam.append(row_for(r, "_keepck", "_capref_canonck",
                                ["unlearning.ref_cap_forget=true"]))
    for r in L:
        if not (r["setting"] == "unwanted" and r["track"] == "tiger+rkmeans"
                and r["config"] == "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"):
            continue
        assert r["method"] == "NAU" and r["status"] == "ok", r
        for w, tok in ((0.1, "0p1"), (1.0, "1p0")):
            sens.append(row_for(r, "_mia", f"_anchor{tok}_canonck", [f"unlearning.lambda_a={w}"]))
    assert len(spam) == 25, len(spam)
    assert len(sens) == 30, len(sens)
    for name, rows in (("capref", spam), ("anchor", sens)):
        rows.sort(key=lambda x: (x[0], x[1], int(x[2]), x[4]))
        tags = [x[4] for x in rows]
        assert len(set(tags)) == len(tags)
        with open(os.path.join(OUT, f"plan_{name}.tsv"), "w") as f:
            for x in rows:
                f.write("\t".join(x) + "\n")
        print(f"{OUT}/plan_{name}.tsv: {len(rows)} rows")
    # sanity: seeds covered
    for ds in ("beauty", "sports", "toys"):
        got = sorted({int(x[2]) for x in sens if x[0] == ds})
        assert got == list(seeds), (ds, got)


if __name__ == "__main__":
    sys.exit(main())
