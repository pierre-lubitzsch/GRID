"""Plan validation re-evaluations of canonical checkpoints (canonical_runs_plan.md step 2).

For every row of paper/plotting/canonical_runs.tsv with a resolvable checkpoint and no
<run>/eval_val, emit the arguments of EXACTLY the validation-eval block of run_valsel_one.sh:

  python -u -m scripts.eval_ckpt_on_test experiment=tiger_train_flat
    data_dir=src/data/amazon_data/<ds>_valview "semantic_id_path='<SID>'"
    "ckpt_path='<OUT>/checkpoints/unlearned.ckpt'" "spam_forget_manifest='<DATA_DIR>/forget_manifest.json'"
    num_hierarchies=4 seed=<UNLEARN_SEED> train=False test=True trainer.devices=1 trainer.strategy=auto
    trainer.sync_batchnorm=false trainer.num_nodes=1 trainer.deterministic=true
    callbacks.model_checkpoint=null callbacks.early_stopping=null hydra.run.dir=<OUT>/eval_val

with SID, DATA_DIR and the seed taken from the run's own .hydra/overrides.yaml (what the runner
printed as "Using semantic_id_path=/data_dir=" and UNLEARN_SEED) and OUT = the absolute run dir.

Verification per row: the run's own TEST eval (<run>/eval/.hydra/overrides.yaml, written by
run_tiger_unlearn_sequential.sh) must be this same override list with data_dir=<ds> (the test
view) instead of <ds>_valview. Rows that fail are reported and left out.

    python scripts/plan_canonical_val_eval.py --out logs/canon_val/plan.tsv [--ledger ...]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIXED_TAIL = ["num_hierarchies=4", "SEED", "train=False", "test=True", "trainer.devices=1",
              "trainer.strategy=auto", "trainer.sync_batchnorm=false", "trainer.num_nodes=1",
              "trainer.deterministic=true", "callbacks.model_checkpoint=null", "callbacks.early_stopping=null"]


def overrides(path):
    return [str(o) for o in yaml.safe_load(open(path))]


def expected(view, sid, out, data_dir, seed):
    return (["experiment=tiger_train_flat", f"data_dir={view}", f"semantic_id_path='{sid}'",
             f"ckpt_path='{out}/checkpoints/unlearned.ckpt'",
             f"spam_forget_manifest='{data_dir}/forget_manifest.json'"]
            + [f"seed={seed}" if x == "SEED" else x for x in FIXED_TAIL])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="paper/plotting/canonical_runs.tsv")
    ap.add_argument("--out", required=True)
    ap.add_argument("--setting", default="spam")
    args = ap.parse_args()
    os.chdir(ROOT)
    rows = [r for r in csv.DictReader(open(args.ledger), delimiter="\t")
            if r["setting"] == args.setting and r["status"] == "ok"]
    plan, bad, have = [], [], 0
    seen = set()
    for r in rows:
        run = os.path.abspath(r["run_dir"])
        if run in seen:
            continue
        seen.add(run)
        if os.path.isfile(os.path.join(run, "eval_val/csv/version_0/metrics.csv")):
            have += 1
            continue
        if not os.path.exists(os.path.join(run, "checkpoints/unlearned.ckpt")):
            continue
        if r["track"] != "tiger+rkmeans":
            bad.append((run, f"track {r['track']} (only TIGER + RQ-KMeans planned)")); continue
        kv = dict(o.split("=", 1) for o in overrides(os.path.join(run, ".hydra/overrides.yaml")))
        sid = kv["semantic_id_path"].strip("'\"")
        data_dir = kv["data_dir"].strip("'\"")
        seed = str(int(kv["seed"]))
        ds = r["dataset"]
        view = f"src/data/amazon_data/{ds}_valview"
        if not os.path.isdir(os.path.join(view, "testing")):
            bad.append((run, f"missing {view}/testing")); continue
        if not os.path.isfile(os.path.join(data_dir, "forget_manifest.json")):
            bad.append((run, f"missing {data_dir}/forget_manifest.json")); continue
        test_ov = os.path.join(run, "eval/.hydra/overrides.yaml")
        if not os.path.isfile(test_ov):
            bad.append((run, "no eval/.hydra/overrides.yaml to verify against")); continue
        got = overrides(test_ov)
        want_test = expected(f"src/data/amazon_data/{ds}", sid, run, data_dir, seed)
        if got != want_test:
            d = [f"{a!r} vs {b!r}" for a, b in zip(got, want_test) if a != b]
            if len(got) != len(want_test):
                d.append(f"{len(got)} vs {len(want_test)} overrides")
            bad.append((run, "test-eval command differs: " + "; ".join(d))); continue
        plan.append([run, ds, view, sid, data_dir, seed, r["config"], r["source"]])
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for p in plan:
            f.write("\t".join(p) + "\n")
    print(f"{len(plan)} validation re-eval(s) planned -> {args.out}; {have} already have eval_val; "
          f"{len(bad)} refused")
    for run, why in bad:
        print(f"  REFUSED {os.path.basename(run)}: {why}")
    return 0 if not bad else 2


if __name__ == "__main__":
    sys.exit(main())
