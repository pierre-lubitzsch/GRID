"""Check paper/plotting/canonical_runs.tsv (scripts/build_canonical_runs.py).

Asserts, per row:
  1. exactly one row per (setting, dataset, track, config, stratum, seed) and no run dir used
     by two rows of different configurations;
  2. the run dir exists (status=pending rows are reported, not failed, unless --strict-pending);
  3. the run is the right CELL: data_dir matches (dataset, stratum, seed) and the poisoned
     checkpoint path carries the stratum and seed (catches the wrong-seed bug);
  4. its resolved config tree (.hydra/config.yaml) equals the group's ANCHOR (the mid seed-2 row,
     else the first row of the same dataset/track/config) up to
       - volatile keys: unlearning_run_tag, paths.*, logger.*, callbacks.*, hydra.*, task_name,
         tags, extras.*, null values; cell keys: data_dir, ckpt_path, seed, spam_forget_manifest;
         semantic_id_path and other *_path keys compared by realpath;
       - inert keys (canonical_runs_plan.md 1.1): unlearning.drift_*; unlearning.val_selection.*
         when not enabled; unlearning.coherence_checkpoint; unlearning.forget_rows_per_batch when
         forget_full_coverage is false; unlearning.sep_* when lambda_s=0 (unified) or for a
         non-unified algorithm; unlearning.coherence_*, neighborhood_count, embedding_path when
         lambda_n=0 or for scif/seif/kookmin/fanchuan;
       - late-added keys at their code default: lambda_r=1.0, code_row_scope=all,
         coherence_union_size=full, forget_full_coverage=False, freeze_sid_table=False,
         coherence_random_seed=0.
     A difference that consists only of keys present on ONE side (older runs predating a key;
     renamed lambda_forget/lambda_sep) or of checkpoint_fractions is the plan's "loose"
     equivalence: printed as LOOSE, not failed. Values that differ on both sides fail.
     Unwanted-item (_mia) rows: cell check is data_dir contains the category and seed.

Exit status 1 on any failure. Light: reads YAML only.

    python scripts/check_canonical_runs.py [--ledger paper/plotting/canonical_runs.tsv] [-v]
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import defaultdict

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
VOLATILE_PREFIX = ("paths.", "logger.", "callbacks.", "hydra.", "extras.")
VOLATILE = {"unlearning_run_tag", "task_name", "tags", "data_dir", "ckpt_path", "seed",
            "spam_forget_manifest", "semantic_id_path"}
LATE_DEFAULTS = {"unlearning.lambda_r": 1.0, "unlearning.code_row_scope": "all",
                 "unlearning.coherence_union_size": "full", "unlearning.forget_full_coverage": False,
                 "unlearning.freeze_sid_table": False, "unlearning.coherence_random_seed": 0}
NON_UNIFIED_NBR = ("scif", "seif", "kookmin", "fanchuan")
RENAMED = {"unlearning.lambda_forget": "unlearning.lambda_f", "unlearning.lambda_sep": "unlearning.lambda_s"}
LOOSE_IGNORE = {"unlearning.checkpoint_fractions"}   # only adds intermediate checkpoints


def norm_value(k, v):
    """Path-valued keys compare by realpath (relative vs absolute spelling of one file)."""
    if isinstance(v, str) and k.endswith("_path") and v:
        return os.path.realpath(os.path.join(ROOT, v.strip("'\"")))
    return v


def flat(d, p=""):
    o = {}
    if isinstance(d, dict):
        for k, v in d.items():
            o.update(flat(v, f"{p}{k}."))
    else:
        o[p[:-1]] = d
    return o


def normalized(run_dir):
    cfg = flat(yaml.safe_load(open(os.path.join(run_dir, ".hydra/config.yaml"))))
    raw = dict(cfg)
    u = {RENAMED.get(k, k): norm_value(k, v) for k, v in cfg.items() if v is not None and k not in VOLATILE
         and not k.startswith(VOLATILE_PREFIX)}
    for k, v in LATE_DEFAULTS.items():
        u.setdefault(k, v)
    algo = u.get("unlearning.algorithm")
    ls = float(u.get("unlearning.lambda_s", 0) or 0)
    ln = float(u.get("unlearning.lambda_n", 0) or 0)
    vs_on = bool(u.get("unlearning.val_selection.enabled", False))
    out = {}
    for k, v in u.items():
        if k.startswith("unlearning.drift_") or k == "unlearning.coherence_checkpoint":
            continue
        if k.startswith("unlearning.val_selection.") and not vs_on:
            continue
        if k == "unlearning.forget_rows_per_batch" and not u.get("unlearning.forget_full_coverage"):
            continue
        if k.startswith("unlearning.sep_") and (algo != "unified" or ls == 0.0):
            continue
        if (k.startswith("unlearning.coherence_") or k in ("unlearning.neighborhood_count", "unlearning.embedding_path")) \
                and (ln == 0.0 or algo in NON_UNIFIED_NBR):
            continue
        out[k] = v
    return out, raw


def cell_problems(row, raw):
    ds, st, sd = row["dataset"], row["stratum"], int(row["seed"])
    data_dir = str(raw.get("data_dir", "")).rstrip("/")
    ckpt = str(raw.get("ckpt_path", ""))
    probs = []
    if row["setting"].startswith("spam"):
        want = (f"{ds}_spam_seed{sd}_pct1_n1" if st == "unpopular" else f"{ds}_spam_tgt{st}_seed{sd}_pct1_n1")
        if os.path.basename(data_dir) != want:
            probs.append(f"data_dir {data_dir} != {want}")
        base = ckpt.split("/checkpoints/")[0]
        toks = {t for t in ("unpopular", "mid", "popular") if f"tgt{t}" in base}
        # unpopular predates the tgt<stratum> naming: some of its trainings carry no token
        if (toks and toks != {st}) or (not toks and st != "unpopular"):
            probs.append(f"poisoned ckpt is not stratum {st}: {ckpt}")
        has_seed = f"_seed{sd}_" in base + "_"
        if sd == 2 and "_seed" in os.path.basename(base):
            probs.append(f"seed-2 row uses a seeded ckpt: {ckpt}")
        if sd != 2 and not has_seed:
            probs.append(f"ckpt lacks _seed{sd}: {ckpt}")
    else:
        cat = row["stratum"]
        if cat not in data_dir or (sd != 2 and f"seed{sd}" not in data_dir):
            probs.append(f"data_dir {data_dir} does not match {cat} seed {sd}")
    if int(raw.get("seed", -1)) not in (sd, 2) and row["setting"].startswith("spam"):
        probs.append(f"unlearning seed {raw.get('seed')} (data seed {sd})")
    return probs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ledger", default="paper/plotting/canonical_runs.tsv")
    ap.add_argument("--strict-pending", action="store_true", help="fail on pending rows")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    os.chdir(ROOT)
    rows = list(csv.DictReader(open(args.ledger), delimiter="\t"))
    fail, warn, loose = [], [], []
    keys, dirs = defaultdict(list), defaultdict(set)
    for r in rows:
        keys[(r["setting"], r["dataset"], r["track"], r["config"], r["stratum"], r["seed"])].append(r)
        if r["run_dir"]:
            dirs[r["run_dir"]].add(r["config"])
    for k, v in keys.items():
        if len(v) != 1:
            fail.append(f"{len(v)} rows for {k}")
    for d, cfgs in dirs.items():
        if len(cfgs) > 1:
            fail.append(f"run dir used by several configs {sorted(cfgs)}: {d}")
    groups = defaultdict(list)
    for r in rows:
        if r["status"] == "pending":
            (fail if args.strict_pending else warn).append(f"pending: {r['dataset']} {r['config']} {r['stratum']} s{r['seed']} ({r['tag']})")
            continue
        if not r["run_dir"] or not os.path.isdir(r["run_dir"]):
            fail.append(f"missing run dir: {r['dataset']} {r['config']} {r['stratum']} s{r['seed']} {r['run_dir']!r}")
            continue
        if r["status"] == "dangling_ckpt":
            warn.append(f"dangling checkpoint symlink: {r['run_dir']}")
        if not int(r["has_test"]):
            fail.append(f"no test eval: {r['run_dir']}")
        groups[(r["setting"], r["dataset"], r["track"], r["config"])].append(r)
    n_cmp = 0
    for g, rs in sorted(groups.items()):
        anchor = next((r for r in rs if r["stratum"] == "mid" and r["seed"] == "2"), rs[0])
        a_cfg, a_raw = normalized(anchor["run_dir"])
        a_sid = os.path.realpath(str(a_raw.get("semantic_id_path", "")).strip("'\""))
        for r in rs:
            cfg, raw = normalized(r["run_dir"])
            n_cmp += 1
            for p in cell_problems(r, raw):
                fail.append(f"{g[1]} {g[3]} {r['stratum']} s{r['seed']}: {p}")
            sid = os.path.realpath(str(raw.get("semantic_id_path", "")).strip("'\""))
            if sid != a_sid:
                fail.append(f"{g[1]} {g[3]} {r['stratum']} s{r['seed']}: SID {sid} != anchor {a_sid}")
            diff = sorted(k for k in set(cfg) | set(a_cfg) if cfg.get(k) != a_cfg.get(k))
            strict = [k for k in diff if k in cfg and k in a_cfg and k not in LOOSE_IGNORE]
            if diff and not strict:
                loose.append(f"{g[1]} {g[2]} {g[3]} {r['stratum']} s{r['seed']} [{r['source']}] vs anchor "
                             f"[{anchor['source']}]: {len(diff)} key(s) present on one side only "
                             f"(e.g. {', '.join(diff[:4])})")
            elif diff:
                fail.append(f"{g[1]} {g[2]} {g[3]} {r['stratum']} s{r['seed']} [{r['source']}] vs anchor "
                            f"{anchor['stratum']} s{anchor['seed']} [{anchor['source']}]: "
                            + "; ".join(f"{k}={cfg.get(k)!r}/{a_cfg.get(k)!r}" for k in diff))
    print(f"{len(rows)} rows, {len(groups)} (dataset, config) groups, {n_cmp} config trees compared")
    for w in warn:
        print("WARN", w)
    for x in loose:
        print("LOOSE", x)
    for f in fail:
        print("FAIL", f)
    print("OK" if not fail else f"{len(fail)} failure(s)")
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
