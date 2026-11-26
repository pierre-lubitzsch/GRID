"""Check that every finished 5e-4 unlearning run (logs/sens5e4/unlearn_plan.tsv) resolved to the same
config as its 1e-4 template run, except for the keys that are supposed to differ (data dir, manifest,
run tag, output paths). Light: reads two .hydra/config.yaml per run.

    python scripts/check_sens5e4_replay.py
"""
import glob
import os
import sys

import yaml

SKIP_SUBSTR = ("data_dir", "forget_manifest", "unlearning_run_tag", "paths.", "hydra", "data_folder",
               "metadata_dir", "output_dir", "log_dir", "save_dir", "dirpath")


def flat(x, p=""):
    if isinstance(x, dict):
        for k, v in x.items():
            yield from flat(v, f"{p}.{k}" if p else str(k))
    else:
        yield p, x


def main():
    bad = ok = pending = 0
    for line in open("logs/sens5e4/unlearn_plan.tsv"):
        f = line.rstrip("\n").split("\t")
        tag, tmpl = f[4], f[12]
        runs = sorted(glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}"))
        if not runs:
            pending += 1
            continue
        a = dict(flat(yaml.safe_load(open(os.path.join(tmpl, ".hydra", "config.yaml")))))
        b = dict(flat(yaml.safe_load(open(os.path.join(runs[-1], ".hydra", "config.yaml")))))
        diff = []
        for k in sorted(set(a) | set(b)):
            if any(s in k for s in SKIP_SUBSTR):
                continue
            va, vb = a.get(k, "<missing>"), b.get(k, "<missing>")
            if isinstance(va, str) and isinstance(vb, str) and va.replace("1e-4", "5e-4") == vb:
                continue
            if va != vb:
                diff.append(f"{k}: {va!r} -> {vb!r}")
        if diff:
            bad += 1
            print(f"DIFF {tag}\n  " + "\n  ".join(diff[:20]))
        else:
            ok += 1
    print(f"identical up to data/manifest/tag: {ok}; differing: {bad}; not run yet: {pending}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
