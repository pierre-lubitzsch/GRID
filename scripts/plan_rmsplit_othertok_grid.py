"""Plan: the FULL validation grid of sensitive item deletion for the other SID tokenizers (2026-11-23).

Per-tokenizer selection on cleaned validation (instead of pinning the TIGER + RQ-KMeans picks). The grid per
(dataset) is the TIGER + RQ-KMeans grid of logs/rmsplit_sens/plan_grid_wave{1,2}.tsv with NAU restricted to the
submitted sensitive grid (lambda_s=0, lambda_n in {0, -0.1, -1, -10}); Beauty additionally gets the Sports TRACER grid
(TIGER never ran TRACER on Beauty; rows are synthesized exactly as plan_rmsplit_othertok.template does). Configurations
that the pinned campaign already ran (logs/rmsplit_sens/plan_othertok_<track>.tsv) are skipped.

Rows are built by plan_rmsplit_othertok.plan_track (unchanged: clones the recorded overrides of the TIGER rmsplit run of
the same (dataset, config, seed), swaps ckpt / SID / TRACER codebook / tag, asserts the corrected protocol and that no
tag exists). Added here: unlearning.tracer_sid_tolerance=0.999 on TRACER rows of RQ-VAE Toys and LETTER Sports (one
item of 11,924 / 18,357 fails phi=0 reproduction there by a quantization tie-break; the spam TRACER runs on these pairs
used the same tolerance).

    python scripts/plan_rmsplit_othertok_grid.py -> logs/rmsplit_sens/plan_othertok_grid_<track>.tsv
"""
import argparse
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_s = importlib.util.spec_from_file_location("P", os.path.join(HERE, "plan_rmsplit_othertok.py"))
P = importlib.util.module_from_spec(_s)
_s.loader.exec_module(P)

WAVES = ("logs/rmsplit_sens/plan_grid_wave1.tsv", "logs/rmsplit_sens/plan_grid_wave2.tsv")
TOL_PAIRS = {("rqvae", "toys"), ("letter", "sports")}


def grid_arms(ds):
    arms = set()
    for p in WAVES:
        for line in open(p):
            f = line.split("\t")
            if f[1] != ds:
                continue
            a = re.match(rf"sens_{ds}_\w+?rm_(.*)_rmsplit_canonck(_seed\d+)?$", f[5]).group(1)
            if a.startswith("unified_hp2_sepgen") and "_ls0.0_" not in a:
                continue                                  # NAU: submitted grid, lambda_s = 0 only
            arms.add(a)
    if ds == "beauty":
        arms |= {a for a in grid_arms("sports") if a.startswith("tracer_")}
    return arms


def done_arms(track, ds):
    p = f"logs/rmsplit_sens/plan_othertok_{track}.tsv"
    out = set()
    for line in open(p):
        f = line.split("\t")
        if f[1] == ds:
            out.add(re.match(rf"\w*?sens_{ds}_\w+?rm_(.*)_rmsplit_canonck(_seed\d+)?$", f[5]).group(1))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=list(P.TRACKS))
    a = ap.parse_args()
    for track in a.tracks:
        todo = {ds: sorted(grid_arms(ds) - done_arms(track, ds)) for ds in P.TRACKS[track]["datasets"]}
        P.arms_of = lambda ds: todo[ds]                   # plan_track uses this list
        rows = P.plan_track(track)
        for r in rows:
            if r[7] == "tracer" and (track, r[1]) in TOL_PAIRS:
                assert "tracer_sid_tolerance" not in r[14]
                r[14] += " unlearning.tracer_sid_tolerance=0.999"
        out = f"logs/rmsplit_sens/plan_othertok_grid_{track}.tsv"
        if os.path.exists(out):
            raise SystemExit(f"{out} exists")
        with open(out, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(out, len(rows), "rows;", {ds: len(v) for ds, v in todo.items()}, "configs per dataset")


if __name__ == "__main__":
    main()
