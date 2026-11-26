"""List the unlearning run dirs whose checkpoints are kept (user decision 2026-11-07) for
archive_unlearn_ckpts.sh (RUN_LIST): every run of paper/plotting/canonical_runs.tsv with a
checkpoint (all settings: spam, unwanted `_mia`, spam_frtie_candidate), the neighbor-source
keepck runs (`*_nau_s0p01_n0p01_{nbrrand,nbrpfx}_keepck*`) and the Figure 1 fix `_canonck` runs.

Only runs with at least one REAL (non-symlink) `checkpoints/unlearned_*.ckpt` on scratch are
printed (one run dir basename per line); already archived runs are skipped.

--defer-readers drops runs that queued/running jobs will read (so the first pass never moves a
file under a reader): the nbrsrc keepck runs (scope jobs 11466392-11466421), the `_mia` runs
(scrub eval array 11465971), and the Beauty NAU / Forget+Repair checkpoints that those scope
jobs and the F+R tie-break scope jobs (run_scope_frtie.sh) load. The second pass (after those
jobs) runs without it. Light: globs + lstat.

    python scripts/list_canonical_ckpt_runs.py [--ledger paper/plotting/canonical_runs.tsv] [--defer-readers] [--sizes]
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "logs/unlearn/runs")


def real_ckpts(run_dir):
    return [f for f in glob.glob(os.path.join(run_dir, "checkpoints", "unlearned_*.ckpt"))
            if os.path.isfile(f) and not os.path.islink(f)]


def beauty_scope_readers(ledger_rows):
    """Run dirs whose checkpoints the queued scope jobs load (Beauty, all 15 cells)."""
    out = set()
    spec = importlib.util.spec_from_file_location("f", os.path.join(ROOT, "scripts/run_fig1_js_eval.py"))
    f = importlib.util.module_from_spec(spec)
    sys.path.insert(0, os.path.join(ROOT, "scripts"))
    spec.loader.exec_module(f)
    for st in ("unpopular", "mid", "popular"):
        for sd in (2, 3, 5, 7, 11):
            try:
                led = f.unlearned_checkpoints("beauty", sd, st)
            except Exception as e:   # noqa: BLE001
                print(f"warning: fig1_js ledger beauty {st} {sd}: {e}", file=sys.stderr)
                continue
            for arm in ("nau_s0p01_n0p01", "forget_repair", "nosep_ln0"):
                # run dir = parent of checkpoints/ of the UNRESOLVED path; resolve() may point into
                # the archive already, so map via the run dir glob of the job id instead
                p = led.get(arm)
                if p:
                    out.add(os.path.basename(os.path.dirname(os.path.dirname(p))))
    for r in ledger_rows:   # run_scope_frtie.sh: canonical NAU and F+R (lambda_f=0.1) on Beauty
        if r["setting"] == "spam" and r["dataset"] == "beauty" and r["config"] in ("nau_ls0.01_ln0.01", "forget_repair_lf0.1") and r["run_dir"]:
            out.add(os.path.basename(r["run_dir"]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "paper/plotting/canonical_runs.tsv"))
    ap.add_argument("--defer-readers", action="store_true")
    ap.add_argument("--sizes", action="store_true", help="print a size summary to stderr")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.ledger), delimiter="\t"))
    groups = {}
    for r in rows:
        if r["run_dir"]:
            groups.setdefault(os.path.basename(r["run_dir"]), f"ledger:{r['setting']}")
    for d in glob.glob(os.path.join(RUNS, "job*_bs1_bw_tgt*_nau_s0p01_n0p01_nbr*_keepck*")):
        groups.setdefault(os.path.basename(d), "nbrsrc_keepck")
    for d in glob.glob(os.path.join(RUNS, "job*_bs1_bw_tgtmid_*_canonck*")):
        # not the rebuttal-variant runs (_capref/_anchor, plan_rebuttal_variants.py):
        # they are not ledger runs and may still be running/evaluating
        if "_capref" in os.path.basename(d) or "_anchor" in os.path.basename(d):
            continue
        groups.setdefault(os.path.basename(d), "canonck")
    defer = set()
    if a.defer_readers:
        defer = beauty_scope_readers(rows)
        defer |= {b for b, g in groups.items() if g == "nbrsrc_keepck" or "_mia" in b}
    tot, n, nd, dbytes, by = 0, 0, 0, 0, {}
    for b, g in sorted(groups.items()):
        fs = real_ckpts(os.path.join(RUNS, b))
        if not fs:
            continue
        size = sum(os.path.getsize(f) for f in fs)
        if b in defer:
            nd += 1; dbytes += size
            continue
        print(b)
        n += 1; tot += size
        by.setdefault(g, [0, 0]); by[g][0] += 1; by[g][1] += size
    if a.sizes:
        print(f"{n} runs, {tot / 1024 ** 3:.1f} GiB listed; deferred {nd} runs, {dbytes / 1024 ** 3:.1f} GiB", file=sys.stderr)
        for g, (k, s) in sorted(by.items()):
            print(f"  {g:28s} {k:4d} runs {s / 1024 ** 3:7.1f} GiB", file=sys.stderr)


if __name__ == "__main__":
    main()
