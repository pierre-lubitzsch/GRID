#!/usr/bin/env python3
"""Reference-capped forgetting (rebuttal R1 W1a, R2 W3/Q3) vs uncapped NAU and Forget+Repair,
Beauty spam, TIGER + RQ-KMeans, per target stratum, test and validation.

Arms (per stratum, 5 seeds each):
  NAU                 canonical NAU run (ledger config nau_ls0.01_ln0.01)
  NAU + cap           same configuration, unlearning.ref_cap_forget=true ref_cap_sep=true
                      (tag ..._capref_canonck)
  NAU + forget cap    forget cap only (popular; ..._caprefF_canonck)
  Forget+Repair       canonical F+R (ledger config forget_repair_lf0.1)
  F+R + forget cap    unlearning.ref_cap_forget=true (popular; ..._sepgen_capref_canonck)

Per cell (stratum, seed): SH@10 and NDCG@10 of the run; the retrained model of the same seed
gives the reference: Delta_SH = |SH - SH_retr| (distance to retrained exposure), signed
SH - SH_retr (negative = over-suppression), tau_P = NDCG / NDCG_retr. Test references:
paper/data/spam_references/beauty[_seed<N>].json (clean_* = retrained); validation:
logs/diagnose/valsel_refs/rkmeans_beauty_<stratum>_s<seed>_clean. Means over seeds; the
paired column counts seeds where the variant is closer to retraining than its uncapped arm.

    PYTHONPATH=. python paper/plotting/capref_table.py [--ledger paper/plotting/canonical_runs.tsv]
-> paper/revision/capref_comparison.csv, paper/revision/capref_comparison.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as S

SEEDS = ["2", "3", "5", "7", "11"]
STR = ["unpopular", "mid", "popular"]
RUNS = "logs/unlearn/runs"


def last_test_metrics(d):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get("test/SH@10") and r.get("test/ndcg@10")]
    return (float(rows[-1]["test/SH@10"]), float(rows[-1]["test/ndcg@10"])) if rows else None


def test_ref(seed, st):
    r = json.load(open("paper/data/spam_references/beauty" + ("" if seed == "2" else f"_seed{seed}") + ".json"))
    return r["clean_SH@10"][st], r["clean_ndcg@10"]


def val_ref(seed, st):
    return last_test_metrics(f"logs/diagnose/valsel_refs/rkmeans_beauty_{st}_s{seed}_clean")


def run_dir_of_tag(tag):
    c = sorted(glob.glob(os.path.join(RUNS, f"job*_bs1_{tag}")))
    return c[-1] if c else ""


def arms(ledger):
    L = [r for r in csv.DictReader(open(ledger), delimiter="\t")
         if r["setting"] == "spam" and r["dataset"] == "beauty" and r["track"] == "tiger+rkmeans"]
    out = {}
    for name, cfg in (("NAU", "nau_ls0.01_ln0.01"), ("Forget+Repair", "forget_repair_lf0.1")):
        for r in L:
            if r["config"] == cfg:
                out.setdefault(name, {})[(r["stratum"], r["seed"])] = r["run_dir"]
                # the variant replays this run with a new tag (plan_rebuttal_variants.py)
                base, sfx = r["tag"].rsplit("_keepck", 1)
                for vname, vtok, only_pop in (
                        ((name + " + cap") if name == "NAU" else "F+R + forget cap", "_capref_canonck", name != "NAU"),
                        ("NAU + forget cap" if name == "NAU" else None, "_caprefF_canonck", True)):
                    if vname is None or (only_pop and r["stratum"] != "popular"):
                        continue
                    out.setdefault(vname, {})[(r["stratum"], r["seed"])] = run_dir_of_tag(base + vtok + sfx)
    return out


def cell(run_dir, split, st, seed):
    sub = "eval" if split == "test" else "eval_val"
    m = last_test_metrics(os.path.join(run_dir, sub)) if run_dir else None
    ref = test_ref(seed, st) if split == "test" else val_ref(seed, st)
    if not m or not ref:
        return None
    sh, n = m
    return dict(sh=sh, n=n, dsh=abs(sh - ref[0]), ssh=sh - ref[0], tp=n / ref[1], shr=ref[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="paper/plotting/canonical_runs.tsv")
    ap.add_argument("--out", default="paper/revision/capref_comparison")
    a = ap.parse_args()
    A = arms(a.ledger)
    order = ["NAU", "NAU + cap", "NAU + forget cap", "Forget+Repair", "F+R + forget cap"]
    base_of = {"NAU + cap": "NAU", "NAU + forget cap": "NAU", "F+R + forget cap": "Forget+Repair"}
    rows = []
    for split in ("test", "val"):
        for st in STR + ["all"]:
            sts = STR if st == "all" else [st]
            for arm in order:
                cells = {}
                for s in sts:
                    for sd in SEEDS:
                        c = cell(A.get(arm, {}).get((s, sd), ""), split, s, sd)
                        if c:
                            cells[(s, sd)] = c
                expect = len(sts) * len(SEEDS)
                if arm in ("NAU + forget cap", "F+R + forget cap") and st != "popular":
                    continue
                if not cells:
                    rows.append(dict(split=split, stratum=st, arm=arm, n=0, status="WAITING"))
                    continue
                mean = lambda k: S.mean(c[k] for c in cells.values())  # noqa: E731
                closer = ""
                if arm in base_of:
                    b = {}
                    for s in sts:
                        for sd in SEEDS:
                            cb = cell(A[base_of[arm]].get((s, sd), ""), split, s, sd)
                            if cb and (s, sd) in cells:
                                b[(s, sd)] = cb
                    if b:
                        closer = f"{sum(cells[k]['dsh'] < b[k]['dsh'] - 1e-12 for k in b)}/{len(b)}"
                        # ties (both exactly at retrained, e.g. 0 vs 0) are not counted as closer
                rows.append(dict(split=split, stratum=st, arm=arm, n=len(cells),
                                 status="ok" if len(cells) == expect else f"PARTIAL({len(cells)}/{expect})",
                                 SH10=100 * mean("sh"), SH10_retr=100 * mean("shr"),
                                 dSH=100 * mean("dsh"), signed_dSH=100 * mean("ssh"),
                                 NDCG10=mean("n"), tau_P=100 * mean("tp"), closer_than_uncapped=closer))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    keys = ["split", "stratum", "arm", "n", "status", "SH10", "SH10_retr", "dSH", "signed_dSH",
            "NDCG10", "tau_P", "closer_than_uncapped"]
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow({k: (round(r[k], 4) if isinstance(r.get(k), float) else r.get(k, "")) for k in keys})
    L = ["# Reference-capped forgetting, Beauty spam (TIGER + RQ-KMeans)", "",
         "SH@10, Delta_SH (= |SH - SH_retr|, distance to retrained exposure) and signed SH - SH_retr in %,",
         "NDCG@10 absolute, tau_P = NDCG / NDCG_retr in %. Means over 5 seeds (15 cells for `all`).",
         "`closer` = seeds where the capped arm is strictly closer to retraining than its uncapped arm.", ""]
    for split in ("test", "val"):
        L += [f"## {split}", "", "| stratum | arm | n | SH@10 | SH@10 retr | Delta_SH | signed | NDCG@10 | tau_P | closer |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["split"] != split:
                continue
            if r["status"] == "WAITING":
                L.append(f"| {r['stratum']} | {r['arm']} | 0 | WAITING | | | | | | |")
                continue
            L.append(f"| {r['stratum']} | {r['arm']} | {r['n']}{'' if r['status'] == 'ok' else ' (partial)'} | "
                     f"{r['SH10']:.3f} | {r['SH10_retr']:.3f} | {r['dSH']:.3f} | {r['signed_dSH']:+.3f} | "
                     f"{r['NDCG10']:.4f} | {r['tau_P']:.1f} | {r['closer_than_uncapped']} |")
        L.append("")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", a.out + ".csv", a.out + ".md")


if __name__ == "__main__":
    main()
