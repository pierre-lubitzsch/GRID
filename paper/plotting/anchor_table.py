#!/usr/bin/env python3
"""Retain-side neighbourhood anchor (rebuttal R2 Q4) vs NAU without it, unwanted-item removal
(Tables 3/22/23 setting: TIGER + RQ-KMeans, NAU lambda_f=0.1, lambda_s=0, lambda_n=-1), Beauty /
Sports / Toys x 5 seeds, test and validation.

Arms: NAU = the canonical `_mia` run of each cell (paper/plotting/canonical_runs.tsv);
NAU + anchor (lambda_a = w) = the same run replayed with unlearning.lambda_a=w
(tag ..._anchor<w>_canonck, scripts/plan_rebuttal_variants.py).

Per cell: UHF@10 (= SHF@10, requesting users), UHR@10 (= SHR@10, everyone else), Delta^retr_UHR =
|UHR - UHR_retr| against the same seed's retrained model, NDCG@10, tau_P = NDCG / NDCG_retr.
Test references through scripts/extract_sensitive.reference with the corrected "before deletion"
reference (GRID_SENSITIVE_BASE_REF_ROOT=logs/eval/sensitive_reference_lastckpt, set here by
default); validation references from logs/diagnose/valsel_sens_refs/<ds>_<role>_s<seed>.
Means over seeds; `UHR lower` counts seeds where the anchored run's UHR is closer to the retrained
UHR than NAU's.

    PYTHONPATH=. python paper/plotting/anchor_table.py [--ledger paper/plotting/canonical_runs.tsv]
-> paper/revision/anchor_comparison.csv, paper/revision/anchor_comparison.md

--view scrub (protocol A, cleaned test histories: requesting users' deleted items removed from their
test input histories, views src/data/amazon_data/*_scrubtestview): test split only; every model,
including Before deletion / Retrained / NAU, is read from the cleaned-view evals under
logs/eval/sensitive_scrub/scrub/ (references: reference/v1/<ds>_<role>[_seedN], evaluated at the
same start checkpoint the runs use; runs: runs/<run dir name>/eval; plans plan.tsv and
plan_anchor.tsv). tau_P and Delta^retr use the retrained model on the cleaned view. Also checks that
the normal-view re-evaluation (logs/eval/sensitive_scrub/normal/runs/...) reproduces each anchor
run's own <run>/eval.

    PYTHONPATH=. python paper/plotting/anchor_table.py --view scrub
-> paper/revision/anchor_comparison_scrub.csv, paper/revision/anchor_comparison_scrub.md
"""
from __future__ import annotations

import argparse
import csv
import glob
import importlib.util
import os
import statistics as S
import sys

os.environ.setdefault("GRID_SENSITIVE_BASE_REF_ROOT", "logs/eval/sensitive_reference_lastckpt")
sys.path.insert(0, "scripts")
_spec = importlib.util.spec_from_file_location("es", "scripts/extract_sensitive.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

SEEDS = [2, 3, 5, 7, 11]
DS = ("beauty", "sports", "toys")
NAU_CFG = "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"
WEIGHTS = (("0p1", 0.1), ("1p0", 1.0))
RUNS = "logs/unlearn/runs"
SCRUB = "logs/eval/sensitive_scrub"
VIEW = "normal"


def metrics_csv(d):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get("test/ndcg@10")]
    return ({k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith("test/")}
            if rows else None)


def ref(ds, role, sd, split):
    if VIEW == "scrub":
        return metrics_csv(f"{SCRUB}/scrub/reference/v1/{ds}_{role}" + ("" if sd == 2 else f"_seed{sd}"))
    if split == "test":
        return es.reference(ds, role, "tiger", "rkmeans", sd, es.CATEGORY[ds])[0]
    return metrics_csv(f"logs/diagnose/valsel_sens_refs/{ds}_{role}_s{sd}")


def cell(m, retr):
    if not m or not retr or "test/SHF@10" not in m:
        return None
    return dict(f=m["test/SHF@10"], r=m["test/SHR@10"], nd=m["test/ndcg@10"],
                tp=m["test/ndcg@10"] / retr["test/ndcg@10"],
                dr=abs(m["test/SHR@10"] - retr["test/SHR@10"]),
                df=abs(m["test/SHF@10"] - retr["test/SHF@10"]))


def run_metrics(d, sub, view):
    """Metrics of run dir d: its own eval (normal view) or the cleaned-view re-evaluation."""
    if view == "scrub":
        return metrics_csv(os.path.join(SCRUB, "scrub", "runs", os.path.basename(d), "eval"))
    return metrics_csv(os.path.join(d, sub))


def repro_check(A):
    """Normal-view re-evaluation vs each anchor run's own test eval: (n compared, max abs diff)."""
    n, worst = 0, 0.0
    for arm, cells in A.items():
        if arm == "NAU":
            continue
        for d in cells.values():
            own = metrics_csv(os.path.join(d, "eval")) if d else None
            re_ = metrics_csv(os.path.join(SCRUB, "normal", "runs", os.path.basename(d), "eval")) if d else None
            if not own or not re_:
                continue
            n += 1
            for k in ("test/SHF@10", "test/SHR@10", "test/ndcg@10"):
                worst = max(worst, abs(own[k] - re_[k]))
    return n, worst


def run_dir_of_tag(tag):
    c = sorted(glob.glob(os.path.join(RUNS, f"job*_bs1_{tag}")))
    return c[-1] if c else ""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default="paper/plotting/canonical_runs.tsv")
    ap.add_argument("--out", default=None)
    ap.add_argument("--view", choices=("normal", "scrub"), default="normal",
                    help="normal (default): recorded test/val evals; scrub: cleaned test histories (protocol A)")
    a = ap.parse_args()
    global VIEW
    VIEW = a.view
    a.out = a.out or ("paper/revision/anchor_comparison" + ("_scrub" if VIEW == "scrub" else ""))
    splits = ("test",) if VIEW == "scrub" else ("test", "val")
    A = {}
    for r in csv.DictReader(open(a.ledger), delimiter="\t"):
        if not (r["setting"] == "unwanted" and r["track"] == "tiger+rkmeans" and r["config"] == NAU_CFG):
            continue
        ds, sd = r["dataset"], int(r["seed"])
        A.setdefault("NAU", {})[(ds, sd)] = r["run_dir"]
        base, sfx = r["tag"].rsplit("_mia", 1)
        for tok, w in WEIGHTS:
            A.setdefault(f"NAU + anchor {w:g}", {})[(ds, sd)] = run_dir_of_tag(f"{base}_anchor{tok}_canonck{sfx}")
    order = ["NAU"] + [f"NAU + anchor {w:g}" for _, w in WEIGHTS]
    rows = []
    for split in splits:
        sub = "eval" if split == "test" else "eval_val"
        for ds in DS:
            # reference rows
            for role, name in (("base", "Before deletion"), ("retrain", "Retrained")):
                cs = []
                for sd in SEEDS:
                    m, rt = ref(ds, role, sd, split), ref(ds, "retrain", sd, split)
                    c = cell(m, rt)
                    if c:
                        cs.append(c)
                if cs:
                    rows.append(dict(split=split, dataset=ds, arm=name, n=len(cs),
                                     **{k: S.mean(c[k] for c in cs) for k in ("f", "r", "nd", "tp", "dr", "df")}))
            base_cells = {}
            for arm in order:
                cs = {}
                for sd in SEEDS:
                    d = A.get(arm, {}).get((ds, sd), "")
                    c = cell(run_metrics(d, sub, VIEW) if d else None, ref(ds, "retrain", sd, split))
                    if c:
                        cs[sd] = c
                if arm == "NAU":
                    base_cells = cs
                if not cs:
                    rows.append(dict(split=split, dataset=ds, arm=arm, n=0))
                    continue
                closer = ""
                if arm != "NAU":
                    k = [sd for sd in cs if sd in base_cells]
                    closer = f"{sum(cs[s]['dr'] < base_cells[s]['dr'] - 1e-12 for s in k)}/{len(k)}"
                rows.append(dict(split=split, dataset=ds, arm=arm, n=len(cs), closer=closer,
                                 **{k: S.mean(c[k] for c in cs.values()) for k in ("f", "r", "nd", "tp", "dr", "df")}))
    keys = ["split", "dataset", "arm", "n", "UHF10", "UHR10", "dUHR_retr", "dUHF_retr", "NDCG10", "tau_P",
            "UHR_closer_than_NAU"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(keys)
        for r in rows:
            if not r["n"]:
                w.writerow([r["split"], r["dataset"], r["arm"], 0] + [""] * 7)
                continue
            w.writerow([r["split"], r["dataset"], r["arm"], r["n"], f"{100 * r['f']:.3f}", f"{100 * r['r']:.3f}",
                        f"{100 * r['dr']:.3f}", f"{100 * r['df']:.3f}", f"{r['nd']:.4f}",
                        f"{100 * r['tp']:.1f}", r.get("closer", "")])
    title = "# Retain-side neighbourhood anchor, unwanted-item removal (TIGER + RQ-KMeans)"
    if VIEW == "scrub":
        title += ", CLEANED test histories (protocol A)"
    L = [title, "",
         "UHF@10 (requesting users), UHR@10 (everyone else), Delta^retr_UHR = |UHR - UHR_retr| and",
         "Delta^retr_UHF in %, tau_P = NDCG@10 / retrained NDCG@10 in %; means over 5 seeds.",
         "Before deletion = corrected last-ckpt reference (test). `closer` = seeds where the anchored",
         "run's UHR is strictly closer to the retrained UHR than NAU's.", ""]
    if VIEW == "scrub":
        n, worst = repro_check(A)
        L += ["All rows on the cleaned test view (requesting users' deleted items removed from their test",
              "input histories): Before deletion (start checkpoint), Retrained, NAU (`_mia`) and the anchor",
              "runs; tau_P and Delta^retr against the retrained model on the cleaned view.",
              f"Reproduction check: normal-view re-evaluation of {n} anchor runs vs their own test eval, "
              f"max |diff| of UHF/UHR/NDCG@10 = {worst:.2e}.", ""]
    for split in splits:
        L += [f"## {split}", "", "| dataset | arm | n | UHF@10 | UHR@10 | Delta^retr_UHR | Delta^retr_UHF | NDCG@10 | tau_P | closer |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["split"] != split:
                continue
            if not r["n"]:
                L.append(f"| {r['dataset']} | {r['arm']} | 0 | WAITING | | | | | | |")
                continue
            L.append(f"| {r['dataset']} | {r['arm']} | {r['n']} | {100 * r['f']:.1f} | {100 * r['r']:.3f} | "
                     f"{100 * r['dr']:.3f} | {100 * r['df']:.1f} | {r['nd']:.4f} | {100 * r['tp']:.1f} | {r.get('closer', '')} |")
        L.append("")
    open(a.out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", a.out + ".csv", a.out + ".md")


if __name__ == "__main__":
    main()
