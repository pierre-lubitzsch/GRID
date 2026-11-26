#!/usr/bin/env python3
"""Sensitive item deletion: current forget/retain split vs the retraining-matched split.

Current split (every reported run, canonical `_mia` executions of paper/plotting/canonical_runs.tsv):
forget set = the deleted interactions plus every later prefix of the requesting user's sequence
(include_context_rows=true), requesting users' whole sequences removed from the retain set.
Retraining-matched split (run_rmsplit_sens_launch.sh, unlearning.split_mode=retrain_matched): forget
set = only the training rows whose label is a deleted interaction (original histories); retain set =
every user, the requesting users with exactly the deleted items removed (the retrained reference's
training data). Each matched run replays its `_mia` run exactly except the split keys.

Methods: NAU (lambda_f=0.1, lambda_s=0, lambda_n=-1) and Forget+Repair (lambda_f=0.1, lambda_s=0,
lambda_n=0); TIGER + RQ-KMeans; Beauty / Sports / Toys x 5 seeds.

Test metrics on the CLEANED test view (protocol A, logs/eval/sensitive_scrub/scrub/): UHF@10
(requesting users; also as pooled user counts k/N), UHR@10 (everyone else), Delta^retr_UHR =
|UHR - UHR_retr| against the same seed's retrained model, NDCG@10, tau_P = NDCG / NDCG_retr.
Validation tau_P from each run's eval_val against logs/diagnose/valsel_sens_refs. MIA: membership
AUC on the deleted category (logs/diagnose/mia_unwanted{,_lastckpt,_rmsplit}/), mean over seeds and
mean |AUC - AUC_retr|.

    PYTHONPATH=. python paper/plotting/rmsplit_table.py
-> paper/revision/rmsplit_comparison.md, paper/revision/rmsplit_comparison.csv
"""
from __future__ import annotations

import csv
import glob
import json
import os
import statistics as S

SEEDS = [2, 3, 5, 7, 11]
DS = ("beauty", "sports", "toys")
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
CFG = {"NAU": "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0", "Forget+Repair": "unified_abl_lf0.1_ls0.0_ln0.0"}
MIA_KEY = {"NAU": "nau", "Forget+Repair": "forget_repair"}
SCRUB = "logs/eval/sensitive_scrub"
RUNS = "logs/unlearn/runs"
# requesting TEST users per (dataset, seed); UHF values are exact multiples of 1/N
# (paper/plotting/critique_followup/sensitive_counts.py; checked again below)
N = {("beauty", 2): 14, ("beauty", 3): 12, ("beauty", 5): 11, ("beauty", 7): 13, ("beauty", 11): 11,
     ("sports", 2): 13, ("sports", 3): 14, ("sports", 5): 14, ("sports", 7): 14, ("sports", 11): 17,
     ("toys", 2): 12, ("toys", 3): 13, ("toys", 5): 13, ("toys", 7): 11, ("toys", 11): 12}


def metrics_csv(d, prefix="test/"):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get(f"{prefix}ndcg@10")]
    return ({k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith(prefix)}
            if rows else None)


def ref_scrub(ds, role, sd):
    return metrics_csv(f"{SCRUB}/scrub/reference/v1/{ds}_{role}" + ("" if sd == 2 else f"_seed{sd}"))


def ref_val(ds, role, sd):
    return metrics_csv(f"logs/diagnose/valsel_sens_refs/{ds}_{role}_s{sd}")


def run_dir_of_tag(tag):
    c = [d for d in glob.glob(os.path.join(RUNS, f"job*_bs1_{tag}"))]
    return max(c, key=lambda d: int(os.path.basename(d).split("_")[0][3:])) if c else ""


def count(uhf, ds, sd):
    v = uhf * N[(ds, sd)]
    assert abs(v - round(v)) < 1e-3, (ds, sd, uhf)
    return round(v)


def mia(root, ds, sd, key):
    f = f"{root}/{ds}_seed{sd}.json"
    if not os.path.isfile(f):
        return None
    return {x["method"]: x["auc"] for x in json.load(open(f))["rows"]}.get(key)


NOTES = """## Notes (written 2026-11-08 from the numbers above; re-check if the tables change)

- Forget-set sizes (all 5 seeds pooled; requesting users 63 / 77 / 62 for Beauty / Sports / Toys): the
  matched split forgets 703 / 540 / 465 training rows (every row whose label is a deleted interaction;
  62 / 94 / 56 deleted interactions are labels, the same items as the MIA members). Under the current
  split the forget batches are drawn from ALL of the requesting users' training rows (the collate
  expands each emitted prefix into every contiguous subsequence, so rows before the first deleted item
  are included as well): 25,635 / 6,559 / 11,663 rows, capped at 512 sampled rows per request. 13 / 16 / 8
  requesting users have their deleted item only at the first position (never a label), so they contribute
  no forget row under the matched split; the last request of Sports seeds 2, 3, 5 has no deleted-label
  row at all and gets no update.
- Same lambda_f, much more concentrated forget signal: every forget row now ascends a deleted label.
  Result: stronger suppression for requesting users AND everyone else (UHR drops further below the
  retrained UHR, so Delta^retr_UHR grows), membership AUC falls BELOW the retrained model's (over-forgetting
  of the deleted interactions), and utility is slightly higher (tau_P up 0.6 to 2.0 points on test).
- Forget+Repair changes most: UHF 11.1 / 15.4 / 11.2 -> 3.2 / 5.6 / 3.2 (users 7 -> 2, 11 -> 4, 7 -> 2).
  NAU changes less: 4.8 / 5.6 / 6.4 -> 6.3 / 4.1 / 4.7 (users 3 -> 4, 4 -> 3, 4 -> 3). So under the matched
  split NAU is NOT the lowest-UHF method on Beauty and Toys (F+R 2 vs NAU 4 and 3 users) and ties it on
  Sports (4 vs 3); all differences are 1 to 2 users out of 61 to 72.
- NAU stays closer to retraining than F+R on Delta^retr_UHR (0.841 vs 1.448, 1.517 vs 2.038; Sports 0.996
  vs 0.778 is the exception) and on MIA (|dAUC| 0.103 vs 0.244, 0.093 vs 0.156, 0.147 vs 0.180), but both
  methods are farther from retraining on these two axes than with the current split.
- Sanity: re-scoring the retrained models in the new MIA jobs reproduces the stored AUCs exactly.
""".splitlines()


def main():
    led = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
           if r["setting"] == "unwanted" and r["track"] == "tiger+rkmeans" and r["config"] in CFG.values()]
    runs = {}   # (arm, ds, sd) -> run dir
    for r in led:
        ds, sd = r["dataset"], int(r["seed"])
        meth = [m for m, c in CFG.items() if c == r["config"]][0]
        runs[(f"{meth} (current split)", ds, sd)] = r["run_dir"]
        base, sfx = r["tag"].rsplit("_mia", 1)
        arm = base[len(f"sens_{ds}_{CAT[ds]}_"):]
        runs[(f"{meth} (retraining-matched)", ds, sd)] = run_dir_of_tag(
            f"sens_{ds}_{CAT[ds]}rm_{arm}_rmsplit_canonck{sfx}")
    arms = ["Before deletion", "Retrained"] + [f"{m} ({s})" for m in CFG for s in ("current split", "retraining-matched")]
    rows, per_seed = [], []
    for ds in DS:
        for arm in arms:
            cs = []
            for sd in SEEDS:
                rt = ref_scrub(ds, "retrain", sd)
                rtv = ref_val(ds, "retrain", sd)
                if arm in ("Before deletion", "Retrained"):
                    role = "base" if arm == "Before deletion" else "retrain"
                    m, mv = ref_scrub(ds, role, sd), ref_val(ds, role, sd)
                    aroot = "logs/diagnose/mia_unwanted_lastckpt" if role == "base" else "logs/diagnose/mia_unwanted"
                    auc = mia(aroot, ds, sd, "original" if role == "base" else "retrained")
                else:
                    d = runs.get((arm, ds, sd), "")
                    m = metrics_csv(os.path.join(SCRUB, "scrub", "runs", os.path.basename(d), "eval")) if d else None
                    mv = metrics_csv(os.path.join(d, "eval_val")) if d else None
                    meth = arm.split(" (")[0]
                    auc = (mia("logs/diagnose/mia_unwanted", ds, sd, MIA_KEY[meth]) if "current" in arm
                           else mia("logs/diagnose/mia_unwanted_rmsplit", ds, sd, MIA_KEY[meth] + "_rm"))
                if not m or not rt or "test/SHF@10" not in m:
                    continue
                auc_r = mia("logs/diagnose/mia_unwanted", ds, sd, "retrained")
                c = dict(sd=sd, f=m["test/SHF@10"], k=count(m["test/SHF@10"], ds, sd), n=N[(ds, sd)],
                         r=m["test/SHR@10"], nd=m["test/ndcg@10"], tp=m["test/ndcg@10"] / rt["test/ndcg@10"],
                         dr=abs(m["test/SHR@10"] - rt["test/SHR@10"]),
                         vtp=(mv["test/ndcg@10"] / rtv["test/ndcg@10"]) if (mv and rtv) else None,
                         auc=auc, dauc=(abs(auc - auc_r) if (auc is not None and auc_r is not None) else None))
                cs.append(c)
                per_seed.append(dict(dataset=ds, arm=arm, **c))
            if not cs:
                rows.append(dict(dataset=ds, arm=arm, n_seeds=0))
                continue
            def mean(key):
                v = [c[key] for c in cs if c[key] is not None]
                return (S.mean(v), len(v)) if v else (None, 0)
            rows.append(dict(dataset=ds, arm=arm, n_seeds=len(cs), k=sum(c["k"] for c in cs),
                             N=sum(c["n"] for c in cs), **{key: mean(key) for key in
                                                          ("f", "r", "dr", "nd", "tp", "vtp", "auc", "dauc")}))
    # control: current-split replay on the same hardware class as the matched runs
    ctl = run_dir_of_tag("sens_beauty_hairlossrm_unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0_userrepl80_canonck_seed3")
    ref_run = runs.get(("NAU (current split)", "beauty", 3), "")
    ctl_txt = "control replay not finished"
    if ctl and ref_run:
        a, b = metrics_csv(os.path.join(ctl, "eval")), metrics_csv(os.path.join(ref_run, "eval"))
        av, bv = metrics_csv(os.path.join(ctl, "eval_val")), metrics_csv(os.path.join(ref_run, "eval_val"))
        if a and b:
            dmax = max(abs(a[k] - b[k]) for k in ("test/SHF@10", "test/SHR@10", "test/ndcg@10"))
            dval = (max(abs(av[k] - bv[k]) for k in ("test/SHF@10", "test/SHR@10", "test/ndcg@10"))
                    if av and bv else float("nan"))
            ctl_txt = (f"control: Beauty seed 3 NAU replayed with the CURRENT split on the matched runs' hardware "
                       f"class reproduces the `_mia` run: max |diff| of UHF/UHR/NDCG@10 = {dmax:.2e} (test, normal "
                       f"view), {dval:.2e} (validation).")

    def f1(x, s=100, d=1):
        return "" if x is None or x[0] is None else f"{s * x[0]:.{d}f}"
    out = "paper/revision/rmsplit_comparison"
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["dataset", "arm", "n_seeds", "UHF10_pct", "UHF_users", "UHR10_pct", "dUHR_retr_pct", "NDCG10",
                    "tau_P_test_pct", "tau_P_val_pct", "MIA_AUC", "MIA_absdiff_retr", "n_mia"])
        for r in rows:
            if not r["n_seeds"]:
                w.writerow([r["dataset"], r["arm"], 0] + [""] * 10)
                continue
            w.writerow([r["dataset"], r["arm"], r["n_seeds"], f1(r["f"], d=2), f"{r['k']}/{r['N']}", f1(r["r"], d=3),
                        f1(r["dr"], d=3), f1(r["nd"], 1, 4), f1(r["tp"]), f1(r["vtp"]), f1(r["auc"], 1, 3),
                        f1(r["dauc"], 1, 3), r["auc"][1]])
    with open(out + "_perseed.csv", "w", newline="") as fh:
        keys = ["dataset", "arm", "sd", "k", "n", "f", "r", "dr", "nd", "tp", "vtp", "auc", "dauc"]
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(per_seed)

    L = ["# Sensitive item deletion: current split vs retraining-matched split", "",
         "Generated by `paper/plotting/rmsplit_table.py` (runs: `run_rmsplit_sens_launch.sh`, evals: "
         "`run_rmsplit_eval.sh`). TIGER + RQ-KMeans, 5 seeds per dataset, means over seeds.", "",
         "- **Current split** (all reported runs): forget set = deleted interactions plus every later prefix of the "
         "requesting user's sequence; the requesting users' whole sequences leave the retain set.",
         "- **Retraining-matched split**: forget set = only the training rows whose label is a deleted interaction "
         "(original histories); retain set = every user, the requesting users with exactly the deleted items "
         "removed (the retrained reference's training data). Each run replays its canonical `_mia` run except "
         "the split keys.", "",
         "Test columns on the cleaned test view (protocol A). UHF@10 = requesting users still shown the category "
         "(also pooled as users k/N), UHR@10 = everyone else, Delta^retr_UHR = |UHR - UHR_retr|, all in %; "
         "tau_P = NDCG@10 / retrained NDCG@10 in % (test, and validation as used for selection). MIA = membership "
         "AUC on the deleted category; |dAUC| = mean |AUC - AUC_retr| (retraining is the target).", "",
         ctl_txt, ""]
    for ds in DS:
        L += [f"## {ds.capitalize()}", "",
              "| arm | seeds | UHF@10 | users | UHR@10 | Delta^retr_UHR | NDCG@10 | tau_P test | tau_P val | MIA AUC | MIA abs diff to retr |",
              "|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            if r["dataset"] != ds:
                continue
            if not r["n_seeds"]:
                L.append(f"| {r['arm']} | 0 | WAITING | | | | | | | | |")
                continue
            L.append(f"| {r['arm']} | {r['n_seeds']} | {f1(r['f'])} | {r['k']}/{r['N']} | {f1(r['r'], d=3)} | "
                     f"{f1(r['dr'], d=3)} | {f1(r['nd'], 1, 4)} | {f1(r['tp'])} | {f1(r['vtp'])} | "
                     f"{f1(r['auc'], 1, 3)} | {f1(r['dauc'], 1, 3)} |")
        L.append("")
    # per-seed paired view: matched minus current, users still shown the category
    L += ["## Per seed: requesting users still shown the category (cleaned view), current -> matched", "",
          "| dataset | method | " + " | ".join(f"s{s}" for s in SEEDS) + " |", "|---|---|" + "---|" * len(SEEDS)]
    ps = {(p["dataset"], p["arm"], p["sd"]): p for p in per_seed}
    for ds in DS:
        for m in CFG:
            cells = []
            for s in SEEDS:
                a, b = ps.get((ds, f"{m} (current split)", s)), ps.get((ds, f"{m} (retraining-matched)", s))
                cells.append(f"{a['k']} -> {b['k']} /{a['n']}" if a and b else "")
            L.append(f"| {ds} | {m} | " + " | ".join(cells) + " |")
        rt = [ps.get((ds, "Retrained", s)) for s in SEEDS]
        L.append(f"| {ds} | Retrained | " + " | ".join(f"{p['k']} /{p['n']}" if p else "" for p in rt) + " |")
    L.append("")
    L += NOTES
    open(out + ".md", "w").write("\n".join(L) + "\n")
    print("\n".join(L))
    print("wrote", out + ".md", out + ".csv", out + "_perseed.csv")


if __name__ == "__main__":
    main()
