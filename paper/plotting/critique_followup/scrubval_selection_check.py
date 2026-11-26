"""Unwanted-item removal: redo the validation selection on the CLEANED validation view.

The paper selects hyperparameters on the original validation histories
(paper/plotting/valsens_candidates_ledger.csv, from valsens_table.py --ledger), while the reported
test numbers use cleaned histories. Job 11540362 evaluated every checkpointed candidate plus the
original and retrained models on the cleaned validation view
(src/data/amazon_data/<ds>_sens_<cat>_1e-4_seed<N>_scrubvalview); plan:
logs/eval/sensitive_scrub/plan_scrubval.tsv.

Per candidate and seed: UHF@10 = test/SHF@10, UHR@10 = test/SHR@10, global SH@10 = test/SH@10,
tau_P = NDCG@10 / NDCG@10 of the retrained model of the same (dataset, seed) on the same cleaned
view. Means over the five seeds (mean of per-seed ratios for tau_P, as valsens_table.py).
Rule (valsens_table.choose): lowest mean UHF@10 (rounded to 1e-9) among configurations with mean
tau_P >= 95%, ties by mean global SH@10, then higher tau_P; if none reaches 95%, lowest UHF@10
overall. Baselines per dataset; NAU lambda_n selected on Beauty and used everywhere (per-dataset
picks reported too).

    python paper/plotting/critique_followup/scrubval_selection_check.py
"""
import csv
import glob
import os
import re
import statistics as S
from collections import defaultdict

PLAN = "logs/eval/sensitive_scrub/plan_scrubval.tsv"
LEDGER = "paper/plotting/valsens_candidates_ledger.csv"
OUT_CSV = "paper/revision/scrubval_selection_check.csv"
SEEDS = [2, 3, 5, 7, 11]
DS = ("beauty", "sports", "toys")
NAME = {"scif": "SCIF", "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan",
        "tracer": "TRACER", "unified": "NAU"}


def metrics(d):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get("test/ndcg@10")]
    return {k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith("test/")} if rows else None


def config_of(arm):
    arm = re.sub(r"_(mia|canonck)$", "", arm)
    m = re.match(r"unified_hp2_sepgen_lf0\.1_ls0\.0_ln(neg)?([0-9.]+)$", arm)
    if m:
        return "NAU", f"lambda_n={'-' if m.group(1) else ''}{m.group(2)}"
    return NAME[arm.split("_")[0]], arm


def main():
    plan = list(csv.DictReader(open(PLAN), delimiter="\t"))
    retr, orig, cells, missing = {}, {}, defaultdict(dict), []
    for r in plan:
        m = metrics(r["out_dir"])
        key = (r["dataset"], int(r["seed"]))
        if m is None:
            missing.append(r["out_dir"])
            continue
        if r["role"] == "retrained":
            retr[key] = m
        elif r["role"] == "original":
            orig[key] = m
        else:
            meth, cfg = config_of(r["arm"])
            cells[(r["dataset"], meth, cfg)][int(r["seed"])] = m
    if missing:
        print(f"WARNING: {len(missing)} evals without metrics:")
        for d in missing:
            print("  ", d)

    def agg(per_seed):
        c = [dict(f=m["test/SHF@10"], r=m["test/SHR@10"], sh=m["test/SH@10"],
                  tp=m["test/ndcg@10"] / retr[(ds_, sd)]["test/ndcg@10"],
                  dr=abs(m["test/SHR@10"] - retr[(ds_, sd)]["test/SHR@10"]))
             for (ds_, sd), m in per_seed if (ds_, sd) in retr]
        return {k: S.mean(x[k] for x in c) for k in c[0]} | {"n": len(c)} if c else None

    new = {}
    for (ds, meth, cfg), v in cells.items():
        new[(ds, meth, cfg)] = agg([((ds, sd), m) for sd, m in v.items()])

    def choose(arms):
        ok = [a for a in arms if arms[a]["tp"] >= 0.95]
        return min(ok or list(arms), key=lambda a: (round(arms[a]["f"], 9), arms[a]["sh"], -arms[a]["tp"])), bool(ok)

    ledger = list(csv.DictReader(open(LEDGER)))
    rows = []
    picks = {}
    for ds in DS:
        for meth in NAME.values():
            arms = {cfg: a for (d, m, cfg), a in new.items() if d == ds and m == meth and a and a["n"] == 5}
            if arms:
                picks[(ds, meth)] = choose(arms)
    nau_beauty = picks.get(("beauty", "NAU"))
    for L in ledger:
        ds, meth, cfg = L["dataset"], L["method"], L["config"]
        a = new.get((ds, meth, cfg))
        full = a is not None and a["n"] == 5
        pk = picks.get((ds, meth))
        rows.append(dict(
            dataset=ds, method=meth, config=cfg, source=L["source"],
            orig_val_UHF=float(L["val_UHF"]), orig_val_tauP=float(L["val_tauP"]),
            orig_selected=L["val_selected"] == "True",
            clean_n=a["n"] if a else 0,
            clean_val_UHF=100 * a["f"] if a else "", clean_val_UHR=100 * a["r"] if a else "",
            clean_val_dUHR=100 * a["dr"] if a else "", clean_val_SH=100 * a["sh"] if a else "",
            clean_val_tauP=100 * a["tp"] if a else "",
            clean_selected_per_dataset=bool(full and pk and pk[0] == cfg),
            clean_selected_paper_rule=bool(full and (nau_beauty[0] == cfg if meth == "NAU" else pk and pk[0] == cfg)),
            clean_floor_met=pk[1] if pk else ""))
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader(); w.writerows(rows)
    print("wrote", OUT_CSV)

    # references on the cleaned view (context): original and retrained
    for ds in DS:
        o = [orig[(ds, s)] for s in SEEDS if (ds, s) in orig]
        r = [retr[(ds, s)] for s in SEEDS if (ds, s) in retr]
        if o and r:
            print(f"{ds:7s} ref  original UHF {100 * S.mean(x['test/SHF@10'] for x in o):6.2f} "
                  f"tauP {100 * S.mean(orig[(ds, s)]['test/ndcg@10'] / retr[(ds, s)]['test/ndcg@10'] for s in SEEDS if (ds, s) in orig and (ds, s) in retr):6.1f} | "
                  f"retrained UHF {100 * S.mean(x['test/SHF@10'] for x in r):6.2f}  (n={len(o)}/{len(r)})")
    fmt = lambda x, p: "   n/a" if x == "" else f"{x:6.{p}f}"
    for r in rows:
        print(f"{r['dataset']:7s} {r['method']:9s} {r['config']:24s} orig UHF {r['orig_val_UHF']:6.2f} tauP {r['orig_val_tauP']:6.1f}"
              f"{' *' if r['orig_selected'] else '  '} | clean UHF {fmt(r['clean_val_UHF'], 2)} tauP {fmt(r['clean_val_tauP'], 1)} "
              f"n={r['clean_n']}{' *' if r['clean_selected_paper_rule'] else ''}{' (per-ds)' if r['clean_selected_per_dataset'] and not r['clean_selected_paper_rule'] else ''}")
    for (ds, meth), (p, ok) in sorted(picks.items()):
        old = [r["config"] for r in rows if r["dataset"] == ds and r["method"] == meth and r["orig_selected"]]
        print(f"PICK {ds:7s} {meth:9s} old {old[0] if old else '-':24s} new {p:24s}"
              f"{'' if ok else ' (floor not met)'}{'  CHANGED' if old and old[0] != p else ''}")


if __name__ == "__main__":
    main()
