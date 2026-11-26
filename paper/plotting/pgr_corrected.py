"""PGR with the corrected poisoned references (the exact last.ckpt files the runs start from).

PGR = (SH_P - SH_U) / (SH_P - SH_R): the share of the attack's exposure gap that unlearning
removes (1 = retrained level, > 1 = below the retrained model, < 0 = worse than poisoned).
Poisoned SH from logs/diagnose/poison_ref_start (run_poison_ref_start.sh), retrained SH from
paper/data/spam_references, unlearned SH from each method's Table 1 runs (same selection as
paper/plotting/extract_table1_valsel.py). Two aggregates per dataset, both fixed before any
method is read:
  pooled  ratio of the 15-cell means (no per-cell denominator)
  landed  mean of per-cell PGR over cells where the attack landed (SH_P - SH_R >= 0.005)

    PYTHONPATH=. python paper/plotting/pgr_corrected.py [--out paper/revision/tables/appendix_pgr.tex]

--ledger paper/plotting/canonical_runs.tsv: SH_U from the ONE canonical execution per cell of Table 1's
configuration (extract_table1_valsel.ledger_cells; ledger picks, no `_useed` replicates); a method with
fewer than 15 canonical cells is skipped (pending). Default output with --ledger:
paper/revision/tables/appendix_pgr_ledger.tex.
"""
import argparse
import csv
import importlib.util
import json
import os
import statistics as S
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t1", os.path.join(HERE, "extract_table1_valsel.py"))
t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

SEEDS, STR = t1.SEEDS, t1.STR
LANDED = 0.005


def poisoned(ds, st, sd):
    r = list(csv.DictReader(open(f"logs/diagnose/poison_ref_start/{ds}_{st}_s{sd}/csv/version_0/metrics.csv")))[-1]
    return float(r["test/SH@10"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ledger", default=None, help="canonical_runs.tsv: Table 1's canonical executions")
    a = ap.parse_args()
    if a.out is None:
        a.out = "paper/revision/tables/appendix_pgr_ledger.tex" if a.ledger else "paper/revision/tables/appendix_pgr.tex"
    sel = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open(os.path.join(HERE, "valsel_selected.csv")))}
    res = {}
    for ds in ("beauty", "sports", "toys"):
        R = {s: t1.ref(ds, s, "rkmeans") for s in SEEDS}
        P = {(st, s): poisoned(ds, st, s) for st in STR for s in SEEDS}
        C = {(st, s): R[s]["clean_SH@10"][st] for st in STR for s in SEEDS}
        landed = {k for k in P if P[k] - C[k] >= LANDED}
        if a.ledger:
            LC = t1.ledger_cells(a.ledger, ds, methods=[m for m in t1.ORDER if m not in ("TRACER", "NAU (ls0.1)")])
        latest = {}
        for r in ([] if a.ledger else run_tags.iter_runs("logs/unlearn/runs", setting="spam", dataset=ds, model="tiger", quantizer="rkmeans")):
            if r.get("done") and not r.get("pkm") and "pkm" not in r["tag"]:
                if r["tag"] not in latest or int(r["jid"]) > int(latest[r["tag"]]["jid"]):
                    latest[r["tag"]] = r
        for m in t1.ORDER:
            if m in ("TRACER", "NAU (ls0.1)"):
                continue
            arm = t1.FIXED.get(m) or sel.get((ds, t1.VALSEL[m]))
            U = {}
            for r in latest.values():
                if r["arm"] == arm and r["stratum"] in STR and str(r["seed"]) in SEEDS:
                    mt = run_tags.last_metrics(r["path"])
                    if mt and "test/SH@10" in mt:
                        U.setdefault((r["stratum"], str(r["seed"])), []).append(float(mt["test/SH@10"]))
            U = {k: S.mean(v) for k, v in U.items()}
            if a.ledger:
                U = {k: v[0] for k, v in LC[m].items()}
            if len(U) < 15:
                print(f"  {ds} {m}: {len(U)} cells"); continue
            pooled = (S.mean(P.values()) - S.mean(U.values())) / (S.mean(P.values()) - S.mean(C.values()))
            per = [(P[k] - U[k]) / (P[k] - C[k]) for k in landed]
            res[(ds, m)] = (pooled, S.mean(per), len(landed))
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{",
         r"Poisoning gap recovered, $\mathrm{PGR} = (\mathrm{SH}_P - \mathrm{SH}_U) / (\mathrm{SH}_P - \mathrm{SH}_R)$: the share of the attack's exposure gap removed by unlearning (1: retrained level; above 1: below it). "
         r"$\mathrm{SH}_P$ is the poisoned model the methods start from. Pooled: ratio of the means over three targets $\times$ five data seeds. "
         r"Landed: mean of per-cell PGR over the cells where the attack raised exposure by at least $0.5$ points (the cell set is fixed before any method is read; numbers in parentheses). Methods at their \Cref{tab:bandwagon-unlearn-all-tiger-rkmeans} settings.}",
         r"\label{tab:appendix-pgr}", r"\begin{tabular}{lcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Beauty} & \multicolumn{2}{c}{Sports} & \multicolumn{2}{c}{Toys} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}"]
    nl = {ds: next((v[2] for (d, m), v in res.items() if d == ds), 0) for ds in ("beauty", "sports", "toys")}
    L.append("Method & " + " & ".join(f"Pooled & Landed ({nl[d]})" for d in ("beauty", "sports", "toys")) + r" \\")
    L.append(r"\midrule")
    for m in t1.ORDER:
        if not any((d, m) in res for d in ("beauty", "sports", "toys")):
            continue
        cells = []
        for d in ("beauty", "sports", "toys"):
            v = res.get((d, m))
            cells += [f"{v[0]:.2f}", f"{v[1]:.2f}"] if v else ["--", "--"]
        L.append((r"NAU \textit{(ours)}" if m == "NAU" else m) + " & " + " & ".join(cells) + r" \\")
        print(f"{m:14s} " + "  ".join(cells))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("landed cells:", nl, "wrote", a.out)


if __name__ == "__main__":
    main()
