"""Appendix table: the retraining noise floor of Delta^retr_SH and tau_P.

The five clean models of a dataset are retrained references trained on identical data with
different training seeds. Scoring each against the other seeds' spam targets
(logs/diagnose/retrain_floor/, run_retrain_floor.sh) gives the distance between two equally
valid retrained models, pooled like Table 1. Methods' Delta^retr_SH (Table 1 data,
paper/plotting/retraining_distance_valsel.csv) are reported as multiples of that floor.

    PYTHONPATH=. python paper/plotting/retrain_floor_table.py [--out paper/revision/tables/retrain_noise_floor.tex]

--data CSV reads the method rows from another Table 1 data file (default unchanged).
--ledger paper/plotting/canonical_runs.tsv: every method row from the ONE canonical execution per
cell of Table 1's configuration (extract_table1_valsel.ledger_cells), and the Poisoned row from the
exact checkpoint the runs start from (logs/diagnose/poison_ref_start/, possible_additions section 14),
independent of how any CSV was produced. A method with fewer than 15 canonical cells is "pending".
Default output with --ledger: paper/revision/tables/retrain_noise_floor_ledger.tex (the old
`_poisonref` variant is the same table now, since the ledger version always uses the exact start).
"""
import importlib.util
import os
import argparse
import csv
import json
import statistics as S

SEEDS = [2, 3, 5, 7, 11]
STR = ["unpopular", "mid", "popular"]
METHODS = ["NAU", "Forget+Repair", "Finetune", "SCIF", "TRACER", "Poisoned"]


def ref(ds, sd):
    return json.load(open(f"logs/eval/spam_references/{ds}" + ("" if sd == 2 else f"_seed{sd}") + ".json"))


def sh(ds, ms, ts, st):
    r = list(csv.DictReader(open(f"logs/diagnose/retrain_floor/{ds}_m{ms}_t{ts}_{st}/csv/version_0/metrics.csv")))[-1]
    return float(r["test/SH@10"])


def floor(ds):
    R = {sd: ref(ds, sd) for sd in SEEDS}
    per_seed = [S.mean(S.mean(abs(sh(ds, ms, ts, st) - R[ts]["clean_SH@10"][st]) for ms in SEEDS if ms != ts)
                       for st in STR) for ts in SEEDS]
    ratios = [R[a]["clean_ndcg@10"] / R[b]["clean_ndcg@10"] for a in SEEDS for b in SEEDS if a != b]
    return 100 * S.mean(per_seed), 100 * min(ratios), 100 * max(ratios)


def ledger_rows(ledger):
    """(dataset, method) -> {"abs_dsh10": str(%)} from the canonical executions; None = pending."""
    here = os.path.dirname(os.path.abspath(__file__))
    spec = importlib.util.spec_from_file_location("t1", os.path.join(here, "extract_table1_valsel.py"))
    t1m = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1m)
    out = {}
    for ds in ("beauty", "sports", "toys"):
        R = {sd: ref(ds, sd) for sd in SEEDS}
        cells = t1m.ledger_cells(ledger, ds, methods=[m for m in METHODS if m != "Poisoned"])
        for m, c in cells.items():
            if m == "TRACER" and ds == "beauty":
                continue                      # own tokenizer, own references: "--" as before
            if len(c) < 15:
                out[(ds, m)] = None
                print(f"  PENDING {ds} {m}: {len(c)} of 15 canonical cells")
                continue
            per_seed = [S.mean(abs(c[(st, str(sd))][0] - R[sd]["clean_SH@10"][st]) for st in STR) for sd in SEEDS]
            out[(ds, m)] = {"abs_dsh10": f"{100 * S.mean(per_seed):.4f}"}
        per_seed = []
        for sd in SEEDS:
            v = []
            for st in STR:
                r = [x for x in csv.DictReader(open(f"logs/diagnose/poison_ref_start/{ds}_{st}_s{sd}/csv/version_0/metrics.csv"))
                     if x.get("test/SH@10")][-1]
                v.append(abs(float(r["test/SH@10"]) - R[sd]["clean_SH@10"][st]))
            per_seed.append(S.mean(v))
        out[(ds, "Poisoned")] = {"abs_dsh10": f"{100 * S.mean(per_seed):.4f}"}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--data", default="paper/plotting/retraining_distance_valsel.csv")
    ap.add_argument("--ledger", default=None, help="canonical_runs.tsv: Table 1's canonical executions")
    a = ap.parse_args()
    if a.out is None:
        a.out = ("paper/revision/tables/retrain_noise_floor_ledger.tex" if a.ledger
                 else "paper/revision/tables/retrain_noise_floor.tex")
    if a.ledger:
        t1 = ledger_rows(a.ledger)
    else:
        t1 = {(r["dataset"], r["method"]): r for r in csv.DictReader(open(a.data)) if r["track"] == "rkmeans"}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{",
         r"Retraining noise floor (TIGER + RQ-KMeans). The five retrained models of a dataset are trained on identical data with different training seeds; "
         r"\emph{Retrained vs.\ retrained} scores each against the other seeds' spam targets, pooled like \Cref{tab:bandwagon-unlearn-all-tiger-rkmeans}, and gives the range of $\tau_P$ between two retrained models. "
         r"The other rows give each method's $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ from \Cref{tab:bandwagon-unlearn-all-tiger-rkmeans} and, in parentheses, its multiple of the noise floor. On Beauty, TRACER uses its own tokenizer, whose retrained references differ, so it is omitted (--).}",
         r"\label{tab:appendix-retrain-floor}",
         r"\begin{tabular}{lccc}", r"\toprule", r" & Beauty & Sports & Toys \\", r"\midrule"]
    fl = {ds: floor(ds) for ds in ("beauty", "sports", "toys")}
    L.append(r"Retrained vs.\ retrained $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}$ & "
             + " & ".join(f"{fl[d][0]:.3f}" for d in fl) + r" \\")
    L.append(r"Retrained vs.\ retrained $\tau_P{\scriptstyle\,(\%)}$ & "
             + " & ".join(f"{fl[d][1]:.1f} to {fl[d][2]:.1f}" for d in fl) + r" \\")
    L.append(r"\midrule")
    for m in METHODS:
        cells = []
        for d in fl:
            r = t1.get((d, m), "absent")
            if r is None:
                cells.append("pending"); continue
            r = None if r == "absent" else r
            cells.append("--" if r is None else f"{float(r['abs_dsh10']):.3f} ({float(r['abs_dsh10']) / fl[d][0]:.1f}$\\times$)")
        label = r"NAU \textit{(ours)}" if m == "NAU" else (r"\textit{Poisoned (no unlearning)}" if m == "Poisoned" else m)
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    for d, v in fl.items():
        print(f"{d}: floor dSH {v[0]:.3f}%, tau_P {v[1]:.1f} to {v[2]:.1f}")
    for m in METHODS:
        print(f"  {m:14s} " + "  ".join("pending" if t1.get((d, m), 0) is None else
                                        ("--" if (d, m) not in t1 else f"{float(t1[(d, m)]['abs_dsh10']):.3f} ({float(t1[(d, m)]['abs_dsh10']) / fl[d][0]:.1f}x)")
                                        for d in fl))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
