"""Table 1 significance: NAU against each baseline, paired cell by cell.

Each dataset has 15 cells (3 targets x 5 data seeds); within a cell NAU and the baseline start
from the same poisoned model and are scored against the same retrained model. Per cell
Delta_SH = |SH_U - SH_R| and tau_P = NDCG_U / NDCG_R; unlearning replicates of a cell are
averaged first. Runs and configurations as in paper/plotting/extract_table1_valsel.py.

Reported per (dataset, baseline) and pooled over the three datasets (45 cells):
  wins / ties / losses of NAU (Delta_SH: lower wins; tau_P: higher wins; |difference| < 1e-6
  is a tie), the median paired difference, and a two-sided exact sign test over non-tied cells.

    PYTHONPATH=. python paper/plotting/table1_significance.py [--out paper/revision/tables/appendix_table1_significance.tex]

--ledger paper/plotting/canonical_runs.tsv: every cell from the ONE canonical execution of Table 1's
configuration (extract_table1_valsel.ledger_cells: picks of valsel_selected_ledger.csv and
nau_valsel_selected_ledger.csv, no `_useed` replicates). A (dataset, baseline) pair with fewer than 15
canonical cells is written as "pending" instead of being tested on a subset (and "All" is pending
too). Also computes the magnitude-aware tests (Wilcoxon signed-rank, two-sided, zero differences split between ranks (zero_method="zsplit", as in possible_additions section 19); 95%
percentile paired bootstrap over cells, 10,000 resamples, seed 0) for every baseline and writes them
to paper/plotting/table1_significance_ledger.csv. Default output with --ledger:
paper/revision/tables/appendix_table1_significance_ledger.tex.
"""
import argparse
import csv
import importlib.util
import os
import statistics as S
import sys
from collections import defaultdict
from math import comb

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("t1", os.path.join(HERE, "extract_table1_valsel.py"))
t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

TIE = 1e-6


def sign_p(w, l):
    n = w + l
    if n == 0:
        return 1.0
    k = min(w, l)
    return min(1.0, 2 * sum(comb(n, i) for i in range(k + 1)) / 2 ** n)


def cells(ds):
    sel = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open(os.path.join(HERE, "valsel_selected.csv")))}
    R = {s: t1.ref(ds, s, "rkmeans") for s in t1.SEEDS}
    latest = {}
    for r in run_tags.iter_runs("logs/unlearn/runs", setting="spam", dataset=ds, model="tiger", quantizer="rkmeans"):
        if r.get("done") and not r.get("pkm") and "pkm" not in r["tag"]:
            if r["tag"] not in latest or int(r["jid"]) > int(latest[r["tag"]]["jid"]):
                latest[r["tag"]] = r
    out = {}
    for m in t1.ORDER:
        if m in ("TRACER", "NAU (ls0.1)"):
            continue
        arm = t1.FIXED.get(m) or sel.get((ds, t1.VALSEL[m]))
        per = defaultdict(list)
        for r in latest.values():
            if r["arm"] != arm or r["stratum"] not in t1.STR or str(r["seed"]) not in t1.SEEDS:
                continue
            mt = run_tags.last_metrics(r["path"])
            if not mt or "test/SH@10" not in mt:
                continue
            s, st = str(r["seed"]), r["stratum"]
            sh, n = float(mt["test/SH@10"]), float(mt["test/ndcg@10"])
            per[(st, s)].append((abs(sh - R[s]["clean_SH@10"][st]), n / R[s]["clean_ndcg@10"]))
        out[m] = {k: (S.mean(x[0] for x in v), S.mean(x[1] for x in v)) for k, v in per.items()}
    return out


def ledger_cells_all(ledger, ds):
    got = t1.ledger_cells(ledger, ds, methods=[m for m in t1.ORDER if m not in ("TRACER", "NAU (ls0.1)")])
    out = {}
    for m, cells in got.items():
        R = {s: t1.ref(ds, s, "rkmeans") for s in t1.SEEDS}
        out[m] = {k: (abs(sh - R[k[1]]["clean_SH@10"][k[0]]), n / R[k[1]]["clean_ndcg@10"]) for k, (sh, n, _rd) in cells.items()}
    return out


def magnitude(nau, base, j, lower_better, n_boot=10000, seed=0):
    """Paired difference (> 0: NAU better), its mean, 95% percentile bootstrap CI over cells, Wilcoxon p."""
    import random
    from scipy.stats import wilcoxon
    keys = sorted(set(nau) & set(base))
    d = [(base[k][j] - nau[k][j]) if lower_better else (nau[k][j] - base[k][j]) for k in keys]
    if not d:
        return None
    rng = random.Random(seed)
    bs = sorted(S.mean(rng.choice(d) for _ in d) for _ in range(n_boot))
    try:
        p = float(wilcoxon(d, zero_method="zsplit").pvalue) if any(abs(x) > 0 for x in d) else 1.0
    except ValueError:
        p = 1.0
    return S.mean(d), bs[int(0.025 * n_boot)], bs[int(0.975 * n_boot) - 1], p, len(d)


def compare(nau, base, j, lower_better):
    keys = sorted(set(nau) & set(base))
    d = [(base[k][j] - nau[k][j]) if lower_better else (nau[k][j] - base[k][j]) for k in keys]   # > 0: NAU better
    w = sum(x > TIE for x in d); l = sum(x < -TIE for x in d); t = len(d) - w - l
    return w, t, l, (S.median(d) if d else float("nan")), sign_p(w, l), len(keys)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ledger", default=None, help="canonical_runs.tsv: Table 1's canonical execution per cell")
    a = ap.parse_args()
    if a.out is None:
        a.out = ("paper/revision/tables/appendix_table1_significance_ledger.tex" if a.ledger
                 else "paper/revision/tables/appendix_table1_significance.tex")
    if a.ledger:
        C = {ds: ledger_cells_all(a.ledger, ds) for ds in ("beauty", "sports", "toys")}
        pending = {(m, ds) for ds in C for m, v in C[ds].items() if len(v) < 15}
        for m, ds in sorted(pending):
            print(f"PENDING {ds} {m}: {len(C[ds][m])} of 15 canonical cells with a test eval")
        if any(m == "NAU" for m, _ in pending):
            raise SystemExit("NAU itself is incomplete on the ledger")
        for m in {m for m, _ in pending}:
            pending |= {(m, "all")}
    else:
        C = {ds: cells(ds) for ds in ("beauty", "sports", "toys")}
        pending = set()
    res, mag = {}, {}
    for m in [x for x in t1.ORDER if x not in ("TRACER", "NAU (ls0.1)", "NAU")]:
        for ds in ("beauty", "sports", "toys", "all"):
            if ds == "all":
                nau = {(d,) + k: v for d in C for k, v in C[d]["NAU"].items()}
                base = {(d,) + k: v for d in C for k, v in C[d].get(m, {}).items()}
            else:
                nau, base = C[ds]["NAU"], C[ds].get(m, {})
            if (m, ds) in pending:
                res[(m, ds)] = None
                continue
            res[(m, ds)] = (compare(nau, base, 0, True), compare(nau, base, 1, False))
            if a.ledger:
                mag[(m, ds)] = (magnitude(nau, base, 0, True), magnitude(nau, base, 1, False))
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\caption{",
         r"NAU against each baseline, paired cell by cell (\Cref{tab:bandwagon-unlearn-all-tiger-rkmeans} settings; per dataset 15 cells, three targets $\times$ five data seeds, both methods starting from the same poisoned model). "
         r"W/T/L: cells where NAU is better / tied / worse ($\Delta^{\mathrm{retr}}_{\mathrm{SH}}$: lower is better; $\tau_P$: higher is better). "
         r"$p$: two-sided exact sign test over the non-tied cells. All: the 45 cells of the three datasets.}",
         r"\label{tab:appendix-table1-significance}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llcccc}", r"\toprule",
         r" & & \multicolumn{2}{c}{$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$} & \multicolumn{2}{c}{$\tau_P$} \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6}",
         r"Baseline & Data & W/T/L & $p$ & W/T/L & $p$ \\", r"\midrule"]
    for m in [x for x in t1.ORDER if x not in ("TRACER", "NAU (ls0.1)", "NAU")]:
        for i, ds in enumerate(("beauty", "sports", "toys", "all")):
            if res[(m, ds)] is None:
                L.append(f"{m if i == 0 else ''} & {ds.capitalize()} & \\multicolumn{{4}}{{c}}{{pending}} \\\\")
                print(f"{m:14s} {ds:7s} PENDING (canonical re-runs not finished)")
                continue
            (w, t, l, md, p, n), (w2, t2, l2, md2, p2, n2) = res[(m, ds)]
            fp = lambda x: "$<$0.001" if x < 0.001 else f"{x:.3f}"
            L.append(f"{m if i == 0 else ''} & {ds.capitalize()} & {w}/{t}/{l} & {fp(p)} & {w2}/{t2}/{l2} & {fp(p2)} \\\\")
            print(f"{m:14s} {ds:7s} dSH W/T/L {w}/{t}/{l} med {100 * md:+.3f} p={p:.4f} | tau W/T/L {w2}/{t2}/{l2} med {100 * md2:+.2f} p={p2:.4f} (cells {n})")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote", a.out)
    if a.ledger:
        out = "paper/plotting/table1_significance_ledger.csv"
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["baseline", "dataset", "metric", "n_cells", "W", "T", "L", "median_diff", "sign_p",
                        "mean_diff", "boot_lo", "boot_hi", "wilcoxon_p"])
            for (m, ds), r in res.items():
                for j, met in ((0, "dSH"), (1, "tauP")):
                    if r is None:
                        w.writerow([m, ds, met, "", "", "", "", "", "", "", "", "", "pending"]); continue
                    ww, tt, ll, md, p, n = r[j]
                    mm = mag[(m, ds)][j]
                    w.writerow([m, ds, met, n, ww, tt, ll, f"{100 * md:.4f}", f"{p:.5f}",
                                f"{100 * mm[0]:.4f}", f"{100 * mm[1]:.4f}", f"{100 * mm[2]:.4f}", f"{mm[3]:.5f}"])
                    if met == "dSH" or m == "Forget+Repair":
                        print(f"  {m:14s} {ds:7s} {met:4s} mean diff {100 * mm[0]:+.3f} [{100 * mm[1]:+.3f}, {100 * mm[2]:+.3f}] Wilcoxon p={mm[3]:.4f}")
        print("wrote", out)


if __name__ == "__main__":
    main()
