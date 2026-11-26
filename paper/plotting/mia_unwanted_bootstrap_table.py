"""Membership inference on sensitive item deletion: AUC minus the retrained model's AUC with
paired bootstrap 95% intervals, for every method (extends
paper/plotting/critique_followup/mia_unwanted_bootstrap.py, same data and resampling scheme).

Data: logs/diagnose/mia_unwanted/<ds>_seed<N>.json (per-example log p of the `_mia` executions);
the "original" row is replaced by logs/diagnose/mia_unwanted_lastckpt/ where present (the checkpoint
unlearning starts from). Bootstrap: 4000 resamples; members resampled within each seed, non-members
(identical held-out cases for every seed) resampled jointly; all models scored on the same resample
(paired); AUC averaged over the five seeds.

Utility column: tau_P (test, cleaned evaluation histories) parsed from
paper/revision/tables/sensitive_<ds>_with_dUHF_scrub_ledger.tex.

    PYTHONPATH=. python paper/plotting/mia_unwanted_bootstrap_table.py
Writes paper/revision/tables/appendix_mia_unwanted_bootstrap.{tex,csv}.
"""
from __future__ import annotations

import csv
import json
import os
import re

import numpy as np

ROOT = "/sc-scratch/sc-scratch-ikim-erase/GRID"
DATASETS = ["beauty", "sports", "toys"]
SEEDS = [2, 3, 5, 7, 11]
ORDER = [("original", "Before deletion"), ("finetune", "Finetune"), ("scif", "SCIF"),
         ("tracer", "TRACER"), ("kookmin", "Kookmin"), ("seif", "SEIF"),
         ("fanchuan", "Fanchuan"), ("forget_only", "Forget only"),
         ("forget_repair", "Forget+Repair"), ("nau", "NAU \\textit{(ours)}")]
TEX_NAME = {"original": "Original (before deletion)", "finetune": "Finetune", "scif": "SCIF",
            "tracer": "TRACER", "kookmin": "Kookmin", "seif": "SEIF", "fanchuan": "Fanchuan",
            "forget_only": "Forget only", "forget_repair": "Forget+Repair", "nau": "NAU \\textit{(Ours)}"}
B = 4000


def auc(m, n):
    m = np.asarray(m)[:, None]
    n = np.asarray(n)[None, :]
    return float((m > n).mean() + 0.5 * (m == n).mean())


def auc_b(M, N):
    g = (M[:, :, None] > N[:, None, :]).mean(axis=(1, 2))
    e = (M[:, :, None] == N[:, None, :]).mean(axis=(1, 2))
    return g + 0.5 * e


def tau_p(ds):
    """method label (as printed in the scrub table) -> tau_P mean."""
    f = f"{ROOT}/paper/revision/tables/sensitive_{ds}_with_dUHF_scrub_ledger.tex"
    out = {}
    if not os.path.isfile(f):
        return out
    for line in open(f):
        if "&" not in line or "\\\\" not in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        name = re.sub(r"\$\^\{\\dagger\}\$", "", cells[0]).strip()
        nums = [re.match(r"[\\textbf{\s]*([0-9.]+)", c.replace("\\textbf{", "").replace("\\underline{", "")) for c in cells[1:]]
        vals = [float(m.group(1)) for m in nums if m]
        if len(vals) >= 6:
            out[name] = vals[5]  # columns: UHF, dUHF, UHR, dUHR, NDCG, tau_P, ...
    return out


def main():
    rng = np.random.default_rng(0)
    rows_csv = []
    res = {}
    counts = {}
    for ds in DATASETS:
        data = []
        for sd in SEEDS:
            j = json.load(open(f"{ROOT}/logs/diagnose/mia_unwanted/{ds}_seed{sd}.json"))
            r = {x["method"]: x for x in j["rows"]}
            of = f"{ROOT}/logs/diagnose/mia_unwanted_lastckpt/{ds}_seed{sd}.json"
            if os.path.isfile(of):
                o = {x["method"]: x for x in json.load(open(of))["rows"]}
                r["original"] = o["original"]
            for k, x in r.items():
                a = auc(x["member_logprob"], x["nonmember_logprob"])
                assert abs(a - x["auc"]) < 1e-6, (ds, sd, k, a, x["auc"])
            data.append((sd, j["n_members"], j["n_nonmembers"], r))
        nn = data[0][2]
        assert all(d[2] == nn for d in data)
        counts[ds] = (sum(d[1] for d in data), nn)
        meths = [m for m, _ in ORDER if all(m in d[3] for d in data)] + ["retrained"]
        point = {m: float(np.mean([d[3][m]["auc"] for d in data])) for m in meths}
        boots = {m: np.zeros(B) for m in meths}
        INN = rng.integers(0, nn, (B, nn))
        for sd, nm, _, r in data:
            im = rng.integers(0, nm, (B, nm))
            for m in meths:
                M = np.asarray(r[m]["member_logprob"])[im]
                N = np.asarray(r[m]["nonmember_logprob"])[INN]
                boots[m] += auc_b(M, N) / len(data)
        res[ds] = {"retrained_auc": point["retrained"]}
        for m in meths:
            if m == "retrained":
                continue
            d = boots[m] - boots["retrained"]
            lo, hi = np.percentile(d, [2.5, 97.5])
            res[ds][m] = (point[m] - point["retrained"], float(lo), float(hi), point[m])
            rows_csv.append({"dataset": ds, "method": m, "auc": round(point[m], 4),
                             "auc_retrained": round(point["retrained"], 4),
                             "diff": round(point[m] - point["retrained"], 4),
                             "lo": round(float(lo), 4), "hi": round(float(hi), 4),
                             "includes_zero": bool(lo <= 0 <= hi)})
        print(ds, "members", counts[ds][0], "non-members", nn)
        for m in meths:
            if m != "retrained":
                v = res[ds][m]
                print(f"  {m:14s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]  auc {v[3]:.3f}")

    out = f"{ROOT}/paper/revision/tables/appendix_mia_unwanted_bootstrap"
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0].keys()))
        w.writeheader()
        w.writerows(rows_csv)

    taus = {ds: tau_p(ds) for ds in DATASETS}

    def cell(ds, m):
        if m not in res[ds]:
            return "--", "--"
        d, lo, hi, _ = res[ds][m]
        s = f"{d:+.3f} {{\\scriptsize[{lo:+.3f}, {hi:+.3f}]}}"
        if lo <= 0 <= hi:
            s = f"\\textbf{{{d:+.3f}}} {{\\scriptsize[{lo:+.3f}, {hi:+.3f}]}}"
        t = taus[ds].get(TEX_NAME[m].replace(" \\textit{(Ours)}", " \\textit{(Ours)}"))
        if t is None:
            t = next((v for k, v in taus[ds].items() if k.startswith(TEX_NAME[m].split(" ")[0])), None)
        return s, ("--" if t is None else f"{t:.1f}")

    lines = [
        "\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
        "\\caption{Membership inference on the deleted interactions in sensitive item deletion "
        "(TIGER + RQ-KMeans, five data seeds). A loss-based attack scores each interaction by the "
        "model's log-probability of the item given the user's preceding history. Members are the "
        f"requesting users' training interactions with the category ({counts['beauty'][0]} / "
        f"{counts['sports'][0]} / {counts['toys'][0]} over five seeds), non-members the "
        f"{counts['beauty'][1]} / {counts['sports'][1]} / {counts['toys'][1]} held-out cases with the "
        "category as next item. Members and non-members come from different users and positions, so "
        "even a model that never saw the members separates them to some degree; the reference is "
        "therefore the retrained model's AUC "
        f"({res['beauty']['retrained_auc']:.3f} / {res['sports']['retrained_auc']:.3f} / "
        f"{res['toys']['retrained_auc']:.3f}), not 0.5. Entries: AUC (mean over seeds) minus the "
        "retrained model's, with paired bootstrap 95\\% intervals (4000 resamples; members within "
        "seed, non-members jointly). Bold: interval includes 0. Values below 0 mean the deleted "
        "interactions became less likely than unseen ones, which itself reveals the deletion. "
        "$\\tau_P$: test utility in \\% of the retrained model (cleaned evaluation histories).}",
        "\\label{tab:appendix-mia-unwanted-bootstrap}",
        "\\begin{tabular}{lcccccc}", "\\toprule",
        " & \\multicolumn{2}{c}{Beauty} & \\multicolumn{2}{c}{Sports} & \\multicolumn{2}{c}{Toys} \\\\",
        "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
        "Method & $\\Delta$AUC & $\\tau_P$ & $\\Delta$AUC & $\\tau_P$ & $\\Delta$AUC & $\\tau_P$ \\\\",
        "\\midrule",
    ]
    for m, _ in ORDER:
        cells = []
        for ds in DATASETS:
            cells += list(cell(ds, m))
        lines.append(TEX_NAME[m] + " & " + " & ".join(cells) + " \\\\")
        if m == "original":
            lines.append("\\midrule")
    lines += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(out + ".tex", "w").write("\n".join(lines) + "\n")
    print("wrote", out + ".tex", out + ".csv")


if __name__ == "__main__":
    main()
