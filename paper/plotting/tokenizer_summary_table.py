"""One summary table for the tokenizer tables (v33 Tables 11 to 17), so the full per-popularity
tables can move to the supplementary material.

Input: the regenerated, pinned single-execution tokenizer tables
paper/revision/tables/bandwagon_unlearn_<ds>_<tok>_ledger.tex (written by
scripts/make_paper_spam_table.py --pin-arms paper/plotting/valsel_selected_ledger.csv --one-execution),
plus the RQ-KMeans reference (Table 1) from paper/plotting/retraining_distance_valsel.csv.

Pooling: each tokenizer table reports per target popularity (unpopular / mid / popular) the mean
over five data seeds; the pooled value is the mean of the three, i.e. the mean over all 15
(seed, target) runs, the same pooling as Table 1.

Columns per (dataset, tokenizer): Delta^retr_SH / tau_P of NAU, of Forget+Repair, and of the
best other baseline (lowest pooled Delta^retr_SH among baselines with pooled tau_P >= 95%), and
NAU's rank by pooled Delta^retr_SH among all methods with pooled tau_P >= 95%.

    python paper/plotting/tokenizer_summary_table.py
Writes paper/revision/tables/appendix_tokenizer_summary.{tex,csv}.
"""
from __future__ import annotations

import csv
import os
import re

ROOT = "/sc-scratch/sc-scratch-ikim-erase/GRID"
TAB = f"{ROOT}/paper/revision/tables"
SPECS = [("beauty", "rqvae", "RQ-VAE"), ("beauty", "letter_letter", "LETTER"),
         ("beauty", "tracerv3", "TRACER"), ("sports", "rqvae", "RQ-VAE"),
         ("sports", "letter_letter", "LETTER"), ("toys", "rqvae", "RQ-VAE"),
         ("toys", "letter_letter", "LETTER")]
STRATA = ["Unpopular", "Mid", "Popular"]
NUM = re.compile(r"([0-9]+\.[0-9]+)")


def parse(path):
    """-> {stratum: {method: (dsh, tau)}}"""
    out, cur = {}, None
    for line in open(path):
        for s in STRATA:
            if f"{{{s}}}" in line and "rotatebox" in line:
                cur = s
                out[cur] = {}
        if cur is None or "&" not in line or "multicolumn" in line or "Method" in line:
            continue
        cells = [c.strip() for c in line.split("\\\\")[0].split("&")]
        if len(cells) < 7:
            continue
        name = cells[1].replace("\\quad", "").strip()
        nums = [NUM.search(c.replace("\\textbf{", "").replace("\\underline{", "")) for c in cells[2:7]]
        if not all(nums[i] for i in (1, 3)):
            continue
        out[cur][name] = (float(nums[1].group(1)), float(nums[3].group(1)))
    return out


def pooled(tab):
    meths = set.intersection(*(set(tab[s]) for s in STRATA))
    return {m: (sum(tab[s][m][0] for s in STRATA) / 3, sum(tab[s][m][1] for s in STRATA) / 3)
            for m in meths}


def rkmeans(ds):
    rows = {}
    for r in csv.DictReader(open(f"{ROOT}/paper/plotting/retraining_distance_valsel.csv")):
        if r["dataset"] == ds and r["track"] == "rkmeans" and r["method"] not in ("Retrain", "Poisoned", "NAU (ls0.1)"):
            rows[r["method"].replace("Forget+Repair", "Forget + Repair")] = (
                float(r["abs_dsh10"]), float(r["tau_P"]))
    return rows


def summarize(ds, tok_label, P):
    nau = P["NAU"]
    fr = P["Forget + Repair"]
    base = {m: v for m, v in P.items() if m not in ("NAU", "NAU, $\\lambda_s=0.1$")}
    ok = {m: v for m, v in base.items() if v[1] >= 95.0 and m != "Forget + Repair"}
    best = min(ok.items(), key=lambda kv: kv[1][0]) if ok else (None, (float("nan"),) * 2)
    elig = {m: v for m, v in P.items() if v[1] >= 95.0 and m != "NAU, $\\lambda_s=0.1$"}
    rank = sorted(elig, key=lambda m: elig[m][0]).index("NAU") + 1 if nau[1] >= 95.0 else None
    return {"dataset": ds, "tokenizer": tok_label,
            "nau_dsh": nau[0], "nau_tau": nau[1], "fr_dsh": fr[0], "fr_tau": fr[1],
            "best_other": best[0], "best_dsh": best[1][0], "best_tau": best[1][1],
            "nau_rank": rank, "n_eligible": len(elig)}


def main():
    rows = []
    for ds in ("beauty", "sports", "toys"):
        rows.append(summarize(ds, "RQ-KMeans (Table 1)", rkmeans(ds)))
        for d2, tok, lab in SPECS:
            if d2 != ds:
                continue
            f = f"{TAB}/bandwagon_unlearn_{ds}_{tok}_ledger.tex"
            if not os.path.isfile(f):
                print("missing", f)
                continue
            rows.append(summarize(ds, lab, pooled(parse(f))))
    with open(f"{TAB}/appendix_tokenizer_summary.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    for r in rows:
        print(r)

    def fmt(d, t):
        return f"{d:.3f} / {t:.1f}"

    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{3pt}",
         "\\caption{Spam removal across SID tokenizers (TIGER, bandwagon attack, three target "
         "popularities $\\times$ five data seeds per row). Each cell is $\\Delta^{\\mathrm{retr}}_{\\mathrm{SH}}$ / "
         "$\\tau_P$ in \\%, pooled over the 15 runs as in \\Cref{tab:bandwagon-unlearn-all-tiger-rkmeans}. Every baseline uses the "
         "configuration selected for TIGER + RQ-KMeans on the validation split of the same dataset, and NAU "
         "its single selected setting; nothing is selected on test. Best other baseline: lowest "
         "$\\Delta^{\\mathrm{retr}}_{\\mathrm{SH}}$ among the remaining baselines with $\\tau_P \\geq 95\\%$. Rank: "
         "NAU's position by $\\Delta^{\\mathrm{retr}}_{\\mathrm{SH}}$ among all methods with $\\tau_P \\geq 95\\%$. "
         "Per-popularity results are in the supplementary material.}",
         "\\label{tab:appendix-tokenizer-summary}",
         "\\begin{tabular}{llcccc}", "\\toprule",
         "Dataset & SIDs & NAU & Forget+Repair & Best other baseline & Rank \\\\", "\\midrule"]
    prev = None
    for r in rows:
        if prev is not None and r["dataset"] != prev:
            L.append("\\midrule")
        ds = r["dataset"].capitalize() if r["dataset"] != prev else ""
        prev = r["dataset"]
        best = "--" if r["best_other"] is None else f"{r['best_other']}: {fmt(r['best_dsh'], r['best_tau'])}"
        rank = "--" if r["nau_rank"] is None else f"{r['nau_rank']} of {r['n_eligible']}"
        L.append(f"{ds} & {r['tokenizer']} & {fmt(r['nau_dsh'], r['nau_tau'])} & "
                 f"{fmt(r['fr_dsh'], r['fr_tau'])} & {best} & {rank} \\\\")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(f"{TAB}/appendix_tokenizer_summary.tex", "w").write("\n".join(L) + "\n")
    print("wrote", f"{TAB}/appendix_tokenizer_summary.tex")


if __name__ == "__main__":
    main()
