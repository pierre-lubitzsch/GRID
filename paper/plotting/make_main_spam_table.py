"""Main spam table (tables/paper_spam_all_tiger_rkmeans_pooled.tex) from
plotting/retraining_distance.csv.

    python plotting/make_main_spam_table.py      # from paper/

Ranking is by distance to retraining, not by remaining exposure:
  * Delta^retr_SH = mean |SH_U - SH_R| (lower is better, bold/underline),
  * tau_P ranked by |tau_P - 100| (utility above retraining is not rewarded),
  * E lower is better.
SH@10 itself is printed without ranking marks: 0 is not the target when the
retrained model still exposes the item. Only rows with tau_P >= 95 (and not the dagger TRACER/Beauty row) are ranked;
rows below it are marked with a double dagger. Reference rows are never ranked.
"""
from __future__ import annotations

import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent
DATA = HERE / "retraining_distance.csv"
OUT = HERE.parent / "tables" / "paper_spam_all_tiger_rkmeans_pooled.tex"

DATASETS = ["beauty", "sports", "toys"]
BASELINES = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF",
             "Kookmin", "Fanchuan", "TRACER"]
OURS = [("NAU w/o Lsep", r"NAU w/o $\mathcal{L}_{\mathrm{sep}}$"),
        ("NAU", r"NAU \textit{(Ours)}")]
# Median seconds per unlearning request, read from each run's own latency
# instrumentation (h2h.request_latency_minutes) over the selected arm's runs.
E = {
    "beauty": {"Finetune": 0.5, "Forget only": 0.3, "Forget+Repair": 0.7,
               "SCIF": 35.3, "SEIF": 0.5, "Kookmin": 0.4, "Fanchuan": 1.6,
               "TRACER": 2.7, "NAU w/o Lsep": 2.5, "NAU": 3.5},
    "sports": {"Finetune": 0.7, "Forget only": 0.3, "Forget+Repair": 0.9,
               "SCIF": 34.8, "SEIF": 0.7, "Kookmin": 0.5, "Fanchuan": 2.1,
               "TRACER": 2.7, "NAU w/o Lsep": 2.9, "NAU": 4.2},
    "toys": {"Finetune": 0.5, "Forget only": 0.3, "Forget+Repair": 0.7,
             "SCIF": 34.3, "SEIF": 0.5, "Kookmin": 0.4, "Fanchuan": 1.6,
             "TRACER": 2.3, "NAU w/o Lsep": 2.5, "NAU": 3.4},
}
FLOOR = 95.0


def load():
    d = {}
    with open(DATA) as f:
        for r in csv.DictReader(f):
            d[(r["dataset"], r["method"], r["track"])] = r
    return d


def get(d, ds, m):
    # TRACER on Beauty only exists on TRACER's own tokenizer (dagger row).
    if m == "TRACER" and ds == "beauty":
        return d.get((ds, m, "tracerv3")), True
    return d.get((ds, m, "rkmeans")), False


def pm(m, sd, prec, width=None):
    s = f"{m:.{prec}f}"
    if width and len(s) < width:
        s = r"\phantom{" + "0" * (width - len(s)) + "}" + s
    return s if sd is None else s + rf"\,{{\scriptsize $\pm${sd:.{prec}f}}}"


def main():
    d = load()
    methods = BASELINES + [k for k, _ in OURS]
    marks = {}
    for ds in DATASETS:
        ranked = []
        for m in methods:
            r, dag = get(d, ds, m)
            if r and not dag and float(r["tau_P"]) >= FLOOR:
                ranked.append((m, r))
        for key, f in (("d", lambda r: float(r["abs_dsh10"])),
                       ("t", lambda r: abs(float(r["tau_P"]) - 100.0)),
                       ("e", None)):
            if key == "e":
                order = sorted(ranked, key=lambda x: E[ds][x[0]])
            else:
                order = sorted(ranked, key=lambda x: f(x[1]))
            if order:
                marks[(ds, key, order[0][0])] = "b"
            if len(order) > 1:
                marks[(ds, key, order[1][0])] = "u"

    def deco(s, ds, key, m):
        k = marks.get((ds, key, m))
        return rf"\textbf{{{s}}}" if k == "b" else (rf"\underline{{{s}}}" if k == "u" else s)

    def cells(ds, m, ref=False):
        r, dag = get(d, ds, m)
        if r is None:
            return ["--"] * 4
        sh = pm(float(r["sh10"]), float(r["sh10_sd"]), 3)
        if m == "Retrain":
            ds_ = "0"
        else:
            ds_ = deco(pm(float(r["abs_dsh10"]), float(r["abs_dsh10_sd"]), 3), ds, "d", m)
        tp = pm(float(r["tau_P"]), None if m == "Retrain" else float(r["tau_P_sd"]), 1, width=5)
        tp = tp if ref else deco(tp, ds, "t", m)
        e = "--" if ref else deco(f"{E[ds][m]:.1f}", ds, "e", m)
        low = (not ref) and float(r["tau_P"]) < FLOOR
        sfx = (r"\textsuperscript{\dag}" if dag else "") + (r"\textsuperscript{\ddag}" if low else "")
        return [sh + sfx, ds_, tp, e]

    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.05}",
         r"\caption{",
         r"Spam removal with TIGER + RQ-KMeans under a 1\% bandwagon attack (five data seeds $\times$ three targets per dataset; mean $\pm$ s.d.\ across seeds).",
         r"The retrained model is the target: lower $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ is better, $\tau_P$ is ranked by its distance from $100\%$, and $\mathrm{SH}@10$ is not ranked. Bold/underline mark the best/second best among rows with $\tau_P\geq95\%$.",
         r"\textsuperscript{\ddag}\,No configuration reaches $\tau_P\geq95\%$. \textsuperscript{\dag}\,Measured on TRACER's own tokenizer and its own references, not matched to the other Beauty rows.",
         r"Among methods that keep utility, NAU is closest to retrained exposure on Beauty and Toys and within $0.02$ points of the best on Sports.",
         r"}",
         r"\label{tab:bandwagon-unlearn-all-tiger-rkmeans}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lcccccccccccc}", r"\toprule",
         r"\multirow{2}{*}[-3pt]{\textbf{Method}} & \multicolumn{4}{c}{\textbf{Beauty}} & \multicolumn{4}{c}{\textbf{Sports}} & \multicolumn{4}{c}{\textbf{Toys}} \\",
         r"\cmidrule(lr){2-5} \cmidrule(lr){6-9} \cmidrule(lr){10-13}",
         " & ".join([""] + [r"$\mathrm{SH}@10{\scriptstyle\,(\%)}$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}$ & $\mathcal{E}{\scriptstyle\,(s)}\downarrow$"] * 3) + r" \\",
         r"\midrule"]
    for m, lbl in (("Poisoned", r"\textit{Poisoned (no unlearning)}"), ("Retrain", r"\textit{Retrain (target)}")):
        L.append(" & ".join([lbl] + sum((cells(ds, m, ref=True) for ds in DATASETS), [])) + r" \\")
    L.append(r"\midrule")
    for m in BASELINES:
        L.append(" & ".join([m] + sum((cells(ds, m) for ds in DATASETS), [])) + r" \\")
    L.append(r"\cmidrule(lr){1-13}")
    for m, lbl in OURS:
        L.append(" & ".join([lbl] + sum((cells(ds, m) for ds in DATASETS), [])) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}"]
    OUT.write_text("\n".join(L) + "\n")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
