"""Sensitive item deletion, larger-request variant (budget 5e-4 instead of the paper's 1e-4).

Reads the evals of run_sens5e4_*.sh (plan rows written by scripts/plan_sens5e4_eval.py,
outputs logs/eval/sensitive_5e-4/<view>/...) and writes
  paper/revision/sensitive_larger_request.md                       summary (both views)
  paper/revision/tables/sensitive_larger_request_<ds>.tex           Table 3/22/23 style (cleaned view)
  paper/revision/tables/sensitive_larger_request_<ds>_normal.tex    same, normal view
  paper/revision/tables/sensitive_larger_request_lambda_n.tex       Table 4 style (lambda_n 0 vs -1)
  paper/plotting/sensitive_larger_request.csv                       per (dataset, view, method, seed)
Metrics per seed (as extract_sensitive.cell_entry): UHF@10 = test/SHF@10, UHR@10 = test/SHR@10,
Delta^retr_UHR = |UHR_U - UHR_R| against the retrained model of the same seed, NDCG@10,
tau_P = NDCG_U / NDCG_R (same seed); mean +- sample std over the 5 data seeds.
Configurations are the canonical (`_mia`) ones; no re-selection. A dagger marks a method whose mean
tau_P is below 95% (the paper's utility floor). For reference the 1e-4 numbers on the same view are
read from logs/eval/sensitive_scrub (plan.tsv, plus plan_canonck.tsv for NAU lambda_n=0) when present.

    PYTHONPATH=. python paper/plotting/sensitive_larger_request_table.py
"""
from __future__ import annotations

import csv
import glob
import json
import math
import os
import statistics as st
from collections import defaultdict

R = "5e-4"
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
CATNAME = {"beauty": "Hair Loss Products", "sports": "Guns \\& Rifles", "toys": "toy weapons"}
SEEDS = (2, 3, 5, 7, 11)
ORDER = [("finetune", "Finetune"), ("forget_only", "Forget only"), ("forget_repair", "Forget+Repair"),
         ("scif", "SCIF"), ("seif", "SEIF"), ("kookmin", "Kookmin"), ("fanchuan", "Fanchuan"),
         ("tracer", "TRACER"), ("nau", "NAU (ours)"), ("nau_ln0", "NAU, lambda_n=0")]
REFS = [("original", "Original (before deletion)"), ("retrained", "Retrain (reference)")]
OUTD = os.environ.get("S5E4_OUTD", "paper/revision")
CSV_OUT = os.environ.get("S5E4_CSV", "paper/plotting/sensitive_larger_request.csv")


def last_metrics(out_dir):
    p = os.path.join(out_dir, "csv", "version_0", "metrics.csv")
    if not os.path.isfile(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("test/ndcg@10")]
    if not rows:
        return None
    r = rows[-1]
    return {k: float(r[k]) for k in ("test/ndcg@10", "test/SHF@10", "test/SHR@10", "test/SH@10") if r.get(k)}


def load(plans):
    """(ds, view, method, seed) -> metrics."""
    out = {}
    for p in plans:
        if not os.path.isfile(p):
            continue
        for r in csv.DictReader(open(p), delimiter="\t"):
            r = {k: (v or "").strip() for k, v in r.items()}
            if (r.get("experiment") or "tiger_train_flat") != "tiger_train_flat":
                continue        # LETTER rows of plan_canonck.tsv (same method names)
            m = last_metrics(r["out_dir"])
            if m:
                out[(r["dataset"], r["view"], r["method"], int(r["seed"]))] = m
    return out


def per_seed(data, ds, view, method):
    """seed -> dict(UHF, UHR, dUHR, NDCG, tauP), only seeds with the retrained reference."""
    res = {}
    for sd in SEEDS:
        m, ref = data.get((ds, view, method, sd)), data.get((ds, view, "retrained", sd))
        if not m or not ref or "test/SHF@10" not in m:
            continue
        res[sd] = dict(UHF=100 * m["test/SHF@10"], UHR=100 * m["test/SHR@10"],
                       dUHR=100 * abs(m["test/SHR@10"] - ref["test/SHR@10"]),
                       NDCG=100 * m["test/ndcg@10"], tauP=100 * m["test/ndcg@10"] / ref["test/ndcg@10"])
    return res


def agg(vals):
    vals = list(vals)
    if not vals:
        return None, None
    return st.mean(vals), (st.stdev(vals) if len(vals) > 1 else 0.0)


def users():
    """(ds, seed) -> (n requesting users, n users touching the category) for R and 1e-4."""
    u = {}
    for ratio in (R, "1e-4"):
        for ds, c in CAT.items():
            for sd in SEEDS:
                p = f"src/data/amazon_data/{ds}/forget_manifest_{c}_{ratio}_seed{sd}.json"
                if os.path.isfile(p):
                    m = json.load(open(p))
                    u[(ratio, ds, sd)] = (m["n_spam_users"], m["n_users_touching_category"], m["n_forget_interactions"])
    return u


def fmt(mu, sd, nd=2):
    return "--" if mu is None else f"{mu:.{nd}f} +- {sd:.{nd}f}"


def tex(mu, sd, nd=2):
    return "--" if mu is None else f"{mu:.{nd}f}\\,{{\\scriptsize$\\pm${sd:.{nd}f}}}"


def write_tex(path, ds, view, data, u):
    seeds_n = [u[(R, ds, sd)][0] for sd in SEEDS if (R, ds, sd) in u]
    nus = f"{min(seeds_n)} to {max(seeds_n)}" if seeds_n else "?"
    vtxt = ("cleaned test histories (the requesting users' deleted items removed from their test input histories)"
            if view == "scrub" else "the original test histories")
    lines = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
             r"\caption{Sensitive item deletion with a larger request on " + ds.capitalize()
             + f" ({CATNAME[ds]}), TIGER + RQ-KMeans, five data seeds: requesting users are added until their "
             r"category interactions reach 0.05\% of all training interactions (" + nus
             + r" users per seed, vs.\ 12 to 18 at 0.01\%). Evaluated on " + vtxt
             + r". Every method runs its canonical configuration from the 0.01\% setting (no re-selection); "
             r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ is the mean over seeds of $|\mathrm{UHR}_U@10-\mathrm{UHR}_R@10|$; "
             r"$\tau_P=\mathrm{NDCG}_U@10/\mathrm{NDCG}_R@10$ per seed; $^{\dagger}$: mean $\tau_P<95\%$.}",
             r"\label{tab:sensitive-larger-" + ds + ("" if view == "scrub" else "-normal") + "}",
             r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lccccc}", r"\toprule",
             r"\textbf{Method} & $\mathrm{UHF}@10{\scriptstyle\,(\%)}\downarrow$ & $\mathrm{UHR}@10{\scriptstyle\,(\%)}$ & "
             r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}{\scriptstyle\,(\%)}\downarrow$ & $\mathrm{NDCG}@10{\scriptstyle\,(\%)}\uparrow$ & $\tau_P$ (\%) \\",
             r"\midrule"]
    for key, name in ORDER:
        if key == "nau_ln0":
            continue
        s = per_seed(data, ds, view, key)
        if not s:
            continue
        if key == "nau":
            lines.append(r"\cmidrule(lr){1-6}")
            name = r"NAU \textit{(Ours)}"
        a = {k: agg(v[k] for v in s.values()) for k in ("UHF", "UHR", "dUHR", "NDCG", "tauP")}
        dag = r"$^{\dagger}$" if a["tauP"][0] < 95 else ""
        nmiss = "" if len(s) == 5 else f" ({len(s)} seeds)"
        lines.append(f"{name}{dag}{nmiss} & " + " & ".join(
            [tex(*a["UHF"]), tex(*a["UHR"], 3), tex(*a["dUHR"], 3), tex(*a["NDCG"], 3), tex(*a["tauP"], 1)]) + r" \\")
    lines.append(r"\midrule")
    for key, name in REFS:
        s = per_seed(data, ds, view, key)
        if not s:
            continue
        a = {k: agg(v[k] for v in s.values()) for k in ("UHF", "UHR", "dUHR", "NDCG", "tauP")}
        lines.append(f"\\textit{{{name}}} & " + " & ".join(
            [tex(*a["UHF"]), tex(*a["UHR"], 3), tex(*a["dUHR"], 3), tex(*a["NDCG"], 3), tex(*a["tauP"], 1)]) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    open(path, "w").write("\n".join(lines))


def lambda_n_rows(data, view):
    out = []
    for ds in CAT:
        a, b = per_seed(data, ds, view, "nau_ln0"), per_seed(data, ds, view, "nau")
        common = sorted(set(a) & set(b))
        if not common:
            continue
        d = [b[s]["UHF"] - a[s]["UHF"] for s in common]
        dec = sum(x < -1e-9 for x in d); inc = sum(x > 1e-9 for x in d)
        out.append(dict(ds=ds, n=len(common),
                        ln0=agg(a[s]["UHF"] for s in common), ln1=agg(b[s]["UHF"] for s in common),
                        tau0=agg(a[s]["tauP"] for s in common), tau1=agg(b[s]["tauP"] for s in common),
                        ur0=agg(a[s]["dUHR"] for s in common), ur1=agg(b[s]["dUHR"] for s in common),
                        diff=agg(d), dec=dec, inc=inc, tie=len(common) - dec - inc))
    return out


def main():
    plans = sorted(glob.glob(f"logs/eval/sensitive_{R}/plan_*.tsv"))
    data = load(plans)
    old = load(["logs/eval/sensitive_scrub/plan.tsv", "logs/eval/sensitive_scrub/plan_canonck.tsv"])
    # the canonck plan labels NAU lambda_n=0 rows by arm; normalise to the method key used here
    for p in ("logs/eval/sensitive_scrub/plan_canonck.tsv",):
        if os.path.isfile(p):
            for r in csv.DictReader(open(p), delimiter="\t"):
                if r.get("arm", "").startswith("unified_hp2_sepgen_lf0.1_ls0.0_ln0.0") and \
                        r.get("experiment", "tiger_train_flat").strip() == "tiger_train_flat":
                    m = last_metrics(r["out_dir"].strip())
                    if m:
                        old[(r["dataset"], r["view"], "nau_ln0", int(r["seed"]))] = m
    u = users()
    os.makedirs(f"{OUTD}/tables", exist_ok=True)
    rows_csv = []
    md = ["# Sensitive item deletion: larger request (budget 0.05% = 5e-4)", "",
          "Generated by `paper/plotting/sensitive_larger_request_table.py` from the run_sens5e4_* jobs "
          "(see PROGRESS_rebuttal.md). TIGER + RQ-KMeans, 5 data seeds, canonical configurations of the 0.01% "
          "setting replayed on the larger request (no re-selection). Numbers in %, mean +- std over seeds.", "",
          "## Request sizes", "", "| dataset | seed | users 5e-4 | interactions 5e-4 | users 1e-4 | category users (training) |",
          "|---|---|---|---|---|---|"]
    for ds in CAT:
        for sd in SEEDS:
            a, b = u.get((R, ds, sd)), u.get(("1e-4", ds, sd))
            if a:
                md.append(f"| {ds} | {sd} | {a[0]} | {a[2]} | {b[0] if b else '?'} | {a[1]} |")
    md.append("")
    for view in ("scrub", "normal"):
        md += [f"## {'Cleaned' if view == 'scrub' else 'Original'} test histories ({view} view)", ""]
        for ds in CAT:
            md += [f"### {ds} ({CAT[ds]})", "",
                   "| method | seeds | UHF@10 | UHR@10 | dUHR | NDCG@10 | tau_P | UHF@10 at 1e-4 (same view) |",
                   "|---|---|---|---|---|---|---|---|"]
            for key, name in ORDER + REFS:
                s = per_seed(data, ds, view, key)
                so = per_seed(old, ds, view, key)
                if not s and not so:
                    continue
                for sd, v in s.items():
                    rows_csv.append(dict(dataset=ds, view=view, method=key, seed=sd, **{k: round(x, 4) for k, x in v.items()}))
                a = {k: agg(v[k] for v in s.values()) for k in ("UHF", "UHR", "dUHR", "NDCG", "tauP")}
                o = agg(v["UHF"] for v in so.values())
                md.append(f"| {name} | {len(s)} | {fmt(*a['UHF'])} | {fmt(*a['UHR'], 3)} | {fmt(*a['dUHR'], 3)} | "
                          f"{fmt(*a['NDCG'], 3)} | {fmt(*a['tauP'], 1)} | {fmt(*o)} ({len(so)}) |")
            md.append("")
            write_tex(f"{OUTD}/tables/sensitive_larger_request_{ds}{'' if view == 'scrub' else '_normal'}.tex",
                      ds, view, data, u)
        md += [f"### lambda_n = 0 vs -1 (Table 4 style, {view} view)", "",
               "| dataset | seeds | UHF ln=0 | UHF ln=-1 | mean diff (-1 minus 0) | seeds UHF decreases / ties / increases | tau_P ln=0 | tau_P ln=-1 | dUHR ln=0 | dUHR ln=-1 |",
               "|---|---|---|---|---|---|---|---|---|---|"]
        for r in lambda_n_rows(data, view):
            md.append(f"| {r['ds']} | {r['n']} | {fmt(*r['ln0'])} | {fmt(*r['ln1'])} | {fmt(*r['diff'])} | "
                      f"{r['dec']} / {r['tie']} / {r['inc']} | {fmt(*r['tau0'], 1)} | {fmt(*r['tau1'], 1)} | "
                      f"{fmt(*r['ur0'], 3)} | {fmt(*r['ur1'], 3)} |")
        old_ln = lambda_n_rows(old, view)
        if old_ln:
            md += ["", f"Same contrast at 1e-4 ({view} view): " + "; ".join(
                f"{r['ds']} {r['ln0'][0]:.2f} -> {r['ln1'][0]:.2f}, UHF decreases on {r['dec']} of {r['n']} seeds"
                for r in old_ln)]
        md.append("")
        if view == "scrub":
            L = [r"\begin{table}[t]", r"\centering", r"\small",
                 r"\caption{Neighborhood term in sensitive item deletion with a larger request (0.05\% of interactions), "
                 r"cleaned test histories, TIGER + RQ-KMeans, five data seeds: NAU with $\lambda_n=0$ vs.\ $\lambda_n=-1$ "
                 r"($\lambda_f=0.1$, $\lambda_s=0$). ``Seeds'' counts the seeds on which $\lambda_n=-1$ lowers / ties / raises UHF@10.}",
                 r"\label{tab:sensitive-larger-lambda-n}", r"\begin{tabular}{lccccc}", r"\toprule",
                 r"Dataset & $\mathrm{UHF}@10$, $\lambda_n=0$ & $\mathrm{UHF}@10$, $\lambda_n=-1$ & $\Delta$ & Seeds $\downarrow$/=/$\uparrow$ & $\tau_P$, $\lambda_n=0$ / $-1$ \\",
                 r"\midrule"]
            for r in lambda_n_rows(data, view):
                L.append(f"{r['ds'].capitalize()} & {tex(*r['ln0'])} & {tex(*r['ln1'])} & {tex(*r['diff'])} & "
                         f"{r['dec']}/{r['tie']}/{r['inc']} & {r['tau0'][0]:.1f} / {r['tau1'][0]:.1f} \\\\")
            L += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
            open(f"{OUTD}/tables/sensitive_larger_request_lambda_n.tex", "w").write("\n".join(L))
    with open(CSV_OUT, "w", newline="") as fh:
        if rows_csv:
            w = csv.DictWriter(fh, list(rows_csv[0]))
            w.writeheader()
            w.writerows(rows_csv)
    open(f"{OUTD}/sensitive_larger_request.md", "w").write("\n".join(md) + "\n")
    print(f"{len(data)} evals read; wrote {OUTD}/sensitive_larger_request.md and tables")


if __name__ == "__main__":
    main()
