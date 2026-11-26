"""Neighborhood rehearsal vs NAU (R3: simpler neighborhood regularizer at equal size and budget).

Arms (Beauty, 3 targets x 5 data seeds, one execution per cell):
  NAU            ledger nau_ls0.01_ln0.01 (canonical)
  Forget+Repair  ledger forget_repair_lf0.1 (canonical)
  F+R + rehearsal r=25% / 50%   bw_tgt<st>_fr_nbrrehearse{25,50}_canonck[_seed<N>]
    (run_nbr_rehearsal_launch.sh: F+R whose retain batches have a share r of rows replaced by
     retain rows whose next item is one of NAU's 8 embedding neighbors; same steps and batch sizes)
Whole set: Delta^retr_SH and tau_P on validation and test, pooled like Table 1 (mean over seeds
of the mean over targets; references as in nau_k_table.py).
Neighborhood: first-token SID neighborhood, Figure 2 protocol (run_scope_nbrreh.sh ->
logs/diagnose/fig1_scope_nbrreh[_val]/), per-query paired differences pooled over the 15 cells,
3000 bootstrap resamples over test cases, in points of the retrained model's neighborhood NDCG.

    PYTHONPATH=. python paper/plotting/nbr_rehearsal_table.py
Writes paper/revision/tables/nbr_rehearsal.tex and paper/revision/nbr_rehearsal.md.
"""
import csv
import glob
import importlib.util
import json
import os
import re
import statistics as S
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vt", os.path.join(HERE, "valsel_table.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
STRATA, SEEDS = ("unpopular", "mid", "popular"), ("2", "3", "5", "7", "11")
ARMS = [("nau", "NAU"), ("forget_repair", "Forget+Repair"),
        ("fr_nbrreh25", "F+R + rehearsal (25\\%)"), ("fr_nbrreh50", "F+R + rehearsal (50\\%)")]


def run_dirs():
    led = list(csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t"))
    out = defaultdict(dict)
    for cfg, arm in (("nau_ls0.01_ln0.01", "nau"), ("forget_repair_lf0.1", "forget_repair")):
        for r in led:
            if r["setting"] == "spam" and r["dataset"] == "beauty" and r["config"] == cfg:
                out[arm][(r["stratum"], r["seed"])] = r["run_dir"]
    jid = lambda p: int(re.search(r"/job(\d+)_", p).group(1))
    for rt in ("25", "50"):
        for st in STRATA:
            for sd in SEEDS:
                sfx = "" if sd == "2" else f"_seed{sd}"
                c = glob.glob(f"logs/unlearn/runs/job*_bs1_bw_tgt{st}_fr_nbrrehearse{rt}_canonck{sfx}")
                if c:
                    out[f"fr_nbrreh{rt}"][(st, sd)] = max(c, key=jid)
    return out


def whole(dirs):
    R = vt.refs()
    test_ref = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("paper/data/spam_references/*.json")}
    res, per_cell = {}, {}
    for arm, cells in dirs.items():
        per = defaultdict(list)
        pc = {}
        for (st, sd), d in cells.items():
            te, va = vt.metrics(d + "/eval"), vt.metrics(d + "/eval_val")
            cv = R.get(("rkmeans", "beauty", st, sd, "clean"))
            rt = test_ref.get("beauty" + ("" if sd == "2" else f"_seed{sd}"))
            if te and va and cv and rt:
                x = (abs(va[0] - cv[0]), va[1] / cv[1], abs(te[0] - rt["clean_SH@10"][st]), te[1] / rt["clean_ndcg@10"])
                per[sd].append(x)
                pc[(st, sd)] = x
        n = sum(len(v) for v in per.values())
        res[arm] = (n, [100 * S.mean(S.mean(x[j] for x in v) for v in per.values()) if n else float("nan")
                        for j in range(4)])
        per_cell[arm] = pc
    return res, per_cell


def rehearsal_stats(dirs):
    out = {}
    for arm in ("fr_nbrreh25", "fr_nbrreh50"):
        share, uses, pool = [], [], []
        for d in dirs.get(arm, {}).values():
            f = glob.glob(d + "/*_info.json") or glob.glob(d + "/scif_info.json")
            if not f:
                continue
            for req in json.load(open(f[0]))["requests"]:
                s = (req.get("scif_info") or {}).get("nbr_rehearsal")
                if s:
                    share.append(s.get("share_replaced", 0.0)); uses.append(s.get("mean_uses_per_pool_row", 0.0))
                    pool.append(s.get("n_pool_rows", 0))
        if share:
            out[arm] = (S.mean(share), S.median(pool), S.mean(uses), len(share))
    return out


def neighborhood(stem, pairs, n_boot=3000):
    pool = {p: {"a": [], "b": [], "ref": []} for p in pairs}
    tau = defaultdict(lambda: {"x": [], "ref": []})
    cells = 0
    for f in sorted(glob.glob(f"logs/diagnose/{stem}/*.json")):
        r = {(x["method"], x["group"]): x for x in json.load(open(f))["rows"]}
        if not all((a, "neighbor") in r for a, _ in ARMS):
            continue
        cells += 1
        ref = r[("clean", "neighbor")]["ndcg_per_query"]
        for a, _ in ARMS:
            tau[a]["x"] += r[(a, "neighbor")]["ndcg_per_query"]; tau[a]["ref"] += ref
        for a, b in pairs:
            pool[(a, b)]["a"] += r[(a, "neighbor")]["ndcg_per_query"]
            pool[(a, b)]["b"] += r[(b, "neighbor")]["ndcg_per_query"]
            pool[(a, b)]["ref"] += ref
    out = {}
    for (a, b), p in pool.items():
        if not p["a"]:
            continue
        rng = np.random.default_rng(0)
        x, y, ref = (np.asarray(p[k]) for k in ("a", "b", "ref"))
        d, s = x - y, 100 / ref.mean()
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[(a, b)] = (s * d.mean(), s * lo, s * hi, len(d))
    taus = {a: 100 * np.mean(v["x"]) / np.mean(v["ref"]) for a, v in tau.items()}
    return out, taus, cells


def main():
    dirs = run_dirs()
    wh, _ = whole(dirs)
    st = rehearsal_stats(dirs)
    pairs = [("nau", "fr_nbrreh25"), ("nau", "fr_nbrreh50"), ("fr_nbrreh25", "forget_repair"),
             ("fr_nbrreh50", "forget_repair"), ("nau", "forget_repair")]
    nb = {sp: neighborhood(stem, pairs) for sp, stem in (("test", "fig1_scope_nbrreh"), ("val", "fig1_scope_nbrreh_val"))}
    name = dict(ARMS)
    fmt = lambda g: f"{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}]" if g else "--"
    # LaTeX
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{",
         r"A simpler neighborhood regularizer at equal neighborhood size and budget (Beauty, TIGER + RQ-KMeans, three targets $\times$ five data seeds). "
         r"Neighborhood rehearsal is Forget+Repair ($\lambda_f=0.1$) whose retain batches have 25\% or 50\% of their rows replaced by retain examples whose next item is one of the same eight embedding neighbors NAU uses; "
         r"the number of optimizer steps, epochs and rows per batch equal Forget+Repair's and NAU's. "
         r"Whole set: $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ and $\tau_P$ on validation and test. "
         r"Neighborhood (first-token \ac{SID} neighborhood, test split): neighborhood $\tau_P$ and paired gains in points of the retrained model's neighborhood NDCG@10 with 95\% bootstrap intervals over test cases.}",
         r"\label{tab:appendix-nbr-rehearsal}", r"\begin{tabular}{lccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} & Neighborhood \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-6}",
         r"Method & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ \\", r"\midrule"]
    md = ["# Neighborhood rehearsal vs NAU (Beauty, 15 cells)", "",
          "Generated by `paper/plotting/nbr_rehearsal_table.py`.", "",
          "| Method | n | val dSH | val tauP | test dSH | test tauP | nbr tauP (test) | nbr tauP (val) |",
          "|---|---|---|---|---|---|---|---|"]
    for a, nm in ARMS:
        n, v = wh.get(a, (0, [float("nan")] * 4))
        tt, tv = nb["test"][1].get(a, float("nan")), nb["val"][1].get(a, float("nan"))
        L.append(f"{nm} & {v[0]:.3f} & {v[1]:.1f} & {v[2]:.3f} & {v[3]:.1f} & {tt:.1f} \\\\")
        md.append(f"| {nm.replace(chr(92) + '%', '%')} | {n} | {v[0]:.3f} | {v[1]:.1f} | {v[2]:.3f} | {v[3]:.1f} | {tt:.1f} | {tv:.1f} |")
    L += [r"\midrule", r"\multicolumn{6}{l}{Paired neighborhood gains (test): } \\"]
    md += ["", f"Neighborhood scope cells with all four arms: test {nb['test'][2]}, val {nb['val'][2]}.", "",
           "| Paired neighborhood gain (points of retrained nbr NDCG) | test | val |", "|---|---|---|"]
    for a, b in pairs:
        g, gv = nb["test"][0].get((a, b)), nb["val"][0].get((a, b))
        lab = f"{name[a]} vs {name[b]}"
        L.append(rf"\multicolumn{{4}}{{l}}{{\quad {lab}}} & \multicolumn{{2}}{{c}}{{{fmt(g)}}} \\")
        md.append(f"| {lab.replace(chr(92) + '%', '%')} | {fmt(g)} | {fmt(gv)} |")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    md += ["", "Rehearsal statistics per request (mean share of retain rows replaced, median unique neighbor-labeled pool rows, mean uses per pool row, requests):"]
    for a, s in st.items():
        md.append(f"- {a}: share {s[0]:.3f}, pool median {s[1]:.0f}, uses/row {s[2]:.1f}, requests {s[3]}")
    os.makedirs("paper/revision/tables", exist_ok=True)
    open("paper/revision/tables/nbr_rehearsal.tex", "w").write("\n".join(L) + "\n")
    open("paper/revision/nbr_rehearsal.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
