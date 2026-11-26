"""SASRec vs TIGER: is NAU's neighborhood damage specific to semantic-ID sharing? (rebuttal)

Both backbones, the same poisoned datasets (Beauty/Sports/Toys, bandwagon 1%, one target per
stratum, data seeds 2/3/5/7/11), the same unlearning objective and weights, and the same
Figure 2 test queries (next item shares the target's first RQ-KMeans SID token, the target's
own cases excluded). Contrasts, each a paired difference over test cases (Appendix G:
3000 bootstrap resamples over pooled queries, 95% percentile interval):

  (i)   forget term's neighborhood cost      Forget+Repair - Finetune
  (ii)  separation term's cost               +L_sep(ls) - Forget+Repair,  ls in {0.01, 0.1}
  (iii) neighborhood term's gain             NAU(ls, ln=0.01) - +L_sep(ls)

reported as neighborhood NDCG@10 in points of the retrained model's neighborhood NDCG (pooled
over the same queries) and as mean log p(true next item) in nats.

SASRec: logs/sasrec_nau_v2/<ds>/<stratum>_seed<N>/{clean,poison,<arm>}.json (scripts/sasrec_nau_v2.py,
run_sasrec_nau_v2.sh). TIGER: the canonical checkpoint of each arm-cell
(paper/plotting/canonical_runs.tsv) matched by path to a test-split, first-token scope file in
logs/diagnose/fig1_scope*/ (per-query arrays of one cell are aligned across files: same queries,
same order). Arm-cells without such a file are listed, never filled in.

    PYTHONPATH=. python paper/plotting/sasrec_vs_tiger_table.py
      -> paper/revision/tables/sasrec_vs_tiger.tex, paper/revision/tables/sasrec_vs_tiger.csv,
         stdout (markdown blocks for paper/revision/sasrec_vs_tiger.md)
"""
import argparse
import csv
import glob
import json
import os
from collections import defaultdict

import numpy as np

DS = ("beauty", "sports", "toys")
ST = ("unpopular", "mid", "popular")
SEEDS = ("2", "3", "5", "7", "11")
# arm -> (TIGER canonical config, display name)
ARMS = {"finetune": ("finetune", "Finetune"),
        "forget_repair": ("forget_repair_lf0.1", "Forget+Repair"),
        "sep_ls0p01": ("nau_ls0.01_ln0.0", r"+$\mathcal{L}_{sep}$ ($\lambda_s$=0.01)"),
        "sep_ls0p1": ("nau_ls0.1_ln0.0", r"+$\mathcal{L}_{sep}$ ($\lambda_s$=0.1)"),
        "nau_ls0p01": ("nau_ls0.01_ln0.01", r"NAU ($\lambda_s$=0.01)"),
        "nau_ls0p1": ("nau_ls0.1_ln0.01", r"NAU ($\lambda_s$=0.1)")}
CONTRASTS = [("forget", "Forget term (F+R $-$ Finetune)", "forget_repair", "finetune"),
             ("sep001", r"Separation term, $\lambda_s$=0.01 (+$\mathcal{L}_{sep}$ $-$ F+R)", "sep_ls0p01", "forget_repair"),
             ("sep01", r"Separation term, $\lambda_s$=0.1 (+$\mathcal{L}_{sep}$ $-$ F+R)", "sep_ls0p1", "forget_repair"),
             ("nbr001", r"Neighborhood term, $\lambda_s$=0.01 (NAU $-$ +$\mathcal{L}_{sep}$)", "nau_ls0p01", "sep_ls0p01"),
             ("nbr01", r"Neighborhood term, $\lambda_s$=0.1 (NAU $-$ +$\mathcal{L}_{sep}$)", "nau_ls0p1", "sep_ls0p1"),
             ("nau_fr", r"NAU ($\lambda_s$=0.01) $-$ F+R (Figure 2)", "nau_ls0p01", "forget_repair")]
PLAIN = {"forget": "Forget term (F+R minus Finetune)",
         "sep001": "Separation term, ls=0.01 (+L_sep minus F+R)",
         "sep01": "Separation term, ls=0.1 (+L_sep minus F+R)",
         "nbr001": "Neighborhood term, ls=0.01 (NAU minus +L_sep)",
         "nbr01": "Neighborhood term, ls=0.1 (NAU minus +L_sep)",
         "nau_fr": "NAU (ls=0.01) minus F+R (Figure 2 contrast)"}


# --------------------------------------------------------------------------- #
def sasrec_cells(root):
    """(ds, st, seed) -> {arm|clean|poison: {"ndcg": arr, "logp": arr, "whole": (SH, NDCG)}}"""
    out = {}
    for ds in DS:
        for st in ST:
            for sd in SEEDS:
                d = os.path.join(root, ds, f"{st}_seed{sd}")
                cell = {}
                for name in ("clean", "poison", *ARMS):
                    f = os.path.join(d, f"{name}.json")
                    if not os.path.isfile(f):
                        continue
                    j = json.load(open(f))
                    pq = j["neighbor_per_query"]
                    cell[name] = {"ndcg": np.asarray(pq["ndcg"]), "logp": np.asarray(pq["logp"]),
                                  "uid": pq["user_id"], "whole": (j["SH@10"], j["ndcg@10"])}
                if cell:
                    out[(ds, st, sd)] = cell
    return out


def tiger_cells(ledger):
    rows = [r for r in csv.DictReader(open(ledger), delimiter="\t")
            if r["setting"] == "spam" and r["track"] == "tiger+rkmeans"]
    led = {(r["config"], r["dataset"], r["stratum"], r["seed"]): r for r in rows}
    # checkpoint -> (file, arm name in file); test split, first-token neighborhood only
    ck2 = defaultdict(list)
    for f in sorted(glob.glob("logs/diagnose/fig1_scope*/*.json")):
        stem = f.split("/")[2]
        if "val" in stem or "knn" in stem or "legacy" in stem:
            continue
        j = json.load(open(f))
        if j.get("neighbor_def", "prefix1") != "prefix1":
            continue
        for m, ck in j["checkpoints"].items():
            ck2[os.path.realpath(ck)].append((f, m))
    cache = {}

    def rows_of(f):
        if f not in cache:
            j = json.load(open(f))
            cache[f] = {(x["method"], x["group"]): x for x in j["rows"]}
        return cache[f]

    refs = {}
    for f in glob.glob("paper/data/spam_references/*.json"):
        b = os.path.basename(f)[:-5]
        for ds in DS:
            for sd in SEEDS:
                if b == ds + ("" if sd == "2" else f"_seed{sd}"):
                    refs[(ds, sd)] = json.load(open(f))
    out, missing = {}, []
    for ds in DS:
        for st in ST:
            for sd in SEEDS:
                cell = {}
                for arm, (cfg, _) in ARMS.items():
                    r = led.get((cfg, ds, st, sd))
                    if r is None:
                        missing.append((ds, st, sd, arm, "no ledger row"))
                        continue
                    whole = None
                    p = os.path.join(r["run_dir"], "eval", "csv", "version_0", "metrics.csv")
                    if not os.path.isfile(p):
                        p = os.path.join(r["run_dir"], "csv", "version_0", "metrics.csv")
                    if os.path.isfile(p):
                        m = list(csv.DictReader(open(p)))[-1]
                        whole = (float(m["test/SH@10"]), float(m["test/ndcg@10"]))
                    hits = ck2.get(os.path.realpath(r["ckpt_resolved"])) if r["ckpt_resolved"] else None
                    if not hits:
                        missing.append((ds, st, sd, arm, "no checkpoint" if r["has_ckpt"] != "1"
                                        else "checkpoint has no scope eval"))
                        cell[arm] = {"whole": whole}
                        continue
                    # prefer a file that also stores log-probabilities
                    hits.sort(key=lambda h: "label_logprob_per_query" not in rows_of(h[0])[(h[1], "neighbor")])
                    f, m = hits[0]
                    rr = rows_of(f)
                    nb = rr[(m, "neighbor")]
                    cl = rr[("clean", "neighbor")]
                    lp = nb.get("label_logprob_per_query")
                    cell[arm] = {"ndcg": np.asarray(nb["ndcg_per_query"]),
                                 "logp": np.asarray(lp) if lp else None, "whole": whole, "file": f}
                    if ("poison", "neighbor") in rr and "poison" not in cell:
                        po = rr[("poison", "neighbor")]
                        cell["poison"] = {"ndcg": np.asarray(po["ndcg_per_query"]),
                                          "logp": np.asarray(po["label_logprob_per_query"])
                                          if po.get("label_logprob_per_query") else None}
                    cell.setdefault("clean", {"ndcg": np.asarray(cl["ndcg_per_query"]),
                                              "logp": np.asarray(cl["label_logprob_per_query"])
                                              if cl.get("label_logprob_per_query") else None})
                ref = refs.get((ds, sd))
                if ref:
                    cell.setdefault("clean", {})["whole"] = (ref["clean_SH@10"][st], ref["clean_ndcg@10"])
                    cell.setdefault("poison", {})["whole"] = (ref["poison_SH@10"][st], ref["poison_ndcg@10"][st])
                out[(ds, st, sd)] = cell
    return out, missing


# --------------------------------------------------------------------------- #
def contrast(cells, a, b, keys, n_boot=3000, seed=0):
    """Pooled paired difference a - b over the neighbor queries of `keys`."""
    da, dl, ref, ncell, neg = [], [], [], 0, 0
    for k in keys:
        c = cells.get(k, {})
        if not all(x in c and c[x].get("ndcg") is not None for x in (a, b, "clean")):
            continue
        if len(c[a]["ndcg"]) == 0:
            continue
        assert len(c[a]["ndcg"]) == len(c[b]["ndcg"]) == len(c["clean"]["ndcg"]), k
        ncell += 1
        da.append(c[a]["ndcg"] - c[b]["ndcg"])
        ref.append(c["clean"]["ndcg"])
        if c[a].get("logp") is not None and c[b].get("logp") is not None:
            d = c[a]["logp"] - c[b]["logp"]
            dl.append(d)
            neg += d.mean() < 0
        else:
            dl.append(np.full(len(c[a]["ndcg"]), np.nan))
    if not ncell:
        return None
    d, ref, l = np.concatenate(da), np.concatenate(ref), np.concatenate(dl)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(d), (n_boot, len(d)))
    s = 100 / ref.mean()
    bs = d[idx].mean(1) * s
    res = {"cells": ncell, "queries": len(d), "ndcg_pts": s * d.mean(),
           "ndcg_lo": np.percentile(bs, 2.5), "ndcg_hi": np.percentile(bs, 97.5)}
    ok = ~np.isnan(l)
    if ok.all():
        bl = l[idx].mean(1)
        res.update(logp=l.mean(), logp_lo=np.percentile(bl, 2.5), logp_hi=np.percentile(bl, 97.5),
                   logp_cells_neg=int(neg))
    else:
        res.update(logp=None, logp_cells_with=int(ok.sum()))
    return res


def whole(cells, arm, keys):
    """Mean over cells of |SH - SH_retr| (in %), NDCG@10 (x100), tau_P (%)."""
    dsh, nd, tau = [], [], []
    for k in keys:
        c = cells.get(k, {})
        if c.get(arm, {}).get("whole") is None or c.get("clean", {}).get("whole") is None:
            continue
        (sh, n), (shc, nc) = c[arm]["whole"], c["clean"]["whole"]
        dsh.append(100 * abs(sh - shc)); nd.append(100 * n); tau.append(100 * n / nc)
    if not dsh:
        return None
    return len(dsh), np.mean(dsh), np.mean(nd), np.mean(tau)


def fmt(r, unit):
    if r is None:
        return "--"
    if unit == "ndcg":
        return f"{r['ndcg_pts']:+.1f} [{r['ndcg_lo']:+.1f}, {r['ndcg_hi']:+.1f}]"
    if r.get("logp") is None:
        return "--"
    return f"{r['logp']:+.3f} [{r['logp_lo']:+.3f}, {r['logp_hi']:+.3f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sasrec_root", default="logs/sasrec_nau_v2")
    ap.add_argument("--ledger", default="paper/plotting/canonical_runs.tsv")
    ap.add_argument("--out", default="paper/revision/tables/sasrec_vs_tiger.tex")
    ap.add_argument("--csv", default="paper/revision/tables/sasrec_vs_tiger.csv")
    a = ap.parse_args()
    S = sasrec_cells(a.sasrec_root)
    T, missing = tiger_cells(a.ledger)
    allk = [(ds, st, sd) for ds in DS for st in ST for sd in SEEDS]
    groups = [("all", allk)] + [(ds, [k for k in allk if k[0] == ds]) for ds in DS]

    print(f"SASRec cells with results: {len(S)} / 45")
    print(f"TIGER arm-cells without a neighborhood eval: {sum(1 for m in missing)}")
    by = defaultdict(list)
    for ds, st, sd, arm, why in missing:
        by[(ds, arm, why)].append(f"{st[:3]}{sd}")
    for (ds, arm, why), v in sorted(by.items()):
        print(f"  TIGER {ds} {arm}: {why}: {len(v)} cells ({' '.join(v)})")

    # ---------------- contrasts ----------------
    table = []
    for key, tex, x, y in CONTRASTS:
        for g, keys in groups:
            for bb, cells in (("TIGER", T), ("SASRec", S)):
                r = contrast(cells, x, y, keys)
                table.append((key, g, bb, r))
    with open(a.csv, "w") as f:
        w = csv.writer(f)
        w.writerow(["contrast", "datasets", "backbone", "cells", "queries", "ndcg_pts", "ndcg_lo", "ndcg_hi",
                    "logp", "logp_lo", "logp_hi", "logp_cells_neg"])
        for key, g, bb, r in table:
            if r is None:
                w.writerow([key, g, bb] + [""] * 9)
            else:
                w.writerow([key, g, bb, r["cells"], r["queries"]] + [
                    "" if r.get(c) is None else f"{r[c]:.4f}" for c in
                    ("ndcg_pts", "ndcg_lo", "ndcg_hi", "logp", "logp_lo", "logp_hi")] + [r.get("logp_cells_neg", "")])

    print("\n### Neighborhood contrasts (paired over test cases, 95% bootstrap CI, 3000 resamples)\n")
    print("NDCG in points of the retrained model's neighborhood NDCG@10; log p in nats. cells = (dataset, stratum, seed) cells in the pool.\n")
    for key, tex, x, y in CONTRASTS:
        print(f"**{PLAIN[key]}**\n")
        print("| datasets | TIGER cells | TIGER NDCG pts | TIGER log p | SASRec cells | SASRec NDCG pts | SASRec log p |")
        print("|---|---|---|---|---|---|---|")
        for g, _ in groups:
            rt = next(r for k, gg, bb, r in table if k == key and gg == g and bb == "TIGER")
            rs = next(r for k, gg, bb, r in table if k == key and gg == g and bb == "SASRec")
            print(f"| {g} | {rt['cells'] if rt else 0} | {fmt(rt, 'ndcg')} | {fmt(rt, 'logp')} | "
                  f"{rs['cells'] if rs else 0} | {fmt(rs, 'ndcg')} | {fmt(rs, 'logp')} |")
        print()

    print("### Whole test set (mean over cells): |dSH| to retrained (%), NDCG@10 (x100), tau_P (%)\n")
    print("| datasets | arm | TIGER cells | TIGER dSH | TIGER NDCG | TIGER tau_P | SASRec cells | SASRec dSH | SASRec NDCG | SASRec tau_P |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for g, keys in groups:
        for arm in ("poison", *ARMS):
            name = "Before deletion" if arm == "poison" else ARMS[arm][1].replace("$", "").replace(r"\mathcal{L}_{sep}", "L_sep").replace(r"\lambda_s", "ls")
            cols = []
            for cells in (T, S):
                w_ = whole(cells, arm, keys)
                cols += ["0", "--", "--", "--"] if w_ is None else [str(w_[0]), f"{w_[1]:.3f}", f"{w_[2]:.3f}", f"{w_[3]:.1f}"]
            print(f"| {g} | {name} | " + " | ".join(cols) + " |")
    print()
    print("### Neighborhood level per arm (pooled, % of retrained neighborhood NDCG; mean log p)\n")
    print("| datasets | arm | TIGER cells | TIGER nbr tau_P | TIGER log p | SASRec cells | SASRec nbr tau_P | SASRec log p |")
    print("|---|---|---|---|---|---|---|---|")
    for g, keys in groups:
        for arm in ("poison", *ARMS):
            cols = []
            for cells in (T, S):
                nd, rf, lp = [], [], []
                for k in keys:
                    c = cells.get(k, {})
                    if c.get(arm, {}).get("ndcg") is None or c.get("clean", {}).get("ndcg") is None:
                        continue
                    nd.append(c[arm]["ndcg"]); rf.append(c["clean"]["ndcg"])
                    if c[arm].get("logp") is not None:
                        lp.append(c[arm]["logp"])
                cols += (["0", "--", "--"] if not nd else
                         [str(len(nd)), f"{100 * np.concatenate(nd).mean() / np.concatenate(rf).mean():.1f}",
                          f"{np.concatenate(lp).mean():.3f}" if lp and len(lp) == len(nd) else "--"])
            print(f"| {g} | {arm} | " + " | ".join(cols) + " |")

    # ---------------- LaTeX ----------------
    def cellfmt(r, unit):
        if r is None or (unit == "logp" and r.get("logp") is None):
            return "--"
        if unit == "ndcg":
            v, lo, hi, p = r["ndcg_pts"], r["ndcg_lo"], r["ndcg_hi"], 1
        else:
            v, lo, hi, p = r["logp"], r["logp_lo"], r["logp_hi"], 3
        s = f"{v:+.{p}f} [{lo:+.{p}f}, {hi:+.{p}f}]"
        s = s.replace("-", "$-$")
        return rf"\textbf{{{s}}}" if (lo > 0 or hi < 0) else s

    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Is the neighborhood damage specific to semantic IDs? TIGER (RQ-KMeans SIDs) and SASRec "
         r"(independent item embeddings), each unlearned with the same objective and weights on the same poisoned "
         r"datasets (Beauty, Sports, Toys; bandwagon 1\%; three targets $\times$ five data seeds; one unlearning run per cell). "
         r"Paired differences on the Figure~2 neighborhood (test cases whose next item shares the target's first SID token, "
         r"the target's own cases excluded; identical queries for both backbones), pooled over cells: neighborhood NDCG@10 "
         r"in points of the retrained model's neighborhood NDCG, and mean $\log p$(true next item) in nats, with 95\% "
         r"bootstrap intervals over test cases (3000 resamples); bold: interval excludes 0. "
         r"Cells: TIGER cells with a neighborhood evaluation for both arms (SASRec: all 45).}",
         r"\label{tab:sasrec-vs-tiger}",
         r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & \multicolumn{3}{c}{TIGER} & \multicolumn{3}{c}{SASRec} \\",
         r"\cmidrule(lr){3-5}\cmidrule(lr){6-8}",
         r"Contrast & Data & cells & NDCG (pts) & $\log p$ (nats) & cells & NDCG (pts) & $\log p$ (nats) \\",
         r"\midrule"]
    for i, (key, tex, x, y) in enumerate(CONTRASTS):
        for j, (g, _) in enumerate(groups):
            rt = next(r for k, gg, bb, r in table if k == key and gg == g and bb == "TIGER")
            rs = next(r for k, gg, bb, r in table if k == key and gg == g and bb == "SASRec")
            lab = (r"\multirow{4}{*}{\parbox{3.6cm}{" + tex + "}}") if j == 0 else ""
            L.append(f"{lab} & {g.capitalize() if g != 'all' else 'All'} & {rt['cells'] if rt else 0} & "
                     f"{cellfmt(rt, 'ndcg')} & {cellfmt(rt, 'logp')} & {rs['cells'] if rs else 0} & "
                     f"{cellfmt(rs, 'ndcg')} & {cellfmt(rs, 'logp')} \\\\")
        if i < len(CONTRASTS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"\nwrote {a.out}, {a.csv}")


if __name__ == "__main__":
    main()
