"""Figure 1 variant with Table 1's Forget only and Fanchuan configurations (plan 3.2, fix (a)).

Same data path and drawing as teaser.ipynb (paper/data/teaser_scope_tauP.csv from the Beauty
mid JSONs in logs/diagnose/fig1_scope_allunrel, query-weighted over the 5 seeds, tau_P = NDCG@10
as % of the retrained model in the same query set), except:
  * forget_only = Table 1 Forget only (lambda_f=1, lambda_r=0) and fanchuan = Fanchuan t0.07, from
    the _canonck runs scored in logs/diagnose/fig1_scope_allunrel_fix (run_fig1_fix_eval.sh);
    the old points were fig1js_forget_only (lambda_f=0.1) and fanchuan_t0.007_keepck;
  * forget_repair is read from the `nosep_ln0` key (the canonical F+R execution,
    unified_lf0.1_ls0.0_ac0_ln0.0_sepgen_keepck); its pooled numbers equal the old
    `forget_repair` key's to the digit (checked below).
The clean and poisoned models are re-scored inside every fix JSON; their pooled numbers are
compared with the original JSONs and printed.

Writes paper/data/teaser_scope_tauP_fix.csv and paper/figures/teaser_fix.pdf (+ .png). Never
touches teaser_scope_tauP.csv or teaser.pdf. Uses LaTeX text only if `latex` is on PATH.

    python paper/plotting/plot_teaser_fix.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import shutil
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC = os.path.join(ROOT, "logs/diagnose/fig1_scope_allunrel")
FIX = os.path.join(ROOT, "logs/diagnose/fig1_scope_allunrel_fix")
OUT_CSV = os.path.join(ROOT, "paper/data/teaser_scope_tauP_fix.csv")
OUT_FIG = os.path.join(ROOT, "paper/figures/teaser_fix")
SEEDS = (2, 3, 5, 7, 11)
METHODS = ["clean", "poison", "forget_only", "retain_only", "forget_repair",
           "scif", "seif", "kookmin", "fanchuan", "nau_s0p01_n0p01"]
# CSV method -> (directory, JSON key)
SOURCE = {m: (SRC, m) for m in METHODS}
SOURCE.update(forget_only=(FIX, "forget_only"), fanchuan=(FIX, "fanchuan"), forget_repair=(SRC, "nosep_ln0"))


def pooled(directory, key):
    acc = defaultdict(lambda: [0.0, 0, 0.0, 0])
    for sd in SEEDS:
        f = os.path.join(directory, f"beauty_seed{sd}.json")
        d = json.load(open(f))
        assert d.get("unrelated_selection") == "all prefix-disjoint rows", f
        rows = [r for r in d["rows"] if r["method"] == key]
        assert rows, f"{key} missing in {f}"
        for r in rows:
            if not r["n"]:   # empty group (e.g. no test case whose next item is the target): sh is None
                continue
            for g in (r["group"], "whole"):
                a = acc[g]
                a[0] += sum(r["ndcg_per_query"]); a[1] += r["n"]; a[2] += r["sh_at_10"] * r["n"]; a[3] += 1
    return {g: (a[0] / a[1], a[2] / a[1], a[1]) for g, a in acc.items()}


def main() -> int:
    miss = [sd for sd in SEEDS if not os.path.isfile(os.path.join(FIX, f"beauty_seed{sd}.json"))]
    if miss:
        print(f"fix JSONs missing for seeds {miss} in {FIX}; run after job fig1fix_scope finishes")
        return 1
    P = {m: pooled(*SOURCE[m]) for m in METHODS}
    # consistency: clean/poison re-scored in the fix JSONs; F+R key switch
    for m in ("clean", "poison"):
        new = pooled(FIX, m)
        for g in ("whole", "neighbor"):
            print(f"check {m:6s} {g:8s} NDCG orig {100 * P[m][g][0]:.6f} fix {100 * new[g][0]:.6f}  "
                  f"SH orig {100 * P[m][g][1]:.6f} fix {100 * new[g][1]:.6f}  n {P[m][g][2]}/{new[g][2]}")
    old_fr = pooled(SRC, "forget_repair")
    for g in ("whole", "neighbor"):
        print(f"check forget_repair vs nosep_ln0 {g}: {100 * old_fr[g][0]:.6f} vs {100 * P['forget_repair'][g][0]:.6f}")
    for m in ("forget_only", "fanchuan"):
        old = pooled(SRC, m)
        for g in ("whole", "neighbor"):
            print(f"changed {m:11s} {g:8s} NDCG {100 * old[g][0]:.4f} -> {100 * P[m][g][0]:.4f}  "
                  f"SH {100 * old[g][1]:.4f} -> {100 * P[m][g][1]:.4f}")
    rows = []
    for g in ("whole", "neighbor"):
        cn = P["clean"][g][0]
        for m in METHODS:
            nd, sh, n = P[m][g]
            rows.append(dict(dataset="beauty", query_set=g, method=m, n_rows=n, n_seeds=len(SEEDS),
                             ndcg10_pct=100 * nd, sh10_pct=100 * sh, tauP_pct=100 * nd / cn))
    with open(OUT_CSV, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    print(f"wrote {OUT_CSV}")

    import logging
    import matplotlib
    matplotlib.use("Agg")
    logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)   # Arial absent off-laptop
    import matplotlib.pyplot as plt
    import numpy as np
    usetex = shutil.which("latex") is not None
    plt.rcParams["pdf.fonttype"] = 42
    plt.rc("text", usetex=usetex)
    plt.rcParams["font.sans-serif"] = "Arial"
    plt.rcParams["font.family"] = "sans-serif"
    if not usetex:
        print("note: no latex on PATH; drawn with mathtext (re-run where latex exists for the paper version)")
    # ---- drawing: identical to teaser.ipynb cell 2 ----
    methods = METHODS
    Y = "tauP_pct"
    names = dict(zip(methods, ["Retrained", "Poisoned", "Forget only", "Retain only", "Forget + Repair",
                               "SCIF", "SEIF", "Kookmin", "Fanchuan", "NAU (ours)"]))
    colors = dict(zip(methods, ["#1b9e77", "#d62728", "#7f7f7f", "#7f7f7f", "#7f7f7f", "#7f7f7f", "#7f7f7f",
                                "#7f7f7f", "#7f7f7f", "#6a51a3"]))
    markers = dict(zip(methods, ["*", "X", "v", "^", 'o', ">", "h", "<", "p", "P"]))
    sizes = dict(zip(methods, np.array([220, 140, 100, 100, 100, 100, 100, 100, 100, 150]) * 2))
    zorders = dict(zip(methods, [10, 10, 3, 3, 3, 3, 3, 3, 3, 9]))
    fig, (ax1, ax2, ax_legend) = plt.subplots(1, 3, gridspec_kw={"width_ratios": [1, 1, 0.7]})
    by = {(r["query_set"], r["method"]): r for r in rows}
    for ax, g in ((ax1, "whole"), (ax2, "neighbor")):
        for m in methods:
            r = by[(g, m)]
            ax.scatter(r["sh10_pct"], r[Y], c=colors[m], s=sizes[m], marker=markers[m], label=names[m],
                       zorder=zorders[m], edgecolor="black", linewidth=0.6, alpha=0.9)
    label_size, tick_size, legend_size = 18, 14, 10
    for ax in (ax1, ax2):
        ax.set_xscale("symlog", linthresh=0.01)
        ax.tick_params(axis='both', which='major', labelsize=tick_size)
        ax.tick_params(axis='both', which='minor', labelsize=tick_size)
        ax.set_xlim((-0.005, 25))
        ax.set_ylim((-6, 118))
        ax.axhline(100, color='#9a9a9a', lw=0.8, ls=(0, (2, 2)), zorder=0)
    pct = r"\%" if usetex else "%"
    ax1.set_title(r'Whole test set', fontsize=label_size)
    ax2.set_title(r'SID neighborhood', fontsize=label_size)
    ax1.set_ylabel(rf'Utility ({pct} of retrained) $\uparrow$', fontsize=label_size)
    ax2.get_yaxis().set_visible(False)
    ax2.set_xlabel(rf'Spam exposure ({pct}) $\downarrow$ (log scale)', fontsize=label_size)
    ax2.xaxis.set_label_coords(-0.0, -0.16)
    displaynames = {"Retrained": "Retrained\nwithout spam", "Poisoned": "Poisoned\nwith spam",
                    "Forget only": "Fine-tuned on\nforget set", "Retain only": "Fine-tuned on\nretain set",
                    "Forget + Repair": "Fine-tuned on\nretain+forget set", "SCIF": "SCIF", "SEIF": "SEIF",
                    "Kookmin": "Kookmin", "Fanchuan": "Fanchuan", "NAU (ours)": "NAU (ours)"}
    handles = [plt.Line2D([0], [0], marker=markers[m], color=colors[m], linestyle='None', markersize=12,
                          label=displaynames[names[m]], markeredgecolor='black', markeredgewidth=0.5)
               for m in methods]
    ax_legend.axis('off')
    ax_legend.legend(handles=handles, loc='center', ncol=1, fontsize=legend_size, frameon=False,
                     handletextpad=0.5, borderpad=0.5, labelspacing=0.4)
    plt.gcf().subplots_adjust(wspace=0.05)
    plt.gcf().set_size_inches(8.25, 2.5)
    for ext in ("pdf", "png"):
        plt.savefig(f"{OUT_FIG}.{ext}", dpi=300, bbox_inches='tight')
    print(f"wrote {OUT_FIG}.pdf/.png")
    return 0


if __name__ == "__main__":
    sys.exit(main())
