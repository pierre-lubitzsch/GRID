"""RQ4 figure: utility on the whole test set vs the SID neighborhood, per method.

Panel (a): tau_P (NDCG@10 as % of the retrained model, within each scope) on the
whole test set and on the SID neighborhood (test cases whose next item shares the
target's first SID token; the target's own cases excluded). Point estimates are
QUERY-WEIGHTED over seeds, i.e. the same numbers as tables/scoped_ablation_compact_*.
Panel (b): paired difference in neighborhood NDCG@10 against Forget+Repair, in
points of the retrained value, with a 95% percentile bootstrap over the shared test
cases. Differences to the retrained model itself are not shown: they are not
resolved at this sample size (about +-40 points).

    PYTHONPATH=. python paper/plotting/plot_scope_locality.py --dataset beauty

Reads logs/diagnose/fig1_scope_allunrel/<ds>_seed*.json (cluster) and writes
paper/plotting/scope_locality_<ds>.csv plus paper/figures/scope_locality_<ds>.pdf.
With --from-csv the figure is redrawn from the CSV alone.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
# (json method key, label), labels as in Figure 1. Only objectives built on
# forget-and-repair are shown; baselines are compared in Figure 1 and Table 1.
# Previously omitted, and said so in the caption: Forget only and Fanchuan (collapse, tau_P < 13%), SEIF and Kookmin (lose
# >= 19% of whole-set utility) and Finetune = fine-tuned on retain set (removes
# nothing, so it says nothing about what forgetting does to the neighborhood).
METHODS = [
    ("forget_repair", "Fine-tuned on\nretain+forget set"),
    ("sep_ln0", "Fine-tuned on\nretain+forget set\n+ separation term"),
    ("nau", "NAU ($\\lambda_s=0.1$)"),
    ("nau_s0p01_n0p01", "NAU (ours)"),
]
REF = "forget_repair"
WHOLE, NBR = "#2a78d6", "#eb6834"   # validated categorical slots 1 and 2 (light)
MS = 70   # marker area (pt^2); was 26, too small at column width
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def compute(ds, n_boot=3000, seed=0):
    V = {}
    for f in sorted(glob.glob(f"logs/diagnose/fig1_scope_allunrel/{ds}_seed*.json")):
        d = json.load(open(f))
        for r in d["rows"]:
            V.setdefault((r["method"], r["group"]), {})[d["seed"]] = np.asarray(r["ndcg_per_query"])
    seeds = sorted(V[("clean", "neighbor")])
    cat = lambda m, g: np.concatenate([V[(m, g)][s] for s in seeds])
    whole = lambda m: np.concatenate([cat(m, "neighbor"), cat(m, "unrelated")])
    ref_w, ref_n = whole("clean").mean(), cat("clean", "neighbor").mean()
    rng = np.random.default_rng(seed)
    rows = []
    for m, lab in METHODS:
        if (m, "neighbor") not in V or set(V[(m, "neighbor")]) != set(seeds):
            print(f"  skip {m}: not scored on all seeds {seeds}")
            continue
        diff = cat(m, "neighbor") - cat(REF, "neighbor")
        bs = np.array([diff[rng.integers(0, len(diff), len(diff))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        rows.append(dict(method=m, label=lab,
                         tauP_whole=100 * whole(m).mean() / ref_w,
                         tauP_nbr=100 * cat(m, "neighbor").mean() / ref_n,
                         d_nbr=100 * diff.mean() / ref_n, d_lo=100 * lo / ref_n, d_hi=100 * hi / ref_n,
                         n_seeds=len(seeds), n_nbr=len(cat("clean", "neighbor"))))
    return rows


def draw(rows, out):
    plt.rcParams.update({"font.size": 8, "pdf.fonttype": 42, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED})
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.5, 2.3), sharey=True,
                               gridspec_kw={"width_ratios": [1.35, 1], "wspace": 0.08})
    y = np.arange(len(rows))[::-1]
    for ax in (a, b):
        ax.grid(axis="x", color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"): ax.spines[s].set_visible(False)
    for yi, r in zip(y, rows):
        a.plot([r["tauP_whole"], r["tauP_nbr"]], [yi, yi], color=GRID, lw=3, zorder=1, solid_capstyle="round")
    a.scatter([r["tauP_whole"] for r in rows], y, s=MS, marker="o", color=WHOLE,
              edgecolor="white", linewidth=1.2, zorder=4, label="Whole test set")
    a.scatter([r["tauP_nbr"] for r in rows], y, s=MS, marker="s", color=NBR,
              edgecolor="white", linewidth=1.2, zorder=3, label="SID neighborhood")
    a.axvline(100, color=MUTED, lw=0.8, ls=(0, (2, 2)), zorder=0)
    a.set_yticks(y, [r["label"] for r in rows])
    a.set_xlabel(r"$\tau_P$ (% of retrained NDCG@10)")
    fig.legend(*a.get_legend_handles_labels(), loc="upper center", ncol=2, frameon=False,
               handletextpad=0.3, columnspacing=1.2, fontsize=7, markerscale=1.0, bbox_to_anchor=(0.5, 1.06))
    a.set_title("(a) utility, whole test set vs. neighborhood", fontsize=8, loc="left", color=INK)
    for yi, r in zip(y, rows):
        b.plot([r["d_lo"], r["d_hi"]], [yi, yi], color=NBR, lw=2.5, solid_capstyle="round", zorder=2)
        b.scatter([r["d_nbr"]], [yi], s=MS, marker="s", color=NBR, edgecolor="white", linewidth=1.2, zorder=3)
    b.axvline(0, color=MUTED, lw=0.8, zorder=1)
    b.set_xlabel("Neighborhood utility gained over\nfine-tuning on retain+forget set\n(points of $\\tau_P$, 95% CI)")
    b.set_title("(b) neighborhood gain vs. retain+forget", fontsize=8, loc="left", color=INK)
    b.tick_params(axis="y", length=0)
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=200)
    print(f"wrote {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--from-csv", action="store_true")
    a = ap.parse_args()
    path = os.path.join(HERE, f"scope_locality_{a.dataset}.csv")
    if a.from_csv:
        rows = list(csv.DictReader(open(path)))
        for r in rows:
            for k in ("tauP_whole", "tauP_nbr", "d_nbr", "d_lo", "d_hi"): r[k] = float(r[k])
    else:
        rows = compute(a.dataset)
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        print(f"wrote {path}")
    draw(rows, os.path.join(HERE, "..", "figures", f"scope_locality_{a.dataset}.pdf"))


if __name__ == "__main__":
    main()
