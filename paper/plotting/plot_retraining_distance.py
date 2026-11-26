"""Figure 1: how far each unlearned model lands from matched retraining.

    python plotting/plot_retraining_distance.py      # from paper/

Reads plotting/retraining_distance.csv (TIGER + RQ-KMeans, 5 data seeds x 3
target strata per dataset, same run selection as the main spam table) and writes
figures/retraining_distance.{pdf,png}.

x = d_forget = mean_cells |SH_U - SH_R| / mean_cells |SH_P - SH_R|
    Retrain = 0, Poisoned = 1, > 1 means the method moved exposure further from
    retraining than the attack did (in either direction).
y = d_util = mean_cells |NDCG_U - NDCG_R| / NDCG_R, in percent. Deviations above
    retraining count as well; nothing is rewarded for exceeding the reference.
Both axes are symlog so that the origin (retraining) is on the plot.
"""
from __future__ import annotations

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
DATA = HERE / "retraining_distance.csv"
OUT = HERE.parent / "figures" / "retraining_distance"

DATASETS = [("beauty", "Beauty"), ("sports", "Sports"), ("toys", "Toys")]
# Fixed categorical order (never cycled). Methods that collapse utility share
# one muted ink and are told apart by marker and direct label.
STYLE = {
    "NAU":           dict(c="#2a78d6", m="D", s=95, z=6, hollow=True),
    "Forget+Repair": dict(c="#eb6834", m="o", s=52, z=5),
    "Finetune":      dict(c="#1baf7a", m="v", s=52, z=4),
    "SCIF":          dict(c="#4a3aa7", m="^", s=52, z=4),
    "TRACER":        dict(c="#008300", m="p", s=56, z=4),
    "SEIF":          dict(c="#8a8a85", m="P", s=48, z=3),
    "Kookmin":       dict(c="#8a8a85", m="X", s=48, z=3),
    "Fanchuan":      dict(c="#8a8a85", m="h", s=48, z=3),
    "Forget only":   dict(c="#8a8a85", m="s", s=70, z=3, hollow=True),
    "Poisoned":      dict(c="#1f1f1d", m="x", s=60, z=5),
}
LABEL = {"Forget+Repair": "Forget+Repair"}
INK, MUTED, GRID = "#1f1f1d", "#6b6b66", "#e4e3dc"


def load():
    rows = {}
    with open(DATA) as f:
        for r in csv.DictReader(f):
            if r["track"] != "rkmeans" or r["method"].startswith("NAU w/o"):
                continue
            rows.setdefault(r["dataset"], []).append(r)
    return rows


def main():
    rows = load()
    plt.rcParams.update({"font.size": 8, "axes.edgecolor": MUTED,
                         "axes.labelcolor": INK, "xtick.color": MUTED,
                         "ytick.color": MUTED})
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.35), sharey=True)
    for ax, (ds, title) in zip(axes, DATASETS):
        ax.set_xscale("symlog", linthresh=1.0, linscale=1.2)
        ax.set_yscale("symlog", linthresh=10.0, linscale=1.0)
        ax.set_xlim(-0.08, 8.0)
        ax.set_ylim(-0.8, 130)
        ax.grid(True, color=GRID, lw=0.6, zorder=0)
        ax.axvline(1.0, color=GRID, lw=1.0, ls="--", zorder=1)
        ax.scatter([0], [0], marker="*", s=220, c="#eda100",
                   edgecolors=INK, linewidths=0.6, zorder=7, clip_on=False)
        ax.annotate("Retrain", (0, 0), xytext=(6, 4), textcoords="offset points",
                    color=INK, fontsize=7, fontweight="bold")
        for r in rows[ds]:
            m = r["method"]
            if m == "Retrain" or m not in STYLE:
                continue
            st = STYLE[m]
            x, y = float(r["d_forget"]), float(r["d_util"])
            if st.get("hollow"):
                # Hollow so a method sitting at the same point (Forget+Repair
                # under NAU, Fanchuan inside Forget only) stays visible.
                ax.scatter([x], [y], marker=st["m"], s=st["s"], facecolors="none",
                           edgecolors=st["c"], linewidths=1.6, zorder=st["z"])
            else:
                ax.scatter([x], [y], marker=st["m"], s=st["s"], c=st["c"],
                           edgecolors="white" if st["m"] != "x" else None,
                           linewidths=0.8, zorder=st["z"])
        ax.set_title(title, fontsize=9, color=INK)
        ax.set_xlabel(r"forgetting distance $d_{\mathrm{forget}}$", fontsize=8)
        ax.set_xticks([0, 0.5, 1, 2, 5])
        ax.set_xticklabels(["0", "0.5", "1", "2", "5"])
        ax.set_yticks([0, 5, 10, 30, 100])
        ax.set_yticklabels(["0", "5", "10", "30", "100"])
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    axes[0].set_ylabel(r"utility distance $d_{\mathrm{util}}$ (%)", fontsize=8)
    handles = [plt.Line2D([], [], ls="", marker=STYLE[m]["m"], color=STYLE[m]["c"],
                          markerfacecolor="none" if STYLE[m].get("hollow") else STYLE[m]["c"],
                          markeredgewidth=1.4 if STYLE[m].get("hollow") else 1.0,
                          markersize=6, label=LABEL.get(m, m))
               for m in STYLE]
    handles.insert(0, plt.Line2D([], [], ls="", marker="*", color="#eda100",
                                 markeredgecolor=INK, markersize=10, label="Retrain"))
    fig.legend(handles=handles, loc="lower center", ncol=11, frameon=False,
               fontsize=6.5, handletextpad=0.2, columnspacing=0.8,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.08, 1, 1), w_pad=0.6)
    OUT.parent.mkdir(exist_ok=True)
    fig.savefig(OUT.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(OUT.with_suffix(".png"), dpi=220, bbox_inches="tight")
    print("wrote", OUT.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
