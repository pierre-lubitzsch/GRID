"""Teaser, pivoted: one panel per popularity stratum, three bars per objective.

`plot_mass_balance` puts the OBJECTIVES on x and splits the three groups across
panels, with the strata as the series. This script transposes that: one figure
per stratum, objectives on x, and the three item groups as the series --

    target item      what was asked to be removed      (red)
    neighborhood     items sharing the target's code   (orange)
    unrelated        prefix-disjoint controls          (light blue)

Read left to right it says what each term buys: `Forget only` drives all three
groups off the chart, and every objective that keeps a retain term holds the
unrelated items flat while still moving the target.

All three groups on ONE axis, which the panelled version cannot do. That is the
point of the pivot: the comparison that matters is target-versus-unrelated
WITHIN an objective, and across panels the reader has to carry a number in their
head to make it.

COLOURS are semantic, not decorative: red is the thing that should go, blue the
thing that should not move, orange the neighbourhood in between. Checked for
colour-vision deficiency rather than eyeballed -- the red/orange pair is the
classic failure, and at #B2182B/#F1A340 it separates by OKLab dE 21.8
(deuteranope) and 23.2 (protanope), against a floor of 8. An earlier
#C0392B/#E67E22 pair passed at only 12.5/13.1.

    python -m scripts.plot_mass_groups --stratum unpopular
    python -m scripts.plot_mass_groups --all-strata
"""
from __future__ import annotations

import argparse
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator

try:                      # in-repo: python -m scripts.plot_mass_groups
    from scripts.plot_mass_balance import ARMS, ASCENT_ARM, ROOT, cell
except ImportError:       # flat bundle: both files side by side, no package
    from plot_mass_balance import ARMS, ASCENT_ARM, ROOT, cell


def _cell_from_csv(path):
    """Build a `cell`-compatible reader over paper/data/mass_groups.csv.

    Lets a collaborator regenerate this figure from the committed CSV with no
    access to logs/. Same signature and same return shape as
    plot_mass_balance.cell, so nothing downstream changes.
    """
    import csv as _csv
    tbl = {}
    with open(path) as fh:
        for r in _csv.DictReader(fh):
            tbl.setdefault((r["stratum"], r["arm"]), {})[r["group"]] = r
    def _cell(stratum, arm, root, field="delta"):
        got = tbl.get((stratum, arm))
        if not got:
            return None
        try:
            d = {k: float(got[k][field]) for k in ("target", "nbr", "ctrl")}
        except (KeyError, TypeError, ValueError):
            return None
        return d
    return _cell

# The POISONED model, i.e. the state every arm starts from. Its `unlearned`
# field holds the poisoned model's own log-mass, so differencing against it
# answers "how much did unlearning move each group", which is a different
# question from "how far is the result from a model that never saw the attack".
POISONED_ARM = "original_vs_retrained"

# group key -> (legend label, colour)
GROUPS = [
    ("target", "Target item", "#B2182B"),
    ("nbr", "Neighborhood item", "#F1A340"),
    ("ctrl", "Unrelated item", "#92C5DE"),
]


def _nice(v: float, floor: float) -> float:
    """Smallest 1-2-5 ladder bound at or above |v|."""
    v = abs(v)
    if v <= 0:
        return floor
    dec = 10.0 ** math.floor(math.log10(v))
    for m in (1, 2, 5, 10):
        if v <= m * dec * (1 + 1e-12):
            return m * dec
    return 10 * dec


def _pow10(v, _pos):
    if v == 0:
        return "0"
    return rf"${'-' if v < 0 else ''}10^{{{int(round(math.log10(abs(v))))}}}$"


def draw(stratum: str, root: str, out: str, linthresh: float,
         with_ascent: bool, vs_poisoned: bool = False,
         cell=cell) -> str:
    # Dropped from the GROUPED figure only: forget + retain + neighborhood with
    # no separation term. It is a partial objective that the teaser does not
    # argue about, and removing it keeps the panel to the contrast that matters
    # (baselines vs the full objective). ARMS itself is shared with
    # plot_mass_balance, so it is filtered here rather than edited there.
    EXCLUDE = {"unified_lf0.1_ls0.0_ac0_ln0.01"}
    arms = ([ASCENT_ARM] + list(ARMS)) if with_ascent else list(ARMS)
    arms = [(a_, l_) for a_, l_ in arms if a_ not in EXCLUDE]
    # Every arm sizes the y range, `Forget only` included, so its -57 nat bars
    # are drawn rather than clipped at the axis edge with a printed annotation.
    # Including it costs one decade on the negative side, which is the better
    # trade than a bar that leaves the frame with a printed annotation.
    bound = set(range(len(arms)))

    x = np.arange(len(arms))
    width = 0.78 / len(GROUPS)
    fig, ax = plt.subplots(figsize=(1.55 * len(arms) + 2.6, 5.0))

    seen, drawn, have = [], [], set()
    if vs_poisoned:
        ref = cell(stratum, POISONED_ARM, root, "unlearned")
        if ref is None:
            raise SystemExit(f"no {POISONED_ARM} cell under {root}")
        raw = [cell(stratum, arm, root, "unlearned") for arm, _l in arms]
        vals = [None if v is None else {k: v[k] - ref[k] for k in v} for v in raw]
    else:
        vals = [cell(stratum, arm, root, "delta") for arm, _l in arms]
    for i, v in enumerate(vals):
        if v is not None:
            have.add(i)

    for k, (key, label, col) in enumerate(GROUPS):
        off = (k - (len(GROUPS) - 1) / 2) * width
        xs = [xi + off for xi, v in zip(x, vals) if v is not None]
        ys = [v[key] for v in vals if v is not None]
        ax.bar(xs, ys, width * 0.92, color=col, label=label,
               edgecolor="white", linewidth=0.8, zorder=3)
        drawn.extend(zip(xs, ys))
        seen.extend(v[key] for i, v in enumerate(vals)
                    if v is not None and i in bound)

    lt = linthresh
    hi = _nice(max(seen), lt) if seen else 1.0
    lo = -_nice(min(seen), lt) if seen and min(seen) < 0 else 0.0
    ax.set_yscale("symlog", linthresh=lt, linscale=0.4)
    ax.set_ylim(lo, hi)
    ax.yaxis.set_major_locator(SymmetricalLogLocator(base=10, linthresh=lt))
    ax.yaxis.set_minor_locator(
        SymmetricalLogLocator(base=10, linthresh=lt,
                              subs=[2, 3, 4, 5, 6, 7, 8, 9]))
    ax.yaxis.set_major_formatter(FuncFormatter(_pow10))
    ax.tick_params(axis="y", which="minor", length=2.6, width=0.6)
    ax.tick_params(axis="y", which="major", length=4.5, labelsize=12)
    ax.axhline(0, color="k", lw=.9)
    ax.grid(alpha=.25, axis="y", zorder=0)
    ax.grid(alpha=.10, axis="y", which="minor", zorder=0)
    ax.set_axisbelow(True)
    ax.set_xlim(-0.5, len(arms) - 0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([l for _a, l in arms],
                       fontsize=13 if len(arms) <= 4 else 12)

    # Bars past the axis keep the range and print the truth; see the note in
    # plot_mass_balance. Only clipped bars are labelled.
    for xi, val in drawn:
        if lo <= val <= hi:
            continue
        over = val > hi
        ax.annotate(f"{val:,.0f}", xy=(xi, hi if over else lo),
                    xytext=(0, -7 if over else 7), textcoords="offset points",
                    ha="center", va="top" if over else "bottom",
                    fontsize=10, rotation=90, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    # An empty slot is not a zero.
    for i in range(len(arms)):
        if i not in have:
            ax.annotate("not measured", xy=(i, 0.5),
                        xycoords=("data", "axes fraction"), ha="center",
                        va="center", fontsize=10, rotation=90,
                        style="italic", color="#999999", zorder=4)

    ax.set_ylabel(("probability mass change\nvs the poisoned model (nats)"
                   if vs_poisoned else
                   "probability mass\nvs the retrained model (nats)"),
                  fontsize=13)
    # NO in-figure title. The claim this figure makes ("baselines either degrade
    # utility or do not remove the influence, and the mass change is bigger in
    # neighbors than in unrelated items") belongs in the LaTeX caption, which
    # also carries the stratum -- the three figures this script writes are
    # otherwise identical and are told apart only by filename.
    # ABOVE the axes, in the band the title used to occupy. Inside the panel is
    # not available: the floor is pinned at -10^2 by request, and the gap between
    # the lowest bar and that floor is ~4% of the axes height, against the ~15% a
    # 13pt row needs -- and on the left the `Forget only` bars run the full depth
    # anyway. Below the axes it has to clear the two-line "NAU (full objective)"
    # tick label, which is what opened the empty band this replaces.
    ax.legend(frameon=False, fontsize=13, ncol=3,
              loc="lower center", bbox_to_anchor=(0.5, 1.005))

    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{out}.{ext}", bbox_inches="tight", dpi=200)
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--stratum", choices=("unpopular", "mid", "popular"),
                    default="unpopular")
    ap.add_argument("--all-strata", action="store_true", dest="all_strata",
                    help="write one figure per stratum.")
    ap.add_argument("--vs-poisoned", action="store_true",
                    help="plot the change from the POISONED model instead of "
                         "the distance to the retrained one")
    ap.add_argument("--no-ascent", action="store_false", dest="with_ascent",
                    help="drop the `Forget only` arm.")
    ap.add_argument("--linthresh", type=float, default=0.01,
                    help="symlog linear-region threshold; 0.01 takes the log "
                         "ticks down to 10^-2. The grouped panel is wide enough\n"
                         "to carry the extra decade; the teaser panel (b) is not\n"
                         "and stays at 0.1.")
    ap.add_argument("--out", default=None, help="output stem")
    ap.add_argument("--data", default=None,
                    help="read paper/data/mass_groups.csv instead of scanning "
                         "--root. Regenerates the published figure with no "
                         "access to logs/.")
    a = ap.parse_args()

    strata = (("unpopular", "mid", "popular") if a.all_strata
              else (a.stratum,))
    for st in strata:
        out = a.out or (f"figures/probability_mass_distribution_{st}"
                        if a.vs_poisoned else
                        f"figures/teaser_mass_groups_{st}")
        if a.out and len(strata) > 1:
            out = f"{a.out}_{st}"
        print("wrote " + draw(st, a.root, out, a.linthresh, a.with_ascent,
                              vs_poisoned=a.vs_poisoned,
                              cell=(_cell_from_csv(a.data) if a.data else cell))
              + ".pdf / .png")


if __name__ == "__main__":
    main()
