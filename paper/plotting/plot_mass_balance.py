"""Teaser: the neighborhood term adds mass to the neighborhood, the separation
term takes it back, and removal deepens throughout.

Two panels over one arm progression, on held-out test queries, scored against the
retrained model so that zero is parity with a model that never saw the attack:

  (a) what the NEIGHBORHOOD gains, over prefix-disjoint controls from the same
      run. Isolating the gap rather than the raw neighborhood value removes the
      part of the change that every item sees.
  (b) what the DELETED ITEM loses, i.e. how much removal was achieved.

Read together they say: lambda_n redirects freed mass into the code neighborhood
in proportion to its weight, and adding lambda_s pulls that back while deepening
removal, so the two terms trade against each other rather than stacking.

MASS, NOT A MEAN OF PER-ITEM DELTAS: a group that half rises and half falls reads
as unchanged under the mean, which also carries a ~0.75 nat per-item estimation
floor. Summing inside the log removes both and is what L_neighborhood optimises.

    python -m scripts.plot_mass_balance
"""
from __future__ import annotations

import argparse
import json
import math
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = "logs/diagnose/drift_test_mass_vs_clean"
STRATA = [("unpopular", "o", "#9EBBD9"), ("mid", "s", "#5B7DB1"),
          ("popular", "^", "#22303F")]
# The per-term progression: repair alone, then forgetting, then the
# neighborhood term, then the separation term on top of it. Named by which term
# is added rather than by lambda values, so the panel reads without the caption.
# Ordered by how much removal pressure the objective carries, so panel (b) is
# monotone: the two lambda_f = 0 arms first, then forgetting, then forgetting
# with the neighborhood term, then the separation term on top. The
# `L_nbhd only` arm is the CONTROL for panel (a): it differs from Finetune in
# lambda_n alone, so the gap between those two points is the neighborhood term
# acting with no forgetting anywhere in the objective.
# The per-term chain: repair alone, then forgetting, then the neighborhood term,
# then the separation term on top of it. Each step adds one term, so panel (a)
# rises then falls and panel (b) falls monotonically.
#
# `unified_lf0.0_ls0.0_ac0_ln0.01_nbronly` (lambda_n with NOTHING else) is
# deliberately NOT plotted: it belongs between Finetune and Forget + Repair by
# removal, but between NAU arms by neighborhood gain, so inserting it makes one
# panel or the other zigzag. It is the cleanest isolation of the neighborhood
# term (it differs from Finetune in lambda_n alone) and is reported in
# tables/drift_mass_beauty.tex instead.
ARMS = [
    ("unified_lf0.0_ls0.0_ac0",                      "Retain only"),
    ("unified_lf0.1_ls0.0_ac0_ln0.0",                "Forget + Retain"),
    # Named by which terms are in the objective rather than by lambda values:
    # the caption already fixes lambda_f = 0.1 and lambda_n = 0.01 throughout,
    # so the values on the ticks were redundant and crowded the axis.
    ("unified_lf0.1_ls0.0_ac0_ln0.01",
     "NAU\n(Forget + Retain\n+ Neighborhood)"),
    ("unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative",
     "NAU\n(full objective)"),
]

# The contrast arm for --with-ascent: lambda_f = 0.1 with EVERY other weight at
# zero, the retain term included. It is the baseline that genuinely removes the
# target, and the ablation records what that costs: SH@10 0.000000 but UR 0.001,
# i.e. 0.1% of clean nDCG retained (tables/baseline_ablation_beauty.md).
#
# Placed FIRST on the axis, so the panels read left to right as "what removal
# costs without any retain term" and then what each added term buys back. That
# breaks the old ordering-by-removal-pressure, under which panel (b) was
# monotone; it no longer is, and the caption should not claim otherwise.
ASCENT_ARM = ("unified_abl_lf0.1_ls0.0_ln0.0_lr0.0", "Forget only")


def cell(stratum, arm, root, field="delta"):
    """Group log-mass for one (stratum, arm).

    `field` selects what is plotted:
      delta     -- unlearned minus the reference, the published view. Zero is
                   parity with the reference, so the panels read as "how far
                   from a model that never saw the attack".
      unlearned -- the ABSOLUTE log-mass the unlearned model puts on each group.
                   No reference is subtracted, so the curves show where the mass
                   actually sits rather than how it moved. Useful for arguing
                   what the terms DO, since a delta hides that two arms can
                   differ by the same amount from the reference while sitting at
                   very different absolute levels.
    """
    f = os.path.join(root, f"beauty_{stratum}_s2_{arm}", "collateral_drift.json")
    if not os.path.isfile(f):
        return None
    m = (json.load(open(f)).get("group_log_mass") or {})
    d = {k: m.get(k, {}).get(field) for k in ("target", "nbr", "ctrl")}
    return None if any(v is None for v in d.values()) else d


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--out", default=None,
                    help="output stem; defaults to a name matching --mode")
    ap.add_argument("--scale", choices=("symlog", "linear"), default="symlog",
                    help="y scale for panel (b) in delta mode. symlog "
                         "(default) compresses the tail so the small early "
                         "steps stay visible next to the final arm; linear "
                         "keeps bar length proportional to value.")
    ap.add_argument("--stratum", choices=("all", "unpopular", "mid", "popular"),
                    default="all",
                    help="restrict to ONE popularity stratum, giving one bar "
                         "per arm instead of three. Easier to read as a teaser; "
                         "the y range then comes from that stratum alone.")
    ap.add_argument("--with-ctrl", action="store_true", dest="with_ctrl",
                    help="add a third panel: what UNRELATED items lose. The "
                         "collateral the other two panels cannot show, since "
                         "(a) plots a gap in which ctrl cancels out.")
    ap.add_argument("--with-ascent", action="store_true", dest="with_ascent",
                    help="append the pure-ascent (lambda_r=0) arm. The y range "
                         "still comes from the other arms only, so its bars "
                         "clip and carry their true value as a label.")
    ap.add_argument("--linthresh-a", type=float, default=0.1, dest="linthresh_a",
                    help="panel (a): magnitude where the linear band around 0 "
                         "ends and the log decades begin (default 0.1).")
    ap.add_argument("--linthresh-b", type=float, default=1.0, dest="linthresh_b",
                    help="panel (b): same, default 1.0.")
    ap.add_argument("--form", choices=("bar", "line"), default="bar",
                    help="bar: grouped bars, three strata per arm (default). "
                         "line: the previous connected-marker form.")
    ap.add_argument("--mode", choices=("delta", "absolute"), default="delta",
                    help="delta: vs the reference (published). absolute: the "
                         "unlearned model's own log-mass, no reference "
                         "subtracted.")
    a = ap.parse_args()
    field = "delta" if a.mode == "delta" else "unlearned"
    # `Forget only` goes at the LEFT, so the index of every other arm shifts by
    # one and `bound_arms` has to shift with it -- computing it as range(len(ARMS))
    # would silently hand the y range to the wrong arms.
    arms = ([ASCENT_ARM] + list(ARMS)) if a.with_ascent else list(ARMS)
    # Arms whose values set the y range. `Forget only` is deliberately NOT among
    # them: letting a -52 nat control collapse size the axis would squash every
    # other arm onto the zero line, which is the comparison the figure is for.
    bound_arms = set(range(1, len(arms))) if a.with_ascent else set(range(len(arms)))
    if a.out is None:
        a.out = ("figures/teaser_mass_balance" if a.mode == "delta"
                 else "figures/teaser_mass_balance_absolute")
        if a.with_ctrl:
            a.out += "_ctrl"
        if a.with_ascent:
            a.out += "_ascent"
        if a.stratum != "all":
            a.out += f"_{a.stratum}"

    x = np.arange(len(arms))
    n_panel = 3 if a.with_ctrl else 2
    # Width per panel scales with the arm count: five arms with names as long as
    # "NAU (Forget + Retain + Neighborhood)" collide at the four-arm width, and
    # the fix belongs here rather than in shorter labels, which were chosen
    # deliberately.
    per_panel = 5.5 + 0.75 * max(0, len(arms) - 4)
    fig, _axes = plt.subplots(1, n_panel, sharex=True,
                              figsize=(per_panel * n_panel, 4.1))
    axA, axB = _axes[0], _axes[1]
    axC = _axes[2] if a.with_ctrl else None
    axes = tuple(_axes)
    # Grouped bars: one group per arm, one bar per stratum. Both panels are
    # signed quantities against a zero baseline, which is the case bars are for:
    # every bar is anchored at 0, so sign and magnitude read off the length
    # directly instead of off a marker's height above a rule.
    #
    # NOTE ON PANEL (b)'s SCALE (--scale, default symlog). On symlog a bar's
    # length is NOT proportional to its value, so the panel should be read off
    # its axis labels rather than by comparing bar lengths: the final arm's ~17
    # nat drop does not look 4x the ~4 nat one that precedes it. The trade is
    # deliberate -- linear is faithful but squashes the first three arms against
    # zero, symlog keeps those early steps legible. `--scale linear` restores
    # proportionality.
    strata = ([t for t in STRATA if t[0] == a.stratum] if a.stratum != "all"
              else list(STRATA))
    n_st = len(strata)
    # 0.78 leaves a visible gutter between groups. Capped so a single-stratum
    # figure does not get one absurdly fat bar per arm.
    width = min(0.78 / n_st, 0.5)
    seen: List[float] = []       # values from `bound_arms` only, for the y range
    drawn: List[Tuple[Any, float, float]] = []   # (ax, x position, true value)
    have: set = set()            # arm indices with at least one measured cell
    for k, (st, mark, col) in enumerate(strata):
        # PER ARM, not per stratum. The old guard dropped a whole stratum if any
        # single arm was missing, so adding an arm before its cells had landed
        # silently emptied the figure instead of showing the arms that exist.
        vals = [cell(st, arm, a.root, field) for arm, _l in arms]
        if all(v is None for v in vals):
            continue
        ya = [None if v is None else v["nbr"] - v["ctrl"] for v in vals]
        yb = [None if v is None else v["target"] for v in vals]
        yc = [None if v is None else v["ctrl"] for v in vals]
        for i, trio in enumerate(zip(ya, yb, yc)):
            if any(v is not None for v in trio):
                have.add(i)
            if i in bound_arms:
                seen.extend(v for v in trio if v is not None)
        if a.form == "line":
            axA.plot(x, ya, marker=mark, ms=6, lw=1.8, color=col,
                     label=f"{st} target")
            axB.plot(x, yb, marker=mark, ms=6, lw=1.8, color=col,
                     label=f"{st} target")
            continue
        # Centre the group on the tick: offsets run -w, 0, +w for three strata.
        off = (k - (n_st - 1) / 2) * width
        panels = [(axA, ya), (axB, yb)] + ([(axC, yc)] if axC is not None else [])
        for ax, y in panels:
            xs = [xi + off for xi, v in zip(x, y) if v is not None]
            vs = [v for v in y if v is not None]
            ax.bar(xs, vs, width * 0.92, color=col,
                   label=(f"{st} target" if ax is axA else None),
                   edgecolor="white", linewidth=0.8, zorder=3)
            drawn.extend((ax, xi, v) for xi, v in zip(xs, vs))
    for ax in axes:
        # A zero line is a reference marker; in absolute mode zero is not a
        # meaningful level (every log-mass is negative), so it is dropped.
        if a.mode == "delta":
            ax.axhline(0, color="k", lw=.9)
        ax.set_xticks(x)
        ax.set_xticklabels([l for _a, l in arms],
                           fontsize=8 if len(arms) <= 4 else 7.5)
        if a.form == "bar":
            # Horizontal only: a vertical rule through a bar group reads as a
            # divider between bars that belong together.
            ax.grid(alpha=.25, axis="y", zorder=0)
            ax.set_axisbelow(True)
            ax.set_xlim(-0.5, len(arms) - 0.5)
        else:
            ax.grid(alpha=.25)
    axA.set_ylabel("probability mass the neighborhood gains\n"
                   "over unrelated items (nats)"
                   if a.mode == "delta" else
                   "log probability mass gap, neighborhood\n"
                   "minus unrelated items (nats)",
                   fontsize=9)
    axA.set_title("(a) Neighborhood probability mass changes after unlearning",
                  loc="left", fontsize=10.5)
    if a.mode == "delta":
        if a.scale == "symlog":
            # linthresh=1.0: linear within +/-1 nat, logarithmic outside, so the
            # sign is preserved and the small early steps are not squashed
            # against zero by the final arm's ~17 nats. Costs proportionality
            # (see the note in the plotting block); `--scale linear` trades back.
            #
            # BOTH panels carry it, so the two are read on one scale rather than
            # on two that merely look alike. Note that every value in panel (a)
            # falls inside the +/-1 nat linear band, so (a) is unchanged in
            # appearance -- symlog IS linear there. The panels still do not share
            # y LIMITS: (a) spans about 0.9 nats and (b) about 25, so a shared
            # range would flatten (a) into the axis.
            # SYMLOG, not log: these are DELTAS and go negative, which a
            # pure log scale cannot draw at all. `linthresh` is where the linear
            # band around zero ends and the decades begin, so it must sit BELOW
            # the smallest magnitude plotted -- otherwise the values land inside
            # the linear band and the axis is logarithmic in name only.
            #
            # ONE scale and ONE range for both panels. That is only readable
            # because the axis is logarithmic: (a) tops out near 0.7 nats and
            # (b) near 17, which is 24x on a linear axis (so (a) would collapse
            # onto the zero line) but about 1.4 decades here, so both panels
            # stay legible while being directly comparable.
            from matplotlib.ticker import FuncFormatter, SymmetricalLogLocator
            # Powers of ten on the decade ticks: 10^-1 rather than 0.1, with
            # the sign carried outside the power so the negative half reads
            # -10^0 and not 10^-0. Zero is the one tick with no exponent.
            def _pow10(v, _pos):
                if v == 0:
                    return "0"
                exp = int(round(math.log10(abs(v))))
                return rf"${'-' if v < 0 else ''}10^{{{exp}}}$"

            _plain = FuncFormatter(_pow10)
            _lt = min(a.linthresh_a, a.linthresh_b)

            def _nice(v: float) -> float:
                """Smallest 1-2-5 ladder bound at or above |v|."""
                import math
                v = abs(v)
                if v <= 0:
                    return _lt
                dec = 10.0 ** math.floor(math.log10(v))
                for m in (1, 2, 5, 10):
                    if v <= m * dec * (1 + 1e-12):
                        return m * dec
                return 10 * dec

            _hi = _nice(max(seen)) if seen else 1.0
            _lo = -_nice(min(seen)) if seen and min(seen) < 0 else 0.0
            for _ax in axes:
                _ax.set_yscale("symlog", linthresh=_lt, linscale=0.4)
                _ax.set_ylim(_lo, _hi)
                _ax.yaxis.set_major_locator(
                    SymmetricalLogLocator(base=10, linthresh=_lt))
                # The 2..9 subdivisions inside each decade: the 0.2, 0.3, ...
                # 0.9 marks between 0.1 and 1, and likewise 2..9 between 1 and 10.
                _ax.yaxis.set_minor_locator(
                    SymmetricalLogLocator(base=10, linthresh=_lt,
                                          subs=[2, 3, 4, 5, 6, 7, 8, 9]))
                _ax.yaxis.set_major_formatter(_plain)
                _ax.tick_params(axis="y", which="minor", length=2.6, width=0.6)
                _ax.tick_params(axis="y", which="major", length=4.5)
                _ax.grid(alpha=.10, axis="y", which="minor", zorder=0)
        axB.set_ylabel("probability mass left on the deleted item,\n"
                       "vs the retrained model (nats)", fontsize=9)
        axB.set_title("(b) Target removal varies across objectives",
                      loc="left", fontsize=10.5)
    else:
        # Absolute log-mass is negative and unbounded below; symlog around 0
        # would compress exactly the region the curves live in.
        axB.set_ylabel("log probability mass on the deleted item\n"
                       "(unlearned model, nats)", fontsize=9)
        axB.set_title("(b) where the deleted item's probability mass ends up",
                      loc="left", fontsize=10.5)
        axB.annotate("lower: less likely to be recommended", xy=(0.97, 0.93),
                     xycoords="axes fraction", ha="right", fontsize=8,
                     style="italic", color="#555555")
    # Bars that run past the axis: keep the range, print the truth. Expanding
    # the range to fit a -52 nat control collapse would squash every other arm
    # onto the zero line, losing the comparison the figure exists to make. So
    # the bar clips at the edge and carries its real value as a label -- the
    # reader sees both that it left the chart and by how much.
    #
    # Only the clipped bars are labelled. A number on every bar would be 24-36
    # of them and would bury the three that matter.
    for ax, xi, val in drawn:
        lo, hi = ax.get_ylim()
        if lo <= val <= hi:
            continue
        over = val > hi
        ax.annotate(f"{val:,.0f}",
                    xy=(xi, hi if over else lo),
                    xytext=(0, -7 if over else 7), textcoords="offset points",
                    ha="center", va="top" if over else "bottom",
                    fontsize=7, rotation=90, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.12", fc="white",
                              ec="none", alpha=0.85))

    # AN EMPTY SLOT IS NOT A ZERO. A bar of height zero and a bar that was never
    # drawn look the same on this chart, so an arm with no cells on disk is
    # labelled rather than left bare -- otherwise "not measured yet" reads as
    # "this objective changed nothing", which for pure ascent is the opposite of
    # the truth (it removes the target and destroys the model).
    for i in range(len(arms)):
        if i in have:
            continue
        for ax in axes:
            ax.annotate("not measured", xy=(i, 0.5), xycoords=("data",
                        "axes fraction"), ha="center", va="center",
                        fontsize=7.5, rotation=90, style="italic",
                        color="#999999", zorder=4)

    if axC is not None:
        axC.set_ylabel("probability mass left on unrelated items,\n"
                       "vs the retrained model (nats)", fontsize=9)
        # The collateral panel. (a) subtracts ctrl to isolate the neighborhood,
        # so a run that drains the WHOLE catalog shows up there as a small gap
        # and looks harmless. Only this panel distinguishes "moved mass around
        # the neighborhood" from "destroyed the model".
        axC.set_title("(c) Collateral to unrelated items",
                      loc="left", fontsize=10.5)

    axA.legend(frameon=False, fontsize=8.5, loc="upper left")
    if n_st == 1:
        # One bar per arm: the colour carries no contrast, so the legend is the
        # only thing naming the stratum. Kept for that reason, not for contrast.
        axA.legend(frameon=False, fontsize=9, loc="upper left")
    os.makedirs(os.path.dirname(a.out) or ".", exist_ok=True)
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(f"{a.out}.{ext}", bbox_inches="tight", dpi=200)
    print(f"wrote {a.out}.pdf / .png")


if __name__ == "__main__":
    main()
