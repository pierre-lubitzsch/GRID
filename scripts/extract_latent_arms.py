#!/usr/bin/env python
"""Comparison tables for the refined-ID-space (latent refiner) L_n arms.

    python -m scripts.extract_latent_arms [--md tables/latent_arms_beauty.md]

Reads the post-unlearn eval metrics out of the SLURM job logs of every
``bw_tgt<strat>_unified_lf*_ls*_ac0_ln*`` run dir under ``logs/unlearn/runs`` and
emits one table per (lambda_f, lambda_s, target) panel, with the neighbourhood
definition and lambda_n as the rows. That layout is the point: the sweep varies
only the NEIGHBOUR SOURCE within a panel, so a panel is the unit in which two
arms are actually comparable.

Why the job log and not the run dir: the sequential unlearn wrapper runs the
post-unlearn eval itself and prints a lightning metrics table to stdout; the run
dir keeps only ``scif_info.json`` plus per-request scratch. The LAST occurrence of
each metric in the log is the post-unlearn one.

Metrics, matching ``scripts/compute_relative_utility.py`` exactly:
  UR     = NDCG@10_run / NDCG@10_clean
  PGR@10 = (SH_poison - SH_run) / (SH_poison - SH_clean)
           1.0 = spam exposure pushed down to the clean model, 0.0 = no change,
           <0 = worse, >1 = overshot below the clean baseline.

Read with the seed noise in mind: WORKFLOW.md measures 1.3-3.2 UR points of
seed-to-seed spread on these models, so a sub-point difference between two arms
at ONE seed is not a result. The footer says so rather than letting the table
imply more precision than it has.
"""

from __future__ import annotations

import argparse
import glob
import os
import re
from collections import OrderedDict
from typing import Dict, List, Optional, Tuple

CLEAN_NDCG10_BEAUTY_W256 = 0.03762655

# References for beauty w256 bandwagon n_target=1 non-PKM (WORKFLOW.md: POISON
# MODELS jobs 9872116-8; CLEAN BASELINE job 9882465).
#
# NOTE on `popular`: its poison gap is only 0.00058 -- the clean model already
# recommends item 1594 at SH@10 0.00125 and the attack only lifts it to 0.00183.
# PGR there is NOISE-DOMINATED: a value like +3.2 means "SH went to zero", NOT
# "3x better unlearning". WORKFLOW.md flags this explicitly.
SH_REF = {
    #            SH_poison  SH_clean
    "unpopular": (0.01400,  0.00000),
    "mid":       (0.01274,  0.00000),
    "popular":   (0.00183,  0.00125),
}
NOISY_PGR = {"popular"}

STRATA = ["unpopular", "mid", "popular"]

# Neighbour-source label per run-dir suffix, in reporting order.
METHOD_LABEL = OrderedDict([
    ("prefix", "N_prefix (shipped default)"),
    ("embedding", "N_emb"),
    ("latent", "N_z (latent, overlap .78)"),
    ("latent_zlow", "N_z (LOW overlap .42)"),
    ("emblatent_full", "union full"),
    ("emblatent_matched", "union matched"),
])

RUN_RE = re.compile(
    r"job(\d+)_.*_bs1_bw_tgt(\w+?)_unified_lf([\d.]+)_ls([\d.]+)_ac0_ln([\d.]+)(?:_(\w+))?$"
)

# The trailing `_useed<N>` (absent for the default seed 2) is stripped BEFORE the
# main regex rather than captured by it. As an optional trailing group it loses:
# the engine happily satisfies the optional method group with "useed3" and leaves
# the seed group empty, so `..._ln0.0_useed3` parses as method="useed3", seed=2 --
# a seeded run silently attributed to the wrong seed and dropped from every table.
SEED_RE = re.compile(r"_useed(\d+)$")


def _metric(txt: str, key: str) -> Optional[float]:
    """Last value of `test/<key>` in a lightning metrics table."""
    hits = re.findall(rf"test/{re.escape(key)}\s*│\s*([\d.eE+-]+)", txt)
    return float(hits[-1]) if hits else None


def _method_of(rest: Optional[str]) -> str:
    """`mass_latent_dz128` -> `latent`; None (the ln0.0 arm) -> ''."""
    if not rest:
        return ""
    r = re.sub(r"^(mass|nll|suppress)_", "", rest)
    r = re.sub(r"_dz\d+", "", r)
    return r


def collect(runs_dir: str, logs_dir: str) -> List[Dict]:
    out = []
    for d in sorted(glob.glob(os.path.join(runs_dir, "job*_bs1_bw_*"))):
        base = os.path.basename(d)
        sm = SEED_RE.search(base)
        seed = sm.group(1) if sm else "2"
        if sm:
            base = base[: sm.start()]
        m = RUN_RE.match(base)
        if not m:
            continue
        jid, strat, lf, ls, ln, rest = m.groups()
        log = os.path.join(logs_dir, f"{jid}.out")
        if not os.path.exists(log):
            continue
        txt = open(log, errors="ignore").read()
        # A run dir with no metrics is EITHER crashed OR still in flight, and
        # calling the second one "crashed" is exactly the misreport the marker
        # exists to prevent. Distinguish on the error signature in the log.
        crashed = bool(re.search(r"CUDA out of memory|Error executing job|Traceback", txt))
        out.append({
            "crashed": crashed,
            "job": jid, "strat": strat, "lf": lf, "ls": ls, "ln": ln,
            "method": _method_of(rest), "seed": seed,
            "sh10": _metric(txt, "SH@10"), "ndcg10": _metric(txt, "ndcg@10"),
            "recall10": _metric(txt, "recall@10"),
        })
    return out


def _agg(vals: List[float], fmt: str, signed: bool = False) -> str:
    """mean +- half-range, the aggregation every other extractor here uses.

    Half-range (not stddev) because n is 1-3: with three points the spread is
    better communicated by its extent than by an estimate of sigma that n=3
    cannot support. n=1 prints the bare value, so a single-seed cell can never be
    mistaken for a replicated one.
    """
    if not vals:
        return "—"
    lo, hi = min(vals), max(vals)
    mean = sum(vals) / len(vals)
    sign = "+" if signed else ""
    if len(vals) == 1:
        return f"{mean:{sign}{fmt}}"
    return f"{mean:{sign}{fmt}} ±{(hi - lo) / 2:{fmt}}"


def _row(label: str, rs: List[Dict], sh_p: float, gap: float,
         clean_ndcg: float) -> str:
    """One table row aggregated over however many seeds exist for this arm."""
    if not rs:
        return f"| {label} | — | — | — | — | — | — |"
    # Run dirs with no metrics in their log are CRASHED runs (e.g. the OOMing
    # emblatent_full arm). Say so instead of dropping them: a silently missing
    # row reads as "not run yet", a different and misleading state.
    good = [r for r in rs if r["sh10"] is not None and r["ndcg10"] is not None]
    if not good:
        jobs = ",".join(r["job"] for r in rs)
        state = "*crashed*" if all(r.get("crashed") for r in rs) else "*running*"
        return f"| {label} | {jobs} | {state} | — | — | — | — |"
    seeds = ",".join(sorted({r["seed"] for r in good}, key=int))
    n = len(good)
    # If SOME seeds of this arm crashed, say so. Reporting only the survivors
    # reads as "n seeds, all fine" and would quietly bias the arm toward whatever
    # the surviving seeds happened to show.
    n_crash = sum(1 for r in rs if r not in good and r.get("crashed"))
    n_wip = len(rs) - n - n_crash
    extra = []
    if n_crash:
        extra.append(f"+{n_crash} crashed")
    if n_wip:
        extra.append(f"+{n_wip} running")
    if extra:
        seeds = f"{seeds} ({', '.join(extra)})"
    sh = _agg([r["sh10"] for r in good], ".5f")
    pgr = _agg([(sh_p - r["sh10"]) / gap for r in good], ".3f", signed=True) if gap else "—"
    ndcg = _agg([r["ndcg10"] for r in good], ".5f")
    ur = _agg([r["ndcg10"] / clean_ndcg for r in good], ".3f")
    bold = "**" if n > 1 else ""
    return (f"| {label} | {bold}s{seeds}{bold} | {sh} | {pgr} | {ndcg} | {ur} "
            f"| {_agg([r['recall10'] for r in good], '.5f')} |")


def main(argv: List[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs-dir", default="logs/unlearn/runs")
    p.add_argument("--logs-dir", default="logs")
    p.add_argument("--clean-ndcg", type=float, default=CLEAN_NDCG10_BEAUTY_W256)
    p.add_argument("--md", default=None, help="also write the markdown here")
    args = p.parse_args(argv)

    rows = collect(args.runs_dir, args.logs_dir)
    if not rows:
        raise SystemExit(f"no matching run dirs under {args.runs_dir!r}")

    # Panels: (lambda_f, lambda_s). lambda_n becomes a row group inside.
    panels: List[Tuple[str, str]] = sorted(
        {(r["lf"], r["ls"]) for r in rows},
        key=lambda t: (float(t[0]), -float(t[1])),
    )
    lns = sorted({r["ln"] for r in rows if r["ln"] != "0.0"}, key=float)

    lines: List[str] = []
    for lf, ls in panels:
        lines.append(f"\n#### lambda_f = {lf}, lambda_s = {ls}\n")
        for strat in STRATA:
            here = [r for r in rows if r["strat"] == strat
                    and r["lf"] == lf and r["ls"] == ls]
            if not here:
                continue
            sh_p, sh_c = SH_REF.get(strat, (float("nan"),) * 2)
            gap = sh_p - sh_c
            flag = ("  **PGR noise-dominated** (gap only %.5f)" % gap
                    if strat in NOISY_PGR else "")
            lines.append(f"**target = {strat}** — SH_poison {sh_p:.5f}, "
                         f"SH_clean {sh_c:.5f}, gap {gap:.5f}.{flag}\n")
            lines.append("| arm | seeds | SH@10 | PGR@10 | NDCG@10 | UR | recall@10 |")
            lines.append("|-----|-------|-------|--------|---------|----|-----------|")
            lines.append(_row("L_n off", [r for r in here if r["ln"] == "0.0"],
                              sh_p, gap, args.clean_ndcg))
            for ln in lns:
                if not any(x["ln"] == ln for x in here):
                    continue  # this lambda_n was never run in this panel
                for meth, label in METHOD_LABEL.items():
                    rs = [x for x in here
                          if x["ln"] == ln and x["method"] == meth]
                    lines.append(_row(f"λ_n={ln} {label}", rs, sh_p, gap,
                                      args.clean_ndcg))
            lines.append("")
    body = "\n".join(lines)
    print(body)

    note = (
        f"_UR = NDCG@10 / {args.clean_ndcg} (clean beauty w256). "
        f"PGR@10 = (SH_poison - SH_run) / (SH_poison - SH_clean). "
        f"{len(rows)} runs found._\n"
        "\n_Seed noise on these models is 1.3-3.2 UR points (WORKFLOW.md, "
        "HOLDOUT_HEAD2HEAD). These are single-seed (useed 2) runs, so a "
        "sub-point gap between two arms is NOT a result._"
    )
    print(note)
    if args.md:
        os.makedirs(os.path.dirname(os.path.abspath(args.md)) or ".", exist_ok=True)
        with open(args.md, "w") as fh:
            fh.write(body + "\n" + note + "\n")
        print(f"\nwrote {args.md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
