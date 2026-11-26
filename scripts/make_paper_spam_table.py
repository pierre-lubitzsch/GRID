"""Paper-format spam table: one row per baseline, ONE row for NAU.

Differs from make_spam_tables.latex_head2head, which is the working table:

* NAU appears ONCE, at the fixed operating point (lambda_f=0.1, lambda_s=0.1,
  lambda_n=0.01, generative separation), not once per configuration family.
  The working table's four NAU rows exist to compare sep scores and the lambda
  grid; in the paper that is the ablation's job, and four rows invite the reader
  to take the best of them, which is a per-cell oracle rather than a method.
  The lambdas are stated in the caption, not in the row label.
* SH@10, NDCG@10 and tau_P are PERCENTAGES, matching the paper's other tables.
* Strata are multirow blocks with a rotated label, split into Baselines / Ours.
* No tau_E column: E (s) already carries cost, and tau_E is 1 - E/E_retrain,
  which is the same number against a denominator the caption can state once.

    python -m scripts.make_paper_spam_table --dataset beauty --model tiger \
        --quantizer rkmeans --out tables/paper_spam_beauty_tiger_rkmeans.tex
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.dirname(_HERE))
import run_tags  # noqa: E402

_spec = importlib.util.spec_from_file_location(
    "h2h", os.path.join(_HERE, "extract_bandwagon_head2head.py"))
h2h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2h)
_spec2 = importlib.util.spec_from_file_location(
    "mst", os.path.join(_HERE, "make_spam_tables.py"))
mst = importlib.util.module_from_spec(_spec2)
_spec2.loader.exec_module(mst)

# The reported operating point, selected by EXACT ARM rather than by family.
# Selecting a family would take that family's best configuration per cell, and
# `unified_sepgenerative` holds two (lambda_s = 0.01 and 0.1), so the row would
# be a best-of-two oracle rather than one method at one setting. The arm below
# is lambda_f=0.1, lambda_s=0.1, lambda_n=0.01 with the generative separation
# score, at 45 runs (3 datasets x 3 strata x 5 seeds) on every track.
#
# LETTER spells it differently: its generative arms come from the NAU-form
# ablation grid (`_nauform`), because the sep-searched families were never run
# on that track. Same objective and same lambdas, different tag.
OURS_ARM = {
    # lambda_s=0.01, lambda_n=0.01: selected 2026-09-25 among full-objective settings
    # (lambda_s, lambda_n > 0) whose validation exposure is statistically tied, by
    # combined whole + neighborhood tau_P (paper/plotting/nau_val_table.py).
    ("tiger", "rkmeans"): "nau_s0p01_n0p01_keepck",
    ("tiger", "rqvae"): "nau_s0p01_n0p01_keepck",
    ("letter", "letter"): "unified_abl_lf0.1_ls0.01_ln0.01_nauform",
    ("tiger", "tracerv3"): "nau_s0p01_n0p01_keepck",
}
OURS_FAMILY = "unified_sepgenerative"   # fallback only
# The weaker-separation setting (lambda_s=0.01, lambda_n=0.01), shown as an UNRANKED
# variant row under NAU: same objective, one knob changed, never bolded or underlined.
VARIANT_ARM = {   # the stronger-separation setting (lambda_s=0.1), unranked
    ("tiger", "rkmeans"): "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative",
    ("tiger", "rqvae"): "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative",
    ("letter", "letter"): "unified_abl_lf0.1_ls0.1_ln0.01_nauform",
    ("tiger", "tracerv3"): "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative",
}
VARIANT_LABEL = r"\quad NAU, $\lambda_s=0.1$"
# `tracer`, not `tracer_fixedsteps`: the latter runs 500 steps on a single
# forget batch, roughly 125x the optimisation the other methods get, and is
# kept a separate family so it cannot be selected into this row.
BASELINES = [("finetune_abl", "Finetune"), ("ga_abl", "Forget only"),
             ("negtrain_abl", "Forget + Repair"), ("scif", "SCIF"),
             ("seif", "SEIF"), ("kookmin", "Kookmin"), ("fanchuan", "Fanchuan"),
             ("tracer", "TRACER")]
STRATA = [("unpopular", "Unpopular"), ("mid", "Mid"), ("popular", "Popular")]


def _agg_arm(runs, arm):
    """Aggregate one ARM's runs into the shape mst._cells consumes.

    h2h.best_per_family returns this shape for the winning arm of a family;
    here the arm is fixed, so the same aggregation is applied directly.
    """
    if not runs:
        return None
    out = {k: h2h.agg(runs, k) for k in ("sh", "n10", "UR", "PGR")}
    out["arm"] = arm
    out["runs"] = runs
    ur = out["UR"]
    out["cleared_floor"] = bool(ur and ur.get("mean") is not None)
    out["empty"] = False
    return out


# family -> method name in valsel_selected.csv; finetune_abl and scif have one config
_PIN_NAME = {"ga_abl": "Forget only", "negtrain_abl": "Forget+Repair", "seif": "SEIF",
             "kookmin": "Kookmin", "fanchuan": "Fanchuan", "tracer": "TRACER"}
_PIN_SINGLE = {"finetune_abl": "unified_abl_lf0.0_ls0.0_ln0.0", "scif": "scif"}


def _pinned_arm(csv_path, dataset, quantizer, fam):
    """TIGER + RQ-KMeans's validation-selected arm of `fam` on `dataset`.

    TRACER is selected on Beauty under its own tokenizer (track tracerv3), so its
    Beauty row is used wherever Beauty has no rkmeans TRACER row.
    """
    import csv
    if fam in _PIN_SINGLE:
        return _PIN_SINGLE[fam]
    rows = {(r["dataset"], r["track"], r["method"]): r["config"]
            for r in csv.DictReader(open(csv_path)) if r.get("selected", "True") == "True"}
    name = _PIN_NAME.get(fam)
    for track in (("tracerv3", "rkmeans") if quantizer == "tracerv3" else ("rkmeans", "tracerv3")):
        if (dataset, track, name) in rows:
            arm = rows[(dataset, track, name)]
            if quantizer == "tracerv3":
                # tracerv3 tags spell out the learning rate; these are the RQ-KMeans
                # runs' defaults (seif_repair_lr 0.0007, kookmin_retain_lr 0.001, fanchuan_lr 0.001)
                arm += {"seif": "_rlr0.0007", "kookmin": "_rlr0.001", "fanchuan": "_lr0.001"}.get(fam, "")
            return arm
    return None


def _family_arms(data, fam):
    """Every arm tag in `data` whose family is `fam`."""
    return sorted({arm for (_st, arm) in data
                   if run_tags.family_of(arm) == fam})


def _pool(data, arms, ur_floor, min_seeds, score=None):
    """Best config for one family, POOLED over the three target strata.

    The per-stratum tables select a config per stratum; pooling keeps the same
    rule (lowest SH@10 among configs clearing the utility floor) but applies it
    once to the union of the strata, so each method is reported at ONE setting
    across the whole table rather than three.

    h2h.agg groups by data seed before taking the spread, so runs of the same
    seed at different strata are averaged first and the reported SD is across
    the five independent seeds, not across 15 correlated cells.
    """
    best = None
    for arm in arms:
        runs = []
        for st, _lbl in STRATA:
            runs.extend(data.get((st, arm), []))
        if not runs:
            continue
        cand = _agg_arm(runs, arm)
        if cand is None:
            continue
        ur, sh = cand["UR"], cand["sh"]
        if ur["n_seeds"] < min_seeds:
            continue
        cleared = ur["mean"] is not None and ur["mean"] >= ur_floor
        cand["cleared_floor"] = cleared
        if score is None:
            obj = sh["mean"] if sh["mean"] is not None else 9e9
        else:
            v = score(arm, runs)
            obj = 9e9 if v is None else v
        key = (0 if cleared else 1, obj)
        if best is None or key < best[0]:
            best = (key, cand)
    return best[1] if best else None


def _num(pm: str, scale: float = 1.0, prec: int = 3):
    """'0.00157±0.00332' -> (mean, sd) scaled, or (None, None)."""
    if not pm or pm in ("n/a", "--"):
        return None, None
    s = pm.replace("±", "|").replace("+-", "|")
    parts = s.split("|")
    try:
        m = float(parts[0]) * scale
        sd = float(parts[1]) * scale if len(parts) > 1 else None
    except ValueError:
        return None, None
    return m, sd


def _fmt(m, sd, prec=3):
    if m is None:
        return "n/a"
    return f"{m:.{prec}f}$\\pm${sd:.{prec}f}" if sd is not None else f"{m:.{prec}f}"


# Distance to retrained exposure, computed PER CELL and then averaged, which is
# the definition Table 1 uses: mean over the (seed, stratum) cells of
# |SH_unlearned - SH_retrained|. The cheap alternative, |mean SH_u - mean SH_r|,
# is a different number: it cancels cells that sit above retraining against
# cells that sit below, so it understates the distance for any method that
# straddles the reference.
#
# make_spam_tables._cells argues no such column is needed because tau_F is the
# signed, gap-normalised version of the same quantity. That holds arithmetically
# (within a block, distance = |tau_F - 1| x gap), but tau_F/PGR is not reportable
# on this data -- it is sound in 1 of 18 blocks -- so the unnormalised distance
# is what Table 1 reports and what these tables must agree with.
def _cell_distance(data, arm, strata, refs_by_seed, require_full=True):
    """(mean, sd) of |SH_u - SH_r| over cells; sd is across seeds.

    require_full rejects an arm that is missing a stratum. It matters because
    this doubles as the SELECTION objective: retrained exposure is 0 on some
    strata, so an arm present only there scores a perfect 0.000 distance and
    wins over a fully-measured arm. `fanchuan_t0.007_keepck` (mid only) beat the
    complete `fanchuan_t0.07` on Beauty exactly this way.
    """
    if not arm:
        return None
    if require_full and any(not data.get((st, arm)) for st in strata):
        return None
    per_seed = {}
    for st in strata:
        for run in data.get((st, arm), []) or []:
            seed, sh = run.get("seed"), run.get("sh")
            ref = refs_by_seed.get(seed, {}).get(st)
            if sh is None or ref is None:
                continue
            per_seed.setdefault(seed, []).append(abs(float(sh) - float(ref)))
    # average the strata within a seed first, so the spread is across seeds
    vals = [100.0 * (sum(v) / len(v)) for v in per_seed.values() if v]
    if not vals:
        return None
    mean = sum(vals) / len(vals)
    if len(vals) < 2:
        return (mean, 0.0)
    var = sum((x - mean) ** 2 for x in vals) / (len(vals) - 1)
    return (mean, var ** 0.5)


def _refs_by_seed(ds, model, quantizer, seeds):
    out = {}
    for sd in sorted(seeds):
        r, _src = h2h.resolve_reference(ds, model, quantizer, sd)
        if r is not None:
            out[sd] = r.get("clean_sh", {})
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--model", default=run_tags.DEFAULT_MODEL)
    ap.add_argument("--quantizer", default=run_tags.DEFAULT_QUANTIZER)
    ap.add_argument("--runs-root", default="logs/unlearn/runs")
    ap.add_argument("--ur-floor", type=float, default=0.95)
    ap.add_argument("--min-seeds", type=int, default=5,
                    help="configs must hold this many data seeds. "
                         "Matches make_spam_tables; at 1 a single-seed "
                         "config prints no spread and can win on noise.")
    ap.add_argument("--aggregate", action="store_true",
                    help="pool the three target strata into one block and "
                         "drop the Target column")
    ap.add_argument("--show-missing", action="store_true",
                    help="render methods with no runs as a 'not run' row "
                         "instead of omitting them")
    ap.add_argument("--out", default=None)
    ap.add_argument("--pin-arms", default=None, metavar="CSV",
                    help="report every baseline at the configuration selected for TIGER + RQ-KMeans "
                         "on the validation split (paper/plotting/valsel_selected.csv) instead of "
                         "selecting one on this track's test split")
    ap.add_argument("--one-execution", action="store_true",
                    help="keep ONE execution per (arm, stratum, data seed): the newest complete run of "
                         "the arm's own tag, never a `_useed` replicate (a replicate is used only if a seed "
                         "has nothing else, then the newest one). Tokenizer tracks are not in "
                         "canonical_runs.tsv; this is the 'latest equivalent execution' rule of "
                         "canonical_runs_plan.md section 3.3. Dropped runs are printed.")
    a = ap.parse_args()

    data, ref, _src, _prov, _sk = mst.collect_dataset(
        a.dataset, a.model, a.quantizer, a.runs_root)
    if a.one_execution:
        n_drop = 0
        for key, runs in list(data.items()):
            by_seed = {}
            for r in runs:
                by_seed.setdefault(r["seed"], []).append(r)
            keep = []
            for sd, rs in by_seed.items():
                main = [r for r in rs if r.get("replicate") is None] or rs
                pick = max(main, key=lambda r: int(r["job"]))
                keep.append(pick)
                for r in rs:
                    if r is not pick:
                        n_drop += 1
                        print(f"  one-execution: {key[1]} {key[0]} s{sd}: drop {os.path.basename(r['dir'])} "
                              f"(keep {os.path.basename(pick['dir'])})", file=sys.stderr)
            data[key] = sorted(keep, key=lambda r: r["seed"])
        print(f"  one-execution: dropped {n_drop} duplicate executions (replicates / older jobs)", file=sys.stderr)
    if ref is None:
        print(f"no reference for {a.dataset}/{a.model}+{a.quantizer}")
        return
    seeds_present = sorted({r["seed"] for runs in data.values() for r in runs
                            if r.get("seed") is not None})
    refs_seed = _refs_by_seed(a.dataset, a.model, a.quantizer, seeds_present)
    track = run_tags.track_of(a.model, a.quantizer)
    # the caption appends \acp{SID} itself, and LETTER's track key is not a display name
    tn = {"tiger+tracerv3": "TIGER + TRACER", "letter+letter": "TIGER + LETTER"}.get(
        track, mst.TRACK_NAME.get(track, track))

    L = [r"\begin{table*}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.05}",
         r"\caption{",
         rf"Spam-removal results on {a.dataset.capitalize()} with {tn} "
         r"\acp{SID} under the bandwagon attack.",
         (r"Each baseline uses the configuration selected for TIGER + RQ-KMeans on the "
          r"validation split of the same dataset (\Cref{tab:appendix-valsel})."
          if a.pin_arms else
          rf"For baseline families, we show the configuration with the smallest "
          rf"$\Delta^{{\mathrm{{retr}}}}_{{\mathrm{{SH}}}}$ meeting mean "
          rf"$\tau_P\geq{a.ur_floor * 100:.0f}\%$ when one exists; otherwise "
          rf"the smallest-$\Delta^{{\mathrm{{retr}}}}_{{\mathrm{{SH}}}}$ "
          rf"configuration is shown without ranking emphasis."),
         r"NAU is reported at its operating point "
         r"($\lambda_f=0.1$, $\lambda_s=0.01$, $\lambda_n=0.01$, "
         r"generative separation score); the indented row below it uses a stronger "
         r"separation weight ($\lambda_s=0.1$) and is not ranked.",
         r"$\mathrm{SH}@10$, $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$, "
         r"$\mathrm{NDCG}@10$, and $\tau_P$ are reported as percentages.",
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ is the mean over "
         r"(seed, target) cells of $|\mathrm{SH}@10-\mathrm{SH}@10_{\mathrm{retr}}|$, "
         r"so it measures closeness to retraining rather than raw suppression; "
         r"it is the same quantity Table~\ref{tab:bandwagon-unlearn-all-tiger-rkmeans} "
         r"reports.",
         r"SH@10 measures target suppression; closeness to retraining also "
         r"depends on retrained exposure. Higher NDCG@10 and $\tau_P$ "
         r"indicate better utility preservation.",
         # No tau_F column: it is PGR, sound in 1 of 18 blocks on this data, and
         # Delta^retr_SH already reports the distance to retraining.
         r"$\mathcal{E}$ reports unlearning time per request in seconds.",
         r"}",
         rf"\label{{tab:bandwagon-unlearn-{a.dataset}-{a.model}-{a.quantizer}}}",
         r"\resizebox{\textwidth}{!}{%",
         (r"\begin{tabular}{lrrrrr}" if a.aggregate
          else r"\begin{tabular}{llrrrrr}"), r"\toprule",
         ((r"Method & $\mathrm{SH}@10$ (\%) & "
           r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ (\%) $\downarrow$ & "
           r"$\mathrm{NDCG}@10$ (\%) & "
           r"$\tau_P$ (\%) & $\mathcal{E}$ (s) \\")
          if a.aggregate else
          (r"Target & Method & $\mathrm{SH}@10$ (\%) & "
           r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ (\%) $\downarrow$ & "
           r"$\mathrm{NDCG}@10$ (\%) & "
           r"$\tau_P$ (\%) & $\mathcal{E}$ (s) \\")),
         r"\midrule"]

    blocks = [(None, None)] if a.aggregate else list(enumerate(STRATA))
    for si, block in enumerate(blocks if a.aggregate else STRATA):
        st, st_label = (None, None) if a.aggregate else block
        # Select on distance to retraining, matching Table 1 and the column the
        # table now reports. Selecting on SH@10 rewarded over-removal.
        def _score(arm, _runs, _st=None):
            strata = [x for x, _l in STRATA] if a.aggregate else [_st]
            d = _cell_distance(data, arm, strata, refs_seed)
            return None if d is None else d[0]

        best = ({} if a.aggregate
                else h2h.best_per_family(
                    data, st, a.ur_floor, a.min_seeds,
                    score=lambda arm, runs, _st=st: _score(arm, runs, _st)))
        # The NAU entry is keyed by arm, so aggregate that arm directly instead
        # of asking the family for its winner.
        ours_arm = OURS_ARM.get((a.model, a.quantizer))
        var_arm = VARIANT_ARM.get((a.model, a.quantizer))
        if a.aggregate:
            # One block over the union of the strata: each family is selected
            # once on the pooled cells, so a method is reported at a single
            # setting rather than three.
            for _fam, _lbl in BASELINES:
                _b = _pool(data, _family_arms(data, _fam), a.ur_floor,
                           a.min_seeds, score=_score)
                if _b is not None:
                    best[_fam] = _b
            ours_runs = []
            var_runs = []
            for _st, _l in STRATA:
                ours_runs.extend(data.get((_st, ours_arm), []) if ours_arm else [])
                var_runs.extend(data.get((_st, var_arm), []) if var_arm else [])
        else:
            # data is keyed by (stratum, arm), not nested by stratum.
            ours_runs = list(data.get((st, ours_arm), [])) if ours_arm else []
            var_runs = list(data.get((st, var_arm), [])) if var_arm else []
        if a.pin_arms:
            # Pinned: each baseline at TIGER + RQ-KMeans's validation-selected setting,
            # the same setting in every stratum; no selection on this track at all.
            for _fam, _lbl in BASELINES:
                _arm = _pinned_arm(a.pin_arms, a.dataset, a.quantizer, _fam)
                if _arm is None:
                    continue
                _runs = [r for _s in ([x for x, _l in STRATA] if a.aggregate else [st])
                         for r in data.get((_s, _arm), [])]
                best[_fam] = _agg_arm(_runs, _arm) if _runs else None
        best["__ours__"] = (h2h.aggregate_arm(ours_runs, ours_arm)
                            if ours_runs and hasattr(h2h, "aggregate_arm")
                            else _agg_arm(ours_runs, ours_arm))
        best["__variant__"] = _agg_arm(var_runs, var_arm) if var_runs else None
        rows = []
        for fam, label in BASELINES + [("__ours__", "NAU"), ("__variant__", VARIANT_LABEL)]:
            c = mst._cells(best.get(fam))
            if c is None:
                # A method with no runs on this track is OMITTED rather than
                # printed as "not run": a blank row in a paper table reads as a
                # failed method, not as missing data. `--show-missing` keeps the
                # placeholder for working copies where the gap is the point.
                if a.show_missing:
                    rows.append((label, None))
                continue
            _seeds, sh, n10, ur, pgr, _arm, _cl = c
            tmin = mst._median_minutes(best.get(fam))
            dist = _cell_distance(
                data, _arm,
                [st] if not a.aggregate else [x for x, _l in STRATA],
                refs_seed)
            rows.append((label, (_num(sh, 100.0), _num(n10, 100.0),
                                 _num(ur, 100.0), pgr, tmin,
                                 _num(pgr)[0],                       # tau_F mean
                                 None if tmin is None else tmin * 60.0,
                                 dist)))
        # Bold = best, underline = second, ranked ONLY among rows that clear the
        # utility floor. Ranked over every row, SH@10 is won outright by
        # `Forget only` and Fanchuan at exactly 0.000 -- they emit nothing
        # recommendable, so no target can appear in a top-10 that is itself
        # empty, and bolding that advertises total model destruction as the best
        # removal. A row below the floor is not a competitor on any column.
        def marks(idx, better, scalar=False):
            def val(r):
                v = r[1][idx]
                return v if scalar else (v[0] if v else None)
            vals = [(i, val(r)) for i, r in enumerate(rows)
                    if r[1] and val(r) is not None and r[0] != VARIANT_LABEL
                    and r[1][2][0] is not None
                    and r[1][2][0] >= a.ur_floor * 100.0]
            # "one": tau_F targets the RETRAINED model, so overshooting it is a
            # departure exactly like undershooting -- ranking it "high" would
            # bold a model that suppressed many times the poison gap. Same
            # convention as h2h.rank_marks.
            if better == "one":
                vals.sort(key=lambda t: abs(t[1] - 1.0))
            else:
                vals.sort(key=lambda t: t[1], reverse=(better == "high"))
            return ({vals[0][0]} if vals else set(),
                    {vals[1][0]} if len(vals) > 1 else set())
        b_sh, u_sh = marks(0, "low")
        b_d, u_d = marks(7, "low")
        b_n10, u_n10 = marks(1, "high")
        b_ur, u_ur = marks(2, "high")
        b_tf, u_tf = marks(5, "one", scalar=True)
        b_e, u_e = marks(6, "low", scalar=True)

        def deco(s, i, bold, und):
            if i in bold:
                return rf"\textbf{{{s}}}"
            return rf"\underline{{{s}}}" if i in und else s

        pre = "" if a.aggregate else " & "
        span = 5 if a.aggregate else 6
        if not a.aggregate:
            L.append(rf"\multirow{{{len(rows) + 2}}}{{*}}"
                     rf"{{\rotatebox[origin=c]{{90}}{{{st_label}}}}}")
        L.append(rf"{pre}\multicolumn{{{span}}}{{l}}{{\textit{{Baselines}}}} \\")
        for i, (label, v) in enumerate(rows):
            if label == "NAU":
                L.append(r"\cmidrule(lr){1-6}" if a.aggregate
                         else r"\cmidrule(lr){2-7}")
                L.append(rf"{pre}\multicolumn{{{span}}}{{l}}{{\textit{{Ours}}}} \\")
            if v is None:
                L.append(rf"{pre}{label} & \multicolumn{{5}}{{c}}{{not run}} \\")
                continue
            (shm, shs), (nm, ns), (um, us), pgr, tmin, _tfm, _esec, dist = v
            d_s = "--" if dist is None else _fmt(dist[0], dist[1])
            t_s = "--" if tmin is None else f"{tmin * 60.0:.1f}"
            pgr_tex = pgr.replace("\u00b1", "$\\pm$")
            L.append(
                rf"{pre}{label} & {deco(_fmt(shm, shs), i, b_sh, u_sh)} & "
                rf"{deco(d_s, i, b_d, u_d)} & "
                rf"{deco(_fmt(nm, ns), i, b_n10, u_n10)} & "
                rf"{deco(_fmt(um, us, 1), i, b_ur, u_ur)} & "
                rf"{deco(t_s, i, b_e, u_e)} \\")
        L.append(r"\bottomrule" if a.aggregate or si == len(STRATA) - 1
                 else r"\midrule")

    L += [r"\end{tabular}}", r"\end{table*}"]
    suffix = "_pooled" if a.aggregate else ""
    out = a.out or (f"tables/paper_spam_{a.dataset}_{a.model}_"
                    f"{a.quantizer}{suffix}.tex")
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    open(out, "w").write("\n".join(L) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
