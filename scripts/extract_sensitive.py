"""Sensitive-category unlearning (use case 2) across the grid, in ERASE's shape.

Sensitive deletion differs from spam in kind: the neighbours of a deleted item
ARE the concept, so leaving them recommendable is failure, not partial success.

ERASE metrics, mapped onto ours (the manifest reuses `target_items`, so the spam
evaluator scores the SENSITIVE set unchanged):

    Sensitive@k  = SH@k against I_s
    RelItemsF@k  = SHF_retrain@k - SHF_unlearn@k  (forget users only; positive = more
                   of the concept removed than retraining removed)
    tau_P@k      = NDCG_unlearn@k / NDCG_retrain@k

UR is reported alongside, on the SPAM convention (NDCG_unlearn / NDCG_BASE, i.e.
against the model before unlearning) so the sensitive and spam tables can be read
with the same yardstick. UR and tau_P differ only in their denominator: UR is
"how much utility survived the deletion", tau_P is "how much utility relative to
a model that never saw the data". Both are kept because retraining itself moves
utility here (beauty 0.03763 -> 0.03644), so UR alone understates it.

tau_P replaced the older RelEff@k column, which was the same quantity minus one
(RelEff = tau_P - 1); one of the two was redundant and tau_P is the framework's
notation. Read tau_P here with the caveat on the retrain reference below.

CAVEAT ON THE RETRAIN REFERENCE. Everything divided by theta_r inherits the
noise in theta_r, and at forget_ratio 1e-4 that noise dominates. The deletion is
13 to 25 interactions out of ~128,773 (0.01% to 0.02%), yet Sensitive@10 for
retrain minus base is +0.0057 (t=+2.30) on toys and -0.0082 (t=-5.06) on sports:
both significant, opposite in sign. A 0.01% data change cannot do that, so the
gap is a property of the training batch, not of the removal (sports trained base
and retrain in one batch on 2026-08-21, toys did not: base 08-12/08-20, retrain
08-26). It shows in utility too -- the toys retrain sits at UR 0.898 against its
own base. Sensitive@k and UR (vs BASE) are therefore the columns to trust.

WHAT CHANGED FOR THE GRID. This script used to hardcode two datasets, their
categories and four training job ids, glob only `*_unified_*`, and print to
stdout. The grid is 8 algorithms x 4 datasets x 5 seeds x 3 tracks, so:

* run dirs are parsed by `scripts/run_tags.py`, which is track-aware (the old
  glob could not see the RQ-VAE or DIGER tracks) and splits
  `sens_food_k5_meat_...` correctly (the old parser read that as dataset `food`,
  category `k5`);
* base and retrain references resolve per (dataset, track, seed) from the
  tokenized dirs `run_sensitive_reference_eval.sh` now writes, and a cell with no
  reference is reported as such instead of borrowing another cell's denominator;
* a run whose category is not the grid's category for that dataset is excluded:
  it answers a different question. The recorded toys runs use `babytoys`, which
  is NO LONGER the grid's choice and so no longer fills a toys cell: that
  category holds ~22% of all top-10 slots, and its RETRAIN reference measured
  ABOVE its base on 3 of 5 seeds, meaning the gold standard was not even a lower
  bound on exposure. toys now uses `toyweapons`, an ERASE-style keyword concept
  whose interaction footprint (0.37%) sits nearer the two cells that work
  (beauty 0.70%, sports 0.76%). The babytoys runs are superseded, not reused.

This writes `tables/sensitive_grid.md`, NOT `tables/sensitive_results.md`: that
file is the unified-only lambda_n SIGN ablation written by
`scripts/write_result_tables.py`, a different table with paired per-arm deltas.

    python scripts/extract_sensitive.py
    python scripts/extract_sensitive.py --datasets beauty,toys --all-arms
    python scripts/extract_sensitive.py --quantizer rqvae --out tables/sensitive_grid.md
"""

from __future__ import annotations

import argparse
import collections
import glob
import importlib.util
import json
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
_spec = importlib.util.spec_from_file_location(
    "h2h", os.path.join(HERE, "extract_bandwagon_head2head.py"))
h2h = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(h2h)
import run_tags  # noqa: E402

# Must stay identical to category_of() in run_unlearn_sweep_sensitive.sh.
CATEGORY = {"beauty": "hairloss", "toys": "toyweapons", "sports": "guns",
            "food_k5": "meat"}
CATEGORY_NOTE = {
    "beauty": "Hair Loss Products, taxonomy level 2, 106 items, 0.88% of catalog",
    "toys": "toy weapons, keyword-matched, 68 items, 0.57% of catalog",
    "sports": "Guns & Rifles, taxonomy level 3, 202 items, 1.10% of catalog",
    "food_k5": "Meat & Seafood, taxonomy level 1, 1904 items, 1.14% of catalog",
}
def spam_tables_sacct():
    """Reuse make_spam_tables.sacct_elapsed rather than duplicating the parser."""
    _s = importlib.util.spec_from_file_location(
        "mst", os.path.join(HERE, "make_spam_tables.py"))
    _m = importlib.util.module_from_spec(_s)
    _s.loader.exec_module(_m)
    return _m.sacct_elapsed()


# Filter is WITHHELD from these tables for now, both modes.
#
# filter_utils.build_filter_mask gated the global mask on
# `deletion_spec == "item"`, but this scenario is "item_pairs", so global fell
# through to forget_shard_items: every item in a forget user's session. On toys
# that blocked 229 items, only 11 of them sensitive, while leaving 57 of the 68
# sensitive items reachable. Sensitive@10 therefore never went to zero and the
# baseline measured neither "remove the concept" nor "honour the pairs". The
# mask is fixed (it now blocks the target set whatever the deletion_spec) and
# every filter cell is re-running; restore the families below once those land.
#
# filter_user_dependent stays out permanently: it masks only each user's own
# requested items, so it leaves the concept reachable and its strong utility
# number reflects how little it blocks rather than a good trade-off.
# The STANDALONE finetune and neg_train are withheld: they run their own step
# budgets rather than the shared one, which on this scenario drove utility to
# tau_P 0.04 (Finetune) and 0.00 (NegTrain), i.e. a destroyed model reported as
# a baseline. finetune_abl / negtrain_abl / ga_abl replace them, expressing the
# same three methods in the unified objective so nothing but the lambdas differs.
_WITHHELD = {"filter_global", "filter_user_dependent", "finetune", "negtrain"}
FAMILIES = [f for f in h2h.FAMILIES if f not in _WITHHELD]
# The single utility threshold, on tau_P = NDCG@10(theta_u)/NDCG@10(theta_r).
# It selects each family's reported configuration AND gates bold/underline, so
# the caption states one rule and one number. It used to be two: selection on
# UR >= 0.95 (base-model denominator) and ranking on tau_P >= 0.9. Since
# tau_P/UR runs 1.035 to 1.057 here, UR >= 0.95 already implied tau_P >= 0.983
# and the 0.9 rank test never bound, while the caption advertised it as if it
# did.
TAU_P_RANK_MIN = 0.95
PRETTY = {"unified": "NAU", "scif": "SCIF", "seif": "SEIF",
          "kookmin": "Kookmin", "fanchuan": "Fanchuan", "negtrain": "NegTrain",
          "tracer": "TRACER", "finetune": "Finetune",
          # The ascent/descent baselines run THROUGH the unified objective, so
          # the step budget, batching and optimizer are identical to ours and
          # only the lambdas differ. They ARE Finetune, NegTrain and gradient
          # ascent, so they carry those names.
          # The ascent/descent baselines, run THROUGH the unified objective so the
          # step budget, batching and optimizer are identical to ours and only
          # the lambdas differ. Named for what they optimize rather than by
          # method name: negtrain_abl has lambda_f=0.1 AND lambda_r=1.0, so it
          # ascends on D_f and descends on D_r simultaneously, which "NegTrain"
          # hides (it reads as ascent only, which is ga_abl). finetune_abl is
          # lambda_f=0, lambda_r=1, i.e. gradient descent on the retain set.
          # "+" not "&": an ampersand needs LaTeX escaping, and the two table builders
          # differ on whether the name passes through esc() (extract_sensitive
          # does, make_spam_tables does not), so r"GA \& GD" came out as
          # "GA \\\&" in one of them and rendered as a literal backslash. "+"
          # is safe in every path and says the same thing.
          "finetune_abl": "Finetune", "negtrain_abl": "Forget + Repair", "ga_abl": "Forget only",
          "unified_abl_grid": "NAU (lambda grid)",
          "filter_global": "Filter"}
TRACK_NAME = {"tiger+rkmeans": "TIGER + RQ-KMeans",
              "tiger+rqvae": "TIGER + RQ-VAE",
              "diger+rqvae": "DIGER (own RQ-VAE)"}
REF_ROOT = os.environ.get("GRID_SENSITIVE_REF_ROOT",
                          "logs/eval/sensitive_reference")
# e.g. logs/eval/sensitive_reference_lastckpt (see reference()); unset = old behavior
BASE_REF_ROOT = os.environ.get("GRID_SENSITIVE_BASE_REF_ROOT", "")
# Offline: a JSON of already-resolved references, {token: {metric: value}},
# written by scripts/export_paper_data.py. Resolution on the cluster walks eval
# DIRECTORIES and checks coordinates.json, which cannot be shipped; this file is
# the resolved answer, so the same selection logic is not re-implemented here.
SENSITIVE_REF_DATA = os.environ.get("GRID_SENSITIVE_REF_DATA", "")
# Same contract as the spam guard: offline without the resolved references, the
# base/retrain denominators vanish and tau_P would be computed against nothing.
if os.environ.get("GRID_RUN_DATA") and not SENSITIVE_REF_DATA:
    raise RuntimeError(
        "GRID_RUN_DATA is set but GRID_SENSITIVE_REF_DATA is not. The sensitive "
        "tables divide by per-cell base and retrain references that are NOT in "
        "runs.csv; without them every tau_P and RelItemsF column is meaningless. "
        "Point it at paper/data/sensitive_references.json.")
_SENS_REF_CACHE = {}


def _sensitive_ref_table():
    if not SENSITIVE_REF_DATA:
        return None
    if _SENS_REF_CACHE.get("path") != SENSITIVE_REF_DATA:
        with open(SENSITIVE_REF_DATA) as fh:
            _SENS_REF_CACHE.clear()
            _SENS_REF_CACHE.update(path=SENSITIVE_REF_DATA, table=json.load(fh))
    return _SENS_REF_CACHE["table"]


def ref_token(ds: str, role: str, model: str, quantizer: str, seed: int) -> str:
    tok = f"{ds}_{role}"
    if model != run_tags.DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != run_tags.DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    if seed != run_tags.DEFAULT_SEED:
        tok += f"_seed{seed}"
    return tok


def reference(ds: str, role: str, model: str, quantizer: str, seed: int,
              category: str):
    """(metrics, provenance) for the base or retrain reference of one cell.

    A dir carrying coordinates.json is trusted only when every coordinate
    matches. The pre-coordinates dirs (`<ds>_base`) are accepted only for the
    default track at seed 2, which is the only cell that existed when they were
    written -- scoring a DIGER run against a TIGER denominator would give a
    plausible number with no meaning.
    """
    tok = ref_token(ds, role, model, quantizer, seed)
    table = _sensitive_ref_table()
    if table is not None:
        got = table.get(tok)
        return (got["metrics"], got.get("source", tok)) if got else (None, "")
    cands = sorted(glob.glob(os.path.join(REF_ROOT, "*", tok)),
                   key=os.path.getmtime, reverse=True)
    # Opt-in (2026-11-07): the "before deletion" (base) reference evaluated at the
    # exact checkpoint the unlearning runs start from (last.ckpt), written by
    # run_sensitive_reference_eval.sh BASE_CKPT=last ROLES=base. Searched FIRST;
    # cells it lacks (Beauty seed 2, whose runs start from an epoch checkpoint that
    # the old reference already used) fall back to REF_ROOT.
    if role == "base" and BASE_REF_ROOT:
        cands = sorted(glob.glob(os.path.join(BASE_REF_ROOT, "*", tok)),
                       key=os.path.getmtime, reverse=True) + cands
    is_default = ((model, quantizer) == (run_tags.DEFAULT_MODEL,
                                         run_tags.DEFAULT_QUANTIZER)
                  and seed == run_tags.DEFAULT_SEED)
    for d in cands:
        coord = os.path.join(d, "coordinates.json")
        if os.path.isfile(coord):
            c = json.load(open(coord))
            if (c.get("dataset") != ds or c.get("role") != role
                    or c.get("model", "tiger") != model
                    or c.get("quantizer", "rkmeans") != quantizer
                    or int(c.get("seed", 2)) != seed
                    or c.get("category", category) != category):
                continue
        elif is_default:
            # Pre-coordinates dir. It carries no category field, but the Hydra
            # config records the manifest it actually scored -- check that,
            # instead of trusting the name. toys/2026-08-18 is exactly why:
            # it has no coordinates.json and was being accepted for the
            # `toyweapons` grid while its config shows
            # forget_manifest_babytoys_1e-4_seed2.json. That scored a
            # toyweapons-unlearned model (68 items) against a babytoys
            # denominator (399 items, ~22% of top-10 slots) and produced a
            # plausible, meaningless row.
            hy = os.path.join(d, ".hydra", "config.yaml")
            man = ""
            if os.path.isfile(hy):
                try:
                    txt = open(hy, errors="ignore").read()
                    mm = re.search(r"spam_forget_manifest:\s*(\S+)", txt)
                    man = mm.group(1).strip("'\"") if mm else ""
                except OSError:
                    man = ""
            if man and category not in os.path.basename(man):
                continue
        else:
            continue
        m = run_tags.last_metrics(d)
        if m:
            return m, d
    return None, ""


def mean_reference_metrics(refs, role: int):
    """Mean reference metrics over the same seed set used by the comparison.

    A single seed's endpoint alongside five-seed method means silently changes
    the reference population. ``role`` is 0 for base and 1 for retrain.
    """
    records = [pair[role] for pair in refs.values() if pair[role]]
    if not records:
        return None
    keys = set().union(*(record.keys() for record in records))
    return {key: statistics.mean(values) for key in keys
            if (values := [r[key] for r in records
                           if isinstance(r.get(key), (int, float))])}


def reference_hint(ds: str, model: str, quantizer: str, seed: int) -> str:
    return (f"MODEL={model} QUANTIZER={quantizer} DATASETS={ds} SEEDS={seed} "
            f"MODELS='{ds}:{seed}:<base_jid>:<retrain_jid>' "
            "sbatch run_sensitive_reference_eval.sh")


def cell_entry(m, base, retr, seed, replicate, jid, path):
    """One run's metric dict as collect() stores it (m = run_tags.last_metrics of the run;
    base/retr = that seed's reference metrics). None when m has no test NDCG."""
    if not m or "test/ndcg@10" not in m:
        return None
    n10, sh = m["test/ndcg@10"], m.get("test/SH@10")
    # SHF/SHR split the global SH by who asked. Deletion here is per user, so
    # SHF (the requesting users) is the efficacy axis and SHR (everyone else)
    # shows their predictions survived. Global SH averages ~19k retain users
    # against ~12 forget users, so it is >95% collateral and cannot show
    # whether the request was honoured: filter_global reaches SHF@10 0.0
    # while its global SH@10 still reads 0.024.
    shf, shr = m.get("test/SHF@10"), m.get("test/SHR@10")
    b_n = base.get("test/ndcg@10")
    b_sh = base.get("test/SH@10")
    b_shf = base.get("test/SHF@10")
    r_n = (retr or {}).get("test/ndcg@10")
    r_sh = (retr or {}).get("test/SH@10")
    r_shf = (retr or {}).get("test/SHF@10")
    r_shr = (retr or {}).get("test/SHR@10")
    # tau_F under F = -Sensitive@k, per the framework:
    #   tau_F = (F(u) - F(o)) / (F(r) - F(o)) = (Sens_o - Sens_u)/(Sens_o - Sens_r)
    # Computed per seed against that seed's own references, then averaged,
    # so a seed is never scored against another seed's reference.
    _den = (b_sh - r_sh) if (b_sh is not None and r_sh is not None) else None
    _denF = (b_shf - r_shf) if (b_shf is not None and r_shf is not None) else None
    return {
        "seed": seed,
        "replicate": replicate,
        "job": jid,
        "dir": path,
        "n10": n10,
        "sens": sh,
        "sensF": shf,
        "sensR": shr,
        "tpm": m.get("test/TPM@10"),
        "UR": n10 / b_n if b_n else None,
        "tau_P": (n10 / r_n) if r_n else None,
        # Forget-scoped, matching SHF@k: the deletion is a per-user
        # request, so the quantity of interest is what the FORGET users
        # still see, not the whole catalogue. Global Sensitive@k is
        # dominated by users who never asked for anything (measured at
        # 95.8% collateral), so a global RelItems mostly reports drift on
        # people unaffected by the request. Built from SHF on both sides.
        "RelItems": (r_shf - shf) if (r_shf is not None and shf is not None) else None,
        # Distance to retraining per cell (this seed's own retrained model),
        # the unwanted-item analogue of Delta^retr_SH in the spam tables.
        # Reported as Delta_UHF / Delta_UHR in the paper (UHF = SHF, UHR = SHR).
        "dUHF": abs(shf - r_shf) if (r_shf is not None and shf is not None) else None,
        "dUHR": abs(shr - r_shr) if (r_shr is not None and shr is not None) else None,
        "tau_F": ((b_sh - sh) / _den
                  if (_den and sh is not None and b_sh is not None) else None),
        # The denominator itself, kept so a reader can see whether tau_F
        # rests on a real gap. It is the amount retraining actually removed.
        "tau_F_den": _den,
        # tau_F on the axis that matters: F = -SHF@k, forget users only.
        "tau_FF": ((b_shf - shf) / _denF
                   if (_denF and shf is not None and b_shf is not None) else None),
        "tau_FF_den": _denF,
    }


def collect(ds: str, model: str, quantizer: str, category: str,
            runs_root: str = "logs/unlearn/runs", include_valsens: bool = False):
    """(arm -> per-run metric dicts, references per seed, seeds lacking one).

    `_valsens` runs (validation-selection replays, run_valsel_batch_sens.sh) are
    replicates of the recorded arms and are skipped unless include_valsens is set;
    otherwise they would enter every table as extra candidate arms.
    """
    cells: dict[str, list[dict]] = collections.defaultdict(list)
    refs: dict[int, tuple] = {}
    skipped: set[int] = set()
    for rec in run_tags.iter_runs(runs_root, setting="sensitive", dataset=ds,
                                  model=model, quantizer=quantizer):
        if rec["category"] != category:
            continue
        if "_valsens" in rec["tag"] and not include_valsens:
            continue
        seed = rec["seed"]
        if seed not in refs:
            base, base_p = reference(ds, "base", model, quantizer, seed, category)
            retr, retr_p = reference(ds, "retrain", model, quantizer, seed, category)
            refs[seed] = (base, retr, base_p, retr_p)
        base, retr, _bp, _rp = refs[seed]
        if base is None:
            skipped.add(seed)
            continue
        e = cell_entry(run_tags.last_metrics(rec["path"]), base, retr, seed, rec["replicate"],
                       rec["jid"], rec["path"])
        if e is None:
            continue
        cells[rec["arm"]].append(e)
    for arm, runs in cells.items():
        by_cell = {}
        for r in sorted(runs, key=lambda x: int(x["job"])):
            by_cell[(r["seed"], r["replicate"])] = r
        cells[arm] = [by_cell[k] for k in sorted(by_cell, key=lambda k: (k[0], k[1] or 0))]
    return cells, refs, sorted(skipped)


def best_per_family(cells, ur_floor: float, min_seeds: int):
    """family -> (arm, aggregates). Lowest SHF@10 subject to tau_P >= floor.

    Selection follows the axis the table reports. Deletion here is per user, so
    efficacy is SHF@10 (the requesting users), not the global SH@10 that averages
    ~12 forget users against ~19k retain users. With only 12 to 18 forget users
    SHF is coarse (steps of 1/n_forget), so ties are common; the global SH@10
    breaks them, and UR breaks any remaining tie. Rows whose runs predate the
    SHF metric fall back to the global SH@10 so they still rank.
    """
    out = {}
    for fam in FAMILIES:
        # Eligibility is tested on the metric the table REPORTS (SHF@10), not
        # on the global SH@10. Testing the global one let a row pass with 5
        # global seeds but only 4 SHF seeds, so its SHF mean and its global mean
        # covered different seed sets and one method's 5-seed SHF was compared
        # against another's 4-seed SHF. SHF@10 landed 2026-08-31, so runs older
        # than that lack it and are correctly not counted here.
        cands = [(arm, runs) for arm, runs in cells.items()
                 if run_tags.family_of(arm) == fam
                 and h2h.agg(runs, "sensF")["n_seeds"] >= min_seeds]
        if not cands:
            out[fam] = None
            continue
        # Floor is tested on tau_P, the column the table PRINTS and the caption
        # names, not on UR. UR divides by the base model and tau_P by the
        # retrained one; on this scenario tau_P/UR runs 1.035 to 1.057, so the
        # two disagree near the floor and the table was gating on a number no
        # column showed.
        scored = [(arm, runs, h2h.agg(runs, "sens"), h2h.agg(runs, "tau_P"),
                   h2h.agg(runs, "sensF")) for arm, runs in cands]
        ok = [x for x in scored if x[3]["mean"] is not None
              and x[3]["mean"] >= ur_floor]
        pool = ok if ok else scored
        def _key(x):
            f = x[4]["mean"] if x[4]["n_seeds"] else None
            # None sorts last: an arm with no SHF cannot outrank one that has it.
            # tau_P is None when a seed has no retrain reference; such an arm
            # must not crash the sort and must not win the tiebreak either.
            tp = x[3]["mean"] if x[3]["mean"] is not None else 0.0
            return (f is None, f if f is not None else 0.0,
                    x[2]["mean"], -tp)
        arm, runs, *_ = min(pool, key=_key)
        agg_d = {f: h2h.agg(runs, f) for f in
                 ("sens", "sensF", "sensR", "n10", "UR", "tau_P",
                  "tau_F", "tau_F_den", "tau_FF", "tau_FF_den",
                  "RelItems", "tpm", "dUHF", "dUHR")}
        # the runs behind the reported row, so wall-clock can be joined by job id
        agg_d["runs"] = runs
        out[fam] = (arm, agg_d, bool(ok))
    return out


def _pm(s: str) -> str:
    return s.replace("+-", "±")


def render(ds: str, track: str, category: str, cells, refs, skipped,
           ur_floor: float, min_seeds: int, all_arms: bool) -> list[str]:
    ea_by_seed = retrain_minutes_by_seed(refs)
    L = [f"### {ds} / {CATEGORY_NOTE.get(ds, category)}, "
         f"{TRACK_NAME.get(track, track)}", ""]
    if not refs:
        L += [f"_No runs on this track. References would come from:_",
              f"`{reference_hint(ds, *track.split('+'), run_tags.DEFAULT_SEED)}`", ""]
        return L
    for seed in sorted(refs):
        base, retr, bp, rp = refs[seed]
        if base:
            L.append(f"- seed {seed} BASE (trained with I_s): NDCG@10 "
                     f"{base.get('test/ndcg@10', float('nan')):.5f}, Sensitive@10 "
                     f"{base.get('test/SH@10', float('nan')):.5f}, from `{bp}`")
        if retr:
            L.append(f"- seed {seed} RETRAIN reference: NDCG@10 "
                     f"{retr.get('test/ndcg@10', float('nan')):.5f}, Sensitive@10 "
                     f"{retr.get('test/SH@10', float('nan')):.5f}, from `{rp}`")
        elif base:
            L.append(f"- seed {seed} RETRAIN reference MISSING, so tau_P and "
                     f"RelItems@10 are n/a. Produce it with: "
                     f"`{reference_hint(ds, *track.split('+'), seed)}`")
    if skipped:
        L.append(f"- seeds {skipped} have runs but no BASE reference, so they are "
                 "excluded rather than scored against another seed's denominator.")
    L.append("")
    if not cells:
        L += ["_(no unlearned runs yet on this track)_", ""]
        return L
    L += ["| algorithm | arm | seeds | SHF@10 | SHR@10 | Sensitive@10 | NDCG@10 | UR (vs base) | "
          "tau_P (vs retrain) | tau_F | RelItemsF@10 | E (s) | tau_E |",
          "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    if all_arms:
        rows = [(run_tags.family_of(arm), arm,
                 {f: h2h.agg(runs, f) for f in
                  ("sens", "sensF", "sensR", "n10", "UR", "tau_P",
                   "tau_F", "tau_F_den", "tau_FF", "tau_FF_den",
                   "RelItems", "tpm", "dUHF", "dUHR")}, True)
                for arm, runs in cells.items()]
        rows.sort(key=lambda r: (FAMILIES.index(r[0]) if r[0] in FAMILIES else 99,
                                 r[2]["sens"]["mean"]))
    else:
        best = best_per_family(cells, ur_floor, min_seeds)
        rows = [(fam, v[0], v[1], v[2]) for fam, v in best.items() if v]
        rows.sort(key=lambda r: FAMILIES.index(r[0]) if r[0] in FAMILIES else 99)
    tau_prec = _tau_prec(rows, ea_by_seed)
    for fam, arm, a, cleared in rows:
        flag = "" if cleared else " ⚠"
        rel = a["tau_P"]
        ritems = a["RelItems"]
        rel_s = "n/a" if rel["n_seeds"] == 0 else _pm(h2h.fmt_pm(rel, 4))
        rit_s = "n/a" if ritems["n_seeds"] == 0 else _pm(h2h.fmt_pm(ritems))
        # No |Sensitive@k - Sensitive@k_retrain| column: RelItems@10 is that
        # residual already, and experimental_setup.tex keeps it SIGNED on
        # purpose -- positive means the unlearned model surfaces sensitive items
        # LESS often than retraining, negative more often, and RelItems = 0 is
        # exactly tau_F = 1. Taking the magnitude throws the direction away.
        # tau_P is NDCG@10(theta_u)/NDCG@10(theta_r), i.e.
        # utility against the RETRAINED model, which is what the framework
        # defines; the older UR column divides by the BASE model instead and is
        # kept beside it because the two answer different questions here (unlike
        # spam, base != retrain in this setting).
        tp_s = ("n/a" if rel["n_seeds"] == 0 or rel["mean"] is None
                else f"{rel['mean']:.3f}")
        tfa = a["tau_F"]
        tf_s = "n/a" if tfa["n_seeds"] == 0 else _pm(h2h.fmt_pm(tfa, 3))
        L.append(f"| {PRETTY.get(fam, fam)}{h2h.lam_suffix(arm, family=fam)} | `{arm}` | "
                 f"{h2h.seed_label(a['sens'])} | "
                 f"{'n/a' if a['sensF']['n_seeds']==0 else _pm(h2h.fmt_pm(a['sensF']))} | "
                 f"{'n/a' if a['sensR']['n_seeds']==0 else _pm(h2h.fmt_pm(a['sensR']))} | "
                 f"{_pm(h2h.fmt_pm(a['sens']))} | "
                 f"{_pm(h2h.fmt_pm(a['n10']))} | {_pm(h2h.fmt_pm(a['UR'], 3))}{flag} "
                 f"| {tp_s} | {tf_s} | {rit_s} | {_tmin(a)} | {h2h.fmt_tau_E(*_eu_ea(a, ea_by_seed), tau_prec)} |")
    L += ["", f"_⚠ = no configuration of that family held tau_P ≥ {ur_floor}, "
          "so its row is never bolded or underlined. "
          "`seeds` is the number of data seeds. `±` is the sample SD across them "
          "when there are 2 or more; with 1 seed it is "
          "x = half-range over unlearning replicates._",
          "",
          "_RelItemsF@10 = SHF@10(theta_r) - SHF@10(theta_u), on the FORGET users' "
          "test entries only (the same scope as SHF@10), is the "
          "forget axis on an absolute scale: 0 means the unlearned and retrained "
          "models surface sensitive items equally often (exactly tau_F = 1), "
          "positive means the unlearned model surfaces them LESS often than "
          "retraining, negative more often. It is deliberately not normalised: "
          "the gap Sensitive@10(theta_o) - Sensitive@10(theta_r) changes SIGN "
          "across seeds (+0.013 to -0.002 on beauty), so a ratio against it is "
          "uninterpretable. tau_P divides by the RETRAINED model per the "
          "framework; UR divides by the BASE model and is kept beside it because "
          "base != retrain in this setting._", ""]
    return L


def _tmin(a, times=None):
    """Median SECONDS PER UNLEARNING REQUEST over this row's runs, else "--".

    Seconds, matching make_spam_tables.py: the same quantity reported in two
    units across the paper's two table families invites a misread, and at these
    magnitudes (0.3 s to 35 s) minutes would print as a row of near-zeros.

    From each run's own instrumentation, not from sacct: a SLURM elapsed time
    also covers model load, the per-request data split, and the post-unlearn
    evaluation, so it overstates the per-request latency E is defined as by
    about an order of magnitude. `times` is accepted and ignored so existing
    callers keep working. Filter has no such record, being a decode-time mask
    rather than a training procedure, so it reports "--".
    """
    mins = [m for r in a.get("runs", [])
            if (m := h2h.request_latency_minutes(r["dir"])) is not None]
    if not mins:
        return "--"
    return f"{statistics.median(mins) * 60.0:.1f}"


def retrain_minutes_by_seed(refs) -> dict:
    """seed -> E(A) minutes for the SENSITIVE retrain, from its own eval refs.

    This scenario's theta_r is a separate fit that excluded the sensitive items,
    NOT the clean model the spam tables use (on sports seed 5, job 10465011 vs
    10465007), so E(A) has to come from the retrain reference's own provenance.
    """
    out = {}
    for seed, tup in (refs or {}).items():
        retr_path = tup[3] if len(tup) > 3 else None
        j = h2h.train_job_from_eval_dir(retr_path)
        t = h2h.train_minutes(j) if j else None
        if t is not None:
            out[seed] = t
    return out


def _eu_ea(a, ea_by_seed: dict):
    """(E(U), E(A)) minutes for one row, E(A) averaged over the seeds it pools."""
    mins = [m for r in a.get("runs", [])
            if (m := h2h.request_latency_minutes(r["dir"])) is not None]
    if not mins:
        return None, None
    ea = [ea_by_seed[sd] for sd in sorted({r["seed"] for r in a.get("runs", [])})
          if sd in (ea_by_seed or {})]
    return statistics.median(mins), (sum(ea) / len(ea) if ea else None)


def _tau_prec(rows, ea_by_seed: dict) -> int:
    """One tau_E precision for the whole table, set by the fastest method."""
    return h2h.tau_E_precision(
        [eu / ea for _f, _arm, a, _c in rows
         for eu, ea in [_eu_ea(a, ea_by_seed)] if eu and ea])


def latex_grid(ds: str, track: str, cells, base, retr, ur_floor: float,
               min_seeds: int, cat_desc: str, times: dict | None = None,
               ea_by_seed: dict | None = None) -> str:
    """One booktabs table per dataset, best arm per algorithm.

    Mirrors `latex_head2head` in make_spam_tables.py: same selection rule, same
    escaping, and the same explicit row when nothing clears the UR floor, so a
    stratum with no qualifying configuration reads as a result rather than as a
    rendering bug.
    """
    # `%` MUST be escaped: it starts a LaTeX comment, and the category notes
    # carry percentages ("0.88%"), which would silently swallow the rest of the
    # caption line.
    def esc(t):
        out = str(t)
        for ch in ("\\", "&", "%", "#"):
            out = out.replace(ch, "\\" + ch)
        return out.replace("_", r"\_")
    strip = lambda t: (t.replace("(r", " (r").replace("±", r"$\pm$")
                        .replace("+-", r"$\pm$"))
    tn = TRACK_NAME.get(track, track)
    b_n10 = (base or {}).get("test/ndcg@10", float("nan"))
    b_sen = (base or {}).get("test/SH@10", float("nan"))
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         # Concise by design: the caption defines the selection rule, the two
         # denominators and the one caveat a reader must carry to read tau_P
         # correctly. The full variance argument lives in this module's
         # docstring and in the experimental setup.
         rf"\caption{{Sensitive-category deletion on {esc(ds)} ({esc(cat_desc)}), "
         rf"{esc(tn)}. Deletion is requested by specific users, so efficacy is "
         rf"SHF@10, the sensitive-item hit rate for those users (lower is "
         rf"better), and SHR@10 is the same rate for everyone else, which should "
         rf"stay near the base model. Per algorithm, the lowest-SHF@10 "
         rf"configuration holding $\tau_{{\mathcal{{P}}}} \geq {ur_floor}$; "
         rf"$^{{\dagger}}$ marks families that miss it (base model NDCG@10 "
         rf"${b_n10:.5f}$). "
         rf"$\tau_{{\mathcal{{P}}}}$ is utility against the retrained model and "
         rf"$\tau_{{\mathcal{{F}}}}$ its forget counterpart under "
         rf"$\mathcal{{F}}=-$SHF@10. With only a handful of requesting users "
         rf"SHF@10 moves in steps of $1/n_{{\text{{forget}}}}$, so its spread is "
         rf"wide. $\mathcal{{E}}$ is seconds per unlearning request and "
         rf"$\tau_{{\mathcal{{E}}}} = 1 - \mathcal{{E}}(U)/\mathcal{{E}}(A)$ the "
         rf"share of retraining cost saved. \textbf{{Bold}} and "
         rf"\underline{{underline}} rank only rows holding "
         rf"$\tau_{{\mathcal{{P}}}} \geq {ur_floor}$, so removal "
         rf"is credited only where the surviving model is still usable.}}",
         rf"\label{{tab:sensitive-{ds}-{track.replace('+', '-')}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
         r"algorithm & SHF@10 & SHR@10 & NDCG@10 & "
         r"$\tau_{\mathcal{P}}$ & $\tau_{\mathcal{F}}$ & RelItemsF@10 & "
         r"$\mathcal{E}$ (s) & $\tau_{\mathcal{E}}$ \\",
         r"\midrule"]
    best = best_per_family(cells, ur_floor, min_seeds)
    rows = [(fam, v[0], v[1], v[2]) for fam, v in best.items() if v]
    rows.sort(key=lambda r: FAMILIES.index(r[0]) if r[0] in FAMILIES else 99)
    if not rows:
        L.append(rf"\multicolumn{{9}}{{l}}{{\emph{{no configuration of any "
                 rf"algorithm held $\tau_{{\mathcal{{P}}}} \geq$ "
                 rf"{ur_floor}}}}} \\")
    # Rank only among floor-clearing rows: an arm that destroyed the model
    # reaches Sensitive@10 0.00000 trivially and would take every bold mark.
    tau_prec = _tau_prec(rows, ea_by_seed or {})
    # Bold/underline uses the SAME tau_P floor that selected the row, so a
    # dagger row (no config held the floor) can never be marked. Written out
    # rather than inferred from the dagger flag so the caption and the code say
    # one thing.
    elig = [(a["tau_P"]["n_seeds"] and a["tau_P"]["mean"] is not None
             and a["tau_P"]["mean"] >= ur_floor)
            for _f, _arm, a, _c in rows]
    gm = lambda k: [(a[k]["mean"] if a[k]["n_seeds"] else None) for _f, _a2, a, _c in rows]
    marks = {
        "sens": h2h.rank_marks(gm("sensF"), "low", elig),
        "sensR": h2h.rank_marks(gm("sensR"), "high", elig),
        "n10":  h2h.rank_marks(gm("n10"),  "high", elig),
        # UR is NOT a column here. It is tau_P with the base model as
        # denominator instead of the retrain, and within a dataset the two are
        # the same number times a constant (tau_P/UR spans 1.035 to 1.057 on
        # toys, 0.997 to 1.012 on sports, 1.019 to 1.047 on beauty), so they
        # rank identically and printing both says one thing twice. UR still
        # decides nothing now: selection and ranking both gate on tau_P.
        #
        # tau_P is ranked HIGH, matching make_spam_tables.py: it is utility
        # retention, so more of it is better. The older RelEff column ranked by
        # closeness to the retrained model, which this scenario cannot support:
        # the retrain reference is dominated by training variance (retrain minus
        # base on Sensitive@10 is +0.0057 on toys and -0.0082 on sports, both
        # significant and opposite, from deleting 0.01% of the interactions).
        "rel":  h2h.rank_marks(gm("tau_P"), "high", elig),
        # closest to 1: retraining is the target, so overshooting it is a
        # departure exactly like undershooting.
        "tauf": h2h.rank_marks(gm("tau_FF"), "one", elig),
    }
    for i, (fam, _arm, a, cleared) in enumerate(rows):
        rel = a["tau_P"]
        rel_s = "n/a" if rel["n_seeds"] == 0 else strip(h2h.fmt_pm(rel, 3))
        tf = a["tau_FF"]
        tf_s = "n/a" if tf["n_seeds"] == 0 else strip(h2h.fmt_pm(tf, 3))
        _ri = a.get("RelItems") or {}
        _ri_s = ("n/a" if not _ri or _ri.get("n_seeds", 0) == 0
                 else strip(h2h.fmt_pm(_ri, 5)))
        sf, sr = a["sensF"], a["sensR"]
        sf_s = "n/a" if sf["n_seeds"] == 0 else strip(h2h.fmt_pm(sf))
        sr_s = "n/a" if sr["n_seeds"] == 0 else strip(h2h.fmt_pm(sr))
        # esc() escapes backslashes, so it MUST run before the \textbf is
        # inserted, or the command itself gets escaped into literal text.
        name = (esc(PRETTY.get(fam, fam)).replace("NAU", r"\textbf{NAU}")
                # the arm is the loop's `_arm`; `a` is the metrics dict and
                # has no "arm" key, so reading it there returned None and the
                # suffix silently vanished from every LaTeX row.
                + h2h.lam_suffix(_arm, latex=True, family=fam))
        # The dagger marks a row that could not meet the floor, so its numbers
        # are the family's lowest-Sensitive@10 config at whatever utility cost.
        mark = "" if cleared else r"$^{\dagger}$"
        mk = lambda col, v: h2h.tex_mark(v, marks[col].get(i))
        L.append(rf"{name}{mark} & "
                 rf"{sf_s if sf_s == 'n/a' else mk('sens', sf_s)} & "
                 rf"{sr_s if sr_s == 'n/a' else mk('sensR', sr_s)} & "
                 rf"{mk('n10', strip(h2h.fmt_pm(a['n10'])))} & "
                 rf"{mk('rel', rel_s) if rel_s != 'n/a' else rel_s} & "
                 rf"{tf_s if tf_s == 'n/a' else mk('tauf', tf_s)} & "
                 rf"{_ri_s} & "
                 rf"{_tmin(a)} & "
                 rf"{h2h.fmt_tau_E(*_eu_ea(a, ea_by_seed or {}), tau_prec)} \\")
    # The RETRAINED model: train from scratch without the sensitive items. It is
    # the gold standard this scenario is measured against, so it belongs in the
    # table rather than only in the caption. Deliberately EXCLUDED from the
    # bold/underline ranking: it is the reference the other rows are compared to,
    # not a competitor, and tau_P is 1 for it by construction.
    if retr and base:
        r_n10 = retr.get("test/ndcg@10")
        r_sen = retr.get("test/SHF@10")
        r_senr = retr.get("test/SHR@10")
        b_n10r = base.get("test/ndcg@10")
        if r_n10 is not None and r_sen is not None and b_n10r:
            # E(A) for this row is MEASURED, not a placeholder: the same
            # per-seed retrain wall-clock that already feeds tau_E for every
            # other row (retrain_minutes_by_seed -> h2h.train_minutes). Printing
            # "full retrain" here while dividing by that very number one column
            # over hid the quantity the comparison rests on. Mean over the seeds
            # this cell actually has, in seconds to match the column.
            _ea = [v for v in (ea_by_seed or {}).values() if v]
            _ea_s = (rf"{statistics.mean(_ea) * 60:,.0f}" if _ea
                     else r"\textit{full retrain}")
            L += [r"\midrule",
                  rf"\textit{{Retrain (reference)}} & {r_sen:.5f} & "
                  rf"{(f'{r_senr:.5f}' if r_senr is not None else '--')} & {r_n10:.5f} & "
                  rf"1.000 & 1.000 & \textit{{0}} & \textit{{{_ea_s}}} & "
                  # tau_E = 1 - E(U)/E(A); for the reference E(U) IS E(A), so
                  # this is exactly 0. It is not a missing value.
                  rf"\textit{{0}} \\"]
    L += [r"\bottomrule", r"\end{tabular}}",
          rf"\par\smallskip\footnotesize $^{{\dagger}}$ no configuration of that "
          rf"family held $\tau_{{\mathcal{{P}}}} \geq {ur_floor}$; the row is "
          rf"its lowest SHF@10 configuration regardless of utility cost, and is "
          rf"never bolded or underlined.",
          r"\end{table}"]
    return "\n".join(L)


# Families that ARE our method. Everything else in FAMILIES is a baseline, so a
# family added later lands in the Baselines block by default rather than being
# silently credited as ours.
_OURS_FAMILIES = {"unified", "unified_sepgenerative", "unified_sepcosine",
                  "unified_abl_grid"}
# Row order inside the Baselines block, matching make_paper_spam_table.py so the
# two main-paper tables read down the same list.
_PAPER_BASELINE_ORDER = ["finetune_abl", "ga_abl", "negtrain_abl", "scif",
                         "seif", "kookmin", "fanchuan", "tracer"]


def _pct_pm(a, places: int = 3, sign: bool = False) -> str:
    """mean +- sd, both scaled to percent. Matches the spam paper tables."""
    if not a or a.get("n_seeds", 0) == 0 or a.get("mean") is None:
        return "n/a"
    fmt = f"{{:+.{places}f}}" if sign else f"{{:.{places}f}}"
    body = fmt.format(a["mean"] * 100.0)
    sd = a.get("sd") if a.get("sd") is not None else a.get("rep_spread")
    if sd is None:
        return body
    return body + r"$\pm$" + f"{sd * 100.0:.{places}f}"


def _plain_pm(a, places: int = 3, sign: bool = True) -> str:
    """mean +- sd on the native scale (tau_F is a ratio, not a percentage)."""
    if not a or a.get("n_seeds", 0) == 0 or a.get("mean") is None:
        return "n/a"
    fmt = f"{{:+.{places}f}}" if sign else f"{{:.{places}f}}"
    body = fmt.format(a["mean"])
    sd = a.get("sd") if a.get("sd") is not None else a.get("rep_spread")
    if sd is None:
        return body
    return body + r"$\pm$" + f"{sd:.{places}f}"


def latex_paper(ds: str, track: str, cells, base, retr, ur_floor: float,
                min_seeds: int, cat_desc: str, times=None,
                ea_by_seed=None) -> str:
    """The sensitive table in the main paper's layout.

    Same selection, same ranking and the same numbers as `latex_grid`; only the
    presentation differs, so the main paper can carry one design across the spam
    and sensitive results. The differences are deliberate: percentages instead of
    raw rates, a Baselines/Ours split instead of one flat list, the lambdas moved
    from the row name into the caption, and tau_E dropped -- the spam tables do
    not carry it and E (s) beside the measured retrain cost already says it.
    """
    def esc(t):
        out = str(t)
        for ch in ("\\", "&", "%", "#"):
            out = out.replace(ch, "\\" + ch)
        return out.replace("_", r"\_")

    best = best_per_family(cells, ur_floor, min_seeds)
    rows = [(fam, v[0], v[1], v[2]) for fam, v in best.items() if v]
    order = lambda f: (0 if f in _OURS_FAMILIES else 1,
                       _PAPER_BASELINE_ORDER.index(f)
                       if f in _PAPER_BASELINE_ORDER else 99)
    rows.sort(key=lambda r: order(r[0]))   # sort() passes the ROW, not the family
    # Ranking is computed over EVERY row at once, then rendered in two blocks:
    # splitting the table must not turn into ranking each block separately, which
    # would bold our row against itself.
    elig = [(a["tau_P"]["n_seeds"] and a["tau_P"]["mean"] is not None
             and a["tau_P"]["mean"] >= ur_floor)
            for _f, _arm, a, _c in rows]
    gm = lambda k: [(a[k]["mean"] if a[k]["n_seeds"] else None)
                    for _f, _a2, a, _c in rows]
    marks = {
        "sensF": h2h.rank_marks(gm("sensF"), "low", elig),
        "sensR": h2h.rank_marks(gm("sensR"), "high", elig),
        "n10": h2h.rank_marks(gm("n10"), "high", elig),
        "tau_P": h2h.rank_marks(gm("tau_P"), "high", elig),
        "tau_FF": h2h.rank_marks(gm("tau_FF"), "one", elig),
        "RelItems": h2h.rank_marks(gm("RelItems"), "high", elig),
    }

    ours = [(i, r) for i, r in enumerate(rows) if r[0] in _OURS_FAMILIES]
    lam = (h2h.lam_suffix(ours[0][1][1], latex=True, family=ours[0][1][0])
           if ours else "")
    lam = lam.strip().lstrip("(").rstrip(")")
    tn = TRACK_NAME.get(track, track)
    ncol = 8
    L = [r"\begin{table}[t]", r"\centering", r"\small",
         r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.05}",
         "\\caption{",
         rf"Sensitive-category deletion on {esc(ds.capitalize())} "
         rf"({esc(cat_desc)}) with {esc(tn)} \acp{{SID}}.",
         rf"Deletion is requested per user, so efficacy is $\mathrm{{SHF}}@10$, "
         rf"the sensitive-item hit rate over the requesting users, and "
         rf"$\mathrm{{SHR}}@10$ is the same rate for everyone else, which should "
         rf"not fall.",
         rf"For each method, we report the lowest-$\mathrm{{SHF}}@10$ "
         rf"configuration among runs satisfying $\tau_P \geq "
         rf"{ur_floor * 100:.0f}\%$; $^{{\dagger}}$ marks methods where no "
         rf"configuration satisfies it, whose row is therefore never ranked.",
         (rf"NAU is reported at its operating point ({lam})." if lam else ""),
         r"$\mathrm{SHF}@10$, $\mathrm{SHR}@10$, $\mathrm{NDCG}@10$ and $\tau_P$ "
         r"are reported as percentages and $\mathrm{RelItemsF}@10$ in points of "
         r"$\mathrm{SHF}@10$ against the retrained model, so positive means more "
         r"of the concept was removed than retraining removed.",
         r"$\tau_F$ measures forgetting relative to the retrained reference, and "
         r"$\mathcal{E}$ reports unlearning time per request.",
         "}",
         rf"\label{{tab:paper-sensitive-{ds}-{track.replace('+', '-')}}}",
         r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{lrrrrrrr}", r"\toprule",
         r"Method & $\mathrm{SHF}@10$ (\%) & $\mathrm{SHR}@10$ (\%) & "
         r"$\mathrm{NDCG}@10$ (\%) & $\tau_P$ (\%) & $\tau_F$ & "
         r"$\mathrm{RelItemsF}@10$ & $\mathcal{E}$ (s) \\",
         r"\midrule"]

    def emit(idx, fam, arm, a, cleared):
        mk = lambda col, v: h2h.tex_mark(v, marks[col].get(idx))
        name = esc(PRETTY.get(fam, fam)) + ("" if cleared else r"$^{\dagger}$")
        L.append(
            rf"{name} & {mk('sensF', _pct_pm(a['sensF']))} & "
            rf"{mk('sensR', _pct_pm(a['sensR']))} & "
            rf"{mk('n10', _pct_pm(a['n10']))} & "
            rf"{mk('tau_P', _pct_pm(a['tau_P'], 1))} & "
            rf"{mk('tau_FF', _plain_pm(a['tau_FF']))} & "
            rf"{mk('RelItems', _pct_pm(a.get('RelItems'), sign=True))} & "
            rf"{_tmin(a, times)} \\")

    if not rows:
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\emph{{no configuration of any "
                 rf"method held $\tau_P \geq {ur_floor}$}}}} \\")
    base_rows = [(i, r) for i, r in enumerate(rows) if r[0] not in _OURS_FAMILIES]
    if base_rows:
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Baselines}}}} \\")
        for i, (fam, arm, a, cleared) in base_rows:
            emit(i, fam, arm, a, cleared)
    if ours:
        L.append(rf"\cmidrule(lr){{1-{ncol}}}")
        L.append(rf"\multicolumn{{{ncol}}}{{l}}{{\textit{{Ours}}}} \\")
        for i, (fam, arm, a, cleared) in ours:
            emit(i, fam, arm, a, cleared)

    if retr and base:
        r_n10 = retr.get("test/ndcg@10")
        r_sen = retr.get("test/SHF@10")
        r_senr = retr.get("test/SHR@10")
        if r_n10 is not None and r_sen is not None:
            _ea = [v for v in (ea_by_seed or {}).values() if v]
            _ea_s = (rf"{statistics.mean(_ea) * 60:,.0f}" if _ea
                     else r"full retrain")
            _p = lambda v: ("--" if v is None else f"{v * 100.0:.3f}")
            L += [r"\midrule",
                  rf"\textit{{Retrain (reference)}} & {_p(r_sen)} & "
                  rf"{_p(r_senr)} & {_p(r_n10)} & 100.0 & +1.000 & "
                  rf"+0.000 & \textit{{{_ea_s}}} \\"]
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(l for l in L if l != "")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="beauty,toys,sports,food_k5")
    ap.add_argument("--model", default=run_tags.DEFAULT_MODEL,
                    choices=(run_tags.DEFAULT_MODEL,) + run_tags.MODELS)
    ap.add_argument("--quantizer", default=run_tags.DEFAULT_QUANTIZER,
                    choices=(run_tags.DEFAULT_QUANTIZER,) + run_tags.QUANTIZERS)
    ap.add_argument("--runs-root", default="logs/unlearn/runs")
    # One knob. --ur-floor is the historical spelling, kept so existing scripts
    # keep working; the quantity it gates is tau_P (see TAU_P_RANK_MIN).
    ap.add_argument("--tau-p-floor", "--ur-floor", dest="ur_floor", type=float,
                    default=TAU_P_RANK_MIN,
                    help="minimum tau_P for a configuration to be selectable "
                         "and for its row to be eligible for bold/underline")
    # 5, matching make_spam_tables.py. At 1 an arm explored on a single seed
    # can win a row on that seed's luck, which is selection on the outcome; the
    # toys rows sat at n=4 for exactly this reason.
    ap.add_argument("--min-seeds", type=int, default=5)
    ap.add_argument("--latex", action="store_true",
                    help="also write tables/sensitive_<ds>.tex (booktabs)")
    ap.add_argument("--all-arms", action="store_true",
                    help="list every arm instead of the best per family")
    ap.add_argument("--out", default=None,
                    help="write markdown here (e.g. tables/sensitive_grid.md)")
    args = ap.parse_args()

    track = run_tags.track_of(args.model, args.quantizer)
    L = ["## Sensitive-category deletion: the grid",
         "",
         "Lower Sensitive@10 and TPM@10 are better (the concept is gone). "
         "tau_P = NDCG_unlearn/NDCG_retrain is utility relative to the retrained "
         "model; RelItems@10 > 0 means more of the concept was removed than "
         "retraining removed. UR is against the BASE model, tau_P against "
         "RETRAIN. At forget_ratio 1e-4 the retrain reference is dominated by "
         "training variance (retrain minus base on Sensitive@10 is +0.0057 on "
         "toys and -0.0082 on sports, both significant and opposite, from "
         "deleting 0.01% of interactions), so read Sensitive@10 and UR as the "
         "primary columns and anything divided by retrain as indicative.",
         ""]
    try:
        _TIMES = spam_tables_sacct()
    except Exception:
        _TIMES = {}
    any_rows = False
    for ds in [d.strip() for d in args.datasets.split(",") if d.strip()]:
        cat = CATEGORY.get(ds)
        if cat is None:
            L += [f"### {ds}", "", f"_no sensitive category defined for {ds}_", ""]
            continue
        cells, refs, skipped = collect(ds, args.model, args.quantizer, cat,
                                       args.runs_root)
        any_rows = any_rows or bool(cells)
        L += render(ds, track, cat, cells, refs, skipped, args.ur_floor,
                    args.min_seeds, args.all_arms)
        if args.latex and cells:
            # Reference endpoints must average the same seeds as the method
            # rows; previously this took only the default seed (seed 2).
            base_mean = mean_reference_metrics(refs, 0)
            retrain_mean = mean_reference_metrics(refs, 1)
            path = run_tags.track_output_path(
                f"tables/sensitive_{ds}.tex", args.model, args.quantizer)
            os.makedirs("tables", exist_ok=True)
            with open(path, "w") as fh:
                fh.write(latex_grid(ds, track, cells, base_mean,
                                    retrain_mean, args.ur_floor,
                                    args.min_seeds, CATEGORY_NOTE.get(ds, cat),
                                    _TIMES,
                                    retrain_minutes_by_seed(refs)) + "\n")
            ppath = run_tags.track_output_path(
                f"tables/paper_sensitive_{ds}.tex", args.model, args.quantizer)
            with open(ppath, "w") as fh:
                fh.write(latex_paper(ds, track, cells, base_mean,
                                     retrain_mean, args.ur_floor,
                                     args.min_seeds, CATEGORY_NOTE.get(ds, cat),
                                     _TIMES,
                                     retrain_minutes_by_seed(refs)) + "\n")
            L += [f"_wrote {path}_", f"_wrote {ppath}_", ""]
    text = "\n".join(L)
    print(text)
    if args.out:
        # Track-scoped, so the RQ-VAE table cannot overwrite the RQ-KMeans one.
        args.out = run_tags.track_output_path(args.out, args.model, args.quantizer)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        print(f"_wrote {args.out}_")
    return 0 if any_rows else 0


if __name__ == "__main__":
    raise SystemExit(main())
