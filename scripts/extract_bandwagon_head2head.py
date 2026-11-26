"""Head-to-head table for the bandwagon n_target=1 unlearning sweeps.

Every algorithm at its best hyperparameters under one common selection rule,
aggregated over seeds, per (dataset, track, target stratum).

TRACKS. The main grid runs three: TIGER+RQ-KMeans, TIGER+RQ-VAE and DIGER (own
RQ-VAE). They are three DIFFERENT IDENTIFIER SPACES, so their numbers are not
comparable and each is scored against its own clean and poisoned references. This
script reports ONE track per invocation (`--model`, `--quantizer`, default
tiger/rkmeans). It used to match run dirs with a regex anchored as
`bs1_<dataset>_bw_tgt...`, which cannot match a tag carrying a model or quantizer
prefix -- so the RQ-VAE and DIGER tracks were invisible rather than reported as
missing. Parsing now lives in `scripts/run_tags.py`.

SEEDS ARE NOT REPLICATES. `seed` is the data-and-training seed: its own poison
generation, clean model and poisoned models, i.e. an independent draw of the
whole experiment. The legacy `_useed<N>` token is the unlearning-side seed on a
FIXED seed-2 poisoned model and is reported as `replicate`. Aggregation averages
replicates WITHIN a seed first and then across seeds, so the +- figure is
seed-to-seed spread and never replicate spread dressed up as it. `--min-seeds 3`
therefore means three independent draws; the 450 beauty `_useed` runs are one
draw replicated three times and no longer satisfy it.

    python scripts/extract_bandwagon_head2head.py --dataset beauty
    python scripts/extract_bandwagon_head2head.py --dataset toys --quantizer rqvae
    python scripts/extract_bandwagon_head2head.py --min-seeds 3 --holdout
"""

from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import statistics
import sys
from typing import Any, Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tags  # noqa: E402

# Legacy hand-recorded reference points. These describe the DEFAULT track
# (tiger + rkmeans) at seed 2 ONLY, and are kept because they are the
# denominators every recorded beauty and rsc15 table was computed with. New
# cells get a file under logs/eval/spam_references/ instead
# (scripts/write_spam_references.py).
LEGACY_REFERENCES = {
    # Clean model: job 9096928 evaluated per strategy (logs/eval/clean_baseline_bandwagon).
    # Poisoned starting checkpoints: jobs 9872116 / 9872117 / 9872118.
    "beauty": {
        "clean_n10": 0.03762655,
        "clean_sh": {"unpopular": 0.0, "mid": 0.0, "popular": 0.00125274},
        "poison_sh": {"unpopular": 0.01400, "mid": 0.01270, "popular": 0.00180},
        "poison_n10": {"unpopular": 0.03670, "mid": 0.03620, "popular": 0.03530},
    },
    # Baseline evals 9986048-53; clean model = resume seg2 9985819. Note the
    # attack only landed on the popular stratum here (see extract_rsc15_unlearn.py).
    "rsc15": {
        "clean_n10": 0.34230,
        "clean_sh": {"unpopular": 0.0, "mid": 0.0, "popular": 0.001951},
        "poison_sh": {"unpopular": 0.000003, "mid": 0.0, "popular": 0.005422},
        "poison_n10": {"unpopular": 0.34525, "mid": 0.34579, "popular": 0.34712},
    },
}
# GRID_REF_DIR points this at an exported copy, so the generators can run
# offline alongside GRID_RUN_DATA. Unset, the cluster path is used.
REF_DIR = os.environ.get("GRID_REF_DIR", "logs/eval/spam_references")

# FAIL LOUDLY. Without references the generators still render -- tau_P, tau_F
# and UR silently fall back to the wrong denominator and landed_seeds returns
# nothing, which collapses every +- spread. The result looks like a table and is
# wrong, so an offline run with no references must not start.
if os.environ.get("GRID_RUN_DATA") and not glob.glob(os.path.join(REF_DIR, "*.json")):
    raise RuntimeError(
        f"GRID_RUN_DATA is set but no spam references were found in {REF_DIR!r}.\n"
        "Every tau_P / tau_F / UR column is a ratio against the clean and "
        "poisoned baselines, which live only in those files, and the +- spreads "
        "come from landed_seeds, which reads them too. Set GRID_REF_DIR to the "
        "exported copy (paper/data/spam_references).")
STRATA = list(run_tags.STRATA)

_LAM_SN = re.compile(r"_ls([0-9.]+)")
# `_lnneg<x>` is how the sensitive sweep spells a NEGATIVE lambda_n (a minus
# sign is not legal in a run-dir token). Matching only `_ln([0-9.]+)` silently
# read `_lnneg0.1` as absent and printed lambda_n=0.0 for an arm whose lambda_n
# is -0.1 -- the opposite sign, on the row we report as ours.
_LAM_N = re.compile(r"_ln(neg)?([0-9.]+)")


# Baselines whose IDENTITY is "lambda_s and lambda_n are both off". Printing
# "(lambda_n=0.0, lambda_s=0.0)" after their names restates the definition and
# adds noise, so it is suppressed for exactly these rows at exactly that point.
_ZERO_LAMBDA_BASELINES = ("finetune_abl", "ga_abl", "negtrain_abl")


def lam_suffix(arm: Optional[str], latex: bool = False,
               family: Optional[str] = None) -> str:
    """" (lambda_n=..., lambda_s=...)" for an arm that carries them, else "".

    Shown on the row NAME so a reader never has to decode the arm tag to learn
    which operating point the selection rule picked. `_ln` is omitted from a tag
    when lambda_n is 0, so an absent token reads as 0.0 rather than unknown --
    the same convention the arm builder uses. Families with no lambdas (SCIF,
    SEIF, Kookmin, Fanchuan, the filters) get an empty string.
    """
    if not arm:
        return ""
    m_s = _LAM_SN.search(arm)
    if not m_s:
        return ""
    m_n = _LAM_N.search(arm)
    ln = ("-" + m_n.group(2) if m_n.group(1) else m_n.group(2)) if m_n else "0.0"
    ls = m_s.group(1)
    if (family in _ZERO_LAMBDA_BASELINES
            and float(ln) == 0.0 and float(ls) == 0.0):
        return ""
    if latex:
        return rf" ($\lambda_n$={ln}, $\lambda_s$={ls})"
    return f" (\u03bb_n={ln}, \u03bb_s={ls})"

FAMILIES = ["unified_sepgenerative", "unified_sepcosine", "unified",
            # Our objective swept over a lambda grid (lambda_s and/or lambda_n
            # active). Kept separate from `unified`, which is the fixed
            # sep-searched operating point, and from the *_abl baseline rows,
            # which are the grid corners where those terms are OFF. Without this
            # entry the grid arms are classified but never reported, so the
            # table showed GA / GA+GD / Finetune and no row for ours.
            "unified_abl_grid",
            "scif", "seif", "kookmin", "fanchuan",
            "negtrain_abl", "ga_abl", "finetune_abl",
            "negtrain", "finetune",
            "tracer", "filter_global", "filter_user_dependent"]

# In the SPAM tables the ascent/descent baselines are reported from the
# `unified_abl_*` arms (families *_abl), which run them through the unified
# objective at an identical step budget, batch construction and optimizer. The
# standalone `negtrain` / `finetune` implementations set their budgets with their
# own knobs (neg_train_steps=200, finetune_steps=500, against unified's
# n_epochs), so reporting both would confound an implementation difference with
# the comparison. They are excluded from the spam tables and still used for the
# SENSITIVE tables, where no ablation arms were run.
SPAM_SUPERSEDED_FAMILIES = ("negtrain", "finetune")

# Filter is reported for the SENSITIVE setting only (user's call, 2026-09-01).
# There a decode-time ban on the item set is a legitimate comparator for a
# deletion request. For SPAM it is not a comparable system: `filter_global`
# refuses to emit I_t for everyone, which trivially forces SH@10 to 0 without
# unlearning anything, and `filter_user_dependent` tracks SCIF to 3 decimals
# because it only masks at decode time for the forget users. Keeping them in the
# spam tables invites reading a decode mask as an unlearning result.
#
# FAMILIES stays the full list because scripts/extract_sensitive.py imports it.
SPAM_EXCLUDED_FAMILIES = ("filter_global", "filter_user_dependent")
SPAM_FAMILIES = [f for f in FAMILIES
                 if f not in SPAM_EXCLUDED_FAMILIES + SPAM_SUPERSEDED_FAMILIES]

family_of = run_tags.family_of

# --- efficiency: E(U) and E(A) from the RUNS' OWN timestamps -----------------
#
# Not from sacct. A SLURM elapsed time for an unlearning job bundles model load,
# data preparation, the whole sequential request stream, and the post-unlearn
# evaluation, so it overstates the per-request latency the setup section defines
# by more than an order of magnitude: measured on beauty/rkmeans SCIF, 0.594
# min/request from the run's own instrumentation against 11.22 min of SLURM
# elapsed for the job -- 19x. Training is the opposite case, where the fit
# dominates and the two agree to a second (216.22 vs 216.23 min), but the
# unlearning side is what tau_E is sensitive to.

def request_latency_minutes(run_dir: str) -> Optional[float]:
    """Mean wall-clock MINUTES per unlearning request for one run.

    Reads `<algo>_info.json`, which stores `wall_seconds` per request under
    `requests[i].<algo>_info`. That timer wraps the unlearning call itself, so it
    excludes checkpoint IO and evaluation.
    """
    for p in glob.glob(os.path.join(run_dir, "*_info.json")):
        try:
            with open(p) as fh:
                d = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        vals = []
        for r in d.get("requests", []) or []:
            if not isinstance(r, dict):
                continue
            for k, v in r.items():
                if k.endswith("_info") and isinstance(v, dict) and "wall_seconds" in v:
                    try:
                        vals.append(float(v["wall_seconds"]))
                    except (TypeError, ValueError):
                        pass
        if vals:
            return sum(vals) / len(vals) / 60.0
    return None


def train_minutes(train_job: str) -> Optional[float]:
    """E(A): retraining MINUTES from the training log's own timestamps.

    run_tiger_train.sh brackets the fit with
    `[<iso>] Starting ... train` and `[<iso>] tiger train finished`, so this
    excludes SLURM queueing and the surrounding shell.
    """
    import datetime as _dt
    f = os.path.join("logs", f"{train_job}.out")
    if not os.path.isfile(f):
        return None
    t0 = t1 = None
    pat = re.compile(r"\[(20[0-9-]+T[0-9:]+\+[0-9:]+)\]\s+(Starting|\S+ train finished)")
    try:
        with open(f, errors="ignore") as fh:
            for ln in fh:
                m = pat.match(ln)
                if not m:
                    continue
                ts = _dt.datetime.fromisoformat(m.group(1))
                if m.group(2) == "Starting" and t0 is None:
                    t0 = ts
                elif "finished" in m.group(2):
                    t1 = ts
    except OSError:
        return None
    return (t1 - t0).total_seconds() / 60.0 if (t0 and t1) else None


def train_job_from_eval_dir(d: str) -> Optional[str]:
    """The training job behind an eval run, from the checkpoint it loaded.

    A training run-dir name carries its own job id, and every eval records the
    checkpoint path it evaluated, so this recovers E(A) for exactly the model a
    reference was measured on -- which differs between scenarios: spam compares
    against the clean fit, the sensitive setting against a separate retrain that
    excluded the sensitive items.
    """
    if not isinstance(d, str):
        return None
    for cfg in ("config.yaml", "overrides.yaml"):
        f = os.path.join(d, ".hydra", cfg)
        if not os.path.isfile(f):
            continue
        try:
            with open(f, errors="ignore") as fh:
                blob = fh.read()
        except OSError:
            continue
        m = re.search(r"job(\d+)_[a-z0-9_]*(?:clean|retrain)", blob)
        if m:
            return m.group(1)
    return None


def tau_E_precision(deficits, sig: int = 3, lo: int = 4, hi: int = 9) -> int:
    """Decimals so the SMALLEST E(U)/E(A) still shows `sig` significant digits.

    tau_E is bounded above by 1 and cannot reach it -- no unlearning procedure
    is free -- so a column that prints 1.0000 is a rounding artefact reporting
    something impossible. One precision is chosen for the whole table rather
    than per row, to keep the column aligned.
    """
    import math
    ds = [d for d in deficits if d is not None and d > 0]
    if not ds:
        return lo
    return max(lo, min(hi, math.ceil(-math.log10(min(ds))) + sig - 1))


def fmt_tau_E(eu: Optional[float], ea: Optional[float], places: int) -> str:
    """tau_E = 1 - E(U)/E(A) at a fixed precision, never rounded up to 1."""
    if eu is None or not ea:
        return "--"
    return f"{1.0 - eu / ea:.{places}f}"


def retrain_train_job(ds: str, model: str, quantizer: str, seed: int) -> Optional[str]:
    """The SLURM job that trained theta_r for this cell, or None.

    Derived, not tabulated: the cell's spam reference records the clean-baseline
    eval dirs it was written from, those evals record the checkpoint they loaded,
    and a training run-dir name carries its own job id. Following that chain
    keeps E(A) attached to the very model the cell's denominators came from, and
    extends to every dataset and track without a hand-maintained map.

    For the poisoning setting theta_r IS the clean model, because the bandwagon
    attack only appends fabricated users; the clean fit is what a practitioner
    would have to redo.
    """
    for path in _ref_candidates(ds, model, quantizer, seed):
        if not os.path.isfile(path):
            continue
        try:
            with open(path) as fh:
                prov = json.load(fh).get("provenance") or {}
        except (OSError, json.JSONDecodeError):
            continue
        for d in list(prov.values()):
            j = train_job_from_eval_dir(d)
            if j:
                return j
    return None


# The admissibility threshold, owned here so collect() and the table share one
# number. make_spam_tables.LANDED_MIN mirrors it.
LANDED_MIN = 1e-4


def landed_seeds(ds: str, model: str, quantizer: str, stratum: str,
                 seeds=(2, 3, 5, 7, 11), landed_min: float = LANDED_MIN):
    """Seeds whose reference gap makes this stratum admissible, and all seeds seen.

    Admissibility is a property of a CELL, which is (dataset, track, stratum,
    SEED): each seed has its own poisoned and clean reference, and the bandwagon
    budget lands on some seeds and not others. Deciding a whole stratum from the
    default seed's reference throws away every other seed's evidence: on
    sports/rkmeans/popular, seed 2's gap is -0.0031 while seeds 3 and 5 are
    +0.0119 and +0.0011, which suppressed 135 finished runs across 27 arms.

    Returns (admissible_seeds, seen_seeds). Callers report SH@k and tau_P over
    all seeds (neither touches the poison denominator) and restrict tau_F to the
    admissible ones.
    """
    ok, seen = [], []
    for sd in seeds:
        ref, _src = resolve_reference(ds, model, quantizer, sd)
        if not ref:
            continue
        ps = ref["poison_sh"].get(stratum)
        cs = ref["clean_sh"].get(stratum)
        if ps is None or cs is None:
            continue
        seen.append(sd)
        if (ps - cs) >= landed_min:
            ok.append(sd)
    return ok, seen


def _ref_candidates(ds: str, model: str, quantizer: str, seed: int) -> List[str]:
    """Reference files that may legitimately describe this cell, best first.

    A file is only allowed to stand in for a cell it actually describes: the
    untokenized `<ds>.json` means tiger + rkmeans + seed 2, so it is never
    offered to another track or seed. Scoring a DIGER run against a TIGER
    denominator would produce a plausible number with no meaning.
    """
    out = [os.path.join(REF_DIR, f"{ds}_{model}_{quantizer}_seed{seed}.json")]
    if seed == run_tags.DEFAULT_SEED:
        out.append(os.path.join(REF_DIR, f"{ds}_{model}_{quantizer}.json"))
    # scripts/write_spam_references.py omits a token that equals its default, so
    # tiger+rqvae lands on `<ds>_rqvae.json`, not `<ds>_tiger_rqvae.json`. Ask
    # for the writer's own spelling via the shared token builder rather than a
    # second private convention -- that mismatch is why the whole RQ-VAE track
    # reported "no reference" while 835 finished runs sat on disk. The token is
    # empty only for tiger+rkmeans, so this still never offers one track's
    # denominator to another.
    tok = run_tags.track_token(model, quantizer)
    out.append(os.path.join(REF_DIR, f"{ds}{tok}_seed{seed}.json"))
    if seed == run_tags.DEFAULT_SEED:
        out.append(os.path.join(REF_DIR, f"{ds}{tok}.json"))
    # dedupe, keeping best-first order
    seen, uniq = set(), []
    for c in out:
        if c not in seen:
            seen.add(c); uniq.append(c)
    return uniq


def resolve_reference(ds: str, model: str, quantizer: str,
                      seed: int) -> Tuple[Optional[Dict[str, Any]], str]:
    """(reference, provenance). reference is None when the cell has none."""
    for path in _ref_candidates(ds, model, quantizer, seed):
        if not os.path.isfile(path):
            continue
        raw = json.load(open(path))
        ref = {"clean_n10": raw.get("clean_ndcg@10"),
               "clean_sh": raw.get("clean_SH@10") or {},
               "poison_sh": raw.get("poison_SH@10") or {},
               "poison_n10": raw.get("poison_ndcg@10") or {}}
        if ref["clean_n10"] is None or not ref["clean_sh"]:
            continue
        return ref, path
    is_default = (model, quantizer) == (run_tags.DEFAULT_MODEL,
                                        run_tags.DEFAULT_QUANTIZER)
    if is_default and seed == run_tags.DEFAULT_SEED and ds in LEGACY_REFERENCES:
        return LEGACY_REFERENCES[ds], f"LEGACY_REFERENCES[{ds!r}] (tiger+rkmeans, seed 2)"
    return None, ""


def reference_hint(ds: str, model: str, quantizer: str, seed: int) -> str:
    return (f"no reference for {ds}/{model}+{quantizer}/seed{seed}. Write one:\n"
            f"    python scripts/write_spam_references.py --dataset {ds} "
            f"--model {model} --quantizer {quantizer} --seed {seed} \\\n"
            f"        --poison-eval unpopular=<eval dir> --poison-eval mid=<...> "
            f"--poison-eval popular=<...>\n"
            f"  (expected at {_ref_candidates(ds, model, quantizer, seed)[0]})")


# --------------------------------------------------------------------------- #
# collection
# --------------------------------------------------------------------------- #
def collect(runs_root: str, dataset: str, model: str, quantizer: str,
            include_pkm: bool = False, include_hpsearch: bool = True,
            ) -> Tuple[Dict[tuple, List[Dict[str, Any]]], Dict[int, str], List[int]]:
    """(target, arm) -> per-run metric dicts, plus reference provenance per seed.

    Every run is scored against ITS OWN seed's references. A seed with no
    reference is skipped and returned in the third element rather than being
    quietly scored against another seed's denominators.
    """
    out: Dict[tuple, List[Dict[str, Any]]] = collections.defaultdict(list)
    refs: Dict[int, Optional[Dict[str, Any]]] = {}
    prov: Dict[int, str] = {}
    skipped: set[int] = set()

    for rec in run_tags.iter_runs(runs_root, setting="spam", dataset=dataset,
                                  model=model, quantizer=quantizer):
        if not include_pkm and rec["pkm"]:
            continue
        if not include_hpsearch and rec["hpsearch"]:
            continue
        seed = rec["seed"]
        if seed not in refs:
            refs[seed], prov[seed] = resolve_reference(dataset, model, quantizer, seed)
        ref = refs[seed]
        if ref is None:
            skipped.add(seed)
            continue
        mm = run_tags.last_metrics(rec["path"])
        if not mm or "test/ndcg@10" not in mm:
            continue
        target = rec["stratum"]
        n10, sh = mm["test/ndcg@10"], mm.get("test/SH@10")
        p_sh = ref["poison_sh"].get(target)
        c_sh = ref["clean_sh"].get(target)
        den = None if (p_sh is None or c_sh is None) else p_sh - c_sh
        out[(target, rec["arm"])].append({
            "seed": seed,
            "replicate": rec["replicate"],
            "job": rec["jid"],
            "dir": rec["path"],
            "sh": sh,
            "asi": mm.get("test/ASI@10"),
            "tpm": mm.get("test/TPM@10"),
            "n10": n10,
            "r5": mm.get("test/recall@5"),
            "UR": n10 / ref["clean_n10"] if ref["clean_n10"] else float("nan"),
            # tau_F is defined only where the attack actually opened a gap on
            # THIS cell (this stratum, this seed). The old `if den` admitted any
            # nonzero denominator, including negative ones and gaps of 1e-6,
            # which is where the +9.900 and -231.492 values came from: an
            # algorithm being credited or blamed for removing an attack that was
            # never there. Declaring admissibility here, before the metric is
            # read, is the condition the setup section describes.
            "PGR": ((p_sh - sh) / den
                    if (den is not None and den >= LANDED_MIN and sh is not None)
                    else float("nan")),
        })

    # One run per (target, arm, seed, replicate) -- keep the newest job.
    for key, runs in out.items():
        by_cell: Dict[tuple, Dict[str, Any]] = {}
        for r in sorted(runs, key=lambda x: int(x["job"])):
            by_cell[(r["seed"], r["replicate"])] = r
        out[key] = [by_cell[k] for k in sorted(by_cell, key=lambda k: (k[0], k[1] or 0))]
    return out, prov, sorted(skipped)


# --------------------------------------------------------------------------- #
# aggregation: replicates within a seed, then across seeds
# --------------------------------------------------------------------------- #
def agg(runs: List[Dict[str, Any]], field: str) -> Dict[str, Any]:
    """Nested aggregate of ``field``.

    Replicates of one poisoned model are averaged first, so ``sd`` is spread
    across INDEPENDENT draws. With a single seed there is no seed-to-seed
    spread to report and ``sd`` is None; ``rep_spread`` then carries the
    half-range across that seed's replicates, which is a different and much
    weaker statement.
    """
    per_seed: Dict[int, List[float]] = collections.defaultdict(list)
    for r in runs:
        v = r.get(field)
        if v is not None and not (isinstance(v, float) and v != v):
            per_seed[r["seed"]].append(v)
    per_seed = {s: v for s, v in per_seed.items() if v}
    if not per_seed:
        return {"mean": float("nan"), "sd": None, "n_seeds": 0, "n_runs": 0,
                "min": float("nan"), "max": float("nan"), "spread": float("nan"),
                "rep_spread": None, "seed_means": {}}
    seed_means = {s: statistics.mean(v) for s, v in per_seed.items()}
    runs_per_seed = {s: len(v) for s, v in per_seed.items()}
    means = list(seed_means.values())
    n_runs = sum(len(v) for v in per_seed.values())
    rep_spread = None
    if len(seed_means) == 1:
        vals = next(iter(per_seed.values()))
        rep_spread = (max(vals) - min(vals)) / 2 if len(vals) > 1 else None
    return {
        "mean": statistics.mean(means),
        "sd": statistics.stdev(means) if len(means) > 1 else None,
        "n_seeds": len(seed_means),
        "n_runs": n_runs,
        "min": min(means),
        "max": max(means),
        "spread": max(means) - min(means),
        "rep_spread": rep_spread,
        "runs_per_seed": runs_per_seed,
        "seed_means": seed_means,
    }


def fmt_pm(a: Dict[str, Any], places: int = 5) -> str:
    """mean +- spread.

    NOTE: as of 2026-08-26 the `(r+-x)` marker is NOT rendered, by request, while
    the results are preliminary. A one-data-seed cell now prints a bare `+-x`
    that is the half-range over UNLEARNING REPLICATES of a single model, not a
    sample SD across data seeds, and it therefore understates uncertainty: it
    cannot see variation from the poisoned model, the target item, or training.
    The `seeds` column still distinguishes them -- `3` is three data seeds,
    `1x3` is one data seed replicated three times -- so read the two together.
    """
    if a["n_seeds"] == 0:
        return "n/a"
    body = f"{a['mean']:.{places}f}"
    if a["sd"] is not None:
        return f"{body}+-{a['sd']:.{places}f}"
    if a["rep_spread"]:
        return f"{body}+-{a['rep_spread']:.{places}f}"
    return body


def rank_marks(vals, better="high", eligible=None):
    """Index -> "best" | "second" for a column of numbers.

    `better` is "high", "low", "zero" (closest to zero wins) or "one" (closest
    to 1 wins, which is what tau_F means: the retrained model is the target, so
    overshooting it is a departure exactly like undershooting. Ranking tau_F
    "high" instead would bold tau_F = +8.9, a model that suppressed nearly nine
    times the poison gap, as the best forgetter).

    `eligible` is a list of booleans, and this is the important part. Ranking is
    restricted to rows that CLEARED the utility floor. Without that restriction a
    destroyed model wins every removal column outright -- NegTrain reaches
    SH@10 0.00000 at UR 0.000 -- and bolding it as "best" would advertise the
    exact misreading the dagger footnote exists to prevent.

    Ties share a mark: every index holding the best value is bold, and "second"
    is the next DISTINCT value, not the next row.
    """
    keyf = {"high": lambda v: -v, "low": lambda v: v, "zero": abs,
            "one": lambda v: abs(v - 1.0)}[better]
    cand = {}
    for i, v in enumerate(vals):
        if v is None or v != v:                      # None or NaN
            continue
        if eligible is not None and not eligible[i]:
            continue
        cand[i] = round(keyf(v), 12)                 # float noise must not split a tie
    if not cand:
        return {}
    order = sorted(set(cand.values()))
    marks = {}
    for i, k in cand.items():
        if k == order[0]:
            marks[i] = "best"
        elif len(order) > 1 and k == order[1]:
            marks[i] = "second"
    return marks


def tex_mark(cell: str, mark) -> str:
    """Wrap a formatted cell for LaTeX. Bold = best, underline = second."""
    if mark == "best":
        return r"\textbf{" + cell + "}"
    if mark == "second":
        return r"\underline{" + cell + "}"
    return cell


def seed_label(a: Dict[str, Any]) -> str:
    """Number of DATA seeds behind the row, as a bare count.

    By request (2026-08-26) this reports data seeds only. The replicate
    structure is NOT shown: a row aggregating 5 data seeds prints "5" whether
    each seed was run once or one of them carries extra unlearning replicates.
    That is safe for the reported statistics -- `agg` averages within a seed
    before going across seeds, so replicates never get extra weight in the mean
    or the SD -- but it does mean the label no longer distinguishes a clean
    5x1 grid from a ragged one. `runs_per_seed` in the aggregate still carries
    the detail for auditing.
    """
    return str(a["n_seeds"])


def _subset(runs: List[Dict[str, Any]],
            seeds: Optional[List[int]]) -> List[Dict[str, Any]]:
    if seeds is None:
        return runs
    keep = set(seeds)
    return [r for r in runs if r["seed"] in keep]


def best_per_family(data, target: str, ur_floor: float, min_seeds: int = 1,
                    select_seeds: Optional[List[int]] = None,
                    report_seeds: Optional[List[int]] = None,
                    score=None,
                    ) -> Dict[str, Optional[Dict[str, Any]]]:
    """family -> {arm, sh, n10, UR, PGR} at its selected config, or None.

    Selection rule: lowest mean SH@10 among configs holding mean UR >= floor;
    if none clears the floor, the lowest mean SH@10 overall.

    `score(arm, runs)` overrides the objective with any value to MINIMISE, so a
    caller can select on distance to retraining instead. Lowest SH@10 rewards
    over-removal -- a config that suppresses far past the retrained model wins
    on SH while sitting further from retraining than a milder one. The default
    is unchanged, because other callers (sensitive tables, head-to-head dumps)
    still select on SH.
    """
    out: Dict[str, Optional[Dict[str, Any]]] = {}
    for fam in FAMILIES:
        cands = []
        for (t, arm), runs in data.items():
            if t != target or family_of(arm) != fam:
                continue
            sel = _subset(runs, select_seeds)
            if not sel:
                continue
            if agg(runs, "sh")["n_seeds"] < min_seeds:
                continue
            cands.append((arm, runs, sel))
        if not cands:
            out[fam] = None
            continue
        scored = [(arm, runs, agg(sel, "sh"), agg(sel, "n10"), agg(sel, "UR"))
                  for arm, runs, sel in cands]
        ok = [x for x in scored if x[4]["mean"] >= ur_floor]
        pool = ok if ok else scored
        if score is None:
            arm, runs, *_ = min(pool, key=lambda x: (x[2]["mean"], -x[3]["mean"]))
        else:
            def _key(x):
                v = score(x[0], x[1])
                return (9e9 if v is None else v, -x[3]["mean"])
            arm, runs, *_ = min(pool, key=_key)
        rep = _subset(runs, report_seeds)
        if not rep:
            out[fam] = {"arm": arm, "empty": True}
            continue
        out[fam] = {"arm": arm, "empty": False,
                    "sh": agg(rep, "sh"), "n10": agg(rep, "n10"),
                    "UR": agg(rep, "UR"), "PGR": agg(rep, "PGR"),
                    "tpm": agg(rep, "tpm"), "asi": agg(rep, "asi"),
                    "cleared_floor": bool(ok),
                    # the runs behind the reported row, so a caller can join them
                    # to sacct wall-clock without re-deriving the selection
                    "runs": rep}
    return out


def print_head2head(data, ref, prov, ur_floor: float, min_seeds: int = 1,
                    select_seeds: Optional[List[int]] = None,
                    report_seeds: Optional[List[int]] = None) -> None:
    holdout = select_seeds is not None or report_seeds is not None
    print(f"\n{'=' * 108}")
    print("HEAD-TO-HEAD: best config per algorithm, common rule "
          f"(min mean SH@10 s.t. mean UR >= {ur_floor}; else min SH)")
    print("+-x is the SAMPLE SD ACROSS DATA SEEDS where the `seeds` column "
          "shows a plain count; where it shows NxR it is the replicate half-range "
          "and x is the half-range over its unlearning replicates, which is not "
          "a seed variance. 'seeds' column: '3' = 3 seeds, '1x3' = 1 seed x 3 "
          "replicates.")
    if min_seeds > 1:
        print(f"RESTRICTED to configs with >= {min_seeds} DATA seeds.")
    if holdout:
        print(f"HOLD-OUT: config SELECTED on seeds {select_seeds}, "
              f"metrics REPORTED on seeds {report_seeds} (disjoint).")
    else:
        print("NO HOLD-OUT: config selected and reported on the same seeds "
              "(selection set overlaps evaluation set).")
    for target in STRATA:
        p_sh, c_sh = ref["poison_sh"].get(target), ref["clean_sh"].get(target)
        p_n10 = ref["poison_n10"].get(target)
        head = f"\n-- target={target}   clean N@10 {ref['clean_n10']:.5f}"
        if c_sh is not None:
            head += f"   clean SH@10 {c_sh:.5f}"
        if p_sh is not None:
            head += f"   poisoned start SH@10 {p_sh:.5f}"
        if p_n10:
            head += f" / N@10 {p_n10:.5f} (UR {p_n10 / ref['clean_n10']:.3f})"
        print(head)
        if p_sh is None:
            print("   (no poisoned reference for this stratum: PGR is n/a)")
        print(f"{'algorithm':<22}{'seeds':>7}  {'SH@10':>21}{'N@10':>21}"
              f"{'UR':>16}{'PGR@10':>9}  best config")
        best = best_per_family(data, target, ur_floor, min_seeds,
                               select_seeds, report_seeds)
        for fam in FAMILIES:
            b = best.get(fam)
            if b is None:
                print(f"{fam:<22}{'--':>7}  (no config with >= {min_seeds} seed(s))")
                continue
            if b["empty"]:
                print(f"{fam:<22}{'0':>7}  {'n/a':>21}{'n/a':>21}{'n/a':>16}"
                      f"{'n/a':>9}  {b['arm']}  [no run in report seeds]")
                continue
            flag = "" if b["cleared_floor"] else "  [UR floor not cleared]"
            pgr = b["PGR"]
            pgr_s = "n/a" if pgr["n_seeds"] == 0 else f"{pgr['mean']:.3f}"
            print(f"{fam:<22}{seed_label(b['sh']):>7}  {fmt_pm(b['sh']):>21}"
                  f"{fmt_pm(b['n10']):>21}{fmt_pm(b['UR'], 3):>16}{pgr_s:>9}  "
                  f"{b['arm']}{flag}")
    print(f"\nreferences: " + ", ".join(f"seed {s}: {p}" for s, p in sorted(prov.items())))


def print_lambda_n(data) -> None:
    """The load-bearing claim: does lambda_n survive seed noise?"""
    print(f"\n{'=' * 108}")
    print("LAMBDA_N CLAIM: unified lf=0.1 ls=0.1, ln=0.0 vs ln=0.01, per run")
    print("Claim under test: ln>0 costs ~nothing on SH but lifts WORST-CASE UR "
          "across target strata.")
    variants = ["unified_lf0.1_ls0.1_ac0_ln0.0", "unified_lf0.1_ls0.1_ac0_ln0.01"]
    worst: Dict[str, Dict[tuple, float]] = {v: {} for v in variants}
    for target in STRATA:
        print(f"\n-- target={target}")
        print(f"{'ln':>6}{'seed':>6}{'rep':>5}{'SH@10':>10}{'N@10':>10}{'UR':>8}"
              f"{'PGR':>8}  job")
        for v in variants:
            runs = data.get((target, v), [])
            if not runs:
                print(f"{v.split('_ln')[1]:>6}   (no runs yet)")
                continue
            for r in runs:
                rep = r["replicate"]
                print(f"{v.split('_ln')[1]:>6}{r['seed']:>6}"
                      f"{(rep if rep is not None else '-'):>5}{r['sh']:>10.5f}"
                      f"{r['n10']:>10.5f}{r['UR']:>8.3f}{r['PGR']:>8.3f}  {r['job']}")
                key = (r["seed"], rep)
                worst[v][key] = min(worst[v].get(key, 1e9), r["UR"])
    print("\nWorst-case UR across the three target strata, per (seed, replicate) "
          "(higher is better):")
    keys = sorted(set().union(*[set(w) for w in worst.values()]))
    if not keys:
        return
    print(f"{'ln':>6}" + "".join(f"{f's{k[0]}r{k[1] or 0}':>12}" for k in keys))
    for v in variants:
        cells = "".join(f"{worst[v][k]:>12.3f}" if k in worst[v] else f"{'--':>12}"
                        for k in keys)
        print(f"{v.split('_ln')[1]:>6}{cells}")
    both = [k for k in keys if all(k in worst[v] for v in variants)]
    if both:
        deltas = [worst[variants[1]][k] - worst[variants[0]][k] for k in both]
        n_seeds = len({k[0] for k in both})
        print(f"\ndelta (ln=0.01 minus ln=0.0) on worst-case UR, over "
              f"{len(both)} paired run(s) spanning {n_seeds} data seed(s): "
              + ", ".join(f"{d:+.4f}" for d in deltas))
        print(f"  mean {statistics.mean(deltas):+.4f}"
              + (f", range {min(deltas):+.4f}..{max(deltas):+.4f}"
                 if len(deltas) > 1 else ""))
        if n_seeds < 2:
            print("  CAUTION: a single data seed. These are unlearning "
                  "replicates of one poisoned model, so this is a statement "
                  "about retain-sampling noise, not about seed variance.")
        elif len(deltas) > 1:
            print("  VERDICT: " + ("holds on every paired run"
                                   if all(d > 0 for d in deltas)
                                   else "NOT consistent across runs"))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--runs-root", default="logs/unlearn/runs")
    ap.add_argument("--dataset", default="beauty")
    ap.add_argument("--model", default=run_tags.DEFAULT_MODEL,
                    choices=(run_tags.DEFAULT_MODEL,) + run_tags.MODELS)
    ap.add_argument("--quantizer", default=run_tags.DEFAULT_QUANTIZER,
                    choices=(run_tags.DEFAULT_QUANTIZER,) + run_tags.QUANTIZERS)
    ap.add_argument("--ur-floor", type=float, default=0.95)
    ap.add_argument("--min-seeds", type=int, default=1,
                    help="only consider configs replicated on >= N DATA seeds")
    ap.add_argument("--include-pkm", action="store_true")
    ap.add_argument("--lambda-n-only", action="store_true")
    ap.add_argument("--json", default=None)
    ap.add_argument("--holdout", action="store_true")
    ap.add_argument("--select-seeds", default="2")
    ap.add_argument("--report-seeds", default=None)
    ap.add_argument("--no-holdout", action="store_true")
    args = ap.parse_args()

    track = run_tags.track_of(args.model, args.quantizer)
    data, prov, skipped = collect(args.runs_root, args.dataset, args.model,
                                  args.quantizer, args.include_pkm)
    if not prov:
        raise SystemExit(reference_hint(args.dataset, args.model,
                                        args.quantizer, run_tags.DEFAULT_SEED))
    ref, _ = resolve_reference(args.dataset, args.model, args.quantizer,
                               run_tags.DEFAULT_SEED)
    if ref is None:
        ref, _ = resolve_reference(args.dataset, args.model, args.quantizer,
                                   min(prov))
    n_runs = sum(len(v) for v in data.values())
    seeds = collections.Counter(r["seed"] for runs in data.values() for r in runs)
    print(f"dataset={args.dataset}  track={track}")
    print(f"Collected {n_runs} runs across {len(data)} (target, config) cells; "
          "runs per data seed: "
          + (", ".join(f"seed {s}: {n}" for s, n in sorted(seeds.items())) or "none"))
    if skipped:
        print(f"\nSKIPPED seeds {skipped}: {reference_hint(args.dataset, args.model, args.quantizer, skipped[0])}")
    if not n_runs:
        print(f"\nNo runs for {args.dataset} on {track}. Run dirs for this track "
              f"carry the tag prefix "
              f"'{'' if args.model == 'tiger' else args.model + '_'}"
              f"{'' if args.quantizer == 'rkmeans' else args.quantizer + '_'}'.")
        return

    def parse_seeds(spec: Optional[str]) -> Optional[List[int]]:
        if spec is None or str(spec).lower() == "all":
            return None
        return sorted({int(x) for x in str(spec).split(",") if x.strip()})

    sel_seeds = rep_seeds = None
    holdout = args.holdout and not args.no_holdout \
        and str(args.select_seeds).lower() != "all"
    if holdout:
        sel_seeds = parse_seeds(args.select_seeds)
        all_seeds = sorted(seeds)
        rep_seeds = (parse_seeds(args.report_seeds) if args.report_seeds
                     else [s for s in all_seeds if s not in set(sel_seeds or [])])
        overlap = set(sel_seeds or []) & set(rep_seeds or [])
        if overlap:
            ap.error(f"--select-seeds and --report-seeds overlap on {sorted(overlap)}")
        if not rep_seeds:
            ap.error(f"no seeds left to report on (select={sel_seeds}, "
                     f"available={all_seeds}); add seeds or use --no-holdout")

    if not args.lambda_n_only:
        print_head2head(data, ref, prov, args.ur_floor, args.min_seeds,
                        sel_seeds, rep_seeds)
    print_lambda_n(data)

    if args.json:
        # Track-scoped for the same reason the tables are.
        args.json = run_tags.track_output_path(args.json, args.model, args.quantizer)
        os.makedirs(os.path.dirname(args.json) or ".", exist_ok=True)
        with open(args.json, "w") as fh:
            json.dump({"dataset": args.dataset, "track": track,
                       "references": prov,
                       "cells": {f"{t}|{a}": runs for (t, a), runs in data.items()}},
                      fh, indent=2)
        print(f"\nWrote {args.json}")


if __name__ == "__main__":
    main()
