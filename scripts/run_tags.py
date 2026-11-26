"""One parser for unlearn run-dir names, shared by every extractor.

Three scripts used to re-derive the run-dir grammar independently
(``extract_bandwagon_head2head.py``, ``extract_sensitive.py``,
``write_experiment_progress.py``) and they disagreed: the head-to-head regex was
anchored as ``bs1_<dataset>_bw_tgt...`` and so matched NOTHING on the RQ-VAE or
DIGER tracks, whose tags carry a model/quantizer prefix. Collecting three tracks
with a beauty-shaped pattern silently reports one of them.

RUN-DIR GRAMMAR
---------------
::

    job<jid>_<date>_<time>_<nanos>_bs1_<TAG>

    TAG  := [<model>_][<quantizer>_] BODY [_seed<N>] [_useed<N>]
    BODY := sens_<dataset>_<category>_<arm>                      (sensitive)
          | [<dataset>_]bw_tgt<stratum>_[pkm<arch>_]<arm>        (spam)

``model`` is absent for tiger and ``diger_`` otherwise (``GRID_MODEL_TAG``, added
by ``run_tiger_unlearn_sequential.sh``). ``quantizer`` is absent for rkmeans and
``rqvae_`` / ``rvq_`` otherwise (``QTOK``). The dataset token is absent for
beauty, which predates the convention. Every absent token is a DEFAULT, not a
missing value, which is why a bare ``*beauty_clean``-style glob matches the diger
runs too and why the guards in the sweep drivers prefix ``GRID_MODEL_TAG``.

THE TWO SEED TOKENS ARE DIFFERENT AXES
--------------------------------------
``_seed<N>``  the data-and-training seed: its own poison generation, its own
              clean and poisoned models. Independent draws of the whole
              experiment, and the axis the main grid averages over.
``_useed<N>`` LEGACY: the unlearning-side seed (retain sampling + request
              order) on a FIXED seed-2 poisoned model. 450 beauty spam runs and
              16 sensitive runs from 2026-08-04..19 carry it. These are
              replicates of one draw, not independent seeds, so pooling them as
              "3 seeds" overstates the evidence -- and the reason the sweeps now
              emit ``_seed<N>`` instead: emitting ``_useed3`` for data seed 3
              would collide with those dirs, the duplicate guard would skip the
              cell, and the table would score a seed-2 poisoned model in the
              seed-3 column.

Records therefore carry ``seed`` (data seed, default 2) and ``replicate``
(unlearn-side seed, or None). Aggregators should count DISTINCT ``seed`` values
for any claim about variance, and may average over ``replicate`` within a seed.
"""

from __future__ import annotations

import csv
import glob
import os
import re
import sys
from typing import Any, Dict, Iterator, List, Optional

# `import src...` below needs the repo ROOT importable. `python scripts/foo.py`
# puts scripts/ on sys.path but NOT the root, so every consumer of this module
# broke with ModuleNotFoundError unless it was run with PYTHONPATH=. from the
# root. Fixing it here rather than in each caller keeps the one-parser property
# this module exists for: a script should be able to `import run_tags` and work.
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import src.models.registry as _registry  # noqa: E402

# Longest first: 'food_k5' must win over 'food', and any dataset whose name
# contains '_' would otherwise be split by a naive token split (the old
# sensitive parser read 'sens_food_k5_meat_...' as dataset='food',
# category='k5').
DATASETS = ("grocery_k3u4", "food_k5", "food_L5", "rsc15", "sports",
            "beauty", "toys", "food")

# Non-default model prefixes, from the registry so this cannot drift: a model
# added there whose prefix is missing HERE parses as tiger, and its runs land in
# tiger's track in every table built on this parser. tiger is the empty prefix.
# Longest first, in case two prefixes ever share a stem.
MODELS = tuple(
    sorted(
        (
            _spec.run_tag_prefix.rstrip("_")
            for _spec in (_registry.resolve_model(_n) for _n in _registry.available_models())
            if _spec.run_tag_prefix
        ),
        key=len,
        reverse=True,
    )
)
# rkmeans is the empty prefix. "letter" is LETTER's own tokenizer: it IS an
# RQ-VAE family quantizer (encoder/decoder MLP + VQ, and load_rq_quantizer
# auto-detects it as rqvae), but it is a DISTINCT identifier space from the
# plain rqvae track, so it needs its own token. Without it, a sweep run with
# QUANTIZER=letter falls back to the rkmeans default and every LETTER cell is
# mislabelled as the rkmeans track in any table keyed on `track`.
# `tracerv3` is a REBUILT RQ-KMeans codebook for beauty, registered as its own
# quantizer because it is a different identifier space, not a variant of the
# original: the width-256 beauty codebook was lost, and a rebuild reproduces
# only 0.01/0.38/0.35% of the stored ids (chance is 1/256 = 0.39%). TRACER
# refuses to run unless phi=0 reproduces the ids exactly, so beauty TRACER needs
# models trained on a codebook that survives. Keeping it a separate track stops
# its runs being scored against the old space's references, which would produce
# plausible, meaningless numbers.
QUANTIZERS = ("rqvae", "rvq", "letter", "tracerv3")
STRATA = ("unpopular", "mid", "popular")

DEFAULT_MODEL = "tiger"
DEFAULT_QUANTIZER = "rkmeans"
DEFAULT_TRACK = f"{DEFAULT_MODEL}+{DEFAULT_QUANTIZER}"
DEFAULT_SEED = 2

RUNDIR_RE = re.compile(r"^job(?P<jid>\d+)_.*_bs1_(?P<tag>.+)$")
SPAM_RE = re.compile(r"^bw_tgt(?P<stratum>unpopular|mid|popular)_(?P<rest>.*)$")
PKM_RE = re.compile(r"^(?P<pkm>pkm[a-z0-9]*)_(?P<rest>.*)$")
SEED_RE = re.compile(r"_seed(\d+)$")
USEED_RE = re.compile(r"_useed(\d+)$")

# Finer than `algo`: the head-to-head table reports the two filter modes as
# separate rows because they are different references, not different configs.
FAMILIES = ("unified", "scif", "seif", "kookmin", "fanchuan", "negtrain",
            "tracer", "finetune", "filter_global", "filter_user_dependent")
ALGOS = ("scif", "seif", "kookmin", "fanchuan", "unified", "neg_train",
         "finetune", "tracer", "filter")


def track_of(model: str, quantizer: str) -> str:
    return f"{model}+{quantizer}"


def algo_of(arm: str) -> str:
    """Canonical algorithm name for a tag's arm.

    Order matters: `negtrain` is the tag spelling of `neg_train`, and the plain
    `unified_lf...` arms must not be confused with the `unified_hp_...` search.
    """
    if arm.startswith("negtrain"):
        return "neg_train"
    for a in ("scif", "seif", "kookmin", "fanchuan", "finetune", "unified",
              "tracer", "filter"):
        if arm.startswith(a):
            return a
    return "other"


def family_of(arm: str) -> str:
    """Algorithm family for head-to-head reporting.

    Splits variants that are different METHODS rather than different
    hyperparameters of one method, so a table row cannot silently mix them:

    * the ascent/descent baselines run THROUGH the unified objective
      (`unified_abl_*`), which are Finetune / NegTrain / GA and not our method,
    * the two filter modes (global vs user dependent), and
    * the two separation scores. `cosine` scores a representation geometry the
      decoder never reads while `generative` scores the model's own sequence
      log-probability, and they differ by up to 7.5 UR points at equal removal.
      Selecting the best per cell across both would report generative on one
      dataset and cosine on another under one "unified" row.
    """
    for fam in ("filter_global", "filter_user_dependent"):
        if arm.startswith(fam):
            return fam
    algo = algo_of(arm)
    if algo == "unified":
        # `unified_abl_*` arms are the ascent/descent BASELINES expressed in the
        # unified objective, so that no implementation difference (step budget,
        # batch construction, optimizer) is confounded with the comparison. They
        # must NOT fall into the `unified` family: they would then compete for the
        # "unified (ours)" row, and a retain-only arm could win it.
        if arm.startswith("unified_abl"):
            body = arm.split("_seed")[0].split("_useed")[0]
            lam_r_zero = body.endswith("_lr0.0") or body.endswith("_lr0")
            lam_f_zero = "_lf0.0_" in body or "_lf0_" in body
            # A baseline is defined by which terms are OFF, so lambda_s and
            # lambda_n have to be read too. The original ablation set was three
            # arms that all had lambda_s = lambda_n = 0, so testing lambda_f and
            # lambda_r alone was sufficient; once a full lambda grid is swept
            # (2026-09-15: 18 quadruples) that test silently files
            # `unified_abl_lf0.1_ls0.1_ln0.1` -- our complete objective -- under
            # `negtrain_abl`, and the table then reports a "baseline" beating
            # our method when the row IS our method with a searched lambda.
            m_s = re.search(r"_ls([0-9.]+)", body)
            m_n = re.search(r"_ln([0-9.]+)", body)
            lam_s_zero = (float(m_s.group(1)) == 0.0) if m_s else True
            lam_n_zero = (float(m_n.group(1)) == 0.0) if m_n else True
            if not (lam_s_zero and lam_n_zero):
                # separation and/or neighbourhood active: this is OUR objective,
                # not an ascent/descent baseline. Kept out of the `unified` row
                # too, which is reserved for the sep-searched operating points.
                return "unified_abl_grid"
            if lam_f_zero:
                # lambda_f = 0, lambda_r = 1: retain term only.
                return "finetune_abl"
            return "ga_abl" if lam_r_zero else "negtrain_abl"
        if arm.endswith("_sepgenerative") or "_sepgenerative_" in arm:
            return "unified_sepgenerative"
        if arm.endswith("_sepcosine") or "_sepcosine_" in arm:
            return "unified_sepcosine"
        return "unified"
    if algo == "tracer":
        # TRACER's step budget is set by `tracer_steps` (500) unless n_epochs is
        # passed, and the run tag records which: an `_ep<N>` token means the
        # budget was matched to ours. A 500-step run on a single forget batch is
        # 500 passes over that batch, so it overfits AND receives ~125x the
        # optimization of the method it is compared against. Those runs are a
        # separate family so they cannot be selected into the TRACER row; the
        # family is deliberately absent from FAMILIES, which drops them.
        body = arm.split("_seed")[0].split("_useed")[0]
        return "tracer" if "_ep" in body else "tracer_fixedsteps"
    return "negtrain" if algo == "neg_train" else algo


def is_hpsearch(arm: str) -> bool:
    """True for a hyperparameter-search arm (a named grid point).

    Frozen-config arms carry a bare algorithm name (`scif`, `unified`) or the
    operating-point spelling (`unified_lf0.1_ls0.1_ac0_ln0.0`); search arms name
    the searched knob.
    """
    return bool(re.match(
        r"^(scif_d|seif_std\S+_rlr|kookmin_ir\S+_rlr|fanchuan_t\S+_lr|"
        r"negtrain_lr|finetune_lr|unified_hp_|tracer_lf)", arm))


def parse_tag(tag: str) -> Optional[Dict[str, Any]]:
    """Parse a run tag into grid coordinates. None if it is not a grid run."""
    rest = tag
    rec: Dict[str, Any] = {"tag": tag}

    rec["model"] = DEFAULT_MODEL
    for m in MODELS:
        if rest.startswith(m + "_"):
            rec["model"] = m
            rest = rest[len(m) + 1:]
            break

    rec["quantizer"] = DEFAULT_QUANTIZER
    for q in QUANTIZERS:
        if rest.startswith(q + "_"):
            rec["quantizer"] = q
            rest = rest[len(q) + 1:]
            break
    rec["track"] = track_of(rec["model"], rec["quantizer"])

    # Seed tokens live at the very end, after the arm. Strip both, in either
    # order, so a future sweep that emits them together still parses.
    rec["seed"], rec["replicate"] = DEFAULT_SEED, None
    for _ in range(2):
        um = USEED_RE.search(rest)
        if um:
            rec["replicate"] = int(um.group(1))
            rest = rest[: um.start()]
            continue
        sm = SEED_RE.search(rest)
        if sm:
            rec["seed"] = int(sm.group(1))
            rest = rest[: sm.start()]
            continue
        break

    if rest.startswith("sens_"):
        body = rest[len("sens_"):]
        ds = next((d for d in DATASETS if body.startswith(d + "_")), None)
        if ds is None:
            return None
        body = body[len(ds) + 1:]
        cat, _, arm = body.partition("_")
        if not arm:
            return None
        rec.update(setting="sensitive", dataset=ds, category=cat,
                   stratum="-", pkm="", arm=arm)
    else:
        ds = next((d for d in DATASETS if rest.startswith(d + "_")), None)
        head = rest if ds is None else rest[len(ds) + 1:]
        sm = SPAM_RE.match(head)
        if not sm:
            return None
        arm = sm.group("rest")
        pm = PKM_RE.match(arm)
        pkm = ""
        if pm:
            pkm, arm = pm.group("pkm"), pm.group("rest")
        if not arm:
            return None
        rec.update(setting="spam", dataset=ds or "beauty", category=None,
                   stratum=sm.group("stratum"), pkm=pkm, arm=arm)

    rec["algo"] = algo_of(rec["arm"])
    rec["family"] = family_of(rec["arm"])
    rec["hpsearch"] = is_hpsearch(rec["arm"])
    return rec


def parse_run_dir(path: str) -> Optional[Dict[str, Any]]:
    """Parse one flat unlearn run dir. None if the name is not a grid run."""
    m = RUNDIR_RE.match(os.path.basename(path))
    if not m:
        return None
    rec = parse_tag(m.group("tag"))
    if rec is None:
        return None
    metrics = os.path.join(path, "eval", "csv", "version_0", "metrics.csv")
    rec.update(jid=m.group("jid"), path=path, metrics_path=metrics,
               done=os.path.isfile(metrics))
    return rec


# OFFLINE MODE. With GRID_RUN_DATA pointing at an exported runs.csv, iter_runs
# and last_metrics read that file instead of scanning logs/. Every generator in
# scripts/extract_*.py then runs with no cluster access and no run directories,
# which is what lets the paper repo rebuild its own tables. Unset, nothing
# changes and the directory scan is used, so recorded invocations are unaffected.
_RUN_DATA_CACHE: Dict[str, Any] = {}


def _run_data() -> Optional[Dict[str, Any]]:
    path = os.environ.get("GRID_RUN_DATA")
    if not path:
        return None
    if _RUN_DATA_CACHE.get("path") != path:
        recs, metrics = [], {}
        with open(path) as fh:
            for row in csv.DictReader(fh):
                rec, met = {}, {}
                for k, v in row.items():
                    if k.startswith("test/") or k in ("epoch", "step"):
                        if v not in ("", None):
                            met[k] = float(v)
                    elif k == "seed" or k == "replicate":
                        rec[k] = int(v) if v not in ("", None) else None
                    elif k == "done":
                        rec[k] = v in ("True", "true", "1")
                    else:
                        rec[k] = v if v != "" else None
                recs.append(rec)
                metrics[rec.get("path")] = met
        _RUN_DATA_CACHE.clear()
        _RUN_DATA_CACHE.update(path=path, recs=recs, metrics=metrics)
    return _RUN_DATA_CACHE


def iter_runs(runs_root: str = "logs/unlearn/runs",
              unparsed: Optional[List[str]] = None,
              **filters: Any) -> Iterator[Dict[str, Any]]:
    """Every parsed run dir under ``runs_root``, optionally filtered.

    Filters are exact-match on any record field, or membership when the value is
    a set/list/tuple: ``iter_runs(setting="spam", seed={2, 3})``. A run dir whose
    name does not parse is appended to ``unparsed`` rather than dropped -- a
    date-stamped dir means the module was called directly instead of through
    run_tiger_unlearn_sequential.sh, which skips the post-eval and still exits 0.
    """
    data = _run_data()
    source = (iter(data["recs"]) if data else
              (parse_run_dir(p) for p in
               sorted(glob.glob(os.path.join(runs_root, "job*_bs1_*")))))
    for rec in source:
        if rec is None:
            if unparsed is not None:
                unparsed.append("<unparsed>")
            continue
        # `_timing` runs (run_timing_launch.sh) only re-measure wall-clock per
        # request; they have no eval and must never enter a table as candidates.
        if "_timing" in (rec.get("tag") or ""):
            continue
        # Rebuttal variants (scripts/plan_rebuttal_variants.py): reference-capped
        # forgetting (`_capref`, `_caprefF`) and the retain-side anchor
        # (`_anchor<w>`). They are new objectives compared only by their own
        # scripts (paper/plotting/capref_table.py, anchor_table.py) and must never
        # enter a table as candidate configurations of an existing method.
        if re.search(r"_capref|_anchor\d", rec.get("tag") or ""):
            continue
        # `randsid` runs (run_randsid_*.sh) unlearn TIGER trained on randomly permuted
        # SIDs (embeddings/<ds>_randsid, Beauty/Sports/Toys); a separate control, never a table candidate.
        if "randsid" in (rec.get("tag") or ""):
            continue
        keep = True
        for key, want in filters.items():
            got = rec.get(key)
            if isinstance(want, (set, frozenset, list, tuple)):
                keep = got in want
            else:
                keep = got == want
            if not keep:
                break
        if keep:
            yield rec


def last_metrics(run_dir: str) -> Optional[Dict[str, float]]:  # noqa: C901
    """Last non-empty value of every numeric column in the post-eval metrics.

    Prefers ``eval/csv/version_0/metrics.csv`` (what run_tiger_unlearn_sequential
    writes); falls back to any nested metrics.csv so a reference eval dir with a
    different layout still reads.
    """
    # Offline: serve runs the export covers, but FALL THROUGH for anything else
    # rather than returning None. Reference eval dirs are read through this same
    # function and are not in runs.csv, so short-circuiting here would silently
    # blank every sensitive reference while the spam tables looked fine.
    data = _run_data()
    if data is not None and run_dir in data["metrics"]:
        return data["metrics"][run_dir] or None
    path = os.path.join(run_dir, "eval", "csv", "version_0", "metrics.csv")
    if not os.path.isfile(path):
        hits = glob.glob(os.path.join(run_dir, "**", "metrics.csv"),
                         recursive=True)
        if not hits:
            return None
        hits.sort(key=lambda p: ("eval" not in p, len(p)))
        path = hits[0]
    best: Dict[str, float] = {}
    with open(path) as fh:
        for row in csv.DictReader(fh):
            for key, val in row.items():
                if val not in (None, ""):
                    try:
                        best[key] = float(val)
                    except (TypeError, ValueError):
                        pass
    return best or None

def track_token(model: str = DEFAULT_MODEL, quantizer: str = DEFAULT_QUANTIZER) -> str:
    """Filename token for a track. Empty for tiger+rkmeans, so the historical
    untokenised names stay valid for the track they actually describe."""
    tok = ""
    if model != DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    return tok


def track_output_path(path: str, model: str = DEFAULT_MODEL,
                      quantizer: str = DEFAULT_QUANTIZER) -> str:
    """Insert the track token into ``path`` before its extension.

    Every table is per-track: the three tracks are different identifier spaces
    with different winners, not three views of one result. Writing them all to
    one filename silently destroys the earlier track's table -- which is exactly
    what happened to tables/hpsearch_beauty_spam.md when the RQ-VAE run
    overwrote the RQ-KMeans one. This makes that unrepresentable: a non-default
    track always lands on its own file, and a path that already carries the
    token is left alone so repeated calls are idempotent.
    """
    tok = track_token(model, quantizer)
    if not tok:
        return path
    root, ext = os.path.splitext(path)
    if root.endswith(tok):
        return path
    return f"{root}{tok}{ext}"
