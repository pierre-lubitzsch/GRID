"""Unwanted-item canonical executions from paper/plotting/canonical_runs.tsv (the `_mia` rows).

Shared by the --ledger options of valsens_table.py (Table 8), make_unwanted_tables.py
(Tables 3/22/23) and make_unwanted_lambda_n_table.py (Table 4). The method of a row is the
ledger's `method` column (written by scripts/build_canonical_runs.py from the run's RESOLVED
config), never run_tags.family_of, which parses `_mia` tags as negtrain_abl (Forget only would
be taken for Forget+Repair).

Run dicts have exactly the fields of extract_sensitive.collect() (built by
extract_sensitive.cell_entry), so every downstream aggregate is unchanged.

VIEWS (protocol A, paper/revision/sensitive_cleaned_protocol.md). set_view(view, *es_modules):
  None      legacy: test metrics from <run>/eval, references as extract_sensitive resolves them
            (GRID_SENSITIVE_REF_ROOT / GRID_SENSITIVE_BASE_REF_ROOT); only the `_mia` rows.
  "normal"  same test metrics; additionally the `unwanted_canonck` rows (canonck_runs()).
  "scrub"   test metrics from the CLEANED test histories, logs/eval/sensitive_scrub/scrub/runs/<run
            basename>/eval (run_sensitive_scrub_eval.sh); original and retrained from
            logs/eval/sensitive_scrub/scrub/reference (original = the exact start checkpoint of the
            runs, i.e. GRID_SENSITIVE_BASE_REF_ROOT=logs/eval/sensitive_reference_lastckpt semantics).
Validation metrics (<run>/eval_val, normal validation view) are the same in every view.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import re
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

_spec = importlib.util.spec_from_file_location("es", "scripts/extract_sensitive.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)

# ledger method -> family key used by the unwanted-item table code
FAM = {"Finetune": "finetune_abl", "Forget only": "ga_abl", "Forget+Repair": "negtrain_abl",
       "SCIF": "scif", "SEIF": "seif", "Kookmin": "kookmin", "Fanchuan": "fanchuan",
       "TRACER": "tracer", "NAU": "unified"}

_REFS: dict = {}
VIEW = None
SCRUB_ROOT = "logs/eval/sensitive_scrub/scrub"
VIEW_SUFFIX = {None: "_ledger", "normal": "_normal_ledger", "scrub": "_scrub_ledger"}


def set_view(view, *es_modules):
    """Select the test view for this module and every extract_sensitive instance given (each table
    script loads its own copy). scrub: references from the cleaned-view reference evals."""
    global VIEW
    assert view in VIEW_SUFFIX, view
    VIEW = view
    _REFS.clear()
    for m in (es, *es_modules):
        if not hasattr(m, "_REF_ROOTS_ENV"):
            m._REF_ROOTS_ENV = (m.REF_ROOT, m.BASE_REF_ROOT)
        if view == "scrub":
            m.REF_ROOT = os.path.join(SCRUB_ROOT, "reference")
            m.BASE_REF_ROOT = ""        # the scrub original IS the start checkpoint
        else:
            m.REF_ROOT, m.BASE_REF_ROOT = m._REF_ROOTS_ENV


def test_metrics(run_dir):
    """Test metrics of one unlearning run on the selected view (None if that eval is missing)."""
    if VIEW == "scrub":
        return _metrics(os.path.join(SCRUB_ROOT, "runs", os.path.basename(run_dir), "eval"))
    return run_tags.last_metrics(run_dir)


def refs_for(ds, seed, model="tiger", quantizer="rkmeans"):
    """(base, retr, base_p, retr_p) of one cell, as extract_sensitive.collect keys them."""
    k = (ds, seed, model, quantizer)
    if k not in _REFS:
        base, bp = es.reference(ds, "base", model, quantizer, seed, es.CATEGORY[ds])
        retr, rp = es.reference(ds, "retrain", model, quantizer, seed, es.CATEGORY[ds])
        _REFS[k] = (base, retr, bp, rp)
    return _REFS[k]


def mia_rows(ledger):
    return [r for r in csv.DictReader(open(ledger), delimiter="\t") if r["setting"] == "unwanted"]


def mia_runs(ledger, ds, sub="eval"):
    """{(family, config): [run dict per seed]} of the canonical `_mia` executions of ds.
    sub="eval" reads the test metrics, sub="eval_val" the validation metrics (the run dict
    is then only meaningful against validation references; see valsens_table.py)."""
    out = defaultdict(list)
    for r in mia_rows(ledger):
        if r["dataset"] != ds:
            continue
        sd = int(r["seed"])
        base, retr, _bp, _rp = refs_for(ds, sd)
        p = os.path.join(r["run_dir"], sub)
        m = test_metrics(r["run_dir"]) if sub == "eval" else _metrics(p)
        jid = re.match(r"job(\d+)_", os.path.basename(r["run_dir"])).group(1)
        e = es.cell_entry(m, base or {}, retr, sd, None, jid, r["run_dir"]) if base is not None else None
        if e is None:
            print(f"  ledger _mia row without usable {sub} metrics: {ds} {r['config']} s{sd} {r['run_dir']}")
            continue
        out[(FAM[r["method"]], r["config"])].append(e)
    for k in out:
        out[k].sort(key=lambda e: e["seed"])
    return out


def canonck_rows(ledger, ds=None, track=None):
    return [r for r in csv.DictReader(open(ledger), delimiter="\t") if r["setting"] == "unwanted_canonck"
            and (ds is None or r["dataset"] == ds) and (track is None or r["track"] == track)]


def canonck_runs(ledger, ds, track="tiger+rkmeans", sub="eval"):
    """{config: [run dict per seed]} of the `unwanted_canonck` executions (Table 4 / Table 8 rows
    re-run with checkpoint) of one dataset and track, on the selected view."""
    model, quant = track.split("+")
    out = defaultdict(list)
    for r in canonck_rows(ledger, ds, track):
        if r["status"] != "ok":
            print(f"  ledger _canonck row {r['status']}: {ds} {track} {r['config']} s{r['seed']}")
            continue
        sd = int(r["seed"])
        base, retr, _bp, _rp = refs_for(ds, sd, model, quant)
        m = test_metrics(r["run_dir"]) if sub == "eval" else _metrics(os.path.join(r["run_dir"], sub))
        jid = re.match(r"job(\d+)_", os.path.basename(r["run_dir"])).group(1)
        e = es.cell_entry(m, base or {}, retr, sd, None, jid, r["run_dir"]) if base is not None else None
        if e is None:
            print(f"  ledger _canonck row without usable {sub} metrics ({VIEW} view): {ds} {r['config']} s{sd}")
            continue
        out[r["config"]].append(e)
    for k in out:
        out[k].sort(key=lambda e: e["seed"])
    return out


def _metrics(d):
    p = os.path.join(d, "csv", "version_0", "metrics.csv")
    if not os.path.isfile(p):
        return None
    rows = [r for r in csv.DictReader(open(p)) if r.get("test/ndcg@10")]
    return {k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith("test/")} if rows else None


def nau_ln(config):
    """'unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0' -> '-1.0'."""
    m = re.search(r"_ln(neg)?([0-9.]+)$", config)
    return ("-" if m.group(1) else "") + m.group(2)
