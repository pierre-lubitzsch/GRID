"""Export the numbers behind the paper's figures and tables as tidy CSVs.

The point is a bundle small enough to commit to the Overleaf repo, so a
collaborator can rebuild or restyle a figure or table WITHOUT cluster access to
`logs/`. Two kinds of file come out:

  mass_groups.csv     the raw per-(stratum, arm, group) log-mass behind
                      plot_mass_groups.py. `plot_mass_groups.py --data` reads
                      this back, so the published figure code runs unchanged off
                      the CSV.
  table_*.csv         one row per rendered table row, i.e. the numbers AFTER
                      arm selection and seed aggregation. Restyling and
                      re-typesetting work off these; re-deriving them from
                      different runs does not, and needs the cluster.

Everything is written under paper/data/. Re-running is idempotent.

    PYTHONPATH=. python scripts/export_paper_data.py
"""
from __future__ import annotations

import csv
import glob
import json
import os
import re
import sys

OUT = "paper/data"

# ---------------------------------------------------------------- figures --
def export_mass_groups() -> str:
    """Per-(stratum, arm, group) log-mass, the input to plot_mass_groups.py."""
    from scripts.plot_mass_balance import ARMS, ASCENT_ARM, ROOT
    from scripts.plot_mass_groups import POISONED_ARM

    # Arm -> the label the figure prints. The poisoned arm carries no label
    # because it is a REFERENCE (--vs-poisoned differences against it), not a
    # bar, but its row has to ship or that mode cannot be rebuilt.
    labels = {a: l for a, l in [ASCENT_ARM] + list(ARMS)}
    labels[POISONED_ARM] = ""

    path = os.path.join(OUT, "mass_groups.csv")
    n = 0
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["stratum", "arm", "arm_label", "group",
                    "base", "unlearned", "delta", "n_items"])
        for stratum in ("unpopular", "mid", "popular"):
            for arm in list(labels):
                f = os.path.join(ROOT, f"beauty_{stratum}_s2_{arm}",
                                 "collateral_drift.json")
                if not os.path.isfile(f):
                    continue
                m = (json.load(open(f)).get("group_log_mass") or {})
                for g in ("target", "nbr", "ctrl"):
                    v = m.get(g) or {}
                    if v.get("delta") is None:
                        continue
                    w.writerow([stratum, arm, labels[arm].replace("\n", " "), g,
                                v.get("base"), v.get("unlearned"),
                                v.get("delta"), v.get("n_items")])
                    n += 1
    print(f"  {path}  ({n} rows)")
    return path


# ------------------------------------------------------------------- runs --
# Exactly the fields run_tags.parse_run_dir produces, so a CSV-backed iter_runs
# hands back records indistinguishable from a directory scan and every
# generator filter keeps working unchanged.
RUN_FIELDS = ["jid", "tag", "path", "dataset", "model", "quantizer", "track",
              "setting", "stratum", "arm", "family", "algo", "seed",
              "replicate", "category", "pkm", "hpsearch", "done"]


def export_runs() -> str:
    """One row per finished unlearn run: parsed tag fields plus final metrics.

    This is the table generators' INPUT, not their output. With
    GRID_RUN_DATA=<this file> the scripts in scripts/extract_*.py run with no
    access to logs/, so the selection rule, the ranking and the captions can all
    be changed off-cluster and the .tex rebuilt.
    """
    from scripts import run_tags

    rows, keys = [], set()
    for rec in run_tags.iter_runs():
        if not rec.get("done"):
            continue
        m = run_tags.last_metrics(rec["path"])
        if not m:
            continue
        keys.update(m)
        rows.append((rec, m))

    cols = RUN_FIELDS + sorted(keys)
    path = os.path.join(OUT, "runs.csv")
    with open(path, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for rec, m in rows:
            w.writerow([rec.get(k, "") if rec.get(k) is not None else ""
                        for k in RUN_FIELDS]
                       + [m.get(k, "") for k in sorted(keys)])
    print(f"  {path}  ({len(rows)} runs, {len(keys)} metric columns)")
    return path


def export_references() -> str:
    """Spam references, reduced to the fields resolve_reference actually reads.

    The originals carry per-item provenance and run to 1.3MB; these are a few KB
    and answer every question the generators ask.
    """
    from scripts.extract_bandwagon_head2head import REF_DIR

    dest = os.path.join(OUT, "spam_references")
    os.makedirs(dest, exist_ok=True)
    n = 0
    for f in sorted(glob.glob(os.path.join(REF_DIR, "*.json"))):
        try:
            raw = json.load(open(f))
        except (OSError, json.JSONDecodeError):
            continue
        keep = {k: raw.get(k) for k in ("clean_ndcg@10", "clean_SH@10",
                                        "poison_SH@10", "poison_ndcg@10")}
        with open(os.path.join(dest, os.path.basename(f)), "w") as fh:
            json.dump(keep, fh, indent=1, sort_keys=True)
        n += 1
    print(f"  {dest}/  ({n} references)")
    return dest


def export_sensitive_references() -> str:
    """Resolve every sensitive base/retrain reference and dump the metrics.

    Resolution walks eval DIRECTORIES and validates coordinates.json against the
    cell, which cannot be shipped to a paper repo. So it is run here, once, and
    the ANSWER is exported: {token: {metrics, source}}. The offline branch in
    extract_sensitive.reference() reads this instead of re-implementing the
    matching, which is where the toys/babytoys mix-up came from.
    """
    import importlib
    from scripts import run_tags
    es = importlib.import_module("scripts.extract_sensitive")

    table = {}
    for ds, cat in es.CATEGORY.items():
        for model, quant in (("tiger", "rkmeans"), ("tiger", "rqvae"),
                             ("letter", "letter")):
            for seed in (2, 3, 5, 7, 11):
                for role in ("base", "retrain"):
                    met, src = es.reference(ds, role, model, quant, seed, cat)
                    if met:
                        table[es.ref_token(ds, role, model, quant, seed)] = {
                            "metrics": met, "source": src}
    path = os.path.join(OUT, "sensitive_references.json")
    with open(path, "w") as fh:
        json.dump(table, fh, indent=1, sort_keys=True)
    print(f"  {path}  ({len(table)} references)")
    return path


# ----------------------------------------------------------------- tables --
_CMD = re.compile(r"\\[a-zA-Z]+\s*(\{([^{}]*)\})?")
# Braceless commands worth keeping as words. Anything else is formatting and
# is dropped; the value it wraps is kept by the braced branch.
_SYMBOL = {"tau": "tau", "lambda": "lambda", "Delta": "Delta",
           "mathcal": "", "dagger": ""}
# \multicolumn{n}{spec}{body} / \multirow{n}{spec}{body}: the body is the third
# group, not the first, so these need stripping before the generic pass.
_SPAN = re.compile(r"\\multi(column|row)\{[^}]*\}\{[^}]*\}\{(.*)\}\s*$")
# \rotatebox[origin=c]{90}{Unpopular}: an OPTIONAL arg plus two braced ones.
# The generic pass keeps the first brace, leaving "[origin=c]90Unpopular".
_ROT = re.compile(r"\\rotatebox(\[[^\]]*\])?\{[^}]*\}\{([^{}]*)\}")


def _cells(line: str):
    r"""Split one LaTeX row into cells, dropping the trailing \\\\."""
    return [c.strip() for c in
            re.split(r"(?<!\\)&", line.rstrip().rstrip("\\").rstrip())]


def _clean(cell: str) -> str:
    r"""Strip LaTeX markup from a rendered cell, keeping the text and numbers.

    Applied repeatedly because the markup nests: \textbf{\underline{1.0}} needs
    two passes, and a single pass would leave a stray brace that then reads as
    part of the value.
    """
    c = cell.replace(r"$\pm$", " +- ").replace(r"$^{\dagger}$", " [floor-miss]")
    c = c.replace("\\ ", " ")      # "\ " is an escaped space, not a command
    c = _SPAN.sub(lambda m: m.group(2), c)
    c = _ROT.sub(lambda m: m.group(2), c)
    def _sub(m):
        if m.group(2) is not None:          # \cmd{body} -> body
            return m.group(2)
        name = m.group(0).lstrip("\\").strip()
        return _SYMBOL.get(name, " ")       # \tau -> tau, \textbf -> ""
    for _ in range(4):
        c2 = _CMD.sub(_sub, c)
        if c2 == c:
            break
        c = c2
    c = (c.replace("$", "").replace(r"\%", "%").replace(r"\_", "_")
          .replace("{", "").replace("}", ""))
    return re.sub(r"\s+", " ", c).strip()


def export_tables() -> list:
    r"""Every tables/*.tex as a CSV of its rendered rows.

    Reading the .tex rather than re-running the generators is deliberate: it
    guarantees the CSV and the typeset table carry the SAME numbers, which
    re-deriving cannot, and it keeps this exporter independent of eight
    different generator entry points.

    Structure is recovered rather than guessed: the header is the row after
    \toprule (it cannot be found by "has no digits" -- SH@10 has one), and a
    full-width \multicolumn is a SECTION heading (Baselines / Ours), which
    becomes a column so the grouping survives into the CSV.
    """
    written = []
    for tex in sorted(glob.glob("tables/*.tex")):
        header, rows, section, group = None, [], "", ""
        seen_toprule = False
        for line in open(tex):
            s = line.strip()
            if s.startswith(r"\toprule"):
                seen_toprule = True
                continue
            if not seen_toprule or s.startswith("%"):
                continue
            # The \multirow row label (the stratum) is on its own line and does
            # NOT end in \\, so it has to be captured before the row filter
            # drops it -- otherwise the Target column is empty on every row.
            if r"\multirow" in s and not s.endswith(r"\\"):
                group = _clean(s)
                continue
            if not s.endswith(r"\\"):
                continue
            cells = [_clean(c) for c in _cells(s)]
            if not any(cells):
                continue
            # A full-width \multicolumn with exactly one non-empty cell is a
            # SECTION heading (Baselines / Ours), not a data row. Detected on the
            # cleaned cells rather than by anchoring a regex at the line start,
            # because the row begins with the empty spanned Target column.
            if r"\multicolumn" in s and sum(1 for c in cells if c) == 1:
                section = next(c for c in cells if c)
                continue
            if header is None:
                header = cells
                continue
            if group and not cells[0]:
                cells[0] = group        # fill the spanned stratum column
            rows.append((section, cells))
        if not rows or header is None:
            continue
        width = max(max(len(c) for _s, c in rows), len(header))
        has_sec = any(s for s, _c in rows)
        path = os.path.join(OUT, "tables", os.path.basename(tex)[:-4] + ".csv")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", newline="") as fh:
            w = csv.writer(fh)
            head = header + [f"col{i}" for i in range(len(header), width)]
            w.writerow((["section"] if has_sec else []) + head)
            for sec, r in rows:
                w.writerow(([sec] if has_sec else []) + r + [""] * (width - len(r)))
        written.append(path)
    print(f"  {OUT}/tables/  ({len(written)} tables)")
    return written


def export_results() -> str:
    """ONE wide CSV with every number the paper's tables report.

    Assembled from the generated .tex, so the CSV and the typeset table cannot
    disagree. Keys (artifact, dataset, track, setting, stratum, method) come from
    the filename and the row; each metric is split into value and spread so the
    numbers are usable without parsing "1.059 +- 0.947".

    This is the READ-ONLY view: right for looking a number up, quoting it in
    prose, or plotting it. Changing which runs a table selects still needs
    runs.csv plus the reference files.
    """
    import re as _re

    def parts(stem):
        """paper_spam_beauty_tiger_rqvae -> (artifact, dataset, track)."""
        ds = next((d for d in ("beauty", "sports", "toys", "food_k5")
                   if f"_{d}" in f"_{stem}"), "")
        track = ""
        for t in ("tiger_rkmeans", "tiger_rqvae", "tiger_tracerv3",
                  "letter_letter", "rqvae", "tracerv3"):
            if stem.endswith("_" + t):
                track = t.replace("tiger_", "") if t.startswith("tiger_") else t
                stem = stem[: -len(t) - 1]
                break
        art = stem.replace(f"_{ds}", "") if ds else stem
        return art, ds, (track or "rkmeans")

    def num(cell):
        """'96.5 +- 6.5' -> (96.5, 6.5); '--' -> ('', '')."""
        if not cell or cell in ("--", "n/a", "not run"):
            return "", ""
        m = _re.match(r"^\s*([+-]?[\d.]+)\s*(?:\+-\s*([\d.]+))?", cell)
        if not m:
            return "", ""
        return m.group(1), (m.group(2) or "")

    # column header (already LaTeX-stripped) -> canonical name
    CANON = {"SH@10 (%)": "sh10", "SH@10": "sh10", "SHF@10 (%)": "shf10",
             "SHF@10": "shf10", "SHR@10 (%)": "shr10", "SHR@10": "shr10",
             "NDCG@10 (%)": "ndcg10", "NDCG@10": "ndcg10", "N@10 (%)": "ndcg10",
             "tau_P (%)": "tau_P", "tau_P": "tau_P", "tau_F": "tau_F",
             "Deltatau_P (%)": "delta_tau_P", "E (s)": "E_s",
             "RelItemsF@10": "rel_items_f10", "tau_E": "tau_E"}
    METRICS = ["sh10", "shf10", "shr10", "ndcg10", "tau_P", "delta_tau_P",
               "tau_F", "rel_items_f10", "E_s", "tau_E"]

    rows = []
    for tex in sorted(glob.glob("tables/*.tex")):
        stem = os.path.basename(tex)[:-4]
        art, ds, track = parts(stem)
        header, section, group = None, "", ""
        seen = False
        for line in open(tex):
            t = line.strip()
            if t.startswith(r"\toprule"):
                seen = True
                continue
            if not seen or t.startswith("%"):
                continue
            if r"\multirow" in t and not t.endswith(r"\\"):
                group = _clean(t)
                continue
            if not t.endswith(r"\\"):
                continue
            cells = [_clean(c) for c in _cells(t)]
            if not any(cells):
                continue
            if r"\multicolumn" in t and sum(1 for c in cells if c) == 1:
                section = next(c for c in cells if c)
                continue
            if header is None:
                header = cells
                continue
            if group and not cells[0]:
                cells[0] = group
            got = {"artifact": art, "dataset": ds, "track": track,
                   "section": section, "stratum": "", "method": "",
                   "source_table": stem}
            for h, c in zip(header, cells):
                key = CANON.get(h)
                if key:
                    got[key], got[key + "_sd"] = num(c)
                elif h in ("Target", "dataset"):
                    got["stratum"] = c
                elif h in ("Method", "algorithm", "Loss terms", "lambda_n"):
                    got["method"] = c
            if got["method"] or got["stratum"]:
                rows.append(got)

    cols = (["artifact", "dataset", "track", "setting", "stratum", "section",
             "method"]
            + [c for m in METRICS for c in (m, m + "_sd")] + ["source_table"])
    path = os.path.join(OUT, "results.csv")
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            r["setting"] = ("sensitive" if "sensitive" in r["artifact"]
                            else "spam")
            w.writerow(r)
    print(f"  {path}  ({len(rows)} rows from {len(glob.glob('tables/*.tex'))} tables)")
    return path


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    print("exporting:")
    export_mass_groups()
    export_runs()
    export_references()
    export_sensitive_references()
    export_tables()
    export_results()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
