"""Compute relative utility metrics for TIGER unlearning runs.

Reads the ``metrics.csv`` files produced by Lightning's ``CSVLogger`` from
multiple ``trainer.test(...)`` runs and prints / saves ``util(unlearned) /
util(reference)`` for every ``test/*`` metric in common.

Typical usage
-------------

::

    python -m scripts.compute_relative_utility \\
        --reference logs/train/runs/2026-05-06/clean/csv/version_0/metrics.csv \\
        --unlearned logs/unlearn/runs/2026-05-06/scif_baseline/csv/version_0/metrics.csv \\
        --label_unlearned scif_baseline \\
        --extra logs/unlearn/runs/2026-05-06/scif_neighborhood/csv/version_0/metrics.csv \\
        --label_extra scif_neighborhood \\
        --extra logs/train/runs/2026-05-06/poisoned/csv/version_0/metrics.csv \\
        --label_extra poisoned \\
        --out_json out/relative_utility.json

How to generate the CSVs
------------------------

For every checkpoint (clean reference, poisoned, each unlearned variant) run
the same ``tiger_train_flat`` experiment with the SCIF-friendly ``train=False
test=True`` overrides::

    python -m src.train experiment=tiger_train_flat \\
        train=False test=True \\
        data_dir=src/data/amazon_data/beauty \\
        semantic_id_path=.../merged_predictions_tensor.pt \\
        ckpt_path=<the_ckpt> \\
        num_hierarchies=4

(``src/train.py`` will skip ``trainer.fit`` and call ``trainer.test`` against
the ckpt; the resulting ``test/*`` metrics land in
``${paths.output_dir}/csv/version_0/metrics.csv``.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
from typing import Dict, List, Optional, Tuple


def _read_test_metrics(csv_path: str) -> Dict[str, float]:
    """Return the last non-NaN value of every ``test/*`` column in ``csv_path``."""
    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"metrics.csv not found at {csv_path!r}")
    out: Dict[str, float] = {}
    with open(csv_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"{csv_path!r} has no header row")
        test_cols = [c for c in reader.fieldnames if c.startswith("test/")]
        if not test_cols:
            raise ValueError(
                f"No test/* columns in {csv_path!r}; got {reader.fieldnames}"
            )
        last: Dict[str, float] = {c: math.nan for c in test_cols}
        for row in reader:
            for col in test_cols:
                cell = row.get(col, "")
                if cell == "" or cell is None:
                    continue
                try:
                    val = float(cell)
                except ValueError:
                    continue
                if math.isnan(val):
                    continue
                last[col] = val
        out = {k: v for k, v in last.items() if not math.isnan(v)}
    if not out:
        raise ValueError(f"{csv_path!r} has no non-NaN test/* metric rows")
    return out


def _ratio(num: float, den: float) -> Optional[float]:
    if den == 0 or math.isnan(den) or math.isnan(num):
        return None
    return num / den


def compute_relative_utility(
    reference_csv: str,
    runs: List[Tuple[str, str]],
) -> Dict[str, Dict[str, Optional[float]]]:
    """Read every CSV in ``runs`` plus ``reference_csv`` and emit ratios.

    Parameters
    ----------
    reference_csv
        Path to the metrics.csv produced by the clean / retrained reference
        ckpt.
    runs
        List of ``(label, csv_path)`` tuples. The label is used as the key
        in the output dict.
    """
    reference_metrics = _read_test_metrics(reference_csv)
    results: Dict[str, Dict[str, Optional[float]]] = {
        "_reference_path": reference_csv,  # type: ignore[dict-item]
        "_reference_metrics": reference_metrics,  # type: ignore[dict-item]
    }
    for label, csv_path in runs:
        run_metrics = _read_test_metrics(csv_path)
        ratios: Dict[str, Optional[float]] = {}
        common = sorted(set(reference_metrics).intersection(run_metrics))
        for metric in common:
            ratios[metric] = _ratio(run_metrics[metric], reference_metrics[metric])
        results[label] = {  # type: ignore[assignment]
            "csv_path": csv_path,
            "metrics": run_metrics,
            "relative_utility": ratios,
        }
    return results


def _print_table(results: Dict[str, Dict[str, Optional[float]]]) -> None:
    ref = results.get("_reference_metrics", {})
    metric_keys = sorted(ref.keys())
    if not metric_keys:
        print("(no reference metrics)")
        return
    labels = [k for k in results if not k.startswith("_")]
    header = ["metric", "reference"] + labels + [f"rel({lab})" for lab in labels]
    rows: List[List[str]] = []
    for m in metric_keys:
        row = [m, _fmt(ref.get(m))]
        for lab in labels:
            row.append(_fmt(results[lab]["metrics"].get(m)))
        for lab in labels:
            ratios = results[lab]["relative_utility"]
            row.append(_fmt(ratios.get(m), is_ratio=True))
        rows.append(row)
    widths = [max(len(r[i]) for r in [header] + rows) for i in range(len(header))]
    fmt = "  ".join(f"{{:<{w}}}" for w in widths)
    print(fmt.format(*header))
    print(fmt.format(*("-" * w for w in widths)))
    for row in rows:
        print(fmt.format(*row))


def _fmt(value: Optional[float], is_ratio: bool = False) -> str:
    if value is None:
        return "n/a"
    if not isinstance(value, (int, float)) or math.isnan(value):
        return "n/a"
    return f"{value:.4f}" if not is_ratio else f"{value:.4f}"


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Compute relative utility = util(unlearned) / util(reference) for "
            "every test/* metric across multiple Lightning CSVLogger runs."
        )
    )
    p.add_argument(
        "--reference",
        required=True,
        help="metrics.csv from the clean / retrained reference ckpt",
    )
    p.add_argument(
        "--unlearned",
        default=None,
        help="metrics.csv from the primary unlearned-model run",
    )
    p.add_argument(
        "--label_unlearned",
        default="unlearned",
        help="Label for the primary unlearned run.",
    )
    p.add_argument(
        "--run",
        action="append",
        default=[],
        help=(
            "Additional run as label:metrics.csv path (repeatable). When any "
            "--run is set, --unlearned is optional and all runs are compared "
            "against --reference."
        ),
    )
    p.add_argument(
        "--extra",
        action="append",
        default=[],
        help=(
            "Extra metrics.csv paths to compare (e.g. poisoned, neighborhood-aware "
            "variant); repeatable."
        ),
    )
    p.add_argument(
        "--label_extra",
        action="append",
        default=[],
        help="Labels for --extra paths, in the same order; repeatable.",
    )
    p.add_argument(
        "--out_json",
        default=None,
        help="Optional path to dump the full results dict as JSON.",
    )
    return p


def _resolve_extras(
    extras: List[str], labels: List[str]
) -> List[Tuple[str, str]]:
    if labels and len(labels) != len(extras):
        raise ValueError(
            f"--label_extra count ({len(labels)}) must match --extra count "
            f"({len(extras)})"
        )
    out: List[Tuple[str, str]] = []
    for i, path in enumerate(extras):
        label = labels[i] if i < len(labels) else f"extra_{i}"
        out.append((label, path))
    return out


def _parse_run_specs(specs: List[str]) -> List[Tuple[str, str]]:
    out: List[Tuple[str, str]] = []
    for spec in specs:
        if ":" not in spec:
            raise ValueError(
                f"Invalid --run {spec!r}; expected label:path/to/metrics.csv"
            )
        label, path = spec.split(":", 1)
        label = label.strip()
        path = path.strip()
        if not label or not path:
            raise ValueError(f"Invalid --run {spec!r}; label and path must be non-empty")
        out.append((label, path))
    return out


def main() -> None:
    args = _build_arg_parser().parse_args()
    if args.run:
        runs = _parse_run_specs(args.run)
    else:
        if not args.unlearned:
            raise SystemExit("Either --unlearned or at least one --run is required.")
        runs = [(args.label_unlearned, args.unlearned)]
        runs.extend(_resolve_extras(args.extra, args.label_extra))

    results = compute_relative_utility(reference_csv=args.reference, runs=runs)
    _print_table(results)

    if args.out_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.out_json)) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nWrote results to {args.out_json}")


if __name__ == "__main__":
    main()
