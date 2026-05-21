"""Collect test metrics from finished unlearning runs (no re-eval).

Sequential unlearning via ``run_tiger_unlearn_sequential.sh`` can write
post-unlearn ``trainer.test`` results under::

    <run_dir>/eval/csv/version_0/metrics.csv

This script discovers those CSVs (and optionally merges baseline metrics from
an existing ``run_tiger_eval_batch_sweep`` output tree), then prints the same
relative-utility table as ``scripts.compute_relative_utility``.

Usage::

    python -m scripts.collect_unlearn_eval_table \\
        --reference logs/eval/batch_sweep/2026-05-20_16-44-14/clean_ref/csv/version_0/metrics.csv \\
        --poisoned logs/eval/batch_sweep/2026-05-20_16-44-14/poisoned/csv/version_0/metrics.csv \\
        --runs-root logs/unlearn/runs \\
        --batch-sweep-dir logs/eval/batch_sweep/2026-05-20_16-44-14 \\
        --out-dir logs/eval/collected/$(date +%Y-%m-%d_%H-%M-%S)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from scripts.compute_relative_utility import (
    _print_table,
    compute_relative_utility,
)

RUN_DIR_RE = re.compile(r"^job\d+")
BS_RE = re.compile(r"_bs(\d+)(?:_|$)")


@dataclass
class RunEvalRecord:
    label: str
    metrics_csv: str
    run_dir: str
    request_batch_size: Optional[int]
    neighborhood_aware: Optional[bool]
    algorithm: Optional[str]
    source: str  # "embedded_eval" | "batch_sweep"


def _parse_batch_size(dirname: str) -> Optional[int]:
    matches = BS_RE.findall(dirname)
    if not matches:
        return None
    return int(matches[-1])


def _read_run_info(run_dir: str) -> Dict:
    for name in ("unlearn_info.json", "scif_info.json"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            try:
                with open(path) as f:
                    return json.load(f)
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _algorithm_for_run(run_dir: str, dirname: str) -> Optional[str]:
    info = _read_run_info(run_dir)
    if info.get("algorithm"):
        return str(info["algorithm"])
    cfg = info.get("unlearning_cfg") or {}
    if cfg.get("algorithm"):
        return str(cfg["algorithm"])
    for prefix in ("finetune", "neg_train", "filter", "unified", "scif"):
        if prefix in dirname:
            return prefix
    return None


def _is_neighborhood_run(run_dir: str, dirname: str) -> Optional[bool]:
    info = _read_run_info(run_dir)
    unlearning_cfg = info.get("unlearning_cfg") or {}
    if "neighborhood_aware" in unlearning_cfg:
        return bool(unlearning_cfg["neighborhood_aware"])
    if "neighborhood_aware" in info:
        return bool(info["neighborhood_aware"])
    if "_nbh_" in dirname or dirname.endswith("_nbh"):
        return True
    return False


def _embedded_metrics_path(run_dir: str) -> str:
    return os.path.join(run_dir, "eval", "csv", "version_0", "metrics.csv")


def _label_for_run(
    dirname: str,
    neighborhood: Optional[bool],
    batch_size: Optional[int],
    algorithm: Optional[str] = None,
) -> str:
    bs = batch_size if batch_size is not None else "?"
    algo = algorithm or "scif"
    if neighborhood:
        return f"{algo}_nbh_bs{bs}"
    return f"{algo}_base_bs{bs}"


def _discover_embedded_evals(runs_root: str) -> List[RunEvalRecord]:
    if not os.path.isdir(runs_root):
        raise FileNotFoundError(f"runs_root not found: {runs_root}")

    found: List[RunEvalRecord] = []
    for name in sorted(os.listdir(runs_root)):
        if not RUN_DIR_RE.match(name):
            continue
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        metrics_csv = _embedded_metrics_path(run_dir)
        if not os.path.isfile(metrics_csv):
            continue
        bs = _parse_batch_size(name)
        nbh = _is_neighborhood_run(run_dir, name)
        algo = _algorithm_for_run(run_dir, name)
        found.append(
            RunEvalRecord(
                label=_label_for_run(name, nbh, bs, algo),
                metrics_csv=os.path.abspath(metrics_csv),
                run_dir=os.path.abspath(run_dir),
                request_batch_size=bs,
                neighborhood_aware=nbh,
                algorithm=algo,
                source="embedded_eval",
            )
        )
    return found


def _discover_batch_sweep_fallback(
    sweep_dir: str,
    *,
    existing_labels: set,
    only_baseline: bool = True,
) -> List[RunEvalRecord]:
    """Use seq_bs* / poisoned / clean_ref trees from a prior batch-sweep eval."""
    if not os.path.isdir(sweep_dir):
        raise FileNotFoundError(f"batch-sweep-dir not found: {sweep_dir}")

    out: List[RunEvalRecord] = []
    for name in sorted(os.listdir(sweep_dir)):
        sub = os.path.join(sweep_dir, name)
        if not os.path.isdir(sub):
            continue
        if only_baseline and not name.startswith("seq_bs"):
            continue
        if not name.startswith("seq_bs"):
            continue
        metrics_csv = os.path.join(sub, "csv", "version_0", "metrics.csv")
        if not os.path.isfile(metrics_csv):
            continue
        m = re.match(r"seq_bs(\d+)$", name)
        if not m:
            continue
        bs = int(m.group(1))
        label = f"base_bs{bs}_sweep"
        if label in existing_labels:
            continue
        out.append(
            RunEvalRecord(
                label=label,
                metrics_csv=os.path.abspath(metrics_csv),
                run_dir=sub,
                request_batch_size=bs,
                neighborhood_aware=False,
                algorithm="scif",
                source="batch_sweep",
            )
        )
    return out


def _dedupe_latest(records: List[RunEvalRecord]) -> List[RunEvalRecord]:
    """Keep the newest run per label (by metrics file mtime)."""
    best: Dict[str, RunEvalRecord] = {}
    best_mtime: Dict[str, float] = {}
    for rec in records:
        mtime = os.path.getmtime(rec.metrics_csv)
        if rec.label not in best or mtime > best_mtime[rec.label]:
            best[rec.label] = rec
            best_mtime[rec.label] = mtime
    return [best[k] for k in sorted(best.keys(), key=lambda x: (int(BS_RE.search(x).group(1)) if BS_RE.search(x) else 0, x))]


def _scan_missing_evals(runs_root: str) -> List[Dict[str, object]]:
    missing: List[Dict[str, object]] = []
    for name in sorted(os.listdir(runs_root)):
        if not RUN_DIR_RE.match(name):
            continue
        run_dir = os.path.join(runs_root, name)
        if not os.path.isdir(run_dir):
            continue
        ckpt = os.path.join(run_dir, "checkpoints", "unlearned.ckpt")
        if not os.path.isfile(ckpt) and not os.path.islink(ckpt):
            continue
        if os.path.isfile(_embedded_metrics_path(run_dir)):
            continue
        missing.append(
            {
                "run_dir": os.path.abspath(run_dir),
                "request_batch_size": _parse_batch_size(name),
                "neighborhood_aware": _is_neighborhood_run(run_dir, name),
                "has_unlearned_ckpt": True,
            }
        )
    return missing


def collect(
    *,
    reference_csv: str,
    poisoned_csv: Optional[str],
    runs_root: str,
    batch_sweep_dir: Optional[str],
    use_sweep_fallback: bool,
    out_dir: Optional[str],
) -> Dict[str, object]:
    embedded = _discover_embedded_evals(runs_root)
    records = list(embedded)

    if use_sweep_fallback and batch_sweep_dir:
        records.extend(
            _discover_batch_sweep_fallback(
                batch_sweep_dir,
                existing_labels={r.label for r in records},
                only_baseline=True,
            )
        )

    records = _dedupe_latest(records)
    missing = _scan_missing_evals(runs_root)

    runs_for_table: List[Tuple[str, str]] = []
    if poisoned_csv:
        runs_for_table.append(("poisoned", poisoned_csv))
    runs_for_table.extend((r.label, r.metrics_csv) for r in records)

    results = compute_relative_utility(reference_csv=reference_csv, runs=runs_for_table)
    _print_table(results)

    payload: Dict[str, object] = {
        "created_at": datetime.utcnow().isoformat() + "Z",
        "reference_csv": os.path.abspath(reference_csv),
        "poisoned_csv": os.path.abspath(poisoned_csv) if poisoned_csv else None,
        "runs_root": os.path.abspath(runs_root),
        "batch_sweep_dir": (
            os.path.abspath(batch_sweep_dir) if batch_sweep_dir else None
        ),
        "records": [
            {
                "label": r.label,
                "metrics_csv": r.metrics_csv,
                "run_dir": r.run_dir,
                "request_batch_size": r.request_batch_size,
                "neighborhood_aware": r.neighborhood_aware,
                "source": r.source,
            }
            for r in records
        ],
        "missing_embedded_eval": missing,
        "relative_utility": {
            k: v for k, v in results.items() if not k.startswith("_")
        },
        "_reference_metrics": results.get("_reference_metrics"),
    }

    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        json_path = os.path.join(out_dir, "relative_utility.json")
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)
        manifest_path = os.path.join(out_dir, "manifest.txt")
        with open(manifest_path, "w") as f:
            f.write(f"reference={reference_csv}\n")
            if poisoned_csv:
                f.write(f"poisoned={poisoned_csv}\n")
            for r in records:
                f.write(f"{r.label}={r.metrics_csv}  # {r.source}\n")
            if missing:
                f.write("\n# Finished unlearn runs without embedded eval:\n")
                for m in missing:
                    f.write(f"missing={m['run_dir']}\n")
        print(f"\nWrote {json_path}")
        print(f"Wrote {manifest_path}")

    if missing:
        print(
            f"\nNote: {len(missing)} finished unlearn run(s) have unlearned.ckpt but no "
            f"eval/csv/version_0/metrics.csv (post-eval was skipped or failed)."
        )
        print("  Re-enable with UNLEARN_RUN_POST_EVAL=true in run_tiger_unlearn_sequential.sh")
        if use_sweep_fallback and batch_sweep_dir:
            print(f"  Baseline metrics may still appear from --batch-sweep-dir {batch_sweep_dir}")

    if not records:
        print("\nNo unlearn eval CSVs found to compare.")

    return payload


def _build_arg_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description=(
            "Build a relative-utility table from existing post-unlearn eval "
            "metrics under logs/unlearn/runs (no GPU re-eval)."
        )
    )
    p.add_argument(
        "--reference",
        required=True,
        help="metrics.csv for the clean reference checkpoint",
    )
    p.add_argument(
        "--poisoned",
        default=None,
        help="Optional metrics.csv for the poisoned (pre-unlearn) checkpoint",
    )
    p.add_argument(
        "--runs-root",
        default="logs/unlearn/runs",
        help="Root directory containing job* unlearning run folders",
    )
    p.add_argument(
        "--batch-sweep-dir",
        default=None,
        help=(
            "Optional prior run_tiger_eval_batch_sweep output; used as fallback "
            "for baseline seq_bs* when embedded eval/ is missing"
        ),
    )
    p.add_argument(
        "--no-sweep-fallback",
        action="store_true",
        help="Do not pull baseline metrics from --batch-sweep-dir",
    )
    p.add_argument(
        "--out-dir",
        default=None,
        help="Directory for relative_utility.json and manifest.txt",
    )
    return p


def main() -> None:
    args = _build_arg_parser().parse_args()
    collect(
        reference_csv=args.reference,
        poisoned_csv=args.poisoned,
        runs_root=args.runs_root,
        batch_sweep_dir=args.batch_sweep_dir,
        use_sweep_fallback=not args.no_sweep_fallback,
        out_dir=args.out_dir,
    )


if __name__ == "__main__":
    main()
