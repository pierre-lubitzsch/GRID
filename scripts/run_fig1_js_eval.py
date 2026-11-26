"""Resolve matched Figure 1 checkpoints and launch the JS evaluator."""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

from plan_frozen_sweep import find_train
from plan_reference_evals import ckpt_of

ROOT = Path(__file__).resolve().parents[1]
LEDGER = ROOT / "paper/plotting/fig1_js_submissions.tsv"
# Item content embeddings the RQ-KMeans SIDs were built from (used by --neighbor-def knn<K>).
EMBEDDINGS = {"beauty": "embeddings/beauty_merged_predictions_tensor_latest.pt",
              "sports": "logs/inference/runs/embeddings/sports_10321537/pickle/merged_predictions_tensor.pt",
              "toys": "logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"}
METHODS = ("nau", "forget_repair", "forget_only", "scif")
# Scored when checkpoints exist, skipped when they do not. Beauty has
# retain-only weights from the keepck sweep; sports and toys do not, and
# requiring them would break those cells for no gain.
OPTIONAL_METHODS = ("fanchuan", "kookmin", "nau_ln0p1", "nau_s0p01_n0p01", "nosep_ln0", "nosep_ln0p01", "nosep_ln0p1", "retain_only", "seif", "sep_ln0", "sep_ln0p01",)


def training_checkpoint(job_id: str) -> str:
    paths = glob.glob(str(ROOT / "logs/train/runs/*" / f"*job{job_id}_*"))
    if len(paths) != 1:
        raise RuntimeError(f"job {job_id}: expected one training run, got {paths}")
    result = ckpt_of(paths[0])
    if not result:
        raise FileNotFoundError(f"job {job_id}: no training checkpoint")
    return str(Path(result).resolve())


def unlearned_checkpoints(dataset: str, seed: int,
                          stratum: str = "mid") -> dict[str, str]:
    with LEDGER.open(newline="") as file:
        rows = [r for r in csv.DictReader(file, delimiter="\t")
                if r["dataset"] == dataset and int(r["seed"]) == seed
                and r["stratum"] == stratum]
    jobs = {r["arm"]: r["slurm_job_id"] for r in rows}
    # METHODS are the Figure-1 baselines, which were only ever run on mid. The
    # off-mid strata carry the ablation arms alone, so requiring METHODS there
    # would reject a complete ablation for missing arms it never needed.
    required = set(METHODS) if stratum == "mid" else set()
    missing = required - set(jobs)
    if missing:
        raise RuntimeError(f"{dataset} seed {seed} ({stratum}): incomplete job "
                           f"ledger, missing {sorted(missing)}")
    if not jobs:
        raise RuntimeError(f"{dataset} seed {seed} ({stratum}): no ledger rows")
    jobs = {k: v for k, v in jobs.items()
            if k in METHODS or k in OPTIONAL_METHODS}
    # A '.' in an arm name becomes a nested-key separator in the Hydra override
    # `+unlearning.js_<arm>_ckpt=...`, which fails to parse. Catch it here rather
    # than 30 seconds into the job.
    bad = [k for k in jobs if "." in k]
    if bad:
        raise ValueError(f"ledger arm names cannot contain '.': {bad} "
                         f"(use e.g. ln0p1 instead of ln0.1)")
    result = {}
    for method, job_id in jobs.items():
        runs = glob.glob(str(ROOT / "logs/unlearn/runs" / f"job{job_id}_*"))
        if len(runs) != 1:
            raise RuntimeError(f"job {job_id}: expected one unlearn run, got {runs}")
        path = Path(runs[0]) / "checkpoints/unlearned.ckpt"
        if not path.is_file():
            raise FileNotFoundError(path)
        result[method] = str(path.resolve())
    return result


def legacy_beauty2_checkpoints() -> dict[str, str]:
    """Saved September diagnostic weights for an early scoped-metric pilot."""
    runs = {
        "nau": ("drift_test_mass_vs_clean",
                "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative"),
        "forget_repair": ("drift_test_mass_vs_clean",
                          "unified_lf0.1_ls0.0_ac0_ln0.0"),
        "forget_only": ("drift_test_mass_vs_clean",
                        "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0"),
        "scif": ("drift", "scif"),
        "no_neighbor": ("drift_test_mass_vs_clean",
                        "unified_lf0.1_ls0.1_ac0_ln0.0_sepgen"),
    }
    found = {}
    for name, (tree, arm) in runs.items():
        record = (ROOT / "logs/diagnose" / tree /
                  f"beauty_mid_s2_{arm}" / "collateral_drift.json")
        with record.open() as file:
            checkpoint = json.load(file)["unlearned_ckpt"]
        if not Path(checkpoint).is_file():
            raise FileNotFoundError(checkpoint)
        found[name] = checkpoint
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, choices=("beauty", "sports", "toys"))
    parser.add_argument("--seed", required=True, type=int, choices=(2, 3, 5, 7, 11))
    parser.add_argument("--stratum", default="mid",
                        choices=("mid", "unpopular", "popular"),
                        help="target popularity stratum (default: mid)")
    parser.add_argument("--n-queries", type=int, default=16)
    parser.add_argument("--item-chunk", type=int, default=128)
    parser.add_argument("--metric", choices=("js", "scope", "mia"), default="js")
    parser.add_argument("--max-neighbors", type=int, default=256)
    parser.add_argument("--scope-batch-size", type=int, default=16)
    parser.add_argument("--out-stem", default=None,
                        help="output tree under logs/diagnose/ (default: the metric's usual "
                             "tree; set a new one to avoid overwriting existing results)")
    parser.add_argument("--all-unrelated", action="store_true",
                        help="score EVERY prefix-disjoint held-out row instead "
                             "of a length-matched sample; writes to "
                             "logs/diagnose/fig1_scope_allunrel/")
    parser.add_argument("--legacy-beauty2", action="store_true",
                        help="pilot only: use saved September Beauty seed 2 weights")
    parser.add_argument("--ckpt", action="append", default=[], metavar="ARM=PATH",
                        help="score these unlearned checkpoints instead of the ledger's "
                             "(repeatable; implies no ledger lookup)")
    parser.add_argument("--data-view", default="",
                        help="suffix of a data-dir view to evaluate on, e.g. _valview "
                             "(testing/ -> the validation split)")
    parser.add_argument("--neighbor-def", default="prefix1",
                        help="prefix1 (default, first SID token), prefix2, or knn<K> (content embedding)")
    parser.add_argument("--embedding-path", default="",
                        help="item content embeddings for knn<K>; defaults to the dataset's embeddings")
    parser.add_argument("--mia-control", action="store_true",
                        help="mia: also score logs/diagnose/mia/control/<ds>_<st>_s<seed> (a second non-member set)")
    parser.add_argument("--sid-path", default="",
                        help="SID tensor the checkpoints were trained with (default: the "
                             "dataset's RQ-KMeans SIDs); e.g. the random-SID control")
    parser.add_argument("--clean-ckpt", default="",
                        help="retrained reference checkpoint (default: the recorded clean run)")
    parser.add_argument("--poison-ckpt", default="",
                        help="poisoned checkpoint (default: the recorded poisoned run)")
    parser.add_argument("--cpu", action="store_true",
                        help="instantiate the Hydra trainer without a GPU")
    args = parser.parse_args()
    dataset, seed, stratum = args.dataset, args.seed, args.stratum
    token = "" if seed == 2 else f"_seed{seed}"
    clean_label = f"{dataset}_clean{token}"
    poison_label = f"{dataset}_poison_pct1_n1{token}_tgt{stratum}"
    if args.clean_ckpt and args.poison_ckpt:
        clean, poison = (str(Path(args.clean_ckpt).resolve()),
                         str(Path(args.poison_ckpt).resolve()))
    else:
        clean_ids = (["9096928"] if (dataset, seed) == ("beauty", 2)
                     else find_train(clean_label, "tiger"))
        poison_ids = find_train(poison_label, "tiger")
        if len(clean_ids) != 1 or len(poison_ids) != 1:
            raise RuntimeError(f"ambiguous reference jobs: clean={clean_ids}, poison={poison_ids}")
        clean = (str(Path(args.clean_ckpt).resolve()) if args.clean_ckpt
                 else training_checkpoint(clean_ids[0]))
        poison = (str(Path(args.poison_ckpt).resolve()) if args.poison_ckpt
                  else training_checkpoint(poison_ids[0]))
    if args.legacy_beauty2 and (dataset, seed, args.metric) != ("beauty", 2, "scope"):
        raise ValueError("--legacy-beauty2 requires --metric scope --dataset beauty --seed 2")
    if args.ckpt:
        methods = {}
        for item in args.ckpt:
            arm, path = item.split("=", 1)
            if "." in arm or not Path(path).is_file():
                raise ValueError(f"bad --ckpt {item}")
            methods[arm] = str(Path(path).resolve())
    else:
        methods = (legacy_beauty2_checkpoints() if args.legacy_beauty2 else
                   unlearned_checkpoints(dataset, seed, stratum))
    # unpopular predates the tgt<stratum> naming and kept the untagged directory;
    # mid and popular carry the token. This mirrors what the unlearn runs on each
    # stratum actually recorded as their data_dir.
    spam = (f"{dataset}_spam_seed{seed}_pct1_n1" if stratum == "unpopular"
            else f"{dataset}_spam_tgt{stratum}_seed{seed}_pct1_n1")
    data_dir = ROOT / "src/data/amazon_data" / (spam + args.data_view)
    sid = (ROOT / args.sid_path if args.sid_path
           else ROOT / "embeddings" / dataset / "merged_predictions_tensor.pt")
    if not data_dir.is_dir() or not sid.is_file():
        raise FileNotFoundError(f"missing data or SID: {data_dir}, {sid}")
    stem = ("fig1_scope_legacy" if args.legacy_beauty2 else
            "fig1_js" if args.metric == "js" else
            "fig1_scope_allunrel" if args.all_unrelated else "fig1_scope")
    module = ("src.measure_retraining_js" if args.metric == "js"
              else "src.measure_membership" if args.metric == "mia"
              else "src.measure_neighborhood_performance")
    if args.metric == "mia" and not args.out_stem:
        stem = "mia"
    if args.out_stem:
        stem = args.out_stem
    name = (f"{dataset}_seed{seed}.json" if stratum == "mid"
            else f"{dataset}_{stratum}_seed{seed}.json")
    out = ROOT / "logs/diagnose" / stem / name
    command = [sys.executable, "-m", module,
               "experiment=tiger_unlearn_scif_flat",
               f"data_dir={data_dir}", f"semantic_id_path={sid}",
               f"ckpt_path='{poison}'", "num_hierarchies=4", f"seed={seed}",
               f"+unlearning.js_dataset={dataset}", f"+unlearning.js_output={out}",
               f"+unlearning.js_n_queries={args.n_queries}",
               f"+unlearning.js_item_chunk={args.item_chunk}",
               f"+unlearning.scope_output={out}",
               f"+unlearning.scope_all_unrelated={bool(args.all_unrelated)}",
               f"+unlearning.scope_max_neighbors={args.max_neighbors}",
               f"+unlearning.scope_stratum={stratum}",
               f"+unlearning.scope_batch_size={args.scope_batch_size}",
               f"+unlearning.js_clean_ckpt='{clean}'",
               f"+unlearning.js_poison_ckpt='{poison}'",
               *[f"+unlearning.js_{method}_ckpt='{checkpoint}'"
                 for method, checkpoint in methods.items()]]
    if args.metric == "mia":
        nonmembers = ROOT / "logs/diagnose/mia/nonmembers" / f"{dataset}_{stratum}_s{seed}"
        if not (nonmembers / "training").is_dir():
            raise FileNotFoundError(f"non-members missing: {nonmembers} (scripts/make_mia_nonmembers.py)")
        command.append(f"+unlearning.mia_nonmember_dir={nonmembers / 'training'}")
        control = ROOT / "logs/diagnose/mia/control" / f"{dataset}_{stratum}_s{seed}" / "training"
        if args.mia_control and control.is_dir():   # second non-member set: AUC must be ~0.5
            command.append(f"+unlearning.mia_control_dir={control}")
    if args.neighbor_def != "prefix1":
        command.append(f"+unlearning.scope_neighbor_def={args.neighbor_def}")
        if args.neighbor_def.startswith("knn"):
            emb = args.embedding_path or EMBEDDINGS[dataset]
            if not (ROOT / emb).is_file():
                raise FileNotFoundError(emb)
            command.append(f"+unlearning.scope_embedding_path={ROOT / emb}")
    if args.cpu:
        command += ["trainer.accelerator=cpu", "trainer.devices=1"]
    print(f"Evaluating {args.metric} for {dataset} {stratum} seed {seed}: {out}",
          flush=True)
    os.chdir(ROOT)
    return subprocess.call(command)


if __name__ == "__main__":
    raise SystemExit(main())
