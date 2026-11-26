"""Plan the reference evaluations the grid's UR / PGR numbers depend on.

Nothing in `tables/spam_grid.md` or `tables/sensitive_grid.md` can be computed
until each cell has its references measured, and a cell is
(dataset, track, seed) because each track is its own identifier space and each
seed its own poison generation. Per spam cell that is one clean-baseline job
(which scores the clean model against all three strata's I_t in one go) plus one
spam eval per poisoned starting checkpoint; per sensitive cell it is one job
covering base and retrain.

This resolves every checkpoint from the training run-dir labels and prints the
exact submission commands. It does NOT submit: pass --submit to do that.

It refuses to guess when a label is ambiguous. Three run dirs are labelled
`beauty_clean` (jobs 9096928 / 9465564 / 9642098) with test NDCG@10 spanning
7.5%, and UR divides by whichever one is chosen, so the choice is named on the
command line rather than left to mtime. --pin <ds>:<track>:<seed>:<role>=<jobid>
settles one; beauty's rkmeans clean is pinned by default to 9096928, the model
every recorded beauty number was computed against.

    python scripts/plan_reference_evals.py                     # all cells
    python scripts/plan_reference_evals.py --datasets sports --seeds 2
    python scripts/plan_reference_evals.py --quantizer rqvae --out submit.sh
"""

from __future__ import annotations

import argparse
import csv
import functools
import getpass
import subprocess
import glob
import os
import re
import sys
from typing import Optional

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tags  # noqa: E402

DATASETS = ["beauty", "toys", "sports", "food_k5"]
SEEDS = [2, 3, 5, 7, 11]
STRATA = list(run_tags.STRATA)
SENSITIVE_CATEGORY = {"beauty": "hairloss", "toys": "toyweapons",
                      "sports": "guns", "food_k5": "meat"}
FORGET_RATIO = "1e-4"
POISON_EVAL_ROOT = "logs/eval/spam_poison"

# Cells where several training runs share one label, pinned to the model the
# recorded numbers were actually computed against. Without these the denominator
# is decided by mtime, and UR divides by it.
#   beauty clean: 3 runs labelled `beauty_clean`, test NDCG@10 spanning 7.5%
#     (9096928 0.03763 / 9465564 0.03615 / 9642098 0.03499). 9096928 is the one
#     LEGACY_REFERENCES and every recorded beauty table use.
#   retrain: the 2026-08-18 pair (10449165 / 10449166) are the models
#     extract_sensitive.py and run_sensitive_reference_eval.sh already name; the
#     2026-08-15 pair (10431613 / 10431614) predate the fixed setup.
DEFAULT_PINS = {
    ("beauty", "tiger+rkmeans", 2, "clean"): "9096928",
    ("beauty", "tiger+rkmeans", 2, "retrain"): "10449165",
    ("toys", "tiger+rkmeans", 2, "retrain"): "10449166",
}


def label(ds: str, seed: int, quantizer: str, kind: str, cat: str = "",
          model: str = run_tags.DEFAULT_MODEL) -> str:
    """Training run-dir label for one cell, matching run_tiger_train.sh.

    The quantizer token is dropped for the model's OWN identifier space, not
    just for rkmeans. LETTER is its own tokenizer -- track letter+letter -- and
    run_tiger_train.sh tags those runs `letter_<ds>_clean`, with the model tag
    in front and no `_letter` suffix (the runner never reads QUANTIZER). Keeping
    the hardcoded rkmeans test here would look for `letter_<ds>_clean_letter`,
    find nothing, and report every LETTER reference as missing.
    """
    seed_tok = "" if seed == 2 else f"_seed{seed}"
    q_tok = ("" if quantizer in (run_tags.DEFAULT_QUANTIZER, model)
             else f"_{quantizer}")
    if kind == "clean":
        return f"{ds}_clean{seed_tok}{q_tok}"
    if kind == "retrain":
        return f"{ds}_clean{seed_tok}{q_tok}_retrain_{cat}_{FORGET_RATIO}"
    return f"{ds}_poison_pct1_n1{seed_tok}{q_tok}_tgt{kind}"


def label_variants(ds: str, seed: int, quantizer: str, kind: str, cat: str = "",
                   model: str = run_tags.DEFAULT_MODEL) -> list[str]:
    """Every spelling a training run for this cell may legitimately carry.

    The `unpopular` stratum has TWO live spellings and both exist on disk.
    run_tiger_train.sh appends `_tgt<strategy>` only when the strategy is not
    the default `unpopular`, so a plain unpopular run is `<ds>_poison_pct1_n1`;
    but a driver that passes RUN_LABEL_SUFFIX containing `_tgt` hits the
    "driver already tagged it" branch and the token survives. The TIGER sweeps
    did the latter (46 `_tgtunpopular` dirs), the LETTER campaign did the
    former, so a single spelling silently finds one model set and not the
    other -- which is why LETTER had no spam references at all and the
    landed-cell table reported "no landed cells" rather than "no reference".
    """
    primary = label(ds, seed, quantizer, kind, cat, model=model)
    out = [primary]
    if kind == "unpopular":
        bare = primary[: -len("_tgtunpopular")] if primary.endswith("_tgtunpopular") else None
        if bare:
            out.append(bare)
    return out


@functools.lru_cache(maxsize=1)
def _completed_jobs() -> frozenset:
    """Job ids that reached COMPLETED, from one cached sacct call."""
    try:
        out = subprocess.run(
            ["sacct", "-a", "-u", getpass.getuser(), "-S", "2026-01-01",
             "-X", "-n", "-P", "-o", "JobID,State"],
            capture_output=True, text=True, timeout=180).stdout
    except Exception:
        return frozenset()
    done = set()
    for line in out.splitlines():
        f = line.split("|")
        if len(f) == 2 and f[1] == "COMPLETED":
            done.add(f[0].split(".")[0])
    return frozenset(done)


def find_runs(lbl: str, model: str) -> list[tuple[str, str]]:
    """(job id, run dir) for every training run with exactly this label."""
    out = []
    prefix = "" if model == "tiger" else f"{model}_"
    for d in glob.glob(f"logs/train/runs/*/*_{prefix}{lbl}"):
        m = re.match(r"^\d{2}-\d{2}-\d{2}_job(\d+)_(.+)$", os.path.basename(d))
        if not m or m.group(2) != f"{prefix}{lbl}":
            continue
        if not glob.glob(os.path.join(d, "checkpoints", "*.ckpt")):
            continue
        # A run whose training job has not COMPLETED is not a usable reference.
        # Two reasons, and the second is the dangerous one:
        #  1. `save_top_k` rotates epoch checkpoints, so the path resolved here
        #     is deleted out from under the eval that was planned against it --
        #     observed 2026-09-14, step=001700 -> 002200 -> 002700 while the
        #     plan still named 001700, and the eval died "clean ckpt missing".
        #  2. If the file HAD survived, the eval would have succeeded and
        #     silently scored an UNCONVERGED model as theta_r, quietly
        #     corrupting every UR/tau_P that divides by it.
        # A non-completed job is simply skipped, so the cell reports as
        # not-yet-ready instead of producing a wrong reference.
        done = _completed_jobs()
        if done and m.group(1) not in done:
            continue
        out.append((m.group(1), d))
    return sorted(out)


def ckpt_of(run_dir: str) -> Optional[str]:
    """Newest epoch checkpoint, which is what the recorded references used."""
    hits = sorted(glob.glob(os.path.join(run_dir, "checkpoints",
                                         "checkpoint_epoch=*.ckpt")),
                  key=os.path.getmtime, reverse=True)
    if hits:
        return hits[0]
    last = os.path.join(run_dir, "checkpoints", "last.ckpt")
    return last if os.path.isfile(last) else None


def ndcg_of(run_dir: str) -> Optional[float]:
    p = os.path.join(run_dir, "csv", "version_0", "metrics.csv")
    if not os.path.isfile(p):
        return None
    best = None
    for row in csv.DictReader(open(p)):
        v = row.get("test/ndcg@10")
        if v not in ("", None):
            try:
                best = float(v)
            except ValueError:
                pass
    return best


def sid_path(ds: str, quantizer: str) -> str:
    suffix = "" if quantizer == run_tags.DEFAULT_QUANTIZER else f"_{quantizer}"
    return f"embeddings/{ds}{suffix}/merged_predictions_tensor.pt"


def poison_eval_dir(ds: str, model: str, quantizer: str, seed: int,
                    stratum: str) -> str:
    tok = ds
    if model != run_tags.DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != run_tags.DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    if seed != run_tags.DEFAULT_SEED:
        tok += f"_seed{seed}"
    return os.path.join(POISON_EVAL_ROOT, f"{tok}_{stratum}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--datasets", default="beauty,toys,sports")
    ap.add_argument("--seeds", default=",".join(str(s) for s in SEEDS))
    ap.add_argument("--model", default=run_tags.DEFAULT_MODEL)
    ap.add_argument("--quantizer", default=None,
                    help="one track only (default: both tiger tracks)")
    ap.add_argument("--pin", action="append", default=[],
                    metavar="DS:TRACK:SEED:ROLE=JOBID")
    ap.add_argument("--setting", default="both",
                    choices=("spam", "sensitive", "both"))
    ap.add_argument("--out", default=None, help="write the commands here")
    ap.add_argument("--submit", action="store_true")
    args = ap.parse_args()

    pins = dict(DEFAULT_PINS)
    for spec in args.pin:
        key, _, jid = spec.partition("=")
        ds, track, seed, role = key.split(":")
        pins[(ds, track, int(seed), role)] = jid

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    quantizers = [args.quantizer] if args.quantizer else ["rkmeans", "rqvae"]

    cmds: list[str] = []
    problems: list[str] = []
    n_cells = n_ready = 0

    for quantizer in quantizers:
        track = run_tags.track_of(args.model, quantizer)
        for ds in datasets:
            cat = SENSITIVE_CATEGORY.get(ds, "")
            for seed in seeds:
                n_cells += 1
                resolved, missing = {}, []
                for role in ["clean", "retrain"] + STRATA:
                    _variants = label_variants(ds, seed, quantizer, role, cat,
                                               model=args.model)
                    lbl = _variants[0]
                    runs = find_runs(lbl, args.model)
                    # Fall back through the other legitimate spellings (see
                    # label_variants) rather than declaring the model missing.
                    for _alt in _variants[1:]:
                        if runs:
                            break
                        runs = find_runs(_alt, args.model)
                        if runs:
                            lbl = _alt
                    pin = pins.get((ds, track, seed, role))
                    if pin:
                        runs = [r for r in runs if r[0] == pin]
                    if not runs:
                        missing.append(role)
                        continue
                    if len(runs) > 1:
                        detail = ", ".join(
                            f"job{j} (NDCG@10 {ndcg_of(d) or float('nan'):.5f})"
                            for j, d in runs)
                        problems.append(
                            f"{ds}/{track}/seed{seed}/{role}: {len(runs)} runs "
                            f"share the label `{lbl}`: {detail}. Pin one with "
                            f"--pin {ds}:{track}:{seed}:{role}=<jobid>.")
                        missing.append(role)
                        continue
                    resolved[role] = runs[0]
                # Only the roles this SETTING actually consumes may block a
                # cell. `retrain` is a sensitive-only reference (the spam
                # references are clean + the poisoned model per stratum), but it
                # used to gate every cell, so a campaign whose retrain models
                # were still training could not plan its SPAM references either
                # -- a sensitive-side dependency silently blocking a spam-side
                # artifact.
                required = ["clean"] + STRATA
                if args.setting in ("sensitive", "both") and cat:
                    required.append("retrain")
                blocking = [r for r in missing if r in required]
                if blocking:
                    if not problems or blocking != ["retrain"]:
                        problems.append(
                            f"{ds}/{track}/seed{seed}: no trained model for "
                            f"{blocking}")
                    continue
                n_ready += 1
                sid = sid_path(ds, quantizer)
                clean_ck = ckpt_of(resolved["clean"][1])

                if args.setting in ("spam", "both"):
                    cmds.append(
                        f"# {ds} / {track} / seed {seed}: clean baseline "
                        f"(clean SH@10 floor per stratum), job "
                        f"{resolved['clean'][0]}")
                    cmds.append(
                        f"DATASET={ds} POISON_SEED={seed} MODEL={args.model} "
                        f"QUANTIZER={quantizer} \\\n"
                        f"  CLEAN_CKPT_OVERRIDE='{clean_ck}' \\\n"
                        f"  SID_OVERRIDE='{sid}' \\\n"
                        f"  sbatch run_clean_baseline_bandwagon.sh")
                    for strat in STRATA:
                        out_dir = poison_eval_dir(ds, args.model, quantizer,
                                                  seed, strat)
                        if os.path.isfile(os.path.join(out_dir, "csv",
                                                       "version_0",
                                                       "metrics.csv")):
                            cmds.append(f"# already done: {out_dir}")
                            continue
                        ck = ckpt_of(resolved[strat][1])
                        cmds.append(
                            f"POISON_METHOD=bandwagon POISONING_RATIO=0.01 "
                            f"N_TARGET_ITEMS=1 POISON_SEED={seed} "
                            f"TARGET_STRATEGY={strat} \\\n"
                            f"  sbatch run_tiger_eval_spam.sh '{ck}' {ds} "
                            f"'{sid}' \\\n"
                            f"    hydra.run.dir='{out_dir}'")
                    cmds.append("")

                if args.setting in ("sensitive", "both") and cat:
                    cmds.append(
                        f"# {ds} / {track} / seed {seed}: sensitive base + "
                        f"retrain references ({cat})")
                    cmds.append(
                        f"MODEL={args.model} QUANTIZER={quantizer} "
                        f"DATASETS={ds} SEEDS={seed} \\\n"
                        f"  MODELS='{ds}:{seed}:{resolved['clean'][0]}:"
                        f"{resolved['retrain'][0]}' \\\n"
                        f"  sbatch run_sensitive_reference_eval.sh")
                    cmds.append("")

    header = [
        "#!/usr/bin/env bash",
        "# Reference evaluations for the grid. Generated by "
        "scripts/plan_reference_evals.py -- do not edit by hand.",
        "#",
        "# Each spam cell needs the clean model's SH@10 floor per stratum AND "
        "the poisoned",
        "# starting checkpoint's SH@10, because PGR = (SH_poison - SH) / "
        "(SH_poison - SH_clean).",
        "# Collect them afterwards with:",
        "#   python scripts/write_spam_references.py --dataset <ds> --model "
        "<m> --quantizer <q> --seed <n>",
        "set -euo pipefail",
        "",
    ]
    text = "\n".join(header + cmds)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
        os.chmod(args.out, 0o755)
        print(f"wrote {args.out}")
    else:
        print(text)

    n_jobs = sum(1 for c in cmds if c.startswith(("DATASET=", "POISON_METHOD=",
                                                  "MODEL=")))
    print(f"\n# cells: {n_ready} ready of {n_cells}; {n_jobs} sbatch jobs",
          file=sys.stderr)
    if problems:
        print(f"\n# {len(problems)} cell(s) cannot be planned:", file=sys.stderr)
        for p in sorted(set(problems)):
            print(f"#   {p}", file=sys.stderr)
    if args.submit:
        print("\n--submit: run the emitted script yourself; this tool does not "
              "queue jobs on your behalf.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
