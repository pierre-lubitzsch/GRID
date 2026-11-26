"""Write the per-(dataset, track, seed) spam reference points UR and PGR need.

UR = NDCG/NDCG_clean and PGR = (SH_poison - SH)/(SH_poison - SH_clean) are
meaningless without four numbers per cell: the clean model's NDCG@10, its SH@10
floor per stratum, and the poisoned model's SH@10 and NDCG@10 per stratum. Those
were previously a hand-maintained dict in extract_bandwagon_head2head.py plus one
hand-written logs/eval/spam_references/toys.json, which does not scale to the
grid's 3 tracks x 4 datasets x 5 seeds and offers no way to tell whether a
reference belongs to the identifier space it is being applied to.

Each track is a DIFFERENT IDENTIFIER SPACE, and each seed has its own poison
generation, its own clean model and its own poisoned models. So references are
keyed by all three and written to::

    logs/eval/spam_references/<ds>[_<model>][_<quantizer>][_seed<N>].json

with the tiger / rkmeans / seed-2 tokens omitted, which keeps the legacy
`toys.json` name valid for the track it actually describes.

    # clean baseline dir auto-resolved, poisoned evals given explicitly
    python scripts/write_spam_references.py --dataset toys \\
        --poison-eval unpopular=logs/eval/runs/2026-08-19/10-11-12 \\
        --poison-eval mid=job:10454300 --poison-eval popular=...

    # re-derive an existing reference from its recorded provenance and diff it
    python scripts/write_spam_references.py --dataset toys --check
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import run_tags  # noqa: E402

STRATA = list(run_tags.STRATA)
REF_DIR = "logs/eval/spam_references"
POISON_EVAL_ROOT = "logs/eval/spam_poison"


def poison_eval_dir(ds: str, model: str, quantizer: str, seed: int,
                    stratum: str) -> str:
    """Where plan_reference_evals.py sends the poisoned-model spam evals."""
    tok = ds
    if model != run_tags.DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != run_tags.DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    if seed != run_tags.DEFAULT_SEED:
        tok += f"_seed{seed}"
    return os.path.join(POISON_EVAL_ROOT, f"{tok}_{stratum}")


def ref_path(ds: str, model: str, quantizer: str, seed: int) -> str:
    tok = ds
    if model != run_tags.DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != run_tags.DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    if seed != run_tags.DEFAULT_SEED:
        tok += f"_seed{seed}"
    return os.path.join(REF_DIR, f"{tok}.json")


def clean_baseline_dir(ds: str, model: str, quantizer: str, seed: int) -> str | None:
    """Newest clean-baseline eval dir for these coordinates.

    Prefers a dir whose coordinates.json matches exactly (written by
    run_clean_baseline_bandwagon.sh); falls back to the directory-name tokens for
    the runs that predate it.
    """
    tok = "" if ds == "beauty" else f"_{ds}"
    if model != run_tags.DEFAULT_MODEL:
        tok += f"_{model}"
    if quantizer != run_tags.DEFAULT_QUANTIZER:
        tok += f"_{quantizer}"
    if seed != run_tags.DEFAULT_SEED:
        tok += f"_seed{seed}"
    cands = sorted(glob.glob(f"logs/eval/clean_baseline_bandwagon{tok}/*"),
                   key=os.path.getmtime, reverse=True)
    for d in cands:
        coord = os.path.join(d, "coordinates.json")
        if os.path.isfile(coord):
            c = json.load(open(coord))
            if (c.get("dataset") == ds and c.get("model", "tiger") == model
                    and c.get("quantizer", "rkmeans") == quantizer
                    and int(c.get("poison_seed", 2)) == seed):
                return d
            continue
        return d      # pre-coordinates dir: the name tokens are all there is
    return None


def resolve_eval_dir(spec: str) -> str | None:
    """A path, or ``job:<slurm id>``, to a directory holding a metrics.csv."""
    if spec.startswith("job:"):
        jid = spec[4:]
        hits = [os.path.dirname(os.path.dirname(os.path.dirname(p)))
                for p in glob.glob(f"logs/**/*{jid}*/**/metrics.csv", recursive=True)]
        if not hits:
            hits = glob.glob(f"logs/**/*{jid}*", recursive=True)
        return hits[0] if hits else None
    return spec if os.path.exists(spec) else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--model", default=run_tags.DEFAULT_MODEL)
    ap.add_argument("--quantizer", default=run_tags.DEFAULT_QUANTIZER)
    ap.add_argument("--seed", type=int, default=run_tags.DEFAULT_SEED)
    ap.add_argument("--clean-eval-dir", default=None,
                    help="clean-baseline eval dir (default: newest matching)")
    ap.add_argument("--poison-eval", action="append", default=[],
                    metavar="STRATUM=DIR|job:ID",
                    help="poisoned-model spam eval, repeatable")
    ap.add_argument("--check", action="store_true",
                    help="re-derive from recorded provenance and report drift")
    ap.add_argument("--allow-partial", action="store_true",
                    help="write even without the poisoned half (disables PGR)")
    ap.add_argument("--force", action="store_true",
                    help="overwrite an existing reference file")
    args = ap.parse_args()

    out = ref_path(args.dataset, args.model, args.quantizer, args.seed)
    track = run_tags.track_of(args.model, args.quantizer)

    if args.check:
        if not os.path.isfile(out):
            print(f"no reference at {out}")
            return 1
        ref = json.load(open(out))
        prov = ref.get("provenance", {})
        print(f"{out}\n  track={ref.get('track', '?')} seed={ref.get('seed', '?')}")
        drift = 0
        for label, d in sorted(prov.items()):
            m = run_tags.last_metrics(d) if d and os.path.isdir(d) else None
            if m is None:
                print(f"  {label:<28} MISSING on disk: {d}")
                drift += 1
                continue
            print(f"  {label:<28} NDCG@10 {m.get('test/ndcg@10', float('nan')):.6f} "
                  f"SH@10 {m.get('test/SH@10', float('nan')):.6f}   {d}")
        return 1 if drift else 0

    clean_dir = args.clean_eval_dir or clean_baseline_dir(
        args.dataset, args.model, args.quantizer, args.seed)
    if not clean_dir:
        print(f"No clean-baseline eval dir for {args.dataset}/{track}/seed{args.seed}. "
              f"Run:  DATASET={args.dataset} POISON_SEED={args.seed} "
              f"MODEL={args.model} QUANTIZER={args.quantizer} "
              "CLEAN_CKPT_OVERRIDE=<ckpt> sbatch run_clean_baseline_bandwagon.sh",
              file=sys.stderr)
        return 2

    clean_sh, clean_n10, provenance = {}, {}, {}
    for s in STRATA:
        d = os.path.join(clean_dir, f"clean_vs_{s}")
        m = run_tags.last_metrics(d)
        if m is None:
            print(f"  missing clean eval: {d}", file=sys.stderr)
            continue
        clean_sh[s] = m.get("test/SH@10")
        clean_n10[s] = m.get("test/ndcg@10")
        provenance[f"clean_vs_{s}"] = d
    if len(clean_sh) != len(STRATA):
        print(f"clean baseline incomplete under {clean_dir}: have "
              f"{sorted(clean_sh)}, need {STRATA}", file=sys.stderr)
        return 2

    # NDCG@10 is a property of (model, test set) and must not vary with which
    # manifest scored SH. If it does, the three dirs are not the same model.
    vals = [v for v in clean_n10.values() if v is not None]
    if max(vals) - min(vals) > 1e-6:
        print(f"WARNING: clean NDCG@10 differs across strata dirs "
              f"({clean_n10}); the three evals are not the same checkpoint.",
              file=sys.stderr)

    poison_sh, poison_n10 = {}, {}
    # Auto-discover the poisoned evals at the conventional path that
    # scripts/plan_reference_evals.py submits them to, so the common case needs
    # no --poison-eval at all. An explicit --poison-eval still wins.
    explicit = {spec.split("=", 1)[0] for spec in args.poison_eval if "=" in spec}
    for s_ in STRATA:
        if s_ in explicit:
            continue
        d = poison_eval_dir(args.dataset, args.model, args.quantizer,
                            args.seed, s_)
        m = run_tags.last_metrics(d) if os.path.isdir(d) else None
        if m is None:
            continue
        if m.get("test/SH@10") is None:
            print(f"  {d} has no test/SH@10: the eval ran without its spam "
                  "manifest, so it measured nothing about exposure. Rerun it "
                  "with POISON_SEED / TARGET_STRATEGY set.", file=sys.stderr)
            continue
        poison_sh[s_] = m["test/SH@10"]
        poison_n10[s_] = m.get("test/ndcg@10")
        provenance[f"poison_{s_}"] = d
    for spec in args.poison_eval:
        if "=" not in spec:
            ap.error(f"--poison-eval wants STRATUM=DIR, got {spec!r}")
        s, _, where = spec.partition("=")
        if s not in STRATA:
            ap.error(f"unknown stratum {s!r} (expected one of {STRATA})")
        d = resolve_eval_dir(where)
        m = run_tags.last_metrics(d) if d else None
        if m is None:
            print(f"  cannot read poisoned eval for {s}: {where}", file=sys.stderr)
            continue
        poison_sh[s] = m.get("test/SH@10")
        poison_n10[s] = m.get("test/ndcg@10")
        provenance[f"poison_{s}"] = d

    # A reference file missing its poisoned half is worse than no file at all:
    # the resolver prefers a file over the legacy constants, so a half-written
    # one would silently disable PGR for a cell that previously had it.
    missing = [s for s in STRATA if s not in poison_sh]
    if missing and not args.allow_partial:
        print(f"REFUSING to write {out}: no poisoned eval for {missing}. PGR "
              "needs SH@10 on the poisoned starting checkpoint. Pass "
              "--poison-eval <stratum>=<eval dir|job:ID> for each stratum, or "
              "--allow-partial to write the clean half only (which leaves PGR "
              "uncomputable for this cell, by design rather than by accident).",
              file=sys.stderr)
        return 2

    ref = {
        "dataset": args.dataset,
        "model": args.model,
        "quantizer": args.quantizer,
        "track": track,
        "seed": args.seed,
        "attack": f"bandwagon pct1 n_target=1 seed{args.seed}",
        "clean_ndcg@10": sorted(vals)[len(vals) // 2],
        "clean_SH@10": clean_sh,
        "poison_SH@10": poison_sh,
        "poison_ndcg@10": poison_n10,
        "provenance": provenance,
        "note": ("Written by scripts/write_spam_references.py. Applies ONLY to "
                 f"track {track} seed {args.seed}: each track is a different "
                 "identifier space and each seed its own poison generation, so "
                 "these numbers are not transferable to another cell."),
    }
    if os.path.isfile(out) and not args.force:
        print(f"{out} exists; pass --force to overwrite", file=sys.stderr)
        return 3
    os.makedirs(REF_DIR, exist_ok=True)
    with open(out, "w") as fh:
        json.dump(ref, fh, indent=2)
        fh.write("\n")
    print(f"Wrote {out}")
    print(f"  clean NDCG@10 {ref['clean_ndcg@10']:.6f}  clean SH@10 "
          + ", ".join(f"{s}={clean_sh[s]:.6f}" for s in STRATA))
    if poison_sh:
        print("  poison SH@10 "
              + ", ".join(f"{s}={poison_sh[s]:.6f}" for s in STRATA if s in poison_sh))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
