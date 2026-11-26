"""Paper-grid summary for the PKM paper: plain TIGER vs item memory (imsc), 3 target
strata x 5 data seeds, test N@10 / SH@10 before deletion and after full unlearning with
the paper's Forget + Repair and NAU, plus each cell's retrained (clean) references.

Usage (from GRID/): python -m scripts.grid_summary [--csv logs/grid_summary.csv]
"""

import argparse
import csv
import glob
import statistics as st

STRATA = ("unpopular", "mid", "popular")
SEEDS = (2, 3, 5, 7, 11)
TOK = {"unpopular": "_up", "mid": "", "popular": "_pop"}
ARMS = {"pre": "pre", "fr": "full_fr_lf0.1", "nau": "full_nau"}


def metrics(tag: str):
    runs = sorted(glob.glob(f"logs/unlearn/runs/*_bs1_{tag}"))
    for run in reversed(runs):  # newest run with an eval
        files = sorted(glob.glob(f"{run}/eval/csv/version_*/metrics.csv"))
        if not files:
            continue
        v = {}
        for r in csv.DictReader(open(files[-1])):
            for k in ("test/ndcg@10", "test/SH@10"):
                if r.get(k) not in (None, ""):
                    v[k] = float(r[k])
        if "test/ndcg@10" in v:
            return v["test/ndcg@10"], v.get("test/SH@10", float("nan"))
    return None


def model_name(arch: str, stratum: str, seed: int) -> str:
    return f"{arch}{TOK[stratum]}" + ("" if seed == 2 else f"_d{seed}")


def ref_tag(arch: str, stratum: str, seed: int) -> str:
    if stratum == "mid" and seed == 2:
        return f"ref_spam_{arch}clean"
    if stratum == "mid" and seed in (3, 5):
        return f"ref_spam_d{seed}_{arch}clean"
    return f"ref_spam_{stratum}_d{seed}_{arch}clean"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for s in STRATA:
        for seed in SEEDS:
            row = {"stratum": s, "seed": seed}
            for arch in ("plain", "imsc"):
                m = model_name(arch, s, seed)
                for k, arm in ARMS.items():
                    r = metrics(f"bw_tgt{s}_ITEMMEM_{m}_{arm}")
                    row[f"{arch}_{k}_N"], row[f"{arch}_{k}_SH"] = r if r else (None, None)
                r = metrics(ref_tag(arch, s, seed))
                row[f"{arch}_ref_N"], row[f"{arch}_ref_SH"] = r if r else (None, None)
            rows.append(row)
    f = lambda x: "  -   " if x is None else f"{x:.4f}"
    print(f"{'cell':14s} {'plain pre':>14s} {'plain F+R':>14s} {'plain NAU':>14s} "
          f"{'imsc pre':>14s} {'imsc F+R':>14s} {'imsc NAU':>14s} {'ref SH p/i':>14s}")
    for r in rows:
        cells = [f"{f(r[f'{a}_{k}_N'])}/{f(r[f'{a}_{k}_SH'])}" for a in ("plain", "imsc")
                 for k in ("pre", "fr", "nau")]
        print(f"{r['stratum'][:5]+' s'+str(r['seed']):14s} " + " ".join(f"{c:>14s}" for c in cells)
              + f" {f(r['plain_ref_SH'])}/{f(r['imsc_ref_SH'])}")
    print("\nper stratum (mean over seeds with both models; paired imsc - plain N@10):")
    for s in STRATA + ("all",):
        sel = [r for r in rows if s == "all" or r["stratum"] == s]
        for k in ("pre", "fr", "nau"):
            pairs = [(r[f"plain_{k}_N"], r[f"imsc_{k}_N"]) for r in sel
                     if r[f"plain_{k}_N"] is not None and r[f"imsc_{k}_N"] is not None]
            if not pairs:
                continue
            d = [100 * (i / p - 1) for p, i in pairs]
            sh_p = [r[f"plain_{k}_SH"] for r in sel if r[f"plain_{k}_SH"] is not None]
            sh_i = [r[f"imsc_{k}_SH"] for r in sel if r[f"imsc_{k}_SH"] is not None]
            sd = st.stdev(d) if len(d) > 1 else float("nan")
            print(f"  {s:9s} {k:4s} n={len(pairs):2d}  plain {st.mean(p for p, _ in pairs):.4f}  "
                  f"imsc {st.mean(i for _, i in pairs):.4f}  imsc vs plain {st.mean(d):+.1f}% "
                  f"(sd {sd:.1f}, min {min(d):+.1f}, max {max(d):+.1f})  "
                  f"SH plain {st.mean(sh_p):.4f} imsc {st.mean(sh_i):.4f}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"\n-> {a.csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
