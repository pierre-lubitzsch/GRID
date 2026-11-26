"""Architecture ablation table for the PKM paper (poisoned Beauty, 3 strata x 5 data seeds):
plain, im, imc, ims, imsc, plainscore before deletion and after full F+R / NAU, test N@10 and
SH@10, plus the same final model with the memory pathways switched off at inference
(eval_off_<mode>/ of the unlearn run, OFF_EVALS=1 in run_tiger_unlearn_sequential.sh).

Gain: paired over the cells where the arm and plain both exist. Memory contribution:
(N_on - N_off) / N_on with every pathway the arm has switched off (both, else the one).

Usage (from GRID/): python -m scripts.itemmem_ablation_table [--cells common|own] [--ds beauty|toys]
"""

import argparse
import csv
import glob
import statistics as st

from scripts.grid_summary import ARMS, SEEDS, STRATA, model_name

ARCHS = ("plain", "im", "imc", "ims", "imsc", "plainscore")
LABEL = {"plain": "TIGER", "im": "+ item memory", "imc": "+ item memory + consistency",
         "ims": "+ item memory + score", "imsc": "+ item memory + score + consistency",
         "plainscore": "TIGER + score only (control)"}
OFF = {"im": "nulltoken", "imc": "nulltoken", "ims": "both", "imsc": "both",
       "plainscore": "scoreoff"}


def _read(path_glob: str):
    files = sorted(glob.glob(path_glob))
    if not files:
        return None
    v = {}
    for r in csv.DictReader(open(files[-1])):
        for k in ("test/ndcg@10", "test/SH@10"):
            if r.get(k) not in (None, ""):
                v[k] = float(r[k])
    return (v["test/ndcg@10"], v.get("test/SH@10", float("nan"))) if "test/ndcg@10" in v else None


def run_metrics(tag: str, sub: str = "eval"):
    """Newest run of ``tag`` that has ``sub``/ metrics (eval or eval_off_<mode>)."""
    for run in reversed(sorted(glob.glob(f"logs/unlearn/runs/*_bs1_{tag}"))):
        r = _read(f"{run}/{sub}/csv/version_*/metrics.csv")
        if r:
            return r
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cells", choices=("own", "common"), default="own",
                    help="own: each arm on its own cells (paired with plain); "
                         "common: only cells where every arm exists")
    ap.add_argument("--ds", choices=("beauty", "toys"), default="beauty")
    a = ap.parse_args()
    dstok = "" if a.ds == "beauty" else f"_{a.ds}"
    D = {}
    for arch in ARCHS:
        for s in STRATA:
            for seed in SEEDS:
                for k, arm in ARMS.items():
                    tag = f"bw_tgt{s}_ITEMMEM{dstok}_{model_name(arch, s, seed)}_{arm}"
                    r = run_metrics(tag)
                    if r:
                        D[arch, s, seed, k, "on"] = r
                    if arch in OFF:
                        r = run_metrics(tag, f"eval_off_{OFF[arch]}")
                        if r:
                            D[arch, s, seed, k, "off"] = r
    cells = [(s, d) for s in STRATA for d in SEEDS]
    if a.cells == "common":
        cells = [c for c in cells if all((x, *c, "pre", "on") in D for x in ARCHS)]
    m = lambda xs: f"{st.mean(xs):.4f}" if xs else "  -   "
    for k, name in (("pre", "before unlearning"), ("fr", "after F+R"), ("nau", "after NAU")):
        print(f"\n{name}   (N@10 / SH@10 mean over cells; gain paired vs TIGER; "
              f"off = pathways off at inference)")
        print(f"  {'architecture':38s} {'n':>3s} {'N@10':>7s} {'gain':>14s} {'SH@10':>7s} "
              f"{'n_off':>5s} {'N off':>7s} {'SH off':>7s} {'mem contrib':>11s}")
        for arch in ARCHS:
            own = [c for c in cells if (arch, *c, k, "on") in D and ("plain", *c, k, "on") in D]
            if not own:
                print(f"  {LABEL[arch]:38s}   0")
                continue
            N = [D[arch, *c, k, "on"][0] for c in own]
            S = [D[arch, *c, k, "on"][1] for c in own]
            g = [100 * (D[arch, *c, k, "on"][0] / D["plain", *c, k, "on"][0] - 1) for c in own]
            gs = "" if arch == "plain" else f"{st.mean(g):+.1f}% ({min(g):+.1f})"
            offc = [c for c in own if (arch, *c, k, "off") in D]
            No = [D[arch, *c, k, "off"][0] for c in offc]
            So = [D[arch, *c, k, "off"][1] for c in offc]
            mc = [100 * (1 - D[arch, *c, k, "off"][0] / D[arch, *c, k, "on"][0]) for c in offc]
            mcs = f"{st.mean(mc):.1f}%" if mc else ("-" if arch == "plain" else "")
            print(f"  {LABEL[arch]:38s} {len(own):3d} {m(N):>7s} {gs:>14s} {m(S):>7s} "
                  f"{len(offc) if arch in OFF else '':>5} {m(No) if arch in OFF else '':>7s} "
                  f"{m(So) if arch in OFF else '':>7s} {mcs:>11s}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
