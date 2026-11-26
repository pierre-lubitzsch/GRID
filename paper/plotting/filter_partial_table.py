"""Two reviewer checks on the spam setting (TIGER + RQ-KMeans, test split).

filter   "Why not block the item at inference?": the poisoned model the methods start from, with
         a decode mask forbidding only the promoted target (run_filter_target.sh ->
         logs/diagnose/filter_target/), against NAU and Forget+Repair (Table 1 runs) in the same
         cells. Per cell Delta_SH = |SH - SH_R| and tau_P = NDCG / NDCG_R, mean over cells.
partial  only a fraction of the fake users known (scripts/make_partial_spam_dirs.py; tags
         bw_tgt<st>_known<f>_<arm>): Delta_SH, tau_P and PGR (against the poisoned model the runs
         start from, logs/diagnose/poison_ref_start/) per known fraction, Beauty mid + unpopular;
         100% = the Table 1 runs of the same cells.

    PYTHONPATH=. python paper/plotting/filter_partial_table.py

--ledger paper/plotting/canonical_runs.tsv: NAU and Forget+Repair (filter part, and the 100% row of the
partial part) from the ONE canonical execution per cell of Table 1's configuration
(extract_table1_valsel.ledger_cells) instead of the latest `_valsel` replay; adds the paired tests of
NAU vs the target-only filter (tau_P difference, 95% percentile bootstrap over cells with 10,000
resamples and seed 0, Wilcoxon signed-rank, cells where NAU is higher). Writes
paper/plotting/filter_partial_ledger.txt (a copy of the printout).
"""
import argparse
import importlib.util
import random
import sys
import csv
import glob
import json
import os
import re
import statistics as S

STRATA = ("unpopular", "mid", "popular")
SEEDS = ("2", "3", "5", "7", "11")
NAU = {"beauty": "valsel_nau_ls0p01_ln0p01", "sports": "valsel_nau_ls0p01_ln0p01", "toys": "valsel_nau_ls0p01_ln0p01"}
FR = "unified_abl_lf0.1_ls0.0_ln0.0_valsel"


def met(p):
    try:
        r = list(csv.DictReader(open(p)))[-1]
        return float(r["test/SH@10"]), float(r["test/ndcg@10"])
    except Exception:
        return None


def ref(ds, sd):
    return json.load(open(f"paper/data/spam_references/{ds}" + ("" if sd == "2" else f"_seed{sd}") + ".json"))


def run(ds, st, sd, arm):
    sfx = "" if sd == "2" else f"_seed{sd}"
    pre = "" if ds == "beauty" else f"{ds}_"
    c = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_{pre}bw_tgt{st}_{arm}{sfx}") if os.path.isdir(d)]
    if not c:
        return None
    return met(max(c, key=lambda x: int(re.search(r"job(\d+)_", x).group(1))) + "/eval/csv/version_0/metrics.csv")


LEDGER = None
_LC = {}


def ledger_run(ds, st, sd, method):
    if ds not in _LC:
        here = os.path.dirname(os.path.abspath(__file__))
        spec = importlib.util.spec_from_file_location("t1", os.path.join(here, "extract_table1_valsel.py"))
        t1 = importlib.util.module_from_spec(spec); spec.loader.exec_module(t1)
        _LC[ds] = t1.ledger_cells(LEDGER, ds, methods=["NAU", "Forget+Repair"])
    v = _LC[ds][method].get((st, sd))
    return None if v is None else (v[0], v[1])


def paired(a, b):
    """a - b over common cells: mean, bootstrap CI, Wilcoxon p, cells a > b, n."""
    from scipy.stats import wilcoxon
    k = sorted(set(a) & set(b))
    d = [a[x] - b[x] for x in k]
    rng = random.Random(0)
    bs = sorted(S.mean(rng.choice(d) for _ in d) for _ in range(10000))
    return S.mean(d), bs[250], bs[9749], float(wilcoxon(d, zero_method="zsplit").pvalue), sum(x > 0 for x in d), len(d)


def filter_part():
    tau = {"Filter (target only)": {}, "NAU": {}, "Forget+Repair": {}}
    print("== Block the target at inference (45 cells max)")
    for ds in ("beauty", "sports", "toys"):
        rows = {"Filter (target only)": [], "NAU": [], "Forget+Repair": [], "Poisoned": []}
        for st in STRATA:
            for sd in SEEDS:
                f = met(f"logs/diagnose/filter_target/{ds}_{st}_s{sd}/csv/version_0/metrics.csv")
                if not f:
                    continue
                R = ref(ds, sd); c_sh, c_n = R["clean_SH@10"][st], R["clean_ndcg@10"]
                p = met(f"logs/diagnose/poison_ref_start/{ds}_{st}_s{sd}/csv/version_0/metrics.csv")
                nau = ledger_run(ds, st, sd, "NAU") if LEDGER else run(ds, st, sd, NAU[ds])
                fr = ledger_run(ds, st, sd, "Forget+Repair") if LEDGER else run(ds, st, sd, FR)
                for name, v in (("Filter (target only)", f), ("NAU", nau),
                                ("Forget+Repair", fr), ("Poisoned", p)):
                    if v:
                        rows[name].append((abs(v[0] - c_sh), v[1] / c_n, v[0]))
                        if name in tau:
                            tau[name][(ds, st, sd)] = (v[1] / c_n, abs(v[0] - c_sh))
        for name, v in rows.items():
            if v:
                print(f"  {ds:7s} {name:22s} cells {len(v):2d}  dSH {100 * S.mean(x[0] for x in v):.3f}  tau {100 * S.mean(x[1] for x in v):5.1f}  SH {100 * S.mean(x[2] for x in v):.3f}")
    if LEDGER:
        for m in ("NAU", "Forget+Repair"):
            for j, lab in ((0, "tau_P (pts, >0: method higher)"), (1, "dSH (>0: filter closer)")):
                a = {k: v[j] for k, v in tau[m].items()}
                b = {k: v[j] for k, v in tau["Filter (target only)"].items()}
                mu, lo, hi, p, pos, n = paired(a, b)
                print(f"  {m} minus filter, {lab}: {100 * mu:+.3f} [{100 * lo:+.3f}, {100 * hi:+.3f}], higher in {pos} of {n} cells, Wilcoxon p={p:.2g}")


def partial_part():
    print("== Only part of the fake users known (Beauty, mid + unpopular)")
    for arm, full in (("nau_s0p01_n0p01", NAU["beauty"]), ("unified_abl_lf0.1_ls0.0_ln0.0", FR)):
        for f in (25, 50, 75, 100):
            v = []
            for st in ("unpopular", "mid"):
                for sd in SEEDS:
                    if f < 100:
                        u = run("beauty", st, sd, f"known{f}_{arm}")
                    elif LEDGER:
                        u = ledger_run("beauty", st, sd, "NAU" if arm.startswith("nau") else "Forget+Repair")
                    else:
                        u = run("beauty", st, sd, full)
                    if not u:
                        continue
                    R = ref("beauty", sd); c_sh, c_n = R["clean_SH@10"][st], R["clean_ndcg@10"]
                    p = met(f"logs/diagnose/poison_ref_start/beauty_{st}_s{sd}/csv/version_0/metrics.csv")
                    v.append((abs(u[0] - c_sh), u[1] / c_n, u[0], p[0], c_sh))
            if v:
                pgr = (S.mean(x[3] for x in v) - S.mean(x[2] for x in v)) / (S.mean(x[3] for x in v) - S.mean(x[4] for x in v))
                print(f"  {arm:32s} known {f:3d}%  cells {len(v):2d}  dSH {100 * S.mean(x[0] for x in v):.3f}  tau {100 * S.mean(x[1] for x in v):5.1f}  PGR {pgr:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=None, help="canonical_runs.tsv: Table 1's canonical executions")
    LEDGER = ap.parse_args().ledger
    if LEDGER:
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            filter_part()
            partial_part()
        sys.stdout.write(buf.getvalue())
        open("paper/plotting/filter_partial_ledger.txt", "w").write(buf.getvalue())
        print("wrote paper/plotting/filter_partial_ledger.txt")
    else:
        filter_part()
        partial_part()
