"""Significance tests for the item-memory study (PKM paper).

Poisoned Beauty (3 strata x 5 data seeds = 15 paired cells), per stage (pre / F+R / NAU):
  each architecture vs TIGER, and the component steps (score: ims vs im, imsc vs imc;
  consistency: imc vs im, imsc vs ims) with memory on and with memory off at inference.
Clean data (5 training seeds per dataset and recipe): im / imc vs plain, imc vs im,
  per dataset and pooled over the 3 datasets (15 pairs).

Per comparison: n, mean relative gain (%) of A over B, 95% bootstrap CI of that mean
(10k resamples of the pairs), wins (A > B), paired t-test p and Wilcoxon signed-rank p
(two-sided, on the raw N@10 differences). Note: with n=5 the smallest possible two-sided
Wilcoxon p is 0.0625, so per-dataset clean tests rely on t / CI; use the pooled rows.

Usage (from GRID/): python -m scripts.itemmem_stats [--ds toys]
"""

import numpy as np
from scipy import stats

from scripts.grid_summary import ARMS, SEEDS, STRATA, model_name
from scripts.itemmem_ablation_table import OFF, run_metrics
from scripts.itemmem_clean_summary import DATASETS, null_ndcg, train_ndcg

RNG = np.random.default_rng(0)


def test(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    n = len(a)
    if n < 3:
        return f"n={n} (too few)"
    rel = 100 * (a / b - 1)
    boot = rel[RNG.integers(0, n, size=(10000, n))].mean(1)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    d = a - b
    pt = stats.ttest_rel(a, b).pvalue
    pw = stats.wilcoxon(d).pvalue if np.any(d != 0) else 1.0
    return (f"n={n:2d}  gain {rel.mean():+6.1f}%  CI [{lo:+6.1f}, {hi:+6.1f}]  "
            f"wins {int((d > 0).sum()):2d}/{n:<2d}  t p={pt:.1e}  W p={pw:.1e}")


def poisoned(ds="beauty"):
    dstok = "" if ds == "beauty" else f"_{ds}"
    D = {}
    for arch in ("plain", "im", "imc", "ims", "imsc", "plainscore"):
        for s in STRATA:
            for seed in SEEDS:
                for k, arm in ARMS.items():
                    tag = f"bw_tgt{s}_ITEMMEM{dstok}_{model_name(arch, s, seed)}_{arm}"
                    r = run_metrics(tag)
                    if r:
                        D[arch, "on", k, s, seed] = r[0]
                    if arch in OFF and (r := run_metrics(tag, f"eval_off_{OFF[arch]}")):
                        D[arch, "off", k, s, seed] = r[0]
    cells = [(s, d) for s in STRATA for d in SEEDS]

    def cmp(A, pa, B, pb, k):
        cs = [c for c in cells if (A, pa, k, *c) in D and (B, pb, k, *c) in D]
        return test([D[A, pa, k, *c] for c in cs], [D[B, pb, k, *c] for c in cs])

    rows = [("vs TIGER", [(x, "on", "plain", "on") for x in ("im", "imc", "ims", "imsc", "plainscore")]),
            ("score head (memory on)", [("ims", "on", "im", "on"), ("imsc", "on", "imc", "on")]),
            ("consistency (memory on)", [("imc", "on", "im", "on"), ("imsc", "on", "ims", "on")]),
            ("consistency (memory off)", [("imc", "off", "im", "off"), ("imsc", "off", "ims", "off")]),
            ("memory off vs on", [(x, "off", x, "on") for x in ("im", "imc", "ims", "imsc")])]
    for k, name in (("pre", "before unlearning"), ("fr", "after F+R"), ("nau", "after NAU")):
        print(f"\n=== Poisoned {ds}, {name}, test N@10 ===")
        for title, comps in rows:
            print(f"  -- {title}")
            for A, pa, B, pb in comps:
                lab = f"{A}{'' if pa == 'on' else '(off)'} vs {B}{'' if pb == 'on' else '(off)'}"
                print(f"    {lab:24s} {cmp(A, pa, B, pb, k)}")


def clean():
    for recipe in ("es", "gp40rk"):
        print(f"\n=== Clean data, recipe _{recipe}, test N@10 (pairs = training seeds) ===")
        V = {(ds, a, s): train_ndcg(ds, a, s, recipe)
             for ds in DATASETS for a in ("plain", "im", "imc") for s in SEEDS}
        V.update({(ds, a + "_off", s): null_ndcg(ds, a, s, recipe)
                  for ds in DATASETS for a in ("im", "imc") for s in SEEDS})
        for A, B in (("im", "plain"), ("imc", "plain"), ("imc", "im"), ("imc_off", "im_off")):
            for ds in (*DATASETS, "pooled"):
                dss = DATASETS if ds == "pooled" else (ds,)
                ps = [(V[d, A, s], V[d, B, s]) for d in dss for s in SEEDS
                      if V.get((d, A, s)) is not None and V.get((d, B, s)) is not None]
                if ps:
                    print(f"    {A + ' vs ' + B:18s} {ds:7s} {test(*zip(*ps))}")


if __name__ == "__main__":
    import sys
    if "--ds" in sys.argv:  # poisoned grid of one dataset only
        poisoned(sys.argv[sys.argv.index("--ds") + 1])
    else:
        poisoned()
        clean()
