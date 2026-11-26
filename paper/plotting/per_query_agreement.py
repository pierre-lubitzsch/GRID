"""Per-prediction agreement with the retrained model (R3 Q1), from the stored scope JSONs.

The Figure 1/2/3 scope evaluations (scripts/run_fig1_js_eval.py --metric scope --all-unrelated)
store, for every arm INCLUDING the retrained ("clean") model, per-query arrays on identical test
queries: ndcg_per_query, sh_per_query (spam target in the top 10), rank_per_query (rank 1..10 of
the true next item, 0 = outside the top 10) and label_logprob_per_query (teacher-forced
log p(true next item | history)). Per test query q this script compares the unlearned model U
with the retrained model R of the same cell:

  mad_ndcg    mean_q |NDCG_U(q) - NDCG_R(q)|, in NDCG points (x100)
  bias_ndcg   mean_q (NDCG_U(q) - NDCG_R(q)), points
  hit_agree   share of q where "true item in top 10" agrees (U and R both hit or both miss), %
  hit_jacc    |hits_U & hits_R| / |hits_U | hits_R|, %  (ignores the both-miss majority)
  rank_same   share of q with the identical rank (both outside the top 10 counts as equal), %
  sh_agree    share of q where "spam target in top 10" agrees, %
  lp_spear    Spearman correlation of log p(true item) between U and R over the queries of a
              cell, averaged over cells (weighted by queries)
  lp_mad      mean_q |log p_U - log p_R| (nats)
Query sets: "whole" = every test query of the cell (neighbor + unrelated + target groups);
"neighbor" = the target's first-SID-token neighborhood (target's own cases excluded), as in
Figures 1-3. Every unlearned arm is paired with the retrained model scored IN THE SAME JSON.

Checkpoints: an arm-cell is used only if the JSON's checkpoint is the CANONICAL execution of that
method's configuration and cell in paper/plotting/canonical_runs.tsv (checked by run dir).
Sources, first canonical match wins: fig1_scope_allunrel_fix (Beauty mid: Table 1 Forget only
lambda_f=1, Fanchuan t0.07), fig1_scope_v3, fig1_scope_v2 (all 9 dataset x stratum blocks).

Scopes in the CSV:
  fig1      Beauty mid, 5 data seeds (the Figure 1 query sets), all methods
  block     each of the 9 (dataset, stratum) blocks, 5 seeds each, methods available there
  all9      pooled over the 45 cells, methods present in all of them (NAU, F+R, Poisoned)
  rr        Beauty data seed 2 (3 strata): every method, plus two further independently retrained
            models of the same seed and data (logs/diagnose/fig1_scope_rr, run_scope_rr.sh):
            the retrained-vs-retrained scale. "Poisoned" (no unlearning) is the other end.
Also prints NAU minus each method on mad_ndcg and hit_agree (paired over queries; 95% percentile
bootstrap with 3000 resamples on the neighborhood, normal approximation on the whole set).

    python paper/plotting/per_query_agreement.py
      -> paper/plotting/per_query_agreement.csv, paper/plotting/per_query_agreement_paired.csv
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
from scipy.stats import spearmanr

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DIAG = os.path.join(ROOT, "logs/diagnose")
SOURCES = ("fig1_scope_allunrel_fix", "fig1_scope_v3", "fig1_scope_v2")
SEEDS = (2, 3, 5, 7, 11)
STRATA = ("unpopular", "mid", "popular")
DATASETS = ("beauty", "sports", "toys")
# display name -> (ledger config, JSON arm names); None config = reference model (no ledger check)
METHODS = [
    ("Poisoned", None, ("poison",)),
    ("Finetune", "finetune", ("retain_only",)),
    ("Forget only", "forget_only_lf1.0", ("forget_only",)),
    ("Forget+Repair", "forget_repair_lf0.1", ("nosep_ln0", "forget_repair")),
    ("SCIF", "scif", ("scif",)),
    ("SEIF", "seif_std0.06", ("seif",)),
    ("Kookmin", "kookmin_ir0.001", ("kookmin",)),
    ("Fanchuan", "fanchuan_t0.07", ("fanchuan",)),
    ("NAU", "nau_ls0.01_ln0.01", ("nau_s0p01_n0p01",)),
    ("NAU (ls0.1)", "nau_ls0.1_ln0.01", ("nau",)),
]
RR = [("Retrained (2nd run)", "clean_b"), ("Retrained (3rd run)", "clean_c")]
GROUPS = ("whole", "neighbor")


def fname(ds, st, sd):
    return f"{ds}_seed{sd}.json" if st == "mid" else f"{ds}_{st}_seed{sd}.json"


def load_ledger(path):
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["setting"] == "spam" and r["run_dir"] and r["track"] == "tiger+rkmeans":
            out[(r["config"], r["dataset"], r["stratum"], int(r["seed"]))] = os.path.basename(r["run_dir"])
    return out


def rundir_of(ckpt):
    return os.path.basename(os.path.dirname(os.path.dirname(ckpt)))


def arrays(rows, arm, groups):
    """Concatenate the per-query arrays of `arm` over `groups` (in a fixed order)."""
    keys = ("ndcg_per_query", "sh_per_query", "rank_per_query", "label_logprob_per_query")
    out = {k: [] for k in keys}
    for g in groups:
        r = rows.get((arm, g))
        if r is None or not r["n"]:
            continue
        for k in keys:
            out[k] += r.get(k) or [np.nan] * r["n"]
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def cell_pairs(ds, st, sd, ledger, log):
    """{method: {group: (U arrays, R arrays)}} for one cell, canonical checkpoints only."""
    files = [(s, os.path.join(DIAG, s, fname(ds, st, sd))) for s in SOURCES]
    docs = []
    for s, f in files:
        if os.path.isfile(f):
            d = json.load(open(f))
            if d.get("unrelated_selection") != "all prefix-disjoint rows":
                continue
            docs.append((s, d, {(r["method"], r["group"]): r for r in d["rows"]}))
    out = {}
    for name, cfg, arms in METHODS:
        hit = None
        for s, d, rows in docs:
            for arm in arms:
                if (arm, "neighbor") not in rows:
                    continue
                if cfg is not None:
                    want = ledger.get((cfg, ds, st, sd))
                    got = rundir_of(d["checkpoints"].get(arm, ""))
                    if want != got:
                        log.append(f"non-canonical {ds} {st} s{sd} {name}: {s}:{arm} = {got} (ledger {want})")
                        continue
                hit = (s, arm, rows)
                break
            if hit:
                break
        if not hit:
            continue
        s, arm, rows = hit
        grp = {"whole": ("neighbor", "unrelated", "target"), "neighbor": ("neighbor",)}
        out[name] = {g: (arrays(rows, arm, gs), arrays(rows, "clean", gs)) for g, gs in grp.items()}
        out[name]["_src"] = f"{s}:{arm}"
    # retrained-vs-retrained reference (Beauty seed 2 only)
    f = os.path.join(DIAG, "fig1_scope_rr", fname(ds, st, sd))
    if os.path.isfile(f):
        d = json.load(open(f))
        rows = {(r["method"], r["group"]): r for r in d["rows"]}
        grp = {"whole": ("neighbor", "unrelated", "target"), "neighbor": ("neighbor",)}
        for name, arm in RR:
            if (arm, "neighbor") in rows:
                out[name] = {g: (arrays(rows, arm, gs), arrays(rows, "clean", gs)) for g, gs in grp.items()}
                out[name]["_src"] = f"fig1_scope_rr:{arm}"
    return out


def measures(pairs):
    """pairs: list of (U, R) per cell for one (method, group)."""
    nd_u = np.concatenate([u["ndcg_per_query"] for u, r in pairs])
    nd_r = np.concatenate([r["ndcg_per_query"] for u, r in pairs])
    sh_u = np.concatenate([u["sh_per_query"] for u, r in pairs])
    sh_r = np.concatenate([r["sh_per_query"] for u, r in pairs])
    rk_u = np.concatenate([u["rank_per_query"] for u, r in pairs])
    rk_r = np.concatenate([r["rank_per_query"] for u, r in pairs])
    hu, hr = nd_u > 0, nd_r > 0
    m = dict(n=int(len(nd_u)), n_cells=len(pairs),
             mad_ndcg=100 * np.mean(np.abs(nd_u - nd_r)), bias_ndcg=100 * np.mean(nd_u - nd_r),
             hit_agree=100 * np.mean(hu == hr),
             hit_jacc=100 * (np.sum(hu & hr) / max(1, np.sum(hu | hr))),
             rank_same=(100 * np.mean(rk_u == rk_r)) if not np.isnan(rk_u).any() else np.nan,
             sh_agree=100 * np.mean((sh_u > 0) == (sh_r > 0)),
             ndcg_u=100 * np.mean(nd_u), ndcg_r=100 * np.mean(nd_r))
    rho, w, lpd = [], [], []
    for u, r in pairs:
        a, b = u["label_logprob_per_query"], r["label_logprob_per_query"]
        if len(a) < 3 or np.isnan(a).any() or np.isnan(b).any():
            continue
        rho.append(spearmanr(a, b).statistic); w.append(len(a)); lpd.append(np.abs(a - b))
    m["lp_spear"] = float(np.average(rho, weights=w)) if rho else np.nan
    m["lp_mad"] = float(np.mean(np.concatenate(lpd))) if lpd else np.nan
    return m


def paired(pa, pb, group, n_boot=3000, seed=0):
    """NAU (pa) minus method (pb) on per-query |dNDCG| and hit agreement; same queries required."""
    def per_q(pairs):
        nd_u = np.concatenate([u["ndcg_per_query"] for u, r in pairs])
        nd_r = np.concatenate([r["ndcg_per_query"] for u, r in pairs])
        return 100 * np.abs(nd_u - nd_r), 100 * ((nd_u > 0) == (nd_r > 0)).astype(float), nd_r
    a1, a2, ra = per_q(pa)
    b1, b2, rb = per_q(pb)
    if len(a1) != len(b1) or not np.allclose(ra, rb):
        return None
    out = {}
    rng = np.random.default_rng(seed)
    for k, (x, y) in {"mad_ndcg": (a1, b1), "hit_agree": (a2, b2)}.items():
        d = x - y
        if group == "neighbor":
            bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
            lo, hi = np.percentile(bs, [2.5, 97.5])
        else:
            se = d.std(ddof=1) / np.sqrt(len(d)); lo, hi = d.mean() - 1.96 * se, d.mean() + 1.96 * se
        out[k] = (d.mean(), lo, hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "paper/plotting/canonical_runs.tsv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/plotting/per_query_agreement.csv"))
    ap.add_argument("--out-paired", default=os.path.join(ROOT, "paper/plotting/per_query_agreement_paired.csv"))
    ap.add_argument("-v", "--verbose", action="store_true")
    a = ap.parse_args()
    ledger = load_ledger(a.ledger)
    log = []
    C = {}
    for ds in DATASETS:
        for st in STRATA:
            for sd in SEEDS:
                C[(ds, st, sd)] = cell_pairs(ds, st, sd, ledger, log)
    if a.verbose:
        for l in sorted(set(log)):
            print(l)
    print(f"{len(set(log))} non-canonical arm-cells skipped (-v lists them)")

    scopes = []   # (scope, label, cells, method list or None=all available in all cells)
    scopes.append(("fig1", "beauty/mid", [("beauty", "mid", sd) for sd in SEEDS]))
    for ds in DATASETS:
        for st in STRATA:
            scopes.append(("block", f"{ds}/{st}", [(ds, st, sd) for sd in SEEDS]))
    scopes.append(("all9", "all 9 blocks", list(C)))
    scopes.append(("rr", "beauty seed 2 (3 strata)", [("beauty", st, 2) for st in STRATA]))
    names = [m[0] for m in METHODS] + [n for n, _ in RR]
    rows, prow = [], []
    for scope, label, cells in scopes:
        for name in names:
            have = [c for c in cells if name in C[c]]
            if len(have) < len(cells):   # a method is reported only where it covers every cell of the scope
                continue
            for g in GROUPS:
                m = measures([C[c][name][g] for c in have])
                rows.append(dict(scope=scope, block=label, method=name, group=g,
                                 sources=";".join(sorted({C[c][name]['_src'] for c in have})), **m))
                if name != "NAU" and "NAU" in C[have[0]] and all("NAU" in C[c] for c in have):
                    p = paired([C[c]["NAU"][g] for c in have], [C[c][name][g] for c in have], g)
                    if p:
                        prow.append(dict(scope=scope, block=label, method=name, group=g,
                                         **{f"nau_minus_{k}{s}": v for k, t in p.items() for s, v in zip(("", "_lo", "_hi"), t)}))
    fields = list(rows[0])
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else v) for k, v in r.items()})
    with open(a.out_paired, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(prow[0])); w.writeheader()
        for r in prow:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, (float, np.floating)) else v) for k, v in r.items()})
    print(f"wrote {a.out} ({len(rows)} rows), {a.out_paired} ({len(prow)} rows)")
    hdr = f"{'method':20s} {'n':>7s} {'|dNDCG|':>8s} {'bias':>7s} {'hitAgr':>7s} {'hitJac':>7s} {'rank=':>7s} {'shAgr':>7s} {'rho_lp':>7s} {'|dlp|':>6s}"
    for scope, label, _ in scopes:
        if scope == "block" and not a.verbose:
            continue
        for g in GROUPS:
            sel = [r for r in rows if r["scope"] == scope and r["block"] == label and r["group"] == g]
            if not sel:
                continue
            print(f"\n== {scope} {label} | {g}\n{hdr}")
            for r in sel:
                print(f"{r['method']:20s} {r['n']:7d} {r['mad_ndcg']:8.3f} {r['bias_ndcg']:7.3f} {r['hit_agree']:7.2f} "
                      f"{r['hit_jacc']:7.2f} {r['rank_same']:7.2f} {r['sh_agree']:7.2f} {r['lp_spear']:7.3f} {r['lp_mad']:6.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
