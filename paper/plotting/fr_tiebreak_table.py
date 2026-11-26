"""R2 Q2: NAU's selection rule (with its neighborhood tie-break) applied to the tuned Forget+Repair grid.

Grid (appendix robustness check, possible_additions.md section 9): Forget+Repair
(unified_abl_lf<lf>_ls0.0_ln0.0[_ep<N>]) on Beauty, lambda_f in {0.01, 0.03, 0.1, 0.3, 1.0} x
epochs in {2, 4, 8} (13 settings), 15 cells (3 targets x 5 data seeds), each with validation and
test metrics (`_valsel` runs, logs/valsel_basegrid*_plan.tsv).

Rule (paper/plotting/nau_val_table.py): among settings with whole-set validation tau_P >= 95%,
those whose validation Delta^retr_SH is not resolvably different from the lowest (paired per-cell
difference, seed-level 95% t-interval over the 5 data seeds includes 0) are tied; among the tied
settings the highest combined utility, mean of whole-set val tau_P and validation-neighborhood tau_P
(first-SID-token neighborhood, target's own cases excluded, tau_P = ratio of pooled NDCG@10), wins.

Whole-set validation numbers per setting come from ONE execution per cell: the canonical one
where it exists (lambda_f=0.1 ep4 = ledger forget_repair_lf0.1; the tie-break candidates = ledger
setting spam_frtie_candidate, the `_canonck` re-runs of run_fr_tiebreak_launch.sh, once complete),
else the grid's `_valsel` run. --grid-only uses the `_valsel` runs for every setting (step 1: the
tie set that decided which settings were re-run with checkpoints).
Neighborhood: logs/diagnose/fig1_scope_frtie_val/ (validation) and fig1_scope_frtie/ (test),
run_scope_frtie.sh; arm-cells are used only if the JSON's checkpoint is the ledger's run dir.
Then NAU (ledger nau_ls0.01_ln0.01) vs the picked F+R setting on the TEST neighborhood: paired
difference of NDCG@10 in points of the retrained model's neighborhood NDCG@10, 95% percentile
bootstrap over the pooled test queries (3000 resamples), as in Appendix G / plot_scope_grid_ls01.py.

    python paper/plotting/fr_tiebreak_table.py [--grid-only] [--mix-weight 0.5]
      -> paper/plotting/fr_tiebreak.csv (one row per setting) + printed summary
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import re
import statistics as S
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SEEDS = ("2", "3", "5", "7", "11")
STRATA = ("unpopular", "mid", "popular")
# scope arm -> (lambda_f, epochs); ledger config of the canonical execution
ARMS = {"fr_lf0p1_ep4": ("0.1", "4"), "fr_lf0p3_ep2": ("0.3", "2"), "fr_lf0p03_ep8": ("0.03", "8")}
LEDGER_CFG = {("0.1", "4"): ("spam", "forget_repair_lf0.1"), ("0.3", "2"): ("spam_frtie_candidate", "frtie_lf0.3_ep2"),
              ("0.03", "8"): ("spam_frtie_candidate", "frtie_lf0.03_ep8")}
NAU_CFG = ("spam", "nau_ls0.01_ln0.01")


def last(f):
    r = list(csv.DictReader(open(f)))[-1]
    return float(r["test/SH@10"]), float(r["test/ndcg@10"])


def val_refs():
    R = {}
    for d in glob.glob(os.path.join(ROOT, "logs/diagnose/valsel_refs/rkmeans_beauty_*_clean")):
        m = re.match(r"rkmeans_beauty_(\w+)_s(\d+)_clean$", os.path.basename(d))
        f = d + "/csv/version_0/metrics.csv"
        if m and os.path.isfile(f):
            R[m.groups()] = last(f)
    return R


def test_refs():
    out = {}
    for sd in SEEDS:
        d = json.load(open(os.path.join(ROOT, "paper/data/spam_references", "beauty" + ("" if sd == "2" else f"_seed{sd}") + ".json")))
        for st in STRATA:
            out[(st, sd)] = (d["clean_SH@10"][st], d["clean_ndcg@10"])
    return out


def grid_runs():
    """(lf, ep) -> {(st, sd): run_dir} from the recorded `_valsel` grid runs (latest job with val)."""
    runs = {}
    for d in glob.glob(os.path.join(ROOT, "logs/unlearn/runs/*_bs1_bw_tgt*_unified_abl_lf*_ls0.0_ln0.0*_valsel*")):
        m = re.search(r"bw_tgt(\w+?)_unified_abl_lf([0-9.]+)_ls0.0_ln0.0(?:_ep(\d+))?_valsel(?:_seed(\d+))?$", os.path.basename(d))
        if not m or m.group(2) == "0.0":
            continue
        if not os.path.isfile(d + "/eval_val/csv/version_0/metrics.csv") or not os.path.isfile(d + "/eval/csv/version_0/metrics.csv"):
            continue
        st, lf, ep, sd = m.group(1), m.group(2), m.group(3) or "4", m.group(4) or "2"
        j = int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))
        cur = runs.setdefault((lf, ep), {}).get((st, sd))
        if cur is None or j > cur[0]:
            runs[(lf, ep)][(st, sd)] = (j, d)
    return {k: {c: v[1] for c, v in cc.items()} for k, cc in runs.items()}


def ledger(path):
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["dataset"] == "beauty" and r["track"] == "tiger+rkmeans":
            out.setdefault((r["setting"], r["config"]), {})[(r["stratum"], r["seed"])] = (r["run_dir"], r["status"])
    return out


def whole(cells, R, val=True):
    """Pooled like Table 1 (mean over seeds of the mean over strata); also per-cell val Delta_SH."""
    per = {}
    for c, d in cells.items():
        f = os.path.join(d, ("eval_val" if val else "eval") + "/csv/version_0/metrics.csv")
        if not d or not os.path.isfile(f) or c not in R:
            continue
        sh, nd = last(f)
        per[c] = (100 * abs(sh - R[c][0]), nd / R[c][1])
    by = {}
    for (st, sd), v in per.items():
        by.setdefault(sd, []).append(v)
    if not per:
        return None, {}
    return (S.mean(S.mean(x[0] for x in v) for v in by.values()),
            100 * S.mean(S.mean(x[1] for x in v) for v in by.values()), len(per)), {c: v[0] for c, v in per.items()}


def tied_with(V, a, b):
    by = {}
    for c in sorted(set(V[a]) & set(V[b])):
        by.setdefault(c[1], []).append(V[a][c] - V[b][c])
    x = np.array([np.mean(v) for v in by.values()])
    if np.allclose(x, 0):
        return True, 0.0, 0.0
    half = 2.776 * x.std(ddof=1) / np.sqrt(len(x))   # t(0.975, df=4)
    return bool(x.mean() - half <= 0 <= x.mean() + half), x.mean() - half, x.mean() + half


def scope(stem, L):
    """Neighborhood per arm from logs/diagnose/<stem>: pooled NDCG ratio, per-query arrays and the
    retrained reference, canonical checkpoints only. Returns {arm: dict(nd, ref, cells, sh)}."""
    out = {}
    want = {a: L.get(LEDGER_CFG[k], {}) for a, k in ARMS.items()}
    want["nau"] = L.get(NAU_CFG, {})
    for f in sorted(glob.glob(os.path.join(ROOT, "logs/diagnose", stem, "beauty*_seed*.json"))):
        d = json.load(open(f))
        st, sd = d["stratum"], str(d["seed"])
        rows = {(r["method"], r["group"]): r for r in d["rows"]}
        c = rows.get(("clean", "neighbor"))
        if not c or not c["n"]:
            continue
        for arm in list(ARMS) + ["nau"]:
            r = rows.get((arm, "neighbor"))
            if r is None:
                continue
            led = want[arm].get((st, sd), ("", ""))[0]
            got = os.path.dirname(os.path.dirname(d["checkpoints"].get(arm, "")))
            if not led or os.path.basename(led) != os.path.basename(got):
                print(f"  {stem}: {arm} {st} s{sd} checkpoint is not the ledger's run, dropped ({os.path.basename(got)})")
                continue
            o = out.setdefault(arm, {"nd": {}, "ref": {}, "sh": {}, "csh": {}})
            o["nd"][(st, sd)] = np.asarray(r["ndcg_per_query"], float)
            o["ref"][(st, sd)] = np.asarray(c["ndcg_per_query"], float)
            o["sh"][(st, sd)] = r["sh_at_10"]; o["csh"][(st, sd)] = c["sh_at_10"]
    return out


def nbr_tau(o):
    cells = sorted(o["nd"])
    nd = np.concatenate([o["nd"][c] for c in cells]); ref = np.concatenate([o["ref"][c] for c in cells])
    return 100 * nd.mean() / ref.mean(), len(cells)


def paired_gain(a, b, n_boot=3000):
    """NAU (a) minus F+R setting (b) on the shared test neighborhood queries, in points of the
    retrained neighborhood NDCG@10; 95% percentile bootstrap over queries."""
    cells = sorted(set(a["nd"]) & set(b["nd"]))
    x = np.concatenate([a["nd"][c] for c in cells]); y = np.concatenate([b["nd"][c] for c in cells])
    ra = np.concatenate([a["ref"][c] for c in cells]); rb = np.concatenate([b["ref"][c] for c in cells])
    assert np.allclose(ra, rb), "retrained reference differs between arms: queries not shared"
    d = x - y; s = 100 / ra.mean()
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    by_cell = [np.mean(a["nd"][c]) - np.mean(b["nd"][c]) for c in cells]
    return s * d.mean(), s * lo, s * hi, len(cells), len(d), sum(v > 0 for v in by_cell), sum(v < 0 for v in by_cell)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=os.path.join(ROOT, "paper/plotting/canonical_runs.tsv"))
    ap.add_argument("--grid-only", action="store_true", help="whole-set val from the grid `_valsel` runs only (step 1)")
    ap.add_argument("--mix-weight", type=float, default=0.5)
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/plotting/fr_tiebreak.csv"))
    a = ap.parse_args()
    Rv, Rt, L, G = val_refs(), test_refs(), ledger(a.ledger), grid_runs()
    src, cells_of = {}, {}
    for k, cells in sorted(G.items()):
        cells_of[k], src[k] = cells, "grid _valsel"
        if not a.grid_only and k in LEDGER_CFG:
            lc = L.get(LEDGER_CFG[k], {})
            ok = {c: v[0] for c, v in lc.items() if v[1] == "ok" and v[0]}
            if len(ok) == 15 and all(os.path.isfile(os.path.join(d, "eval_val/csv/version_0/metrics.csv")) for d in ok.values()):
                cells_of[k], src[k] = {c: os.path.join(ROOT, d) if not os.path.isabs(d) else d for c, d in ok.items()}, f"ledger {LEDGER_CFG[k][1]}"
            elif LEDGER_CFG[k][0] != "spam":
                src[k] = f"grid _valsel (ledger {LEDGER_CFG[k][1]}: {len(ok)}/15 done)"
    W, V, T = {}, {}, {}
    for k, cells in cells_of.items():
        w, v = whole(cells, Rv, val=True)
        t, _ = whole(cells, Rt, val=False)
        if w and w[2] == 15:
            W[k], V[k], T[k] = w, v, t
    elig = [k for k in W if W[k][1] >= 95.0]
    best = min(elig, key=lambda k: W[k][0])
    tie = {k: tied_with(V, k, best) for k in elig}
    tied = [k for k in elig if tie[k][0]]
    print(f"validation whole set ({'grid only' if a.grid_only else 'canonical where available'}); "
          f"eligible (tau_P >= 95): {len(elig)} of {len(W)}; lowest Delta_SH: lf={best[0]} ep={best[1]}")
    print(f"{'lf':>5s} {'ep':>2s} {'val dSH':>8s} {'val tau':>7s} {'test dSH':>8s} {'test tau':>8s}  tie-interval (vs lowest)   source")
    for k in sorted(W, key=lambda k: (float(k[0]), int(k[1]))):
        flag = ("TIED " if k in tied else "     ") if k in elig else "<95  "
        iv = f"[{tie[k][1]:+.3f}, {tie[k][2]:+.3f}]" if k in elig else ""
        print(f"{k[0]:>5s} {k[1]:>2s} {W[k][0]:8.3f} {W[k][1]:7.1f} {T[k][0]:8.3f} {T[k][1]:8.1f}  {flag}{iv:22s} {src[k]}")
    print("tie set:", ", ".join(f"lf={k[0]} ep={k[1]}" for k in tied))
    if a.grid_only:
        return 0

    Nv, Nt = scope("fig1_scope_frtie_val", L), scope("fig1_scope_frtie", L)
    karm = {v: k for k, v in ARMS.items()}
    rows, mix = [], {}
    for k in sorted(W, key=lambda k: (float(k[0]), int(k[1]))):
        arm = karm.get(k)
        nv = nbr_tau(Nv[arm]) if arm in Nv else (None, 0)
        nt = nbr_tau(Nt[arm]) if arm in Nt else (None, 0)
        if k in tied and nv[1] == 15:
            mix[k] = (1 - a.mix_weight) * W[k][1] + a.mix_weight * nv[0]
        rows.append(dict(lambda_f=k[0], epochs=k[1], source=src[k], val_dsh=W[k][0], val_tau=W[k][1],
                         test_dsh=T[k][0], test_tau=T[k][1], eligible=int(k in elig), tied=int(k in tied),
                         val_nbr_tau=nv[0], val_nbr_cells=nv[1], test_nbr_tau=nt[0], test_nbr_cells=nt[1],
                         combined=mix.get(k)))
    missing = [k for k in tied if k not in mix]
    pick = max(mix, key=mix.get) if mix and not missing else None
    for r in rows:
        r["picked"] = int(pick is not None and (r["lambda_f"], r["epochs"]) == pick)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader()
        for r in rows:
            w.writerow({k: (f"{v:.4f}" if isinstance(v, float) else ("" if v is None else v)) for k, v in r.items()})
    print(f"wrote {a.out}")
    for k in tied:
        arm = karm.get(k)
        nv = nbr_tau(Nv[arm]) if arm in Nv else (float("nan"), 0)
        print(f"  tied lf={k[0]} ep={k[1]}: val whole tau {W[k][1]:.1f}, val nbr tau {nv[0]:.1f} ({nv[1]} cells), "
              f"combined {mix.get(k, float('nan')):.1f}")
    if "nau" in Nv:
        print(f"  NAU (ledger nau_ls0.01_ln0.01) val nbr tau {nbr_tau(Nv['nau'])[0]:.1f} ({nbr_tau(Nv['nau'])[1]} cells)")
    if missing:
        print("WAITING: validation neighborhood incomplete for", ", ".join(f"lf={k[0]} ep={k[1]}" for k in missing),
              "(run_scope_frtie.sh SPLIT=val); no pick yet")
        return 0
    grid_w = [x / 1000 for x in range(1001)]
    hold = [x for x in grid_w if max(tied, key=lambda k: (1 - x) * W[k][1] + x * nbr_tau(Nv[karm[k]])[0]) == pick]
    print(f"PICK: Forget+Repair lambda_f={pick[0]}, {pick[1]} epochs (holds for neighborhood weight w in "
          f"[{min(hold):.3f}, {max(hold):.3f}]; used w={a.mix_weight})")
    print(f"  pick whole set test: Delta_SH {T[pick][0]:.3f}, tau_P {T[pick][1]:.1f}")
    if "nau" not in Nt:
        print("WAITING: test neighborhood (run_scope_frtie.sh SPLIT=test)")
        return 0
    for k in tied:
        arm = karm[k]
        if arm not in Nt:
            continue
        g = paired_gain(Nt["nau"], Nt[arm])
        print(f"  NAU minus F+R lf={k[0]} ep={k[1]}{' (PICK)' if k == pick else ''} on the TEST neighborhood: "
              f"{g[0]:+.1f} [{g[1]:+.1f}, {g[2]:+.1f}] points of retrained nbr NDCG ({g[3]} cells, {g[4]} queries; "
              f"NAU higher in {g[5]}, lower in {g[6]} cells); test nbr tau_P NAU {nbr_tau(Nt['nau'])[0]:.1f} vs {nbr_tau(Nt[arm])[0]:.1f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
