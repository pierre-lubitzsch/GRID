"""Reviewer QzxM follow-up 1 (v2): per-query target presence vs the retrained model, with the
retrained-vs-retrained reference from FOUR further independently retrained models per cell.

Sources
  Unlearned arms (Poisoned, Forget+Repair, NAU) and the cell's canonical retrained model R:
    paper/plotting/per_query_agreement.cell_pairs (fig1_scope_allunrel_fix / v3 / v2, canonical
    checkpoint check against paper/plotting/canonical_runs.tsv).
  Alternative retrained models: logs/diagnose/fig1_scope_rr5/<cell>.json (run_scope_rr5.sh), arms
    clean_s<o> (the canonical retrained models of the other data seeds of the same dataset; all five
    are trained on identical data with different training seeds). Each of them is compared with the
    canonical "clean" arm re-scored IN THE SAME rr5 JSON. That re-scored R must be on the same queries
    as R in the v2/v3 JSON of the cell (asserted per group: same n; agreement of the per-query
    target-presence and NDCG arrays is reported).

Per-query fields: sh_per_query (promoted target in the top 10 for that test query). Whole test set =
groups neighbor + unrelated + target, concatenated in a fixed order (per_query_agreement.arrays).
Queries are pooled over the 5 data seeds of a (dataset, stratum) block.

Measures (all %), model M vs reference R:
  exposure  P(M shows target)
  kept      P(M shows | R shows)          does M preserve R's remaining target recommendations
  extra     P(M shows | R does not)
  agree     P(M and R agree on presence)
  jacc      |M & R| / |M or R| on target presence
  lp_rho    Spearman of log p(true next item) between M and R (per cell, weighted by queries)
Alternative retrained row: "mean" pools all (cell, alternative) pairs (each query 4x); "min"/"max"
pool, over the 5 seeds, each cell's lowest / highest of its 4 alternatives on that measure (the
denominators are identical across the alternatives of a cell, so this is a per-cell envelope).

    python paper/plotting/critique_followup/popular_target_agreement_rr5.py [--rr-dir fig1_scope_rr5]
      -> popular_target_agreement_rr5.csv (next to this script)
      -> paper/revision/response_QzxM_followup_Q1_tables.md
"""
import argparse
import csv
import json
import os
import sys

import numpy as np
from scipy.stats import spearmanr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import per_query_agreement as pqa  # noqa: E402

GROUPS = ("neighbor", "unrelated", "target")
UNL = ["Poisoned", "Forget+Repair", "NAU"]
DSN = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys"}
STN = {"popular": "Popular", "mid": "Mid-popularity", "unpopular": "Unpopular"}


def counts(su, sr):
    su, sr = su > 0, sr > 0
    return dict(n=len(su), r_shown=int(sr.sum()), m_shown=int(su.sum()), both=int((su & sr).sum()),
                m_only=int((su & ~sr).sum()), n_agree=int((su == sr).sum()), union=int((su | sr).sum()))


def pct(c):
    return dict(exposure=100 * c["m_shown"] / c["n"],
                kept=100 * c["both"] / c["r_shown"] if c["r_shown"] else np.nan,
                extra=100 * c["m_only"] / (c["n"] - c["r_shown"]),
                agree=100 * c["n_agree"] / c["n"],
                jacc=100 * c["both"] / c["union"] if c["union"] else np.nan)


def add(cs):
    return {k: sum(c[k] for c in cs) for k in cs[0]}


def lp_rho(pairs):
    rho, w = [], []
    for a, b in pairs:
        if len(a) >= 3 and not (np.isnan(a).any() or np.isnan(b).any()):
            rho.append(spearmanr(a, b).statistic); w.append(len(a))
    return float(np.average(rho, weights=w)) if rho else np.nan


def rr_cell(rr_dir, ds, st, sd, ref_rows):
    """Alternative retrained models of one cell: {arm: (M arrays, R arrays)} plus alignment check."""
    f = os.path.join(pqa.DIAG, rr_dir, pqa.fname(ds, st, sd))
    if not os.path.isfile(f):
        return None, None
    d = json.load(open(f))
    assert d.get("unrelated_selection") == "all prefix-disjoint rows", f
    rows = {(r["method"], r["group"]): r for r in d["rows"]}
    # same queries as the v2/v3 JSON of the cell
    match = []
    for g in GROUPS:
        a = pqa.arrays(rows, "clean", (g,)); b = pqa.arrays(ref_rows, "clean", (g,))
        assert len(a["sh_per_query"]) == len(b["sh_per_query"]), (f, g, len(a["sh_per_query"]), len(b["sh_per_query"]))
        match.append((a["sh_per_query"] == b["sh_per_query"]).mean() if len(a["sh_per_query"]) else 1.0)
        match.append(np.isclose(a["ndcg_per_query"], b["ndcg_per_query"]).mean() if len(a["ndcg_per_query"]) else 1.0)
    R = pqa.arrays(rows, "clean", GROUPS)
    alts = sorted({m for m, g in rows if m.startswith("clean_")})
    return {m: (pqa.arrays(rows, m, GROUPS), R) for m in alts}, min(match)


def fmt(v, dec=1):
    return "n/a" if v is None or (isinstance(v, float) and np.isnan(v)) else f"{v:.{dec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rr-dir", default="fig1_scope_rr5")
    ap.add_argument("--out", default=os.path.join(HERE, "popular_target_agreement_rr5.csv"))
    ap.add_argument("--md", default=os.path.join(pqa.ROOT, "paper/revision/response_QzxM_followup_Q1_tables.md"))
    a = ap.parse_args()
    ledger = pqa.load_ledger(os.path.join(pqa.ROOT, "paper/plotting/canonical_runs.tsv"))
    log, out, missing, align = [], [], [], []
    res = {}
    for st in ("popular", "mid", "unpopular"):
        for ds in pqa.DATASETS:
            C = {k: [] for k in ["Retrained (R)"] + UNL}
            LP = {k: [] for k in UNL}
            alt_cells, alt_lp = [], []   # per cell: list of 4 count dicts
            seeds_ok = []
            for sd in pqa.SEEDS:
                cell = pqa.cell_pairs(ds, st, sd, ledger, log)
                if not all(m in cell for m in UNL):
                    missing.append(f"{ds}/{st}/s{sd}: unlearned arms {[m for m in UNL if m not in cell]}")
                    continue
                src = cell["NAU"]["_src"].split(":")[0]
                ref = json.load(open(os.path.join(pqa.DIAG, src, pqa.fname(ds, st, sd))))
                ref_rows = {(r["method"], r["group"]): r for r in ref["rows"]}
                # every unlearned arm must share R (same queries) with the NAU source JSON
                R0 = cell["NAU"]["whole"][1]["sh_per_query"]
                for m in UNL:
                    assert np.array_equal(cell[m]["whole"][1]["sh_per_query"], R0), (ds, st, sd, m)
                alts, mt = rr_cell(a.rr_dir, ds, st, sd, ref_rows)
                if alts is None or len(alts) == 0:
                    missing.append(f"{ds}/{st}/s{sd}: {a.rr_dir} JSON")
                    continue
                align.append((ds, st, sd, mt, len(alts)))
                seeds_ok.append(sd)
                u, r = cell["NAU"]["whole"]
                C["Retrained (R)"].append(counts(r["sh_per_query"], r["sh_per_query"]))
                for m in UNL:
                    u, r = cell[m]["whole"]
                    C[m].append(counts(u["sh_per_query"], r["sh_per_query"]))
                    LP[m].append((u["label_logprob_per_query"], r["label_logprob_per_query"]))
                alt_cells.append([counts(M["sh_per_query"], R["sh_per_query"]) for M, R in alts.values()])
                alt_lp += [(M["label_logprob_per_query"], R["label_logprob_per_query"]) for M, R in alts.values()]
            if not seeds_ok:
                continue
            blk = {}
            for m, cs in C.items():
                c = add(cs)
                blk[m] = dict(c=c, p=pct(c), lp=lp_rho(LP[m]) if m in LP else 1.0)
            c = add([x for cc in alt_cells for x in cc])
            p = pct(c)
            env = {}
            for k in ("exposure", "kept", "extra", "agree", "jacc"):
                lo, hi = [], []
                for cc in alt_cells:
                    vals = [(pct(x)[k], x) for x in cc]
                    vals = [v for v in vals if not np.isnan(v[0])]
                    if vals:
                        lo.append(min(vals, key=lambda t: t[0])[1]); hi.append(max(vals, key=lambda t: t[0])[1])
                env[k] = (pct(add(lo))[k] if lo else np.nan, pct(add(hi))[k] if hi else np.nan)
            blk["Retrained (R)"]["per_seed"] = [c["r_shown"] for c in C["Retrained (R)"]]
            blk["Other retrained"] = dict(c=c, p=p, lp=lp_rho(alt_lp), env=env, n_alt=sum(len(cc) for cc in alt_cells))
            res[(st, ds)] = (blk, seeds_ok)
            for m, b in blk.items():
                row = dict(stratum=st, dataset=ds, model=m, seeds=" ".join(map(str, seeds_ok)), **b["c"],
                           **{k: round(v, 4) for k, v in b["p"].items()}, lp_rho=round(b["lp"], 4))
                if "env" in b:
                    for k, (lo, hi) in b["env"].items():
                        row[f"{k}_min"], row[f"{k}_max"] = round(lo, 4), round(hi, 4)
                    row["n_alt_pairs"] = b["n_alt"]
                out.append(row)

    fields = []
    for r in out:
        fields += [k for k in r if k not in fields]
    with open(a.out, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader(); w.writerows(out)

    # markdown tables: one per stratum
    order = ["Retrained (R)", "Other retrained", "Poisoned", "Forget+Repair", "NAU"]
    disp = {"Retrained (R)": "Retrained R", "Other retrained": "Other retrained (4 per cell)"}
    md = ["# QzxM follow-up Q1: per-query target presence vs the retrained model R", "",
          "All values in %. Test queries pooled over the 5 data seeds of each block (whole test set: "
          "neighbor + unrelated + target-item queries). R = the cell's canonical retrained model. "
          "Other retrained = the 4 other independently retrained models of the dataset (same data, "
          "different training seed), each compared with R on the same queries: mean, with min to max "
          "(each cell's lowest and highest of its 4 alternatives, pooled over seeds) in brackets. "
          "Exposure = queries with the target in the top 10. Kept = of the queries where R shows the "
          "target, share where the model also shows it. Agree = queries where model and R agree on "
          "target presence. Jaccard = overlap of the two models' target-showing query sets.", ""]
    for st in ("popular", "mid", "unpopular"):
        dss = [ds for ds in pqa.DATASETS if (st, ds) in res]
        if not dss:
            continue
        md.append(f"## {STN[st]} targets")
        md.append("")
        n_r = ", ".join(f"{DSN[ds]} {res[(st, ds)][0]['Retrained (R)']['c']['r_shown']} of "
                        f"{res[(st, ds)][0]['Retrained (R)']['c']['n']}" for ds in dss)
        md.append(f"Queries where R shows the target (denominator of Kept): {n_r}. Per seed (2, 3, 5, 7, 11): "
                  + "; ".join(f"{DSN[ds]} " + ", ".join(map(str, res[(st, ds)][0]['Retrained (R)']['per_seed'])) for ds in dss)
                  + ". Kept is n/a where R never shows the target.")
        md.append("")
        for k, title, dec in (("exposure", "Exposure", 2), ("kept", "Kept", 1), ("jacc", "Jaccard", 1), ("agree", "Agree", 2)):
            md.append(f"| {title} | " + " | ".join(DSN[ds] for ds in dss) + " |")
            md.append("|---|" + "---|" * len(dss))
            for m in order:
                if m == "Retrained (R)" and k != "exposure":
                    continue
                cells = []
                for ds in dss:
                    b = res[(st, ds)][0][m]
                    s = fmt(b["p"][k], dec)
                    if "env" in b:
                        lo, hi = b["env"][k]
                        s += f" [{fmt(lo, dec)} to {fmt(hi, dec)}]"
                    cells.append(s)
                md.append(f"| {disp.get(m, m)} | " + " | ".join(cells) + " |")
            md.append("")
    if missing:
        md += ["Missing cells: " + "; ".join(missing), ""]
    with open(a.md, "w") as fh:
        fh.write("\n".join(md))

    print("\n".join(md))
    print("\nalignment (min share of identical per-query sh/ndcg arrays, rr5 clean vs v2/v3 clean; n alts):")
    for x in align:
        flag = "" if x[3] == 1.0 else "  <-- not identical"
        print(f"  {x[0]}/{x[1]}/s{x[2]}: {x[3]:.4f}  alts={x[4]}{flag}")
    print("\nlp_rho (Spearman of log p(true next item) vs R):")
    for (st, ds), (blk, _) in res.items():
        print(f"  {st:9s} {ds:6s} " + "  ".join(f"{m}={blk[m]['lp']:.3f}" for m in order[1:]))
    if missing:
        print("\nMISSING:", *missing, sep="\n  ")
    print(f"\nwrote {a.out}, {a.md}")


if __name__ == "__main__":
    main()
