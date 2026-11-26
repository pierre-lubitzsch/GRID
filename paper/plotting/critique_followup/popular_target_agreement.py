"""Reviewer QzxM follow-up 1: per-query target exposure when the retrained model still shows the target.

For each (dataset, stratum, seed) cell, the stored scope JSONs (same sources and canonical-checkpoint
check as paper/plotting/per_query_agreement.py) hold sh_per_query (target in the top 10) on every test
query for the retrained model R and each unlearned arm U. Per method, pooled over the cells of a block:
  R_shown   queries where R shows the target
  kept      P(U shows | R shows), %: does U preserve the retrained model's remaining target exposure
  extra     P(U shows | R does not), %: exposure R does not have (leftover attack)
  agree     P(U and R agree on target presence), %
Reference: two further retrained models of Beauty seed 2 (fig1_scope_rr), compared with R the same way.

    python paper/plotting/critique_followup/popular_target_agreement.py
      -> paper/plotting/critique_followup/popular_target_agreement.csv
"""
import csv
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import per_query_agreement as pqa  # noqa: E402

OUT = os.path.join(HERE, "popular_target_agreement.csv")
SHOW = ["Poisoned", "Finetune", "Forget+Repair", "NAU", "Retrained (2nd run)", "Retrained (3rd run)"]


def stats(pairs):
    su = np.concatenate([u["sh_per_query"] for u, r in pairs]) > 0
    sr = np.concatenate([r["sh_per_query"] for u, r in pairs]) > 0
    n_r = int(sr.sum())
    return dict(n=len(su), n_cells=len(pairs), R_shown=n_r, U_shown=int(su.sum()),
                kept=100 * (su & sr).sum() / n_r if n_r else np.nan,
                extra=100 * (su & ~sr).sum() / max(1, (~sr).sum()),
                agree=100 * np.mean(su == sr))


def main():
    ledger = pqa.load_ledger(os.path.join(os.path.dirname(HERE), "canonical_runs.tsv"))
    log, rows = [], []
    for st in ("popular", "mid", "unpopular"):
        for ds in pqa.DATASETS:
            acc = {}
            for sd in pqa.SEEDS:
                cell = pqa.cell_pairs(ds, st, sd, ledger, log)
                for name, g in cell.items():
                    acc.setdefault(name, []).append(g["whole"])
            for name in SHOW:
                if name in acc:
                    rows.append(dict(stratum=st, dataset=ds, method=name, **stats(acc[name])))
            # same-cell comparison for the retrained-vs-retrained reference (Beauty seed 2)
            cell = pqa.cell_pairs(ds, st, 2, ledger, [])
            if "Retrained (2nd run)" in cell:
                for name in ("Poisoned", "Forget+Repair", "NAU"):
                    if name in cell:
                        rows.append(dict(stratum=st, dataset=ds, method=name + " [seed 2 only]",
                                         **stats([cell[name]["whole"]])))
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    for r in rows:
        print(f"{r['stratum']:9s} {r['dataset']:6s} {r['method']:30s} cells={r['n_cells']} n={r['n']:6d} "
              f"R_shown={r['R_shown']:5d} U_shown={r['U_shown']:5d} kept={r['kept']:5.1f} "
              f"extra={r['extra']:5.2f} agree={r['agree']:6.2f}")
    if log:
        print(f"{len(log)} non-canonical arm-cells skipped")


if __name__ == "__main__":
    main()
