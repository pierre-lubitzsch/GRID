"""Normal vs cleaned test histories, every row of the sensitive item deletion tables (3, 4, 8, 22, 23).

One line per (dataset, track, method/config) and view, from the ledger's checkpointed executions
(`unwanted` = `_mia`, `unwanted_canonck` = Table 4 / Table 8 NAU re-runs) plus the original and
retrained references: mean over seeds of UHF@10, UHR@10, Delta^retr_UHR, NDCG@10 and tau_P
(against the retrained model of the same seed and view), and the per-seed UHF@10. Normal view =
the runs' own test evals with the original at its start checkpoint
(GRID_SENSITIVE_BASE_REF_ROOT=logs/eval/sensitive_reference_lastckpt semantics; set it in the env);
scrub = logs/eval/sensitive_scrub/scrub (unwanted_ledger.set_view).

    GRID_SENSITIVE_BASE_REF_ROOT=logs/eval/sensitive_reference_lastckpt PYTHONPATH=.:paper/plotting \
        python paper/plotting/sensitive_view_compare.py --ledger paper/plotting/canonical_runs.tsv
        [--csv paper/revision/tables/sensitive_view_compare.csv]
"""
import argparse
import csv
import os
import statistics as S

import unwanted_ledger as ul

DS = ("beauty", "sports", "toys")


def summarize(runs):
    if not runs:
        return None
    m = lambda k: S.mean(100 * r[k] for r in runs if r[k] is not None)  # noqa: E731
    return dict(n=len(runs), UHF=m("sensF"), UHR=m("sensR"), dUHR=m("dUHR"), NDCG=m("n10"), tauP=m("tau_P"),
                per_seed=" ".join(f"s{r['seed']}:{100 * r['sensF']:.1f}" for r in runs))


def ref_entries(ds, model, quant, seeds):
    out = {"original": [], "retrained": []}
    for sd in seeds:
        base, retr, _bp, _rp = ul.refs_for(ds, sd, model, quant)
        if not base or not retr:
            continue
        for role, m in (("original", base), ("retrained", retr)):
            out[role].append(ul.es.cell_entry(m, base, retr, sd, None, "0", ""))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--csv", default=None)
    a = ap.parse_args()
    rows = []
    for view in ("normal", "scrub"):
        ul.set_view(view)
        for ds in DS:
            mia = ul.mia_runs(a.ledger, ds, "eval")
            for (fam, cfg), rr in sorted(mia.items()):
                rows.append(dict(view=view, dataset=ds, track="tiger+rkmeans", source="mia", method=fam, config=cfg,
                                 **(summarize(rr) or {})))
            for track in ("tiger+rkmeans", "letter+letter"):
                cc = ul.canonck_runs(a.ledger, ds, track, "eval")
                for cfg, rr in sorted(cc.items()):
                    meth = next((f for f in ("scif", "seif", "kookmin", "fanchuan", "tracer") if cfg.startswith(f)), "unified")
                    rows.append(dict(view=view, dataset=ds, track=track, source="canonck", method=meth,
                                     config=cfg, **(summarize(rr) or {})))
                model, quant = track.split("+")
                for role, rr in ref_entries(ds, model, quant, (2, 3, 5, 7, 11)).items():
                    rows.append(dict(view=view, dataset=ds, track=track, source="reference", method=role, config=role,
                                     **(summarize(rr) or {})))
    key = lambda r: (r["dataset"], r["track"], r["source"], r["method"], r["config"])  # noqa: E731
    by = {}
    for r in rows:
        by.setdefault(key(r), {})[r["view"]] = r
    print(f"{'dataset':7s} {'track':14s} {'method':13s} {'config':42s} {'n':>3s}  UHF normal->scrub   UHR n->s     "
          f"dUHR n->s     tauP n->s")
    for k, v in by.items():
        nv, sv = v.get("normal", {}), v.get("scrub", {})
        g = lambda d, f, p=2: f"{d[f]:6.{p}f}" if f in d else "   n/a"  # noqa: E731
        print(f"{k[0]:7s} {k[1]:14s} {k[3]:13s} {k[4][:42]:42s} {sv.get('n', nv.get('n', 0)):3d}  "
              f"{g(nv, 'UHF')} -> {g(sv, 'UHF')}  {g(nv, 'UHR')} -> {g(sv, 'UHR')}  "
              f"{g(nv, 'dUHR')} -> {g(sv, 'dUHR')}  {g(nv, 'tauP', 1)} -> {g(sv, 'tauP', 1)}")
    if a.csv:
        os.makedirs(os.path.dirname(a.csv) or ".", exist_ok=True)
        cols = ["view", "dataset", "track", "source", "method", "config", "n", "UHF", "UHR", "dUHR", "NDCG", "tauP",
                "per_seed"]
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, cols, lineterminator="\n")
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)


if __name__ == "__main__":
    main()
