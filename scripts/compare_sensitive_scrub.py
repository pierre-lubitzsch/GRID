"""Normal vs scrubbed test history, unwanted-item track (TIGER + RQ-KMeans).

Reads logs/eval/sensitive_scrub/plan.tsv and the metrics.csv each row produced
(run_sensitive_scrub_eval.sh). Per dataset and method, mean over seeds:
    UHF@10 (= test/SHF@10), UHR@10 (= test/SHR@10), NDCG@10,
    tau_P = NDCG / NDCG(retrained, same seed, SAME view)
on the normal and the scrubbed view, plus per-seed UHF@10. Also checks that the
normal-view re-evaluation reproduces the recorded numbers (the `_mia` runs' own
post-eval; the retrained reference eval), which is the same-code sanity check.

    python scripts/compare_sensitive_scrub.py [--csv out.csv]
"""
import argparse
import csv
import os
import statistics as St
import sys



PLAN = "logs/eval/sensitive_scrub/plan.tsv"
ORDER = ["original", "retrained", "finetune", "forget_only", "forget_repair", "scif", "seif", "kookmin",
         "fanchuan", "tracer", "nau"]
KEYS = {"UHF": "test/SHF@10", "UHR": "test/SHR@10", "NDCG": "test/ndcg@10"}


def _read(p):
    if not os.path.isfile(p):
        return None
    best = {}
    with open(p) as fh:
        for row in csv.DictReader(fh):
            for k, v in row.items():
                try:
                    best[k] = float(v)
                except (TypeError, ValueError):
                    pass
    return best or None


def recorded(row):
    """The number the paper pipeline recorded for this model on the normal view."""
    if row["role"] == "unlearned":
        run = os.path.join("logs/unlearn/runs", row["out_dir"].split("/runs/")[1].split("/")[0])
        return _read(os.path.join(run, "eval", "csv", "version_0", "metrics.csv"))
    if row["role"] == "retrained":
        return _read(os.path.join(row["source"], "csv", "version_0", "metrics.csv"))
    return None


def mean(v):
    v = [x for x in v if x is not None]
    return (St.mean(v), len(v)) if v else (None, 0)


def f(x, s=100, p=2):
    return "  n/a " if x is None else f"{s * x:6.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--csv", default=None, help="also write the per-seed rows here")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.plan), delimiter="\t"))
    M = {}   # (ds, method, view, seed) -> metrics
    missing = 0
    repro = []
    for r in rows:
        m = _read(os.path.join(r["out_dir"], "csv", "version_0", "metrics.csv"))
        if m is None:
            missing += 1
            continue
        M[(r["dataset"], r["method"], r["view"], int(r["seed"]))] = m
        if r["view"] == "normal":
            rec = recorded(r)
            if rec:
                repro.append((r["dataset"], r["method"], int(r["seed"]),
                              max(abs(m.get(k, 0) - rec.get(k, 0)) for k in KEYS.values())))
    print(f"{len(rows) - missing}/{len(rows)} evals present")
    if repro:
        worst = max(repro, key=lambda x: x[3])
        print(f"normal-view re-eval vs recorded: {len(repro)} models compared, "
              f"max |diff| over UHF/UHR/NDCG = {worst[3]:.5f} ({worst[0]} {worst[1]} s{worst[2]})")
    # Pipeline check: only the requesting users' rows differ between the views, so
    # UHR@10 (all other users) must match exactly while UHF@10 may move. A UHR gap
    # means the view changed other users' rows or the eval is nondeterministic.
    gaps = [(k[0], k[1], k[3], abs(m[KEYS["UHR"]] - M[(k[0], k[1], "normal", k[3])][KEYS["UHR"]]))
            for k, m in M.items() if k[2] == "scrub" and (k[0], k[1], "normal", k[3]) in M]
    if gaps:
        g = max(gaps, key=lambda x: x[3])
        moved = sum(1 for k, m in M.items() if k[2] == "scrub" and (k[0], k[1], "normal", k[3]) in M
                    and m[KEYS["UHF"]] != M[(k[0], k[1], "normal", k[3])][KEYS["UHF"]])
        print(f"view check: {len(gaps)} model pairs, max |UHR scrub - UHR normal| = {g[3]:.6f} "
              f"({g[0]} {g[1]} s{g[2]}; expect 0); UHF changed in {moved} pairs")
    out = []
    for ds in ("beauty", "sports", "toys"):
        seeds = sorted({s for (d, _m, _v, s) in M if d == ds})
        if not seeds:
            continue
        print(f"\n== {ds}  (x100; tau_P in %, vs retrained on the same view; mean over seeds)")
        print(f"{'method':14s} {'view':6s} {'n':>2s} {'UHF@10':>7s} {'UHR@10':>7s} {'NDCG@10':>8s} {'tau_P':>6s}"
              f"   per-seed UHF@10 ({', '.join(map(str, seeds))})")
        for meth in ORDER:
            for view in ("normal", "scrub"):
                per = {}
                for sd in seeds:
                    m = M.get((ds, meth, view, sd))
                    rr = M.get((ds, "retrained", view, sd))
                    if not m:
                        continue
                    tp = (m[KEYS["NDCG"]] / rr[KEYS["NDCG"]]) if rr and rr.get(KEYS["NDCG"]) else None
                    per[sd] = dict(UHF=m.get(KEYS["UHF"]), UHR=m.get(KEYS["UHR"]), NDCG=m.get(KEYS["NDCG"]), tau_P=tp)
                    out.append(dict(dataset=ds, method=meth, view=view, seed=sd, **per[sd]))
                if not per:
                    continue
                agg = {k: mean([p[k] for p in per.values()])[0] for k in ("UHF", "UHR", "NDCG", "tau_P")}
                ps = " ".join(f(per[s]["UHF"], p=1) if s in per else "   -  " for s in seeds)
                print(f"{meth:14s} {view:6s} {len(per):2d} {f(agg['UHF'])} {f(agg['UHR'])} "
                      f"{f(agg['NDCG'], p=3):>8s} {f(agg['tau_P'], p=1)}   {ps}")
    if a.csv:
        with open(a.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, ["dataset", "method", "view", "seed", "UHF", "UHR", "NDCG", "tau_P"])
            w.writeheader()
            w.writerows(out)
        print(f"\nwrote {a.csv}")


if __name__ == "__main__":
    main()
