"""Table 1 data with baselines selected on the VALIDATION split.

Same aggregation as extract_retraining_distance.py (latest run per tag, unlearning
replicates included; per seed the mean over rows, then mean +- s.d. over the five data
seeds), but from a fresh scan of logs/unlearn/runs (run_tags.iter_runs) so recent runs
are included, and with each baseline's configuration taken from
paper/plotting/valsel_selected.csv (validation selection, valsel_table.py). Finetune and
SCIF have a single configuration. NAU = lambda_s=0.01, lambda_n=0.01 (validation-selected,
nau_val_table.py); "NAU (ls0.1)" is the unranked stronger-separation variant.
TRACER on Beauty uses its own tokenizer (tracerv3), as in Table 1.

Writes paper/plotting/retraining_distance_valsel.csv (the format make_main_spam_table_v16.py reads).

    PYTHONPATH=. python paper/plotting/extract_table1_valsel.py

--ledger paper/plotting/canonical_runs.tsv: test numbers from the ONE canonical execution per
(configuration, cell) of the ledger (no "latest run per tag", no `_useed` replicates). Baseline
picks come from valsel_selected_ledger.csv (valsel_table.py --ledger) and NAU's from
nau_valsel_selected_ledger.csv (nau_val_table.py --ledger; the Beauty pick is used on every
dataset). A pick is PROVISIONAL while any candidate cell of that method lacks validation metrics
(or the ledger selection file is missing, then the v33 pick of valsel_selected.csv / lambda_s =
lambda_n = 0.01 is used). A method whose canonical cells are not all 15 present with a test eval
is printed as WAITING and left out. Writes paper/plotting/retraining_distance_valsel_ledger.csv
and paper/plotting/table1_ledger_picks.csv (with --v33-picks: the v33 picks on the canonical
executions, *_ledger_v33picks files).
"""
import csv
import json
import os
import statistics as S
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

D = "paper/data/"
SEEDS = ["2", "3", "5", "7", "11"]
STR = ["unpopular", "mid", "popular"]
FIXED = {"Finetune": "unified_abl_lf0.0_ls0.0_ln0.0", "SCIF": "scif",
         "NAU": "nau_s0p01_n0p01_keepck", "NAU (ls0.1)": "unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative"}
VALSEL = {"Forget only": "Forget only", "Forget+Repair": "Forget+Repair", "SEIF": "SEIF",
          "Kookmin": "Kookmin", "Fanchuan": "Fanchuan", "TRACER": "TRACER"}
ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU", "NAU (ls0.1)"]


POISON_START = os.environ.get("GRID_POISON_REF_START", "")  # e.g. logs/diagnose/poison_ref_start


def ref(ds, seed, q):
    t = "" if q == "rkmeans" else "_" + q
    r = json.load(open(D + f"spam_references/{ds}{t}" + ("" if seed == "2" else f"_seed{seed}") + ".json"))
    # Opt-in (2026-11-07): the poisoned ("no unlearning") reference evaluated at the EXACT
    # checkpoint the unlearning runs start from (run_poison_ref_start.sh), instead of the
    # best-validation checkpoint. Only the Poisoned row reads poison_*; retrained is unchanged.
    if POISON_START and q == "rkmeans":
        for st in ("unpopular", "mid", "popular"):
            rows = list(csv.DictReader(open(os.path.join(
                POISON_START, f"{ds}_{st}_s{seed}", "csv", "version_0", "metrics.csv"))))
            m = [x for x in rows if x.get("test/SH@10")][-1]
            r["poison_SH@10"][st] = float(m["test/SH@10"])
            r["poison_ndcg@10"][st] = float(m["test/ndcg@10"])
    return r


def msd(v):
    return S.mean(v), (S.stdev(v) if len(v) > 1 else 0.0)


def ledger_cfg(arm):
    """valsel_selected naming -> canonical_runs.tsv config."""
    import re
    m = re.match(r"unified_abl_lf([0-9.]+)_ls0\.0_ln0\.0(_lr0\.0)?$", arm)
    if m:
        return ("forget_only_lf" if m.group(2) else "forget_repair_lf") + m.group(1)
    return arm


def has_metrics(run_dir, sub):
    import os
    return bool(run_dir) and os.path.isfile(os.path.join(run_dir, sub, "csv/version_0/metrics.csv"))


def ledger_selection(path, v33_picks=False):
    """Table 1's configuration per (dataset, method) on the ledger, as main_ledger uses it.

    Returns (by_cfg, cfg_of, status_of): by_cfg maps (ds, track, config) -> {(stratum, seed): run_dir};
    cfg_of(ds, method) gives the ledger config of `method` (TRACER on Beauty: the tracerv3 config);
    status_of(ds, method) the pick status string. Shared by the analyses that must read Table 1's
    canonical executions (table1_significance.py, retrain_floor_table.py, pgr_corrected.py,
    filter_partial_table.py)."""
    L = [r for r in csv.DictReader(open(path), delimiter="\t") if r["setting"] == "spam"]
    by_cfg = defaultdict(dict)
    for r in L:
        by_cfg[(r["dataset"], r["track"], r["config"])][(r["stratum"], r["seed"])] = r["run_dir"]
    val_missing = defaultdict(int)
    for r in L:
        if r["method"] in VALSEL.values() and not has_metrics(r["run_dir"], "eval_val"):
            val_missing[r["method"]] += 1
    f = "paper/plotting/valsel_selected_ledger.csv"
    led = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open(f))} if os.path.isfile(f) and not v33_picks else {}
    v33 = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open("paper/plotting/valsel_selected.csv"))}
    picks, status = {}, {}
    for ds in ("beauty", "sports", "toys"):
        for m in VALSEL.values():
            if (ds, m) in led:
                picks[(ds, m)] = ledger_cfg(led[(ds, m)])
                status[(ds, m)] = "PROVISIONAL(%d val cells missing)" % val_missing[m] if val_missing[m] else "final"
            else:
                picks[(ds, m)] = ledger_cfg(v33[(ds, m)])
                status[(ds, m)] = "PROVISIONAL(v33 pick; no ledger selection)"
    f = "paper/plotting/nau_valsel_selected_ledger.csv"
    if os.path.isfile(f) and not v33_picks:
        n = list(csv.DictReader(open(f)))[0]
        nau_cfg, nau_st = n["config"], ("PROVISIONAL(nau grid incomplete)" if n["preliminary"] == "1" else "final")
    else:
        nau_cfg, nau_st = "nau_ls0.01_ln0.01", "PROVISIONAL(v33 pick; nau_val_table.py --ledger has not selected)"
    fixed = {"Finetune": "finetune", "SCIF": "scif", "NAU": nau_cfg, "NAU (ls0.1)": "nau_ls0.1_ln0.01"}

    def cfg_of(ds, m):
        return fixed.get(m) or picks[(ds, m)]

    def status_of(ds, m):
        return nau_st if m == "NAU" else status.get((ds, m), "single config" if m in fixed else "")
    return by_cfg, cfg_of, status_of


def ledger_cells(path, ds, methods=None, v33_picks=False):
    """{method: {(stratum, seed): (SH@10, NDCG@10, run_dir)}} from the canonical execution of each
    Table 1 configuration (TIGER + RQ-KMeans; TRACER on Beauty from tracerv3). Cells without a
    test eval are absent, so a method with fewer than 15 cells is incomplete (pending)."""
    by_cfg, cfg_of, _ = ledger_selection(path, v33_picks)
    out = {}
    for m in (methods or ORDER):
        track = "tiger+tracerv3" if (m == "TRACER" and ds == "beauty") else "tiger+rkmeans"
        cells = by_cfg.get((ds, track, cfg_of(ds, m)), {})
        got = {}
        for (st, s), rd in cells.items():
            mt = run_tags.last_metrics(rd) if has_metrics(rd, "eval") else None
            if mt and "test/SH@10" in mt:
                got[(st, s)] = (float(mt["test/SH@10"]), float(mt["test/ndcg@10"]), rd)
        out[m] = got
    return out


def main_ledger(path, v33_picks=False):
    import os
    L = [r for r in csv.DictReader(open(path), delimiter="\t") if r["setting"] == "spam"]
    by_cfg = defaultdict(dict)   # (ds, track, config) -> {(stratum, seed): run_dir}
    for r in L:
        by_cfg[(r["dataset"], r["track"], r["config"])][(r["stratum"], r["seed"])] = r["run_dir"]
    # validation completeness per (method): every candidate config of the method, every dataset
    val_missing = defaultdict(int)
    for r in L:
        if r["method"] in VALSEL.values() and not has_metrics(r["run_dir"], "eval_val"):
            val_missing[r["method"]] += 1
    picks, status = {}, {}
    f = "paper/plotting/valsel_selected_ledger.csv"
    led = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open(f))} if os.path.isfile(f) and not v33_picks else {}
    v33 = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open("paper/plotting/valsel_selected.csv"))}
    for ds in ("beauty", "sports", "toys"):
        for m in VALSEL.values():
            if (ds, m) in led:
                picks[(ds, m)] = ledger_cfg(led[(ds, m)])
                status[(ds, m)] = "PROVISIONAL(%d val cells missing)" % val_missing[m] if val_missing[m] else "final"
            else:
                picks[(ds, m)] = ledger_cfg(v33[(ds, m)])
                status[(ds, m)] = "PROVISIONAL(v33 pick; no ledger selection)"
    f = "paper/plotting/nau_valsel_selected_ledger.csv"
    if os.path.isfile(f) and not v33_picks:
        n = list(csv.DictReader(open(f)))[0]
        nau_cfg, nau_st = n["config"], ("PROVISIONAL(nau grid incomplete)" if n["preliminary"] == "1" else "final")
    else:
        nau_cfg, nau_st = "nau_ls0.01_ln0.01", "PROVISIONAL(v33 pick; nau_val_table.py --ledger has not selected)"
    fixed = {"Finetune": "finetune", "SCIF": "scif", "NAU": nau_cfg, "NAU (ls0.1)": "nau_ls0.1_ln0.01"}
    rows, used, waiting = [], [], []
    for ds in ("beauty", "sports", "toys"):
        for q in ("rkmeans", "tracerv3"):
            if q == "tracerv3" and ds != "beauty":
                continue
            R = {s: ref(ds, s, q) for s in SEEDS}
            shR = {s: [R[s]["clean_SH@10"][st] for st in STR] for s in SEEDS}
            nR = {s: R[s]["clean_ndcg@10"] for s in SEEDS}
            if q == "rkmeans":
                pd = [S.mean(abs(R[s]["poison_SH@10"][st] - shR[s][i]) for i, st in enumerate(STR)) for s in SEEDS]
                pu = [S.mean(abs(R[s]["poison_ndcg@10"][st] - nR[s]) / nR[s] for st in STR) for s in SEEDS]
                psh = [S.mean(R[s]["poison_SH@10"][st] for st in STR) for s in SEEDS]
                pn = [S.mean(R[s]["poison_ndcg@10"][st] for st in STR) for s in SEEDS]
                gap = S.mean(pd)
                rows.append(dict(dataset=ds, track=q, method="Retrain", sh=msd([S.mean(shR[s]) for s in SEEDS]),
                                 ndcg=msd(list(nR.values())), tauP=(1.0, 0.0), dsh=(0, 0), d_forget=0.0, d_util=(0, 0)))
                rows.append(dict(dataset=ds, track=q, method="Poisoned", sh=msd(psh), ndcg=msd(pn),
                                 tauP=msd([pn[i] / nR[s] for i, s in enumerate(SEEDS)]), dsh=msd(pd), d_forget=1.0, d_util=msd(pu)))
            track = "tiger+" + q
            for m in ORDER:
                if (q == "tracerv3") != (m == "TRACER" and ds == "beauty"):
                    continue
                cfg = fixed.get(m) or picks[(ds, m)]
                st_ = nau_st if m == "NAU" else status.get((ds, m), "single config" if m in fixed else "")
                cells = by_cfg.get((ds, track, cfg), {})
                per, miss = defaultdict(list), []
                for s in SEEDS:
                    for i, st in enumerate(STR):
                        rd = cells.get((st, s), "")
                        mt = run_tags.last_metrics(rd) if has_metrics(rd, "eval") else None
                        if not mt or "test/SH@10" not in mt:
                            miss.append(f"{st}/s{s}")
                            continue
                        sh, n = float(mt["test/SH@10"]), float(mt["test/ndcg@10"])
                        per[s].append((sh, n, abs(sh - shR[s][i]), abs(n - nR[s]) / nR[s], n / nR[s]))
                if miss:
                    waiting.append((ds, q, m, cfg, len(miss)))
                    continue
                agg = lambda j: msd([S.mean(x[j] for x in per[s]) for s in per])  # noqa: E731
                dsh = agg(2)
                rows.append(dict(dataset=ds, track=q, method=m, sh=agg(0), ndcg=agg(1), tauP=agg(4), dsh=dsh,
                                 d_forget=dsh[0] / gap, d_util=agg(3)))
                used.append((ds, q, m, cfg, st_))
    sfx = "_ledger_v33picks" if v33_picks else "_ledger"
    out = f"paper/plotting/retraining_distance_valsel{sfx}.csv"
    write_rows(out, rows)
    with open(f"paper/plotting/table1{sfx}_picks.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "track", "method", "config", "status"])
        for u in used:
            w.writerow(u)
        for u in waiting:
            w.writerow([*u[:4], f"WAITING({u[4]} of 15 cells without test eval)"])
    for u in used:
        print(f"  {u[0]:7s} {u[1]:9s} {u[2]:13s} {u[3]:40s} {u[4]}")
    for u in waiting:
        print(f"  WAITING {u[0]:7s} {u[1]:9s} {u[2]:13s} {u[3]:40s} {u[4]} of 15 cells without test eval")
    print("wrote", out, f"and paper/plotting/table1{sfx}_picks.csv")


def write_rows(out, rows):
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "track", "method", "sh10", "sh10_sd", "ndcg10", "ndcg10_sd", "tau_P", "tau_P_sd",
                    "abs_dsh10", "abs_dsh10_sd", "d_forget", "d_util", "d_util_sd"])
        for r in rows:
            w.writerow([r["dataset"], r["track"], r["method"], *[f"{100 * x:.4f}" for x in r["sh"]],
                        *[f"{100 * x:.4f}" for x in r["ndcg"]], *[f"{100 * x:.2f}" for x in r["tauP"]],
                        *[f"{100 * x:.4f}" for x in r["dsh"]], f"{r['d_forget']:.4f}", *[f"{100 * x:.3f}" for x in r["d_util"]]])


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledger", default=None, help="canonical_runs.tsv: one canonical execution per (config, cell)")
    ap.add_argument("--v33-picks", action="store_true",
                    help="with --ledger: keep the v33 picks (valsel_selected.csv, NAU 0.01/0.01) to isolate the "
                         "effect of the canonical executions; writes *_ledger_v33picks* files")
    a = ap.parse_args()
    if a.ledger:
        return main_ledger(a.ledger, a.v33_picks)
    sel = {(r["dataset"], r["method"]): r["config"] for r in csv.DictReader(open("paper/plotting/valsel_selected.csv"))}
    rows, used = [], []
    for ds in ("beauty", "sports", "toys"):
        for q in ("rkmeans", "tracerv3"):
            if q == "tracerv3" and ds != "beauty":
                continue
            R = {s: ref(ds, s, q) for s in SEEDS}
            shR = {s: [R[s]["clean_SH@10"][st] for st in STR] for s in SEEDS}
            nR = {s: R[s]["clean_ndcg@10"] for s in SEEDS}
            if q == "rkmeans":
                pd = [S.mean(abs(R[s]["poison_SH@10"][st] - shR[s][i]) for i, st in enumerate(STR)) for s in SEEDS]
                pu = [S.mean(abs(R[s]["poison_ndcg@10"][st] - nR[s]) / nR[s] for st in STR) for s in SEEDS]
                psh = [S.mean(R[s]["poison_SH@10"][st] for st in STR) for s in SEEDS]
                pn = [S.mean(R[s]["poison_ndcg@10"][st] for st in STR) for s in SEEDS]
                gap = S.mean(pd)
                rows.append(dict(dataset=ds, track=q, method="Retrain", sh=msd([S.mean(shR[s]) for s in SEEDS]),
                                 ndcg=msd(list(nR.values())), tauP=(1.0, 0.0), dsh=(0, 0), d_forget=0.0, d_util=(0, 0)))
                rows.append(dict(dataset=ds, track=q, method="Poisoned", sh=msd(psh), ndcg=msd(pn),
                                 tauP=msd([pn[i] / nR[s] for i, s in enumerate(SEEDS)]), dsh=msd(pd), d_forget=1.0, d_util=msd(pu)))
            latest = {}
            for r in run_tags.iter_runs("logs/unlearn/runs", setting="spam", dataset=ds, model="tiger", quantizer=q):
                if not r.get("done") or r.get("pkm") or "pkm" in r["tag"]:
                    continue
                if r["tag"] not in latest or int(r["jid"]) > int(latest[r["tag"]]["jid"]):
                    latest[r["tag"]] = r
            for m in ORDER:
                if q == "tracerv3" and m != "TRACER":
                    continue
                arm = FIXED.get(m) or sel.get((ds, VALSEL[m]))
                if arm is None:
                    print(f"  no validation pick for {ds} {m}"); continue
                per = defaultdict(list)
                for r in latest.values():
                    if r["arm"] != arm or r["stratum"] not in STR or str(r["seed"]) not in SEEDS:
                        continue
                    mt = run_tags.last_metrics(r["path"])
                    if not mt or "test/SH@10" not in mt:
                        continue
                    s = str(r["seed"]); i = STR.index(r["stratum"])
                    sh, n = float(mt["test/SH@10"]), float(mt["test/ndcg@10"])
                    per[s].append((sh, n, abs(sh - shR[s][i]), abs(n - nR[s]) / nR[s], n / nR[s]))
                if len(per) < 5:
                    if q == "rkmeans" or m == "TRACER":
                        print(f"  {ds}/{q} {m} ({arm}): only {len(per)} seeds")
                    continue
                agg = lambda j: msd([S.mean(x[j] for x in per[s]) for s in per])
                dsh = agg(2)
                rows.append(dict(dataset=ds, track=q, method=m, sh=agg(0), ndcg=agg(1), tauP=agg(4), dsh=dsh,
                                 d_forget=dsh[0] / gap, d_util=agg(3)))
                used.append((ds, q, m, arm, sum(len(v) for v in per.values())))
    out = "paper/plotting/retraining_distance_valsel.csv"
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "track", "method", "sh10", "sh10_sd", "ndcg10", "ndcg10_sd", "tau_P", "tau_P_sd",
                    "abs_dsh10", "abs_dsh10_sd", "d_forget", "d_util", "d_util_sd"])
        for r in rows:
            w.writerow([r["dataset"], r["track"], r["method"], *[f"{100 * x:.4f}" for x in r["sh"]],
                        *[f"{100 * x:.4f}" for x in r["ndcg"]], *[f"{100 * x:.2f}" for x in r["tauP"]],
                        *[f"{100 * x:.4f}" for x in r["dsh"]], f"{r['d_forget']:.4f}", *[f"{100 * x:.3f}" for x in r["d_util"]]])
    for u in used:
        print(f"  {u[0]:7s} {u[1]:9s} {u[2]:13s} {u[3]:48s} rows={u[4]}")
    print("wrote", out)


if __name__ == "__main__":
    main()
