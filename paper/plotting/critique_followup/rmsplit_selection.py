"""Sensitive item deletion: hyperparameter selection for the RETRAINING-MATCHED split grid,
on CLEANED validation histories (2026-11-14).

Candidates: every run of logs/rmsplit_sens/plan_grid_wave{1,2}.tsv (tags
sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN]; the latest run dir with checkpoints/unlearned.ckpt).
Metrics: cleaned validation  logs/eval/sensitive_scrub/scrubval/runs/<run>/eval
         cleaned test        logs/eval/sensitive_scrub/scrub/runs/<run>/eval
References (TIGER + RQ-KMeans, same view): logs/eval/sensitive_scrub/<view>/reference/v1/<ds>_{base,retrain}[_seedN]
(read directly, not from plan_rmgrid.tsv; coordinates.json asserted model=tiger, quantizer=rkmeans, same seed and view.)

Averaging and rule exactly as paper/plotting/valsens_table.py: per seed, tau_P = NDCG@10 / NDCG@10 of
that seed's retrained model on the same view and Delta_UHR = |UHR_U - UHR_R|; then the mean over the
five seeds (mean of per-seed ratios). Per (dataset, method), rule SEL_RULE (see choose()):
  rank (alternative, appendix check; was the default 2026-11-16 only): among configurations with mean validation tau_P >= 95% (all, if none
  reaches it), the best average rank in the per-seed distances to retraining |UHF@10 - UHF_R@10| and
  |UHR@10 - UHR_R@10| (ties: smaller UHF distance, then higher tau_P);
  uhf (default, the submitted rule): lowest mean validation UHF@10 among tau_P >= 95%, else lowest overall.
Only configurations with all five
validation seeds are ranked. NAU (lambda_f=0.1; NAU_GRID=submitted: lambda_s=0 x lambda_n in {0, -0.1, -1, -10})
is selected once on Beauty and used on all datasets; the per-dataset NAU pick is reported too.
Forget+Repair lambda_f=0.1 is NAU's lambda_s=0, lambda_n=0 run.

    PYTHONPATH=. python paper/plotting/critique_followup/rmsplit_selection.py
      -> paper/revision/rmsplit_selection.md, paper/revision/rmsplit_selection.csv,
         paper/revision/rmsplit_selected.csv
"""
import csv
import glob
import os
import re
import statistics as S
from collections import defaultdict

SEEDS = (2, 3, 5, 7, 11)
DS = ("beauty", "sports", "toys")
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
ROOT = "logs/eval/sensitive_scrub"
PLANS = ("logs/rmsplit_sens/plan_grid_wave1.tsv", "logs/rmsplit_sens/plan_grid_wave2.tsv")
K = {"f": "test/SHF@10", "r": "test/SHR@10", "sh": "test/SH@10", "n": "test/ndcg@10"}
ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU"]
OUT_MD = "paper/revision/rmsplit_selection.md"
OUT_CSV = "paper/revision/rmsplit_selection.csv"
OUT_SEL = "paper/revision/rmsplit_selected.csv"
OLD_LEDGER = "paper/plotting/valsens_candidates_ledger.csv"
# old split, original validation: fixed single configurations of the ablation rows
OLD_FIXED = {"Finetune": "unified_hp2_ft_lf0.0_ls0.0_ln0.0", "Forget only": "unified_hp2_fo_lf0.1_ls0.0_ln0.0_lr0.0",
             "Forget+Repair": "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0"}


def methods_of(arm):
    """arm -> [(method, config label)]; NAU's ls0/ln0 run is also Forget+Repair lambda_f=0.1."""
    if arm.startswith("unified_hp2_sepgen_lf0.1_"):
        out = [("NAU", arm)]
        if arm == "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0":
            out.append(("Forget+Repair", arm))
        return out
    for pre, m in (("unified_hp2_fr_", "Forget+Repair"), ("unified_hp2_fo_", "Forget only"),
                   ("unified_hp2_ft_", "Finetune"), ("scif_", "SCIF"), ("seif_", "SEIF"),
                   ("kookmin_", "Kookmin"), ("fanchuan_", "Fanchuan"), ("tracer_", "TRACER")):
        if arm.startswith(pre):
            return [(m, arm)]
    raise ValueError(arm)


def metrics_csv(d):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get(K["n"])]
    return {k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith("test/")} if rows else None


def ref(view, ds, role, sd):
    """TIGER + RQ-KMeans reference on the given cleaned view; the coordinates are checked (never a LETTER model)."""
    import json
    d = f"{ROOT}/{view}/reference/v1/{ds}_{role}" + ("" if sd == 2 else f"_seed{sd}")
    c = json.load(open(f"{d}/coordinates.json"))
    assert c["model"] == "tiger" and c["quantizer"] == "rkmeans" and "letter" not in c["ckpt"], d
    assert c["dataset"] == ds and int(c["seed"]) == sd and c["view"] == view, d
    return metrics_csv(d)


def cell(m, retr):
    if not m or not retr or K["f"] not in m:
        return None
    return dict(f=m[K["f"]], r=m[K["r"]], sh=m.get(K["sh"], float("nan")), tp=m[K["n"]] / retr[K["n"]],
                dr=abs(m[K["r"]] - retr[K["r"]]), df=abs(m[K["f"]] - retr[K["f"]]))


def run_dir(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}") if os.path.exists(f"{d}/checkpoints/unlearned.ckpt")]
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))) if c else None


def n_req(ds, sd):
    import json
    return json.load(open(f"src/data/amazon_data/{ds}/forget_manifest_{CAT[ds]}_1e-4_seed{sd}.json"))["n_spam_users"]


def hits(cells, ds):
    """requesting users with a deleted-category item in their top 10, summed over seeds (UHF@10 x n_req)."""
    return sum(c["f"] * n_req(ds, sd) for sd, c in cells.items())


def agg(cells):
    return {k: S.mean(c[k] for c in cells.values()) for k in ("f", "r", "sh", "tp", "dr", "df")} if cells else None


RULE = os.environ.get("SEL_RULE", "uhf")    # uhf = the submitted rule (decision 2026-11-17, main protocol) | rank (distance to retraining; optional appendix check)


def choose(score):
    """score: {config: val agg} -> (pick, floor met).
    rank (default since 2026-11-16): the retrained model is the reference for both user groups, so among the
      configurations with validation tau_P >= 95% (all, if none reaches it) take the best average rank in
      distance to retraining: per-seed |UHF@10 - UHF_R@10| (requesting users) and |UHR@10 - UHR_R@10| (other
      users); ties: smaller UHF distance, then higher tau_P.
    uhf (previous): lowest validation UHF@10 among tau_P >= 95%, else lowest UHF@10 overall."""
    ok = [c for c in score if score[c]["tp"] >= 0.95]
    pool = ok or list(score)
    if RULE == "uhf":
        return min(pool, key=lambda c: (round(score[c]["f"], 9), score[c]["sh"], -score[c]["tp"])), bool(ok)
    def ranks(k):
        vals = sorted(round(score[c][k], 9) for c in pool)
        return {c: vals.index(round(score[c][k], 9)) for c in pool}     # ties share the lower rank
    rf, rr = ranks("df"), ranks("dr")
    return min(pool, key=lambda c: (rf[c] + rr[c], round(score[c]["df"], 9), -score[c]["tp"])), bool(ok)


# NAU's selection grid. "submitted" (default, decision 2026-11-17): the submitted sensitive-deletion grid,
# lambda_s = 0 with lambda_n selected; the lambda_s in {0.01, 0.1} runs stay in the lambda_n ablation table only.
# "extended": every lambda_s of the retraining-matched grid.
NAU_GRID = os.environ.get("NAU_GRID", "submitted")


def nau_in_grid(method, cfg):
    return method != "NAU" or NAU_GRID == "extended" or "_ls0.0_" in cfg


def order(score):
    """Configurations in the order the rule ranks them (pool first, then the rest; same keys as choose())."""
    ok = [c for c in score if score[c]["tp"] >= 0.95]
    pool = ok or list(score)
    rest = [c for c in score if c not in pool]
    def keyf(grp):
        if RULE == "uhf":
            return lambda c: (round(score[c]["f"], 9), score[c]["sh"], -score[c]["tp"])
        def ranks(k):
            vals = sorted(round(score[c][k], 9) for c in grp)
            return {c: vals.index(round(score[c][k], 9)) for c in grp}
        rf, rr = ranks("df"), ranks("dr")
        return lambda c: (rf[c] + rr[c], round(score[c]["df"], 9), -score[c]["tp"])
    out = sorted(pool, key=keyf(pool))
    if rest:
        out += sorted(rest, key=keyf(rest))
    return out, bool(ok)


def old_picks():
    old = {}
    for r in csv.DictReader(open(OLD_LEDGER)):
        if r["val_selected"] != "True":
            continue
        c = r["config"]
        if r["method"] == "NAU":
            v = float(c.split("=")[1])
            c = "unified_hp2_sepgen_lf0.1_ls0.0_ln" + (f"neg{-v}" if v < 0 else f"{v}")
        old[(r["dataset"], r["method"])] = c
    for ds in DS:
        for m, c in OLD_FIXED.items():
            old[(ds, m)] = c
    return old


def pretty(m, c):
    mm = re.match(r"unified_hp2_sepgen_lf0\.1_ls([0-9.]+)_ln(neg)?([0-9.]+)", c)
    if m == "NAU" and mm:
        return f"lambda_s={float(mm.group(1)):g}, lambda_n={'-' if mm.group(2) else ''}{float(mm.group(3)):g}"
    mm = re.match(r"unified_hp2_(?:fr|fo|sepgen)_lf([0-9.]+)", c)
    if m in ("Forget only", "Forget+Repair") and mm:
        return f"lambda_f={float(mm.group(1)):g}"
    if m == "Finetune":
        return "lambda_f=0"
    return c


RULE_TXT = {
    "rank": ("Rule per (dataset, method) (SEL_RULE=rank, alternative check): among configurations with mean validation "
             "tau_P >= 95% (all configurations, if none reaches it), the best average rank in the two per-seed distances to "
             "retraining, Delta_UHF = |UHF@10 - UHF_R@10| (requesting users) and Delta_UHR = |UHR@10 - UHR_R@10| (other "
             "users), each averaged over seeds; ties: smaller Delta_UHF, then higher tau_P."),
    "uhf": ("Rule per (dataset, method) (SEL_RULE=uhf, the submitted rule, main protocol): lowest mean validation UHF@10 among configurations "
            "with mean validation tau_P >= 95%, ties by global SH@10 then higher tau_P; if none reaches 95%, the lowest "
            "UHF@10 overall."),
}


def main():
    import sys
    nmin = 1 if "--partial" in sys.argv else 5     # --partial: also rank configurations with < 5 seeds (debug only)
    refs = {(v, ds, sd): ref(v, ds, "retrain", sd) for v in ("scrubval", "scrub") for ds in DS for sd in SEEDS}
    miss_refs = [k for k, v in refs.items() if not v]
    data = defaultdict(lambda: {"val": {}, "test": {}})          # (ds, method, config) -> split -> seed -> cell
    missing = []
    planned = defaultdict(set)                                    # (ds, method) -> planned configs
    for p in PLANS:
        for line in open(p):
            f = line.rstrip("\n").split("\t")
            ds, sd, tag = f[1], int(f[3]), f[5]
            arm = re.match(rf"sens_{ds}_{CAT[ds]}rm_(.*)_rmsplit_canonck(?:_seed\d+)?$", tag).group(1)
            for m, cfg in methods_of(arm):
                planned[(ds, m)].add(cfg)
            d = run_dir(tag)
            if not d:
                missing.append(f"{tag}: no run with checkpoint"); continue
            run = os.path.basename(d)
            for split, view in (("val", "scrubval"), ("test", "scrub")):
                c = cell(metrics_csv(f"{ROOT}/{view}/runs/{run}/eval"), refs[(view, ds, sd)])
                if c:
                    for m, cfg in methods_of(arm):
                        data[(ds, m, cfg)][split][sd] = c
                else:
                    missing.append(f"{run}: no {view} metrics")
    old = old_picks()
    rows, sel = [], []
    beauty_nau = None
    per_ds_pick = {}
    for m in ORDER:
        for ds in DS:
            score = {cfg: agg(v["val"]) for (d, mm, cfg), v in data.items() if d == ds and mm == m and len(v["val"]) >= nmin
                     and nau_in_grid(m, cfg)}
            complete = len(score) == len([c for c in planned[(ds, m)] if nau_in_grid(m, c)]) or nmin < 5
            if score and complete:
                per_ds_pick[(ds, m)] = choose(score)
    if ("beauty", "NAU") in per_ds_pick:
        beauty_nau = per_ds_pick[("beauty", "NAU")][0]
    for m in ORDER:
        for ds in DS:
            cfgs = sorted(cfg for (d, mm, cfg) in data if d == ds and mm == m)
            if not cfgs:
                continue
            own = per_ds_pick.get((ds, m))
            pick = beauty_nau if m == "NAU" else (own[0] if own else None)
            for cfg in cfgs:
                v = data[(ds, m, cfg)]
                A, T = agg(v["val"]), agg(v["test"])
                rows.append(dict(dataset=ds, method=m, config=cfg, label=pretty(m, cfg),
                                 n_val=len(v["val"]), n_test=len(v["test"]),
                                 val_UHF=100 * A["f"] if A else "", val_tauP=100 * A["tp"] if A else "",
                                 val_dUHR=100 * A["dr"] if A else "", val_dUHF=100 * A["df"] if A else "",
                                 val_SH=100 * A["sh"] if A else "",
                                 val_UHF_users=hits(v["val"], ds) if A else "",
                                 test_UHF=100 * T["f"] if T else "", test_UHR=100 * T["r"] if T else "",
                                 test_dUHR=100 * T["dr"] if T else "", test_tauP=100 * T["tp"] if T else "",
                                 new_pick=cfg == pick, own_dataset_pick=bool(own and cfg == own[0]),
                                 old_pick=cfg == old.get((ds, m))))
            if pick:
                sel.append(dict(dataset=ds, method=m, new_config=pick, new_label=pretty(m, pick),
                                selected_on="beauty" if m == "NAU" else ds,
                                floor_met=(per_ds_pick[("beauty", "NAU")][1] if m == "NAU" else own[1]),
                                own_dataset_pick=own[0] if own else "",
                                old_config=old.get((ds, m), ""), old_label=pretty(m, old[(ds, m)]) if (ds, m) in old else "",
                                agrees=pick == old.get((ds, m))))
    pending = sorted({(ds, m) for (ds, m) in planned if planned[(ds, m)] and (ds, m) not in per_ds_pick and not (m == "NAU" and beauty_nau)})
    # NAU near-tie check on Beauty: margin to the runner-up in the pool the rule ranks
    tie = []
    for ds in DS:
        sc = {cfg: v["val"] for (d, mm, cfg), v in data.items() if d == ds and mm == "NAU" and len(v["val"]) >= nmin
              and nau_in_grid("NAU", cfg)}
        if len(sc) < 2:
            continue
        A = {c: agg(v) for c, v in sc.items()}
        rk, floor = order(A)
        assert rk[0] == choose(A)[0]
        tie.append((ds, rk[0], rk[1], A[rk[0]], A[rk[1]], floor))
    if not rows:
        print("no candidate has cleaned-validation metrics yet;", len(missing), "missing"); return
    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    for path, rr in ((OUT_CSV, rows), (OUT_SEL, sel)):
        if not rr:
            continue
        with open(path, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    f2 = lambda x, p=2: "n/a" if x == "" else f"{x:.{p}f}"
    L = ["# Sensitive item deletion: selection with the retraining-matched split (cleaned validation)", "",
         "Generated by `paper/plotting/critique_followup/rmsplit_selection.py` "
         "(CSV: `paper/revision/rmsplit_selection.csv`, picks: `paper/revision/rmsplit_selected.csv`).", "",
         "Protocol: `unlearning.split_mode=retrain_matched` grid (plan_grid_wave1/2.tsv), five data seeds. "
         "Selection on CLEANED validation histories (requesting users' deleted interactions removed; "
         "`*_scrubvalview`), reported test numbers on cleaned test histories (`*_scrubtestview`). "
         "tau_P = NDCG@10 / NDCG@10 of the same seed's retrained model on the same view, averaged over seeds "
         "(as valsens_table.py). " + RULE_TXT[RULE] + " NAU is selected once on Beauty and used on "
         "all datasets (the own-dataset NAU pick is marked separately). Forget+Repair lambda_f=0.1 is NAU's "
         "lambda_s=0, lambda_n=0 run. Old picks: old split, original validation "
         "(`paper/plotting/valsens_candidates_ledger.csv`; Finetune / Forget only / Forget+Repair were fixed at "
         "lambda_f = 0 / 0.1 / 0.1). All numbers in %.", ""]
    L += ["## Selected configurations", "", "| Dataset | Method | New pick | Old pick | Agree | tau_P floor met | Own-dataset pick (NAU) |",
          "|---|---|---|---|---|---|---|"]
    for s in sel:
        L.append(f"| {s['dataset']} | {s['method']} | {s['new_label']} | {s['old_label'] or 'n/a'} | "
                 f"{'yes' if s['agrees'] else 'NO'} | {'yes' if s['floor_met'] else 'no'} | "
                 f"{pretty('NAU', s['own_dataset_pick']) if s['method'] == 'NAU' else ''} |")
    if pending:
        L += ["", "PENDING (grid or cleaned-view evals incomplete, no pick yet): " +
              ", ".join(f"{m} ({ds})" for ds, m in pending)]
    if tie:
        L += ["", "## NAU: the rule's pick and its runner-up (cleaned validation, %)", "",
              "NAU is selected on Beauty only; the Sports and Toys rows show what the rule would pick there.", "",
              "| Dataset | Pool | Pick | Delta_UHF | Delta_UHR | tau_P | Runner-up | Delta_UHF | Delta_UHR | tau_P |",
              "|---|---|---|---|---|---|---|---|---|---|"]
        for ds, a, b, Aa, Ab, floor in tie:
            L.append(f"| {ds} | {'tau_P >= 95%' if floor else 'all (fallback: no setting reaches 95%)'} | "
                     f"{pretty('NAU', a)} | {100 * Aa['df']:.2f} | {100 * Aa['dr']:.3f} | {100 * Aa['tp']:.1f} | "
                     f"{pretty('NAU', b)} | {100 * Ab['df']:.2f} | {100 * Ab['dr']:.3f} | {100 * Ab['tp']:.1f} |")
    for ds in DS:
        L += ["", f"## {ds.capitalize()}", "",
              "| Method | Configuration | n (val/test) | val UHF@10 (users) | val Delta_UHF | val Delta_UHR | val tau_P | test UHF@10 | test UHR@10 | test Delta_UHR | test tau_P | New | Old |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for r in (r for r in rows if r["dataset"] == ds):
            mark = ("selected" if r["new_pick"] else "") + (" (own-dataset pick)" if r["method"] == "NAU" and r["own_dataset_pick"] else "")
            L.append(f"| {r['method']} | {r['label']} | {r['n_val']}/{r['n_test']} | {f2(r['val_UHF'])} ({f2(r['val_UHF_users'], 1)}) | {f2(r['val_dUHF'])} | {f2(r['val_dUHR'], 3)} | {f2(r['val_tauP'], 1)} | "
                     f"{f2(r['test_UHF'])} | {f2(r['test_UHR'])} | {f2(r['test_dUHR'])} | {f2(r['test_tauP'], 1)} | "
                     f"{mark.strip()} | {'old pick' if r['old_pick'] else ''} |")
    L += ["", "## References (cleaned views, mean over seeds)", "",
          "| Dataset | Model | val UHF@10 | test UHF@10 | test UHR@10 | test tau_P |", "|---|---|---|---|---|---|"]
    for ds in DS:
        for role, name in (("base", "Before deletion"), ("retrain", "Retrained")):
            v = [ref("scrubval", ds, role, sd) for sd in SEEDS]
            t = [ref("scrub", ds, role, sd) for sd in SEEDS]
            if all(v) and all(t):
                L.append(f"| {ds} | {name} | {100 * S.mean(x[K['f']] for x in v):.2f} | {100 * S.mean(x[K['f']] for x in t):.2f} | "
                         f"{100 * S.mean(x[K['r']] for x in t):.2f} | "
                         f"{100 * S.mean(x[K['n']] / refs[('scrub', ds, sd)][K['n']] for x, sd in zip(t, SEEDS)):.1f} |")
    if missing or miss_refs:
        L += ["", "## Missing", ""] + [f"- {x}" for x in missing] + [f"- reference {k}" for k in miss_refs]
    open(OUT_MD, "w").write("\n".join(L) + "\n")
    for s in sel:
        print(f"{s['dataset']:7s} {s['method']:14s} new {s['new_label']:32s} old {s['old_label']:32s} "
              f"{'agree' if s['agrees'] else 'DIFFERENT'}{'' if s['floor_met'] else '  (no config reaches 95%)'}")
    for t in tie:
        print("NAU %s: pick %s (dUHF %.2f dUHR %.3f tauP %.1f) vs runner-up %s (dUHF %.2f dUHR %.3f tauP %.1f)" % (
            t[0], t[1], 100 * t[3]["df"], 100 * t[3]["dr"], 100 * t[3]["tp"], t[2], 100 * t[4]["df"], 100 * t[4]["dr"], 100 * t[4]["tp"]))
    print("pending:", pending)
    print(f"missing: {len(missing)}; wrote {OUT_MD}, {OUT_CSV}, {OUT_SEL}")


if __name__ == "__main__":
    main()
