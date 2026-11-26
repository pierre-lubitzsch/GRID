"""Hyperparameter selection on the VALIDATION split (appendix table + selection CSV).

Reads the *_valsel runs from run_valsel_launch.sh (each has <run>/eval = test and
<run>/eval_val = validation metrics of the same unlearned model) and the validation
references from run_valsel_refs.sh (logs/diagnose/valsel_refs/). For every candidate
configuration, pooled over 3 strata x 5 seeds like Table 1:

    Delta_SH  = |SH_U@10 - SH_R@10| (retrained model's SH on the same split)
    tau_P     = NDCG_U@10 / NDCG_R@10

Selection rule (validation): smallest Delta_SH among configurations with tau_P >= 95%,
ties broken by higher tau_P; if none reaches 95%, smallest Delta_SH overall. The test
columns of the selected configuration are what Table 1 reports.

Writes paper/plotting/valsel_candidates.csv, paper/plotting/valsel_selected.csv and
paper/tables/appendix_valsel.tex.

    PYTHONPATH=. python paper/plotting/valsel_table.py
"""
from __future__ import annotations

import csv
import glob
import os
import re
import statistics as St
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
SEEDS = ("2", "3", "5", "7", "11"); STRATA = ("unpopular", "mid", "popular")
LABEL = {"ga_abl": "Forget only", "negtrain_abl": "Forget+Repair", "seif": "SEIF",
         "kookmin": "Kookmin", "fanchuan": "Fanchuan", "tracer": "TRACER", "nau": "NAU"}


def metrics(path):
    p = os.path.join(path, "csv", "version_0", "metrics.csv")
    if not os.path.isfile(p):
        return None
    rows = list(csv.DictReader(open(p)))
    r = rows[-1]
    return float(r["test/SH@10"]), float(r["test/ndcg@10"])


def refs():
    R = {}
    for d in glob.glob("logs/diagnose/valsel_refs/*"):
        m = re.match(r"(\w+?)_(beauty|sports|toys)_(\w+)_s(\d+)_(clean|poison)$", os.path.basename(d))
        if m and metrics(d):
            R[m.groups()] = metrics(d)
    return R


def config_of(rec):
    tag = rec["tag"]
    m = re.search(r"valsel_nau_ls(\w+?)_ln(\w+?)(?:_seed\d+)?$", tag)
    if m:
        return "nau", f"lambda_s={m.group(1).replace('p', '.')}, lambda_n={m.group(2).replace('p', '.')}"
    arm = rec["arm"].replace("_valsel", "")
    return run_tags.family_of(arm), arm


def ledger_config(cfg):
    """canonical_runs.tsv config -> list of (family, config) as config_of() names them.
    Forget+Repair (lambda_f=0.1) is also the NAU grid's (0, 0) setting: one execution, two rows."""
    if cfg.startswith("forget_only_lf"):
        return [("ga_abl", f"unified_abl_lf{cfg[len('forget_only_lf'):]}_ls0.0_ln0.0_lr0.0")]
    if cfg.startswith("forget_repair_lf"):
        lf = cfg[len("forget_repair_lf"):]
        out = [("negtrain_abl", f"unified_abl_lf{lf}_ls0.0_ln0.0")]
        return out + ([("nau", "lambda_s=0.0, lambda_n=0.0")] if lf == "0.1" else [])
    m = re.match(r"nau_ls([0-9.]+)_ln([0-9.]+)$", cfg)
    if m:
        return [("nau", f"lambda_s={m.group(1)}, lambda_n={m.group(2)}")]
    for fam in ("seif", "kookmin", "fanchuan", "tracer"):
        if cfg.startswith(fam):
            return [(fam, cfg)]
    return []   # finetune, scif: single configuration, nothing to select


def ledger_runs(path):
    """One canonical execution per (config, cell) from paper/plotting/canonical_runs.tsv."""
    runs, skipped = defaultdict(dict), 0
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["setting"] != "spam" or not r["run_dir"]:
            continue
        te, va = metrics(os.path.join(r["run_dir"], "eval")), metrics(os.path.join(r["run_dir"], "eval_val"))
        if not te or not va:
            skipped += int(bool(ledger_config(r["config"])))
            continue
        tr = "tracerv3" if r["track"] == "tiger+tracerv3" else "rkmeans"
        for fam, cfg in ledger_config(r["config"]):
            runs[(r["dataset"], tr, fam, cfg)][(r["stratum"], r["seed"])] = (te, va, 0)
    print(f"ledger {path}: {skipped} candidate row(s) without test+val metrics yet (configs with < 15 cells are dropped)")
    return runs


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true",
                    help="preliminary: compare each method's configurations on the cells they ALL have "
                         "(not the full 15), and mark the output as preliminary")
    ap.add_argument("--ledger", default=None,
                    help="read ONE canonical execution per (config, cell) from this canonical_runs.tsv "
                         "instead of globbing *_valsel* runs; outputs get a _ledger suffix "
                         "(valsel_candidates_ledger.csv, valsel_selected_ledger.csv, appendix_valsel_ledger.tex)")
    ap.add_argument("--per-dataset", action="store_true",
                    help="select one configuration per (method, dataset), as the submitted paper (v33) "
                         "states; default pools the three datasets")
    args = ap.parse_args()
    R = refs()
    runs = defaultdict(dict)   # (ds, track, fam, config) -> {(st, sd): (test, val)}
    if args.ledger:
        runs = ledger_runs(args.ledger)
    for d in ([] if args.ledger else glob.glob("logs/unlearn/runs/*_valsel*")):
        if "_valselck_" in d:      # tie-break reruns with kept checkpoints, not candidates
            continue
        rec = run_tags.parse_run_dir(d)
        if not rec:
            continue
        te, va = metrics(os.path.join(d, "eval")), metrics(os.path.join(d, "eval_val"))
        if not te or not va:
            continue
        fam, cfg = config_of(rec)
        tr = "tracerv3" if rec["track"] == "tiger+tracerv3" else "rkmeans"
        key = (rec["dataset"], tr, fam, cfg)
        cell = (rec["stratum"], str(rec["seed"]))
        if cell not in runs[key] or int(rec["jid"]) > runs[key][cell][2]:
            runs[key][cell] = (te, va, int(rec["jid"]))
    test_ref = {}
    import json
    for f in glob.glob("paper/data/spam_references/*.json"):
        test_ref[os.path.basename(f)[:-5]] = json.load(open(f))
    def usable(ds, tr, cell):
        st, sd = cell
        name = ds + ("" if tr == "rkmeans" else "_" + tr) + ("" if sd == "2" else f"_seed{sd}")
        return R.get((tr, ds, st, sd, "clean")) is not None and test_ref.get(name) is not None

    groups = defaultdict(list)   # (ds, method) -> configs
    for (ds, tr, fam, cfg), cc in runs.items():
        groups[(ds, LABEL.get(fam, fam))].append((tr, fam, cfg, cc))
    out = []
    for (ds, meth), cfgs in sorted(groups.items()):
        cellsets = [{c for c in cc if usable(ds, tr, c)} for tr, fam, cfg, cc in cfgs]
        common = set.intersection(*cellsets) if cellsets else set()
        if not args.partial:
            common = None
        for (tr, fam, cfg, cc), cs in zip(cfgs, cellsets):
            use = cs if common is None else common
            if (common is None and len(cs) < 15) or not use:
                continue
            per_v, per_t = defaultdict(list), defaultdict(list)
            for (st, sd) in use:
                te, va, _j = cc[(st, sd)]
                cv = R[(tr, ds, st, sd, "clean")]
                name = ds + ("" if tr == "rkmeans" else "_" + tr) + ("" if sd == "2" else f"_seed{sd}")
                rt = test_ref[name]
                per_v[sd].append((abs(va[0] - cv[0]), va[1] / cv[1]))
                per_t[sd].append((abs(te[0] - rt["clean_SH@10"][st]), te[1] / rt["clean_ndcg@10"]))
            agg = lambda per, j: St.mean(St.mean(x[j] for x in v) for v in per.values())
            out.append(dict(dataset=ds, track=tr, method=meth, config=cfg, n_cells=len(use), n_have=len(cs),
                            val_dsh=100 * agg(per_v, 0), val_tau=100 * agg(per_v, 1),
                            test_dsh=100 * agg(per_t, 0), test_tau=100 * agg(per_t, 1)))
    # ONE configuration per method for the whole spam scenario (all datasets, and via
    # --pin-arms all tokenizers): among configurations run on every dataset the method
    # has, the smallest mean-over-datasets validation Delta_SH among those whose
    # mean-over-datasets validation tau_P >= 95%, ties by higher tau_P; if none reaches
    # 95%, the smallest mean Delta_SH overall. valsel_selected.csv keeps one row per
    # dataset (all with the same configuration) for the downstream readers.
    sel = []
    for meth in sorted({r["method"] for r in out}):
        C = [r for r in out if r["method"] == meth]
        dss = sorted({r["dataset"] for r in C})
        by = defaultdict(dict)
        for r in C:
            by[r["config"]][r["dataset"]] = r
        if getattr(args, "per_dataset", False):
            # v33 protocol ("selected separately per dataset"): one pick per (method, dataset)
            # among that dataset's configurations, same rule as below.
            for ds in dss:
                D = {r["config"]: r for r in C if r["dataset"] == ds}
                ok = [c for c in D if D[c]["val_tau"] >= 95.0]
                pick = min(ok or list(D), key=lambda c: (round(D[c]["val_dsh"], 6), -D[c]["val_tau"]))
                for c, r in D.items():
                    r["selected"] = c == pick
                    r["pooled_val_dsh"], r["pooled_val_tau"] = r["val_dsh"], r["val_tau"]
                sel.append(D[pick])
            continue
        full = {c: d for c, d in by.items() if len(d) == len(dss)}
        score = {c: (St.mean(x["val_dsh"] for x in d.values()), St.mean(x["val_tau"] for x in d.values()))
                 for c, d in full.items()}
        ok = [c for c in score if score[c][1] >= 95.0]
        pick = min(ok or list(score), key=lambda c: (round(score[c][0], 6), -score[c][1]))
        for r in C:
            r["selected"] = r["config"] == pick
            r["pooled_val_dsh"], r["pooled_val_tau"] = score.get(r["config"], (None, None))
        sel += [full[pick][ds] for ds in dss]
    sfx = ("_preliminary" if args.partial else "") + ("_ledger" if args.ledger else "")
    for name, rows in ((f"valsel_candidates{sfx}.csv", out), (f"valsel_selected{sfx}.csv", sel)):
        if rows:
            with open(os.path.join(HERE, name), "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    # appendix table
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\caption{",
         (r"\\textbf{Preliminary} (runs still in progress; each method's configurations are compared on the (seed, target) cells they all have, given in parentheses after the configuration). " if args.partial else "") +
         r"Hyperparameter selection on the validation split (TIGER + RQ-KMeans; TRACER on Beauty with its own tokenizer). "
         r"For every candidate configuration, pooled over five data seeds and three targets: distance to the retrained model's exposure "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ and $\tau_P$, on the validation split (used for selection) and on the test split (reported in \Cref{tab:bandwagon-unlearn-all-tiger-rkmeans}). "
         r"Each method uses one configuration (\checkmark) on all datasets and tokenizers: the one with the smallest validation $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$, averaged over the three datasets, among those whose averaged validation $\tau_P$ is at least $95\%$, ties broken by higher $\tau_P$; if none reaches $95\%$, the smallest averaged validation $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ overall. "
         r"Finetune and SCIF have a single configuration; NAU's selection is shown in \Cref{tab:appendix-nau-valsel}.}",
         r"\label{tab:appendix-valsel}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{lllrrrrc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} & \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"Dataset & Method & Configuration & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}\downarrow$ & $\tau_P\uparrow$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}\downarrow$ & $\tau_P\uparrow$ & Sel. \\",
         r"\midrule"]
    esc = lambda s: s.replace("_", r"\_").replace("lambda", r"$\lambda$")
    ncell = lambda r: (" (%d)" % r["n_cells"]) if args.partial else ""
    for ds in ("beauty", "sports", "toys"):
        for r in [r for r in out if r["dataset"] == ds and r["method"] != "NAU"]:   # NAU: tab:appendix-nau-valsel
            mark = "\\checkmark" if r["selected"] else ""
            L.append(f"{ds.capitalize()} & {r['method']} & {esc(r['config'])}{ncell(r)} & {r['val_dsh']:.3f} & {r['val_tau']:.1f} & "
                     f"{r['test_dsh']:.3f} & {r['test_tau']:.1f} & {mark} " + "\\\\")
        L.append(r"\midrule")
    L[-1] = r"\bottomrule"
    L += [r"\end{tabular}}", r"\end{table}"]
    os.makedirs(os.path.join(HERE, "..", "tables"), exist_ok=True)
    open(os.path.join(HERE, "..", "tables", f"appendix_valsel{sfx}.tex"), "w").write("\n".join(L) + "\n")
    for r in sel:
        print(f"{r['dataset']:7s} {r['method']:14s} -> {r['config']:45s} cells {r['n_cells']:2d} val dSH {r['val_dsh']:.3f} tau {r['val_tau']:.1f} | test dSH {r['test_dsh']:.3f} tau {r['test_tau']:.1f}")


if __name__ == "__main__":
    main()
