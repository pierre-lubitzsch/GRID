"""Unwanted-item removal: hyperparameter selection on the VALIDATION split.

The candidates are the recorded unwanted-item arms (TIGER + RQ-KMeans, categories as in
the paper), replayed with a `_valsens` tag by run_valsel_batch_sens.sh, which evaluates
every unlearned model on the test split (<run>/eval) and on the validation split
(<run>/eval_val, via src/data/amazon_data/<ds>_valview). NAU additionally gets
lambda_n=-10 (templated from the lambda_n=-1 arms).

References per (dataset, seed): validation from logs/diagnose/valsel_sens_refs/
(run_valsel_sens_refs.sh), test from extract_sensitive.reference.

Per arm, over the five data seeds: UHF@10 (test/SHF@10), UHR@10 (test/SHR@10),
tau_P = NDCG@10 / NDCG@10 of that seed's retrained model, and
Delta_UHR = |UHR_U - UHR_R|. Rule (as extract_sensitive.best_per_family, on validation):
lowest mean UHF@10 among arms with mean tau_P >= 95%, ties by global SH@10, then higher
tau_P; if none reaches 95%, the lowest UHF@10 overall.

    PYTHONPATH=. python paper/plotting/valsens_table.py [--partial] [--tex paper/v24_edits/tables/appendix_valsens.tex]

--ledger paper/plotting/canonical_runs.tsv: for every configuration that has a canonical `_mia`
execution in the ledger (the selected configurations), BOTH the validation and the test columns
come from that execution (<run>/eval_val and <run>/eval); the other candidates keep their
`_valsens` replay (no checkpoint) and are listed as NOT canonical-with-checkpoint. Method from the
ledger's method column (not run_tags.family_of). Outputs get a _ledger suffix
(valsens_candidates_ledger.csv, valsens_selected_ledger.csv, <tex>_ledger.tex), with a `source`
column (mia | valsens). If a selection moves to a configuration without `_mia`, it is printed.

--view {normal,scrub} (with --ledger): additionally, every non-selected candidate reads val + test from
its `unwanted_canonck` re-run (checkpointed; NAU lambda_n=0 / -10 are the same executions Table 4
reads), source `canonck`; `_valsens` remains only where no `_canonck` run exists yet.
Validation columns always come from the normal validation view (selection unchanged in kind).
scrub: test columns on the CLEANED test histories (logs/eval/sensitive_scrub/scrub, tau_P against the
retrained model on the same view); candidates without a checkpointed execution have no cleaned-view
test eval and print n/a. Outputs get a _normal_ledger / _scrub_ledger suffix.
"""
import argparse
import csv
import glob
import importlib.util
import os
import re
import statistics as S
import sys
from collections import defaultdict

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

_spec = importlib.util.spec_from_file_location("es", "scripts/extract_sensitive.py")
es = importlib.util.module_from_spec(_spec); _spec.loader.exec_module(es)

SEEDS = [2, 3, 5, 7, 11]
DS = ("beauty", "sports", "toys")
NAME = {"scif": "SCIF", "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan",
        "tracer": "TRACER", "unified": "NAU"}
CHECK = r"\checkmark"
KEYS = {"f": "test/SHF@10", "r": "test/SHR@10", "sh": "test/SH@10", "n": "test/ndcg@10"}


def metrics_csv(d):
    fs = sorted(glob.glob(os.path.join(d, "csv", "version_*", "metrics.csv")))
    if not fs:
        return None
    rows = [r for r in csv.DictReader(open(fs[-1])) if r.get("test/ndcg@10")]
    return {k: float(v) for k, v in rows[-1].items() if v not in ("", None) and k.startswith("test/")} if rows else None


def val_ref(ds, role, sd):
    return metrics_csv(f"logs/diagnose/valsel_sens_refs/{ds}_{role}_s{sd}")


def test_ref(ds, role, sd):
    return es.reference(ds, role, "tiger", "rkmeans", sd, es.CATEGORY[ds])[0]


def cell(m, retr):
    if not m or not retr or KEYS["f"] not in m:
        return None
    return dict(f=m[KEYS["f"]], r=m[KEYS["r"]], sh=m[KEYS["sh"]], tp=m[KEYS["n"]] / retr[KEYS["n"]],
                dr=abs(m[KEYS["r"]] - retr[KEYS["r"]]))


DUPLICATE_ARMS = {("toys", "tracer", "tracer_ep4"): ("toys", "tracer_lf1.0_lc1.0_ep4")}


VIEW = None


def test_dir(run_dir):
    """<run>/eval, or its cleaned-view evaluation with --view scrub."""
    if VIEW == "scrub":
        import unwanted_ledger as ul
        return os.path.join(ul.SCRUB_ROOT, "runs", os.path.basename(run_dir), "eval")
    return os.path.join(run_dir, "eval")


def canonck_override(out, ledger):
    """Every non-selected candidate (TIGER + RQ-KMeans; NAU lambda_n=0 / -0.1 / -10 and the baseline
    alternatives): val + test from its `unwanted_canonck` re-run, replacing the `_valsens` replay."""
    import unwanted_ledger as ul
    canon = set()
    for r in ul.canonck_rows(ledger, track="tiger+rkmeans"):
        if r["status"] != "ok":
            print(f"  ledger _canonck row {r['status']}: {r['dataset']} {r['config']} s{r['seed']}")
            continue
        ds, sd = r["dataset"], int(r["seed"])
        fam = ul.FAM[r["method"]]
        key = (ds, fam, f"lambda_n={ul.nau_ln(r['config'])}" if fam == "unified" else r["config"])
        if key not in canon:
            out[key] = {"val": {}, "test": {}}
            canon.add(key)
        for split, sub, rf in (("val", os.path.join(r["run_dir"], "eval_val"), val_ref), ("test", test_dir(r["run_dir"]), test_ref)):
            c = cell(metrics_csv(sub), rf(ds, "retrain", sd))
            if c:
                out[key][split][sd] = c
            else:
                print(f"  ledger _canonck run without {split} metrics ({VIEW} view): {r['run_dir']}")
    return canon


def ledger_override(out, ledger):
    """Replace val+test cells of configurations that have a canonical `_mia` run; returns the set
    of (ds, family, arm) keys that are canonical."""
    import unwanted_ledger as ul
    canon = set()
    # duplicate executions of a canonical configuration under another tag (resolved .hydra/config.yaml
    # identical up to run tag and paths, checked on all five seeds): drop them as candidates
    for (ds, fam, arm), (ds2, arm2) in DUPLICATE_ARMS.items():
        if (ds, fam, arm) in out:
            print(f"  dropping duplicate candidate {ds} {arm} (same configuration as {arm2})")
            del out[(ds, fam, arm)]
    for r in ul.mia_rows(ledger):
        fam = ul.FAM[r["method"]]
        if fam not in NAME:          # Finetune / Forget only / Forget+Repair: single configs, not in Table 8
            continue
        ds, sd = r["dataset"], int(r["seed"])
        arm = f"lambda_n={ul.nau_ln(r['config'])}" if fam == "unified" else r["config"]
        key = (ds, fam, arm)
        if key not in canon:
            out[key] = {"val": {}, "test": {}}
            canon.add(key)
        for split, sub, rf in (("val", os.path.join(r["run_dir"], "eval_val"), val_ref), ("test", test_dir(r["run_dir"]), test_ref)):
            c = cell(metrics_csv(sub), rf(ds, "retrain", sd))
            if c:
                out[key][split][sd] = c
            else:
                print(f"  ledger _mia run without {split} metrics: {sub}")
    return canon


def collect():
    """(ds, family, arm) -> {"val": {seed: cell}, "test": {seed: cell}} from the _valsens runs."""
    latest = {}
    for d in glob.glob("logs/unlearn/runs/*_bs1_sens_*_valsens*"):
        m = re.match(r".*/job(\d+)_.*_bs1_(sens_.*)$", d)
        if not m:
            continue
        tag = m.group(2)
        if tag not in latest or int(m.group(1)) > int(latest[tag][0]):
            latest[tag] = (m.group(1), d)
    out = defaultdict(lambda: {"val": {}, "test": {}})
    refs = {}
    for tag, (_j, d) in latest.items():
        m = re.match(r"sens_([a-z]+)_([a-z]+)_(.*)_valsens(?:_seed(\d+))?$", tag)
        if not m:
            print("  unparsed", tag); continue
        ds, _cat, arm, sd = m.group(1), m.group(2), m.group(3), int(m.group(4) or 2)
        fam = run_tags.family_of(arm)
        fam = "unified" if fam.startswith("unified") else fam
        cand_arm, key = arm, sd
        if fam == "unified":
            # NAU is defined with the generative separation score (as for spam removal); at
            # lambda_s=0 that score is inactive, and the sepcos arms are the same objective
            # re-run, so only the sepgen arms are NAU's candidates.
            mm = re.search(r"_sep(cos|gen)_lf([0-9.]+)_ls0\.0_ln(neg)?([0-9.]+)$", arm)
            assert mm, arm
            if mm.group(1) != "gen":
                continue
            cand_arm = f"lambda_n={'-' if mm.group(3) else ''}{mm.group(4)}"
        for split, sub, rf in (("val", "eval_val", val_ref), ("test", "eval", test_ref)):
            k = (ds, "retrain", sd, split)
            if k not in refs:
                refs[k] = rf(ds, "retrain", sd)
            c = cell(metrics_csv(os.path.join(d, sub)), refs[k])
            if c:
                out[(ds, fam, cand_arm)][split][key] = c
    return out


def agg(cells):
    return {k: S.mean(c[k] for c in cells.values()) for k in ("f", "r", "sh", "tp", "dr")} if cells else None


def select(arms, split):
    """arms: [(arm, agg)] -> selected arm under the table's rule, and whether the floor was met."""
    ok = [x for x in arms if x[1]["tp"] >= 0.95]
    pool = ok or arms
    return min(pool, key=lambda x: (x[1]["f"], x[1]["sh"], -x[1]["tp"]))[0], bool(ok)


def pretty(arm):
    if arm.startswith("lambda_n="):
        v = float(arm.split("=")[1])
        return rf"$\lambda_n={v:g}$"
    arm = arm.replace("unified_hp2_", "").replace("lnneg", "ln-")
    return arm.replace("_", r"\_")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--partial", action="store_true", help="also rank arms with < 5 validation seeds")
    ap.add_argument("--pooled", action="store_true", help="one baseline configuration for all datasets")
    ap.add_argument("--tex", default="paper/v24_edits/tables/appendix_valsens.tex")
    ap.add_argument("--ledger", default=None, metavar="TSV",
                    help="canonical `_mia` executions (val + test) for the configurations that have one; "
                         "outputs get a _ledger suffix")
    ap.add_argument("--view", choices=("normal", "scrub"), default=None,
                    help="with --ledger: also the `_canonck` NAU rows; test columns on this view (scrub = cleaned "
                         "test histories); outputs get a _normal_ledger / _scrub_ledger suffix")
    a = ap.parse_args()
    global VIEW
    if a.view:
        assert a.ledger, "--view needs --ledger"
        import unwanted_ledger as ul
        ul.set_view(a.view, es)
        VIEW = a.view
    data = collect()
    if VIEW == "scrub":
        for v in data.values():         # `_valsens` replays: no checkpoint, no cleaned-view test eval
            v["test"] = {}
    canon = ledger_override(data, a.ledger) if a.ledger else set()
    canon_ck = canonck_override(data, a.ledger) if a.view else set()
    sfx = ul.VIEW_SUFFIX[a.view] if a.view else ("_ledger" if a.ledger else "")
    if a.ledger:
        a.tex = a.tex[:-4] + f"{sfx}.tex" if a.tex.endswith(".tex") else a.tex + sfx
    nseed = lambda v: len({k[0] if isinstance(k, tuple) else k for k in v})
    rows, sel_rows = [], []
    for fam in NAME:
        cand = defaultdict(dict)            # arm -> ds -> (val agg, test agg, n)
        for (d, f, arm), v in data.items():
            if f == fam and (nseed(v["val"]) == 5 or (a.partial and v["val"])):
                cand[arm][d] = (agg(v["val"]), agg(v["test"]) if len(v["test"]) == len(v["val"]) else None, nseed(v["val"]))
        if not cand:
            continue
        dss = [d for d in DS if any(d in x for x in cand.values())]
        # NAU is selected on Beauty only and used unchanged elsewhere (as for spam removal);
        # every baseline gets ONE configuration for the scenario, scored on the mean over datasets
        def choose(sel_ds):
            full = {arm: x for arm, x in cand.items() if all(d in x for d in sel_ds)}
            score = {arm: {k: S.mean(x[d][0][k] for d in sel_ds) for k in ("f", "sh", "tp")} for arm, x in full.items()}
            ok = [arm for arm in score if score[arm]["tp"] >= 0.95]
            return min(ok or list(score), key=lambda arm: (round(score[arm]["f"], 9), score[arm]["sh"], -score[arm]["tp"])), bool(ok)
        # NAU: lambda_n selected on Beauty, used everywhere; baselines: per dataset (or pooled with --pooled)
        fixed = choose(["beauty"]) if fam == "unified" else (choose(dss) if a.pooled else None)
        for d in dss:
            pick, okd = fixed if fixed else choose([d])
            ok = okd
            for arm, x in sorted(cand.items()):
                if d not in x:
                    continue
                A, T, n = x[d]
                rows.append(dict(dataset=d, method=NAME[fam], config=arm, n_val=n,
                                 val_UHF=100 * A["f"], val_UHR=100 * A["r"], val_dUHR=100 * A["dr"], val_tauP=100 * A["tp"],
                                 test_UHF=100 * T["f"] if T else "", test_UHR=100 * T["r"] if T else "",
                                 test_dUHR=100 * T["dr"] if T else "", test_tauP=100 * T["tp"] if T else "",
                                 val_selected=arm == pick,
                                 **({"source": "mia" if (d, fam, arm) in canon else
                                     "canonck" if (d, fam, arm) in canon_ck else "valsens"} if a.ledger else {})))
            if pick in cand and d in cand[pick]:
                sel_rows.append(dict(dataset=d, method=NAME[fam], config=pick, floor_met=bool(ok),
                                     selected_on="beauty" if fam == "unified" else ("all datasets" if a.pooled else d)))
    os.makedirs("paper/plotting", exist_ok=True)
    for name, rr in ((f"valsens_candidates{sfx}.csv", rows), (f"valsens_selected{sfx}.csv", sel_rows)):
        if rr:
            with open(f"paper/plotting/{name}", "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(rr[0])); w.writeheader(); w.writerows(rr)
    fmt = lambda x, p: "   n/a" if x == "" else f"{x:6.{p}f}"
    for r in rows:
        print(f"{r['dataset']:7s} {r['method']:9s} {r['config']:40s} n={r['n_val']}  "
              f"val UHF {r['val_UHF']:6.2f} dUHR {r['val_dUHR']:5.2f} tauP {r['val_tauP']:6.1f}  | "
              f"test UHF {fmt(r['test_UHF'], 2)} dUHR {fmt(r['test_dUHR'], 2)} tauP {fmt(r['test_tauP'], 1)}"
              f"{'  <- selected' if r['val_selected'] else ''}")
    for s_ in sel_rows:
        print(f"SELECTED {s_['dataset']:7s} {s_['method']:9s} {s_['config']:40s} (on {s_['selected_on']})"
              f"{'' if s_['floor_met'] else '  (no config reaches 95%)'}")
    if a.ledger:
        nc = [r for r in rows if r["source"] not in ("mia", "canonck")]
        print(f"NOT canonical-with-checkpoint (validation + test from the `_valsens` replay, no checkpoint): {len(nc)} candidate rows")
        for r in nc:
            print(f"  {r['dataset']:7s} {r['method']:9s} {r['config']}{'   <- SELECTED: pick moved to a config without a canonical _mia run' if r['val_selected'] else ''}")
    if not rows:
        return
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{0.92}", r"\caption{",
         r"Hyperparameter selection for unwanted-item removal on the validation split (TIGER + RQ-KMeans). "
         r"For every candidate configuration, averaged over five data seeds: $\mathrm{UHF}@10$ and $\tau_P$ on the "
         r"validation split (used for selection) and on the test split (reported in \Cref{sec:appendix-unwanted}). "
         r"The selected configuration (\checkmark) has the lowest validation $\mathrm{UHF}@10$ among those with "
         r"validation $\tau_P \geq 95\%$; if none reaches $95\%$, the lowest validation $\mathrm{UHF}@10$ overall. "
         r"NAU ($\lambda_f=0.1$, $\lambda_s=0$) selects $\lambda_n$ once on the Beauty validation split with the same rule "
         r"and uses it on all datasets. "
         r"Finetune, Forget only and Forget+Repair have a single configuration."
         + (r" Test columns are measured on cleaned test histories (the requesting users' deleted items removed from their test input histories); n/a: no checkpointed execution of that configuration." if VIEW == "scrub" else "") + "}",
         r"\label{tab:appendix-valsens}", r"\begin{tabular}{lllrrrrc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} & \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"Dataset & Method & Configuration & $\mathrm{UHF}@10\downarrow$ & $\tau_P\uparrow$ & $\mathrm{UHF}@10\downarrow$ & $\tau_P\uparrow$ & Sel. \\",
         r"\midrule"]
    for i, ds in enumerate(DS):
        rr = [r for r in rows if r["dataset"] == ds]
        if not rr:
            continue
        if i and L[-1] != r"\midrule":
            L.append(r"\midrule")
        for r in rr:
            t = (f"{r['test_UHF']:.2f} & {r['test_tauP']:.1f}") if r["test_UHF"] != "" else "n/a & n/a"
            L.append(f"{ds.capitalize()} & {r['method']} & {pretty(r['config'])} & {r['val_UHF']:.2f} & "
                     f"{r['val_tauP']:.1f} & {t} & {CHECK if r['val_selected'] else ''} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(a.tex), exist_ok=True)
    open(a.tex, "w").write("\n".join(L) + "\n")
    print("wrote", a.tex)


if __name__ == "__main__":
    main()
