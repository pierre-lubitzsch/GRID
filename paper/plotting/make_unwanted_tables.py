"""Unwanted-item removal tables (main Beauty table + appendix Beauty/Sports/Toys).

Paper notation: UHF@10 / UHR@10 are the unwanted-item hit rates of the requesting
and the remaining users (the code and metrics.csv still call them SHF / SHR), and
the unwanted set is I_u. tau_F and RelItemsF are replaced by distances to the
retrained model, computed per seed against that seed's own retrained model and
then averaged (the analogue of Delta^retr_SH in the spam tables):

    Delta_UHR = |UHR_U - UHR_R|   (always shown)
    Delta_UHF = |UHF_U - UHF_R|   (only with --with-dUHF; see the caveat below)

Selection is unchanged (extract_sensitive.best_per_family): per method, the
lowest-UHF@10 configuration among those with tau_P >= 95%.

Caveat for --with-dUHF: on Beauty and Toys the retrained model's UHF@10 is HIGHER
than the original model's (12 to 14 requesting users; retraining removes nothing
measurable), so "closest to retraining" on UHF rewards removing nothing.

    PYTHONPATH=. python paper/plotting/make_unwanted_tables.py [--with-dUHF] [--out-dir DIR]

--ledger paper/plotting/canonical_runs.tsv: every method row is the ledger's canonical `_mia`
execution of that dataset (one configuration per method, five seeds; method from the ledger's
method column, not run_tags.family_of). References are unchanged. With --pin, the pinned
configurations are only cross-checked against the ledger (a mismatch is printed: the pick moved
to a configuration without a canonical `_mia` run). Files get a _ledger suffix.

--view scrub (with --ledger): every number on the CLEANED test histories (protocol A; requesting
users' deleted items removed from their test input histories): runs from
logs/eval/sensitive_scrub/scrub/runs, original (exact start checkpoint) and retrained from
logs/eval/sensitive_scrub/scrub/reference (see paper/plotting/unwanted_ledger.set_view). Files get a
_scrub_ledger suffix; --view normal writes the normal-view counterpart as _normal_ledger. Without
--view, unchanged.
"""
from __future__ import annotations

import argparse
import importlib.util
import os
import statistics as St
import sys

sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

_spec = importlib.util.spec_from_file_location("es", "scripts/extract_sensitive.py")
es = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(es)
h2h = es.h2h

NAME = {"finetune_abl": "Finetune", "ga_abl": "Forget only", "negtrain_abl": "Forget+Repair",
        "scif": "SCIF", "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan",
        "tracer": "TRACER"}
ORDER = ["finetune_abl", "ga_abl", "negtrain_abl", "scif", "seif", "kookmin", "fanchuan", "tracer"]
OURS = {"unified", "unified_sepgenerative", "unified_sepcosine", "unified_abl_grid"}
VIEW = None
CAT = {"beauty": "Hair Loss Products", "sports": "Guns \\& Rifles", "toys": "toy weapons"}


def pm(m, sd, places=3):
    if m is None:
        return "n/a"
    return f"{m:.{places}f}" + ("" if sd is None else rf"\,{{\scriptsize$\pm${sd:.{places}f}}}")


def ref_row(refs, role):
    """(mean, sd) over seeds for the original (role 0) or retrained (role 1) model."""
    keys = {"f": "test/SHF@10", "r": "test/SHR@10", "n": "test/ndcg@10"}
    per = {k: [] for k in ("f", "r", "n", "df", "dr", "tp")}
    for _sd, (base, retr, *_p) in refs.items():
        if not base or not retr:
            continue
        m = base if role == 0 else retr
        for k, key in keys.items():
            per[k].append(100 * m[key])
        per["df"].append(100 * abs(m[keys["f"]] - retr[keys["f"]]))
        per["dr"].append(100 * abs(m[keys["r"]] - retr[keys["r"]]))
        per["tp"].append(100 * m[keys["n"]] / retr[keys["n"]])
    return {k: (St.mean(v), St.stdev(v) if len(v) > 1 else 0.0) for k, v in per.items() if v}


PIN_FAM = {"SCIF": "scif", "SEIF": "seif", "Kookmin": "kookmin", "Fanchuan": "fanchuan", "TRACER": "tracer"}
# NAU for unwanted items: lambda_f=0.1, lambda_s=0, lambda_n selected on the Beauty validation
# split (valsens_table.py), with the generative separation score that defines NAU (inactive at
# lambda_s=0).
NAU_ARMS = ("unified_hp2_sepgen_lf0.1_ls0.0_ln{ln}",)


def _row(runs, arm):
    agg_d = {f: es.h2h.agg(runs, f) for f in ("sens", "sensF", "sensR", "n10", "UR", "tau_P", "tau_F", "tau_F_den",
                                           "tau_FF", "tau_FF_den", "RelItems", "tpm", "dUHF", "dUHR")}
    agg_d["runs"] = runs
    tp = agg_d["tau_P"]["mean"]
    return (arm, agg_d, tp is not None and tp >= es.TAU_P_RANK_MIN)


def pinned(ds, cells, pin_csv):
    """Validation-selected rows: each baseline at its valsens_selected.csv config (the recorded
    test runs of that arm), NAU at the Beauty-selected lambda_n; single-config families as is."""
    import csv as _csv
    sel = {r["method"]: r["config"] for r in _csv.DictReader(open(pin_csv)) if r["dataset"] == ds}
    best = {f: v for f, v in es.best_per_family(cells, es.TAU_P_RANK_MIN, 5).items()
            if f in ("finetune_abl", "ga_abl", "negtrain_abl")}
    for meth, fam in PIN_FAM.items():
        if meth in sel:
            arm = sel[meth]
            assert cells.get(arm), f"{ds}: no recorded runs of pinned {arm}"
            best[fam] = _row(cells[arm], arm)
    ln = sel["NAU"].split("=")[1]
    ln = ("neg" + ln[1:]) if ln.startswith("-") else ln
    runs = [r for a in NAU_ARMS for r in cells.get(a.format(ln=ln), [])]
    assert len(runs) == 5, (ds, ln, len(runs))
    best["unified"] = _row(runs, f"NAU lambda_n={sel['NAU'].split('=')[1]}")
    return best


def ledger_best(ds, ledger, pin_csv=None):
    """family -> (arm, aggregates, ok) from the ledger's `_mia` rows; refs per seed."""
    import unwanted_ledger as ul
    runs = ul.mia_runs(ledger, ds, "eval")
    best = {}
    for (fam, cfg), rr in runs.items():
        assert fam not in best, (ds, fam, "two canonical configurations")
        assert len(rr) == 5, (ds, cfg, len(rr))
        best[fam] = _row(rr, f"NAU lambda_n={ul.nau_ln(cfg)}" if fam == "unified" else cfg)
    if pin_csv:
        import csv as _csv
        sel = {r["method"]: r["config"] for r in _csv.DictReader(open(pin_csv)) if r["dataset"] == ds}
        for meth, fam in list(PIN_FAM.items()) + [("NAU", "unified")]:
            want = sel.get(meth)
            have = next((c for (f, c) in runs if f == fam), None)
            if want is None or have is None:
                continue
            if fam == "unified":
                same = float(want.split("=")[1]) == float(ul.nau_ln(have))
            else:
                same = want == have
            if not same:
                print(f"  WARNING {ds} {meth}: pick {want} has no canonical _mia run (ledger has {have})")
    seeds = sorted({e["seed"] for rr in runs.values() for e in rr})
    refs = {sd: ul.refs_for(ds, sd) for sd in seeds}
    return best, refs


def table(ds, with_duhf, main, times, pin_csv=None, ledger=None):
    if ledger:
        best, refs = ledger_best(ds, ledger, pin_csv)
    else:
        cells, refs, _sk = es.collect(ds, run_tags.DEFAULT_MODEL, run_tags.DEFAULT_QUANTIZER, es.CATEGORY[ds])
        best = pinned(ds, cells, pin_csv) if pin_csv else es.best_per_family(cells, es.TAU_P_RANK_MIN, 5)
    for f, v in best.items():
        if v:
            print(f"  {ds:7s} {f:14s} {v[0]}")
    rows = []
    for fam, v in best.items():
        if not v:
            continue
        arm, a, ok = v
        g = lambda k: (None, None) if not a[k]["n_seeds"] else (100 * a[k]["mean"], 100 * (a[k].get("sd") or 0.0))
        rows.append(dict(fam=fam, ok=ok, f=g("sensF"), df=g("dUHF"), r=g("sensR"), dr=g("dUHR"),
                         n=g("n10"), tp=g("tau_P"), e=es._tmin(a, times)))
    rows.sort(key=lambda r: (r["fam"] in OURS, ORDER.index(r["fam"]) if r["fam"] in ORDER else 99))
    cols = [("f", "low", 3), ("df", "low", 3), ("r", None, 3), ("dr", "low", 3), ("n", "high", 3), ("tp", "high", 1)]
    if not with_duhf:
        cols = [c for c in cols if c[0] != "df"]
    elig = [r["ok"] and r["tp"][0] is not None and r["tp"][0] >= 100 * es.TAU_P_RANK_MIN for r in rows]
    marks = {}
    for k, better, _p in cols:
        if better:
            marks[k] = h2h.rank_marks([r[k][0] for r in rows], better, elig)
    hdr = {"f": r"$\mathrm{UHF}@10{\scriptstyle\,(\%)}\downarrow$",
           "df": r"$\Delta^{\mathrm{retr}}_{\mathrm{UHF}}{\scriptstyle\,(\%)}\downarrow$",
           "r": r"$\mathrm{UHR}@10{\scriptstyle\,(\%)}$",
           "dr": r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}{\scriptstyle\,(\%)}\downarrow$",
           "n": r"$\mathrm{NDCG}@10\uparrow$", "tp": r"$\tau_P$ (\%)"}
    ncol = len(cols) + 2
    nau = next((r for r in rows if r["fam"] in OURS), None)
    lam = r"NAU uses $\lambda_n=-1$, $\lambda_s=0$" if nau else ""
    dtxt = (r"$\Delta^{\mathrm{retr}}_{\mathrm{UHF}}$ and $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ are "
            if with_duhf else r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ is ")
    dform = (r"$|\mathrm{UHF}_U@10-\mathrm{UHF}_R@10|$ and $|\mathrm{UHR}_U@10-\mathrm{UHR}_R@10|$"
             if with_duhf else r"$|\mathrm{UHR}_U@10-\mathrm{UHR}_R@10|$")
    cap = [rf"Unwanted item removal on {ds.capitalize()} ({CAT[ds]}) with TIGER + RQ-KMeans, five data seeds.",
           r"$\mathrm{UHF}@10$ and $\mathrm{UHR}@10$ are the unwanted item hit rates of the requesting and the remaining users; "
           + dtxt + r"the mean over seeds of " + dform + r", the distance to the retrained model of the same seed.",
           (r"Each method uses the configuration selected on the validation split (\Cref{tab:appendix-valsens}; $^{\dagger}$: $\tau_P<95\%$, never ranked)"
            if (pin_csv or ledger) else
            r"Each method is at its lowest-$\mathrm{UHF}@10$ configuration with $\tau_P\geq95\%$ ($^{\dagger}$: none reaches it, never ranked)")
           + (f"; {lam}." if lam else ".")]
    if VIEW == "scrub":
        cap.append(r"All models are evaluated on cleaned test histories: the requesting users' deleted items are removed from their test input histories.")
    rr, ro = ref_row(refs, 1), ref_row(refs, 0)
    if ro["f"][0] <= rr["f"][0]:
        cap.append(r"Retraining does not lower $\mathrm{UHF}@10$ below the original model.")
    label = (f"tab:paper-sensitive-{ds}-tiger-rkmeans" if main else f"tab:sensitive-{ds}-tiger-rkmeans")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\renewcommand{\arraystretch}{1.05}", r"\caption{", *cap, "}", rf"\label{{{label}}}",
         r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{l" + "c" * (ncol - 1) + "}", r"\toprule",
         r"\textbf{Method} & " + " & ".join(hdr[k] for k, *_ in cols) + r" & $\mathcal{E}{\scriptstyle\,(s)}\downarrow$ \\",
         r"\midrule"]

    def line(name, vals, e, idx=None):
        out = []
        for k, _b, p in cols:
            s = pm(*vals[k], places=p) if k in vals else "--"
            if idx is not None and k in marks:
                s = h2h.tex_mark(s, marks[k].get(idx))
            out.append(s)
        return f"{name} & " + " & ".join(out) + f" & {e} \\\\"

    for i, r in enumerate(rows):
        if r["fam"] in OURS:
            L.append(rf"\cmidrule(lr){{1-{ncol}}}")
            name = r"NAU \textit{(Ours)}"
        else:
            name = NAME.get(r["fam"], r["fam"])
        name += "" if r["ok"] else r"$^{\dagger}$"
        L.append(line(name, r, r["e"], i))
    L.append(r"\midrule")
    L.append(line(r"\textit{Original (before deletion)}", ro, "--"))
    ea = [v for v in es.retrain_minutes_by_seed(refs).values() if v]
    ref_vals = {"f": (rr["f"][0], rr["f"][1]), "df": (0.0, None), "r": (rr["r"][0], rr["r"][1]),
                "dr": (0.0, None), "n": rr["n"], "tp": (100.0, None)}
    L.append(line(r"\textit{Retrain (reference)}", ref_vals,
                  rf"\textit{{{St.mean(ea) * 60:,.0f}}}" if ea else "--"))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return "\n".join(L) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--with-dUHF", dest="duhf", action="store_true")
    ap.add_argument("--out-dir", default="paper/v20_edits/tables")
    ap.add_argument("--pin", default=None, metavar="CSV",
                    help="validation-selected configurations (paper/plotting/valsens_selected.csv)")
    ap.add_argument("--ledger", default=None, metavar="TSV",
                    help="read the canonical `_mia` executions from paper/plotting/canonical_runs.tsv; "
                         "files get a _ledger suffix")
    ap.add_argument("--view", choices=("normal", "scrub"), default=None,
                    help="with --ledger: test view (scrub = cleaned test histories); default unchanged")
    a = ap.parse_args()
    global VIEW
    if a.view:
        assert a.ledger, "--view needs --ledger"
        import unwanted_ledger as ul
        ul.set_view(a.view, es)
        VIEW = a.view
    os.makedirs(a.out_dir, exist_ok=True)
    try:
        times = es.spam_tables_sacct()
    except Exception:
        times = {}
    sfx = ("_with_dUHF" if a.duhf else "") + ("_ledger" if a.ledger else "")
    if a.view:
        sfx = ("_with_dUHF" if a.duhf else "") + ul.VIEW_SUFFIX[a.view]
    for ds in ("beauty", "sports", "toys"):
        for main_tab in ((True, False) if ds == "beauty" else (False,)):
            fn = f"{'paper_' if main_tab else ''}sensitive_{ds}{sfx}.tex"
            open(os.path.join(a.out_dir, fn), "w").write(table(ds, a.duhf, main_tab, times, a.pin, a.ledger))
            print("wrote", os.path.join(a.out_dir, fn))


if __name__ == "__main__":
    main()
