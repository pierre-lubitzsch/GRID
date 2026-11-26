"""Appendix table: every baseline tuned on a 3 x 3 grid on Beauty validation, like NAU.

Each baseline's two main knobs on a 3 x 3 grid centered on its earlier setting
(scripts/make_valsel_basegrid_plan.py; logs/valsel_basegrid_plan.tsv), plus a second round
beyond the grid edges and SCIF's step multiplier (--ext; logs/valsel_basegrid_ext_plan.tsv), 15 Beauty cells
(3 targets x 5 data seeds, TIGER + RQ-KMeans), each run evaluated on validation and test.
Metrics as in Table 1 (paper/plotting/valsel_table.py helpers). Selection per method with the
baseline rule of the paper, on Beauty validation: smallest Delta^retr_SH among settings with
tau_P >= 95%, ties by higher tau_P; if none reaches 95%, smallest Delta^retr_SH overall.

    PYTHONPATH=. python paper/plotting/baseline_grid_table.py [--out paper/revision/tables/appendix_baseline_grid.tex]

--ledger paper/plotting/canonical_runs.tsv: the "v33 setting" test columns (and the Selected columns
when the grid picks the v33 setting) are Table 1's canonical execution of that configuration
(extract_table1_valsel.ledger_cells), not the grid's own run of it; selection itself is unchanged
(validation of the grid runs). Default output then paper/revision/tables/appendix_baseline_grid_ledger.tex.
"""
import argparse
import csv
import glob
import importlib.util
import json
import os
import re
import statistics as S
import sys
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vt", os.path.join(HERE, "valsel_table.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
sys.path.insert(0, "scripts")
import run_tags  # noqa: E402

LABEL = {"finetune_abl": "Finetune", "ga_abl": "Forget only", "negtrain_abl": "Forget+Repair", "scif": "SCIF",
         "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan", "nau": r"NAU"}
ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "NAU"]
# Table 1 (v33) Beauty configuration per method; Finetune and SCIF had one setting each.
PAPER = {"Finetune": "unified_abl_lf0.0_ls0.0_ln0.0_ulr0.0001_ep4", "SCIF": "scif_d0.01_n1.0"}


def knobs(fam, cfg):
    """Readable knob values of a configuration (defaults filled in)."""
    g = lambda pat, d: (re.search(pat, cfg).group(1) if re.search(pat, cfg) else d)
    ep = g(r"_ep(\d+)", "4")
    if fam in ("ga_abl", "negtrain_abl"):
        return rf"$\lambda_f$={g(r'_lf([0-9.]+)', '?')}, ep={ep}"
    if fam == "finetune_abl":
        return f"lr={g(r'_ulr([0-9.]+)', '0.0001')}, ep={ep}"
    if fam == "scif":
        return f"damping={g(r'_d([0-9.]+)', '0.01')}, max norm={g(r'_n([0-9.]+)', '1.0')}, step x{g(r'_ts([0-9.]+)', '1')}"
    if fam == "seif":
        return f"std={g(r'_std([0-9.]+)', '?')}, repair lr={g(r'_rlr([0-9.]+)', '0.0007')}"
    if fam == "kookmin":
        return f"init rate={g(r'_ir([0-9.]+)', '?')}, retain lr={g(r'_rlr([0-9.]+)', '0.001')}"
    if fam == "fanchuan":
        return f"temp.={g(r'_t([0-9.]+)', '?')}, lr={g(r'_lr([0-9.]+)', '0.001')}"
    return cfg


def collect():
    R = vt.refs()
    test_ref = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("paper/data/spam_references/*.json")}
    runs = defaultdict(dict)
    for d in glob.glob("logs/unlearn/runs/*_bs1_bw_*_valsel*"):
        if "_valselck" in d:   # checkpoint-kept reruns of picks, not candidates
            continue
        rec = run_tags.parse_run_dir(d)
        if not rec or rec["dataset"] != "beauty" or rec["track"] != "tiger+rkmeans":
            continue
        fam, cfg = vt.config_of(rec)
        if fam == "nau" and cfg != "lambda_s=0.01, lambda_n=0.01":
            continue
        te, va = vt.metrics(d + "/eval"), vt.metrics(d + "/eval_val")
        if not te or not va or fam not in LABEL:
            continue
        cell = (rec["stratum"], str(rec["seed"]))
        if cell not in runs[(fam, cfg)] or int(rec["jid"]) > runs[(fam, cfg)][cell][2]:
            runs[(fam, cfg)][cell] = (te, va, int(rec["jid"]))
    out = []
    for (fam, cfg), cc in runs.items():
        if len(cc) < 15:
            print(f"skip {fam} {cfg}: {len(cc)} cells", file=sys.stderr)
            continue
        pv, pt = defaultdict(list), defaultdict(list)
        for (st, sd), (te, va, _j) in cc.items():
            cv = R[("rkmeans", "beauty", st, sd, "clean")]
            rt = test_ref["beauty" + ("" if sd == "2" else f"_seed{sd}")]
            pv[sd].append((abs(va[0] - cv[0]), va[1] / cv[1]))
            pt[sd].append((abs(te[0] - rt["clean_SH@10"][st]), te[1] / rt["clean_ndcg@10"]))
        agg = lambda per, j: 100 * S.mean(S.mean(x[j] for x in v) for v in per.values())
        out.append(dict(method=LABEL[fam], fam=fam, config=cfg, val_dsh=agg(pv, 0), val_tau=agg(pv, 1),
                        test_dsh=agg(pt, 0), test_tau=agg(pt, 1)))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--ledger", default=None)
    a = ap.parse_args()
    if a.out is None:
        a.out = ("paper/revision/tables/appendix_baseline_grid_ledger.tex" if a.ledger
                 else "paper/revision/tables/appendix_baseline_grid.tex")
    rows = collect()
    canon = {}
    if a.ledger:
        sp = importlib.util.spec_from_file_location("t1", os.path.join(HERE, "extract_table1_valsel.py"))
        t1 = importlib.util.module_from_spec(sp); sp.loader.exec_module(t1)
        LC = t1.ledger_cells(a.ledger, "beauty", methods=ORDER)
        R = {sd: t1.ref("beauty", sd, "rkmeans") for sd in t1.SEEDS}
        for m, c in LC.items():
            if len(c) < 15:
                print(f"PENDING {m}: {len(c)} canonical cells"); continue
            canon[m] = (100 * S.mean(S.mean(abs(c[(st, sd)][0] - R[sd]["clean_SH@10"][st]) for st in t1.STR) for sd in t1.SEEDS),
                        100 * S.mean(S.mean(c[(st, sd)][1] / R[sd]["clean_ndcg@10"] for st in t1.STR) for sd in t1.SEEDS))
    paper = {r["method"]: r["config"] for r in csv.DictReader(open(os.path.join(HERE, "valsel_selected.csv")))
             if r["dataset"] == "beauty" and r["track"] == "rkmeans"}
    paper.update(PAPER)
    by = defaultdict(list)
    for r in rows:
        by[r["method"]].append(r)
    summary = []
    for m in ORDER:
        C = by.get(m, [])
        if not C:
            continue
        ok = [r for r in C if r["val_tau"] >= 95.0]
        # No setting reaches the bar: the paper's fallback (smallest Delta overall) would show a
        # destroyed model (tau_P ~1%), so report the highest-utility setting instead (marked).
        pick = (min(ok, key=lambda r: (round(r["val_dsh"], 6), -r["val_tau"])) if ok
                else max(C, key=lambda r: (round(r["val_tau"], 6), -r["val_dsh"])))
        old = next((r for r in C if r["config"] == paper.get(m)), None) if m != "NAU" else pick
        if m == "NAU" and m in canon and pick["config"] == "lambda_s=0.01, lambda_n=0.01":
            print(f"  NAU: test from the canonical execution {canon[m][0]:.3f}/{canon[m][1]:.1f} "
                  f"(grid's own run {pick['test_dsh']:.3f}/{pick['test_tau']:.1f})")
            pick = old = dict(pick, test_dsh=canon[m][0], test_tau=canon[m][1])
        if m in canon and old is not None and m != "NAU":
            print(f"  {m}: v33 setting test from the canonical execution {canon[m][0]:.3f}/{canon[m][1]:.1f} "
                  f"(grid's own run {old['test_dsh']:.3f}/{old['test_tau']:.1f})")
            new = dict(old, test_dsh=canon[m][0], test_tau=canon[m][1])
            if pick is old:
                pick = new
            old = new
        summary.append((m, len(C), len(ok), pick, old))
        print(f"\n{m}: {len(C)} settings, {len(ok)} with val tau>=95")
        for r in sorted(C, key=lambda r: (r["val_dsh"], -r["val_tau"])):
            flag = ("PICK " if r is pick else "     ") + ("v33 " if old is not None and r is old else "    ")
            print(f"  {flag}{knobs(r['fam'], r['config']):38s} val {r['val_dsh']:.3f} {r['val_tau']:5.1f} | test {r['test_dsh']:.3f} {r['test_tau']:5.1f}")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3pt}", r"\caption{",
         r"Baselines tuned with the same budget as NAU (Beauty, TIGER + RQ-KMeans, three targets $\times$ five data seeds). "
         r"Each baseline's two main hyperparameters are searched on a $3 \times 3$ grid centered on its earlier setting, the same number of settings as NAU's grid (\Cref{tab:appendix-nau-valsel}), "
         r"and selected on the validation split with the baseline rule of \Cref{tab:appendix-valsel} (smallest $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ among settings with $\tau_P \geq 95\%$, ties by higher $\tau_P$). "
         r"Selected: the grid's choice; v33: the setting used in \Cref{tab:bandwagon-unlearn-all-tiger-rkmeans}. Test-split values. "
         r"$\geq 95\%$: settings meeting the utility bar on validation. "
         r"$^{\dagger}$: no setting reaches $95\%$; the setting with the highest validation $\tau_P$ is shown.}",
         r"\label{tab:appendix-baseline-grid}", r"\resizebox{\columnwidth}{!}{%", r"\begin{tabular}{llcccccc}", r"\toprule",
         r" & & & \multicolumn{2}{c}{Selected (grid)} & \multicolumn{2}{c}{v33 setting} & \\",
         r"\cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"Method & Selected setting & $\geq 95\%$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & Changed \\", r"\midrule"]
    for m, n, nok, pick, old in summary:
        name = r"NAU \textit{(ours)}" if m == "NAU" else m + ("" if nok else r"$^{\dagger}$")
        setting = r"$\lambda_s$=0.01, $\lambda_n$=0.01" if m == "NAU" else knobs(pick["fam"], pick["config"])
        oc = (f"{old['test_dsh']:.3f} & {old['test_tau']:.1f}") if old else "-- & --"
        ch = "" if m == "NAU" else ("no" if old is pick else "yes")
        if m == "NAU":
            L.append(r"\midrule")
        L.append(f"{name} & {setting} & {nok}/{n} & {pick['test_dsh']:.3f} & {pick['test_tau']:.1f} & {oc} & {ch} \\\\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
