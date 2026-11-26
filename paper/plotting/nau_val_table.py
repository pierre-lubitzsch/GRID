"""Appendix table: NAU hyperparameter selection on the VALIDATION split (Beauty, RQ-KMeans).

Rows: the lambda_s x lambda_n grid at lambda_f=0.1 (3 targets x 5 data seeds each).
Columns, all on the validation split:
  whole set     Delta^retr_SH and tau_P, from the validation-grid runs (<run>/eval_val),
                pooled like Table 1 (mean over seeds of the mean over targets)
  neighborhood  Delta^retr_SH and tau_P on the target's SID neighborhood (the target's own
                cases excluded), from logs/diagnose/fig1_scope_val/, tau_P as a ratio of
                pooled NDCG; "--" until that setting has been scored there.
Selection (2026-09-25): among settings of the full objective (lambda_s > 0 and lambda_n > 0)
with whole-set tau_P >= 95%, a setting counts as tied with the lowest whole-set Delta^retr_SH
when their paired per-cell difference is not resolved (seed-level 95% t-interval over the
five data seeds includes 0); among tied settings the highest combined utility
(1-w) tau_P(whole) + w tau_P(neighborhood), w=0.5, is selected. The script prints the
range of w for which the pick holds.

    PYTHONPATH=. python paper/plotting/nau_val_table.py [--out paper/v24_edits/tables/appendix_nau_valsel.tex]

--ledger paper/plotting/canonical_runs.tsv: ONE canonical execution per (setting, cell). Whole-set
columns and the per-cell Delta_SH of the tie rule come from each canonical run's eval_val (instead
of valsel_candidates_preliminary.csv and the *_valsel_nau_* runs; (0, 0) = the ledger's
forget_repair_lf0.1); neighborhood columns from the same fig1_scope_val{,2} JSONs, but an arm-cell
is only used when the JSON's `checkpoints` entry is the ledger's run dir for that setting and cell
(mismatches are printed and dropped). Same selection and tie rule. Outputs get a _ledger suffix:
<out>_ledger.tex and paper/plotting/nau_valsel_selected_ledger.csv (the pick, for
extract_table1_valsel.py --ledger). Settings with < 15 validation cells are printed as WAITING and
the selection is marked PRELIMINARY until all are complete.
"""
import argparse
import csv
import glob
import json
import os
import statistics as S

import numpy as np

GRID = [(s, n) for s in ("0.0", "0.01", "0.1") for n in ("0.0", "0.01", "0.1")]
# scope-eval arm name -> (lambda_s, lambda_n); both the tie-break names and the ledger names
ARM = {"tie_ln0": ("0.1", "0.0"), "tie_ln0p01": ("0.1", "0.01"), "reported_s0p01_n0p01": ("0.01", "0.01"),
       "sep_ln0": ("0.1", "0.0"), "nau": ("0.1", "0.01"), "nau_s0p01_n0p01": ("0.01", "0.01"),
       "nau_ln0p1": ("0.1", "0.1"), "sep_ln0p01": ("0.01", "0.0"), "nosep_ln0": ("0.0", "0.0"),
       "nosep_ln0p01": ("0.0", "0.01"), "nosep_ln0p1": ("0.0", "0.1")}
HERE = os.path.dirname(os.path.abspath(__file__))


def whole():
    out = {}
    for r in csv.DictReader(open(os.path.join(HERE, "valsel_candidates_preliminary.csv"))):
        if r["dataset"] == "beauty" and r["method"] == "NAU":
            ls = r["config"].split("lambda_s=")[1].split(",")[0].strip()
            ln = r["config"].split("lambda_n=")[1].strip()
            out[(ls, ln)] = (float(r["val_dsh"]), float(r["val_tau"]), int(r["n_cells"]))
    return out


def per_cell_dsh():
    """(lambda_s, lambda_n) -> {(stratum, seed): validation |SH_U - SH_R| in points}."""
    import re
    R = {}
    for d in glob.glob("logs/diagnose/valsel_refs/rkmeans_beauty_*_clean"):
        m = re.match(r"rkmeans_beauty_(\w+)_s(\d+)_clean$", os.path.basename(d))
        R[m.groups()] = float(list(csv.DictReader(open(d + "/csv/version_0/metrics.csv")))[-1]["test/SH@10"])
    V = {}
    for d in glob.glob("logs/unlearn/runs/*_valsel_nau_*"):
        m = re.search(r"bw_tgt(\w+?)_valsel_nau_ls(\w+?)_ln(\w+?)(?:_seed(\d+))?$", os.path.basename(d))
        f = d + "/eval_val/csv/version_0/metrics.csv"
        if not m or not os.path.isfile(f):
            continue
        st, ls, ln, sd = m.groups(); sd = sd or "2"
        key = (ls.replace("p", "."), ln.replace("p", "."))
        V.setdefault(key, {})[(st, sd)] = 100 * abs(float(list(csv.DictReader(open(f)))[-1]["test/SH@10"]) - R[(st, sd)])
    return V


def tied_with(V, a, b):
    """Is the paired validation Delta_SH difference a-b unresolved (seed-level 95% t-interval includes 0)?"""
    cells = sorted(set(V[a]) & set(V[b]))
    by = {}
    for c in cells:
        by.setdefault(c[1], []).append(V[a][c] - V[b][c])
    x = np.array([np.mean(v) for v in by.values()])
    if np.allclose(x, 0):
        return True
    half = 2.776 * x.std(ddof=1) / np.sqrt(len(x))   # t(0.975, df=4)
    return x.mean() - half <= 0 <= x.mean() + half


def neighborhood():
    nd, sh, cells = {}, {}, {}
    ref_nd, ref_sh = [], {}
    for f in sorted(glob.glob("logs/diagnose/fig1_scope_val/*.json") + glob.glob("logs/diagnose/fig1_scope_val2/*.json")):
        d = json.load(open(f))
        rows = {(r["method"], r["group"]): r for r in d["rows"]}
        c = rows.get(("clean", "neighbor"))
        if not c or c["n"] == 0:
            continue
        cell = (d["stratum"], str(d["seed"]))
        # the retrained reference is identical in both folders; count each cell once
        if cell not in ref_sh:
            ref_nd += c["ndcg_per_query"]
        ref_sh[cell] = c["sh_at_10"]
        for arm, key in ARM.items():
            r = rows.get((arm, "neighbor"))
            if r is None or cells.get(key, set()) & {cell}:
                continue
            nd.setdefault(key, []).extend(r["ndcg_per_query"])
            sh.setdefault(key, {})[cell] = abs(r["sh_at_10"] - c["sh_at_10"])
            cells.setdefault(key, set()).add(cell)
    out = {}
    for key in nd:
        by = {}
        for (st, sd), v in sh[key].items():
            by.setdefault(sd, []).append(v)
        out[key] = (100 * S.mean(S.mean(v) for v in by.values()), 100 * np.mean(nd[key]) / np.mean(ref_nd),
                    len(cells[key]))
    return out


def ledger_beauty(path):
    """(lambda_s, lambda_n) -> {(stratum, seed): run_dir} for Beauty RQ-KMeans canonical rows."""
    import re
    out = {}
    for r in csv.DictReader(open(path), delimiter="\t"):
        if r["setting"] != "spam" or r["dataset"] != "beauty" or r["track"] != "tiger+rkmeans":
            continue
        if r["config"] == "forget_repair_lf0.1":
            key = ("0.0", "0.0")
        else:
            m = re.match(r"nau_ls([0-9.]+)_ln([0-9.]+)$", r["config"])
            if not m:
                continue
            key = m.groups()
        out.setdefault(key, {})[(r["stratum"], r["seed"])] = r["run_dir"]
    return out


def _last(f):
    r = list(csv.DictReader(open(f)))[-1]
    return float(r["test/SH@10"]), float(r["test/ndcg@10"])


def ledger_whole(LB):
    """Whole-set validation columns and per-cell Delta_SH from the canonical runs' eval_val.
    Returns W {key: (dsh, tau, n_cells)}, V {key: {cell: dsh}}, waiting {key: n_missing}."""
    R = {}
    for d in glob.glob("logs/diagnose/valsel_refs/rkmeans_beauty_*_clean"):
        import re
        m = re.match(r"rkmeans_beauty_(\w+)_s(\d+)_clean$", os.path.basename(d))
        f = d + "/csv/version_0/metrics.csv"
        if m and os.path.isfile(f):
            R[m.groups()] = _last(f)
    W, V, waiting = {}, {}, {}
    for key, cells in LB.items():
        per = {}
        for cell, rd in cells.items():
            f = os.path.join(rd, "eval_val/csv/version_0/metrics.csv") if rd else ""
            if not rd or not os.path.isfile(f) or cell not in R:
                continue
            sh, nd = _last(f)
            per[cell] = (100 * abs(sh - R[cell][0]), nd / R[cell][1])
        waiting[key] = 15 - len(per)
        if not per:
            continue
        V[key] = {c: v[0] for c, v in per.items()}
        by = {}
        for (st, sd), v in per.items():
            by.setdefault(sd, []).append(v)
        W[key] = (S.mean(S.mean(x[0] for x in v) for v in by.values()),
                  100 * S.mean(S.mean(x[1] for x in v) for v in by.values()), len(per))
    return W, V, waiting


def ledger_neighborhood(LB):
    """neighborhood() restricted to arm-cells whose JSON checkpoint is the ledger's run dir."""
    nd, sh, cells = {}, {}, {}
    ref_nd, ref_sh = [], {}
    bad = []
    for f in sorted(glob.glob("logs/diagnose/fig1_scope_val/*.json") + glob.glob("logs/diagnose/fig1_scope_val2/*.json")):
        d = json.load(open(f))
        rows = {(r["method"], r["group"]): r for r in d["rows"]}
        c = rows.get(("clean", "neighbor"))
        if not c or c["n"] == 0:
            continue
        cell = (d["stratum"], str(d["seed"]))
        if cell not in ref_sh:
            ref_nd += c["ndcg_per_query"]
        ref_sh[cell] = c["sh_at_10"]
        for arm, key in ARM.items():
            r = rows.get((arm, "neighbor"))
            if r is None or cells.get(key, set()) & {cell}:
                continue
            led = LB.get(key, {}).get(cell)
            got = os.path.dirname(os.path.dirname(d["checkpoints"].get(arm, "")))
            if not led or os.path.abspath(led) != os.path.abspath(got):
                bad.append((f, arm, got, led))
                continue
            nd.setdefault(key, []).extend(r["ndcg_per_query"])
            sh.setdefault(key, {})[cell] = abs(r["sh_at_10"] - c["sh_at_10"])
            cells.setdefault(key, set()).add(cell)
    for b in bad:
        print("  scope arm-cell NOT the ledger's checkpoint, dropped:", *b)
    out = {}
    for key in nd:
        by = {}
        for (st, sd), v in sh[key].items():
            by.setdefault(sd, []).append(v)
        out[key] = (100 * S.mean(S.mean(v) for v in by.values()), 100 * np.mean(nd[key]) / np.mean(ref_nd),
                    len(cells[key]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/v24_edits/tables/appendix_nau_valsel.tex")
    ap.add_argument("--mix-weight", type=float, default=0.5, help="weight w of neighborhood tau_P in the tie-break")
    ap.add_argument("--ledger", default=None,
                    help="read ONE canonical execution per (setting, cell) from this canonical_runs.tsv; "
                         "writes <out>_ledger.tex and paper/plotting/nau_valsel_selected_ledger.csv")
    a = ap.parse_args()
    prelim = False
    if a.ledger:
        LB = ledger_beauty(a.ledger)
        W, V, waiting = ledger_whole(LB)
        N = ledger_neighborhood(LB)
        a.out = a.out[:-4] + "_ledger.tex" if a.out.endswith(".tex") else a.out + "_ledger"
        for k in GRID:
            if waiting.get(k, 15):
                print(f"  WAITING: ls={k[0]} ln={k[1]}: {waiting.get(k, 15)} of 15 cells without validation metrics")
        prelim = any(waiting.get(k, 15) for k in GRID if k in LB)
        if prelim:
            print("  PRELIMINARY: selection below uses incomplete settings")
    else:
        W, N = whole(), neighborhood()
        V = per_cell_dsh()
    full = [k for k in GRID if k in W and W[k][1] >= 95.0 and float(k[0]) > 0 and float(k[1]) > 0]
    if not full:
        print("no full-objective setting with validation tau_P >= 95% has validation metrics yet; nothing written")
        for k in GRID:
            if k in W:
                print(f"  ls={k[0]:5s} ln={k[1]:5s} whole {W[k][0]:.4f} {W[k][1]:5.1f} ({W[k][2]} cells) | nbr "
                      + (f"{N[k][0]:.3f} {N[k][1]:5.1f} ({N[k][2]} cells)" if k in N else "--"))
        return
    best = min(full, key=lambda k: W[k][0])
    tied = [k for k in full if k == best or tied_with(V, k, best)]
    w = a.mix_weight
    mix = lambda k, w=w: (1 - w) * W[k][1] + w * N[k][1] if k in N else float("-inf")
    pick = max(tied, key=mix)
    # range of w in [0, 1] for which the same setting wins among the tied ones
    grid_w = [x / 1000 for x in range(1001)]
    hold = [x for x in grid_w if max(tied, key=lambda k: mix(k, x)) == pick]
    print("full-objective candidates:", full, "| lowest exposure:", best, "| tied:", tied)
    f2 = lambda k, i, p: (f"{N[k][i]:.{p}f}" if k in N and N[k][2] == 15 else
                          (f"{N[k][i]:.{p}f}$^{{*}}$" if k in N else "--"))
    L = ([f"% PRELIMINARY (generated from {a.ledger} with settings still missing validation cells)"] if prelim else []) + \
        [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{",
         r"Selection of NAU's weights on the validation split (Beauty, TIGER + RQ-KMeans, $\lambda_f=0.1$, three targets $\times$ five data seeds).",
         r"Whole set: $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ and $\tau_P$ on all validation cases; neighborhood: the same on the validation cases whose next item shares the target's first \ac{SID} token (the target's own cases excluded), with $\tau_P$ as a ratio of pooled $\mathrm{NDCG}@10$.",
         (r"Selection is among settings of the full objective ($\lambda_s, \lambda_n > 0$) that keep whole-set $\tau_P \geq 95\%$: settings whose whole-set $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ is not resolvably different from the lowest one (paired 95\% interval over the five seeds includes 0) are treated as tied, and among them the highest combined utility, the mean of whole-set and neighborhood $\tau_P$, is selected (\checkmark); the choice is the same for any weight above @@WMIN@@ on the neighborhood."),
         r"Settings below $95\%$ whole-set utility are shown in gray; -- marks settings not scored on the neighborhood.}",
         r"\label{tab:appendix-nau-valsel}", r"\begin{tabular}{llccccc c}", r"\toprule",
         r" & & \multicolumn{2}{c}{Whole set} & \multicolumn{2}{c}{SID neighborhood} & Combined & \\",
         r"\cmidrule(lr){3-4} \cmidrule(lr){5-6} \cmidrule(lr){7-7}",
         r"$\lambda_s$ & $\lambda_n$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & Sel. \\",
         r"\midrule"]
    for k in GRID:
        if k not in W:
            continue
        gray = W[k][1] < 95.0
        comb = f"{mix(k):.1f}" if k in N else "--"
        cells = [k[0], k[1], f"{W[k][0]:.3f}", f"{W[k][1]:.1f}", f2(k, 0, 3), f2(k, 1, 1), comb, r"\checkmark" if k == pick else ""]
        if gray:
            cells = [r"\textcolor{gray}{" + c + "}" if c else c for c in cells]
        L.append(" & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    if any(k in N and N[k][2] < 15 for k in GRID):
        L.append(r"\par\smallskip\footnotesize $^{*}$ scored on fewer than 15 cells.")
    L.append(r"\end{table}")
    wmin = min(hold); assert hold == [x for x in grid_w if x >= wmin], "pick is not monotone in w"
    if wmin == 0:
        L = [l.replace("; the choice is the same for any weight above @@WMIN@@ on the neighborhood.", "; the choice is the same for any weighting of the two.") for l in L]
    L = [l.replace("@@WMIN@@", f"{100 * wmin:.1f}\\%") for l in L]
    print(f"tie-break pick holds for neighborhood weight w >= {wmin:.3f} (of [0, 1]); used w={w}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    open(a.out, "w").write("\n".join(L) + "\n")
    print(f"wrote {a.out}; selected lambda_s={pick[0]}, lambda_n={pick[1]}" + (" (PRELIMINARY)" if prelim else ""))
    if a.ledger:
        sel = os.path.join(HERE, "nau_valsel_selected_ledger.csv")
        with open(sel, "w", newline="") as f:
            wr = csv.writer(f)
            wr.writerow(["dataset", "lambda_s", "lambda_n", "config", "val_dsh", "val_tau", "n_cells", "preliminary"])
            wr.writerow(["beauty", pick[0], pick[1], f"nau_ls{pick[0]}_ln{pick[1]}", f"{W[pick][0]:.4f}",
                         f"{W[pick][1]:.2f}", W[pick][2], int(prelim)])
        print(f"wrote {sel}")
    for k in GRID:
        if k in W:
            print(f"  ls={k[0]:5s} ln={k[1]:5s} whole {W[k][0]:.4f} {W[k][1]:5.1f} | nbr "
                  + (f"{N[k][0]:.3f} {N[k][1]:5.1f} ({N[k][2]} cells)" if k in N else "--"))


if __name__ == "__main__":
    main()
