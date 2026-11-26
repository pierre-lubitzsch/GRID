"""Sensitive item deletion tables for the RETRAINING-MATCHED split (unlearning.split_mode=retrain_matched),
selected on cleaned validation and reported on cleaned test histories (2026-11-16).

Reuses the loaders of rmsplit_selection.py (run_dir, metrics_csv, ref, CAT, SEEDS). Every configuration
comes from paper/revision/rmsplit_selected.csv (per dataset; NAU's is the Beauty pick on every dataset),
written by rmsplit_selection.py with its default SEL_RULE=uhf (the submitted rule, main protocol since
2026-11-17): the lowest validation UHF@10 among configurations with validation tau_P >= 95% (else lowest overall).
The rank-rule (distance to retraining) outputs of 2026-11-16 are in paper/revision/rankrule_backup_20261117/.
Writes files (suffix _rmsplit) under paper/revision/tables/:
  paper_sensitive_beauty_rmsplit.tex       Table 3 (main text, Beauty)
  sensitive_sports_rmsplit.tex             appendix, with the Delta^retr_UHF column
  sensitive_toys_rmsplit.tex               appendix, with the Delta^retr_UHF column
  ablation_unified_sensitive_lambda_n_all_rmsplit.tex   Table 4 (lambda_n at NAU's lambda_s first, then the others)
  sensitive_user_counts_rmsplit.tex        pooled requesting users shown the category + McNemar range
  appendix_valsens_rmsplit.tex             selection table, validation only
  appendix_sensitive_anchor_rmsplit.tex    retain-side anchor at the new NAU pick
  appendix_brand_takedown_rmsplit.tex      brand takedown (Beauty, e.l.f.)
  appendix_sensitive_split_robustness_rmsplit.tex   OPTIONAL: old split vs retraining-matched split
and paper/revision/rmsplit_tables_numbers.csv (every number printed, per method and dataset).

    python paper/plotting/critique_followup/rmsplit_tables.py
"""
import csv
import glob
import importlib.util
import json
import os
import re
import statistics as S
import sys

from scipy.stats import binomtest

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("rsel", os.path.join(HERE, "rmsplit_selection.py"))
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)
sys.path.insert(0, "scripts")
_s = importlib.util.spec_from_file_location("h2h", "scripts/extract_bandwagon_head2head.py")
h2h = importlib.util.module_from_spec(_s)
_s.loader.exec_module(h2h)

OUT = "paper/revision/tables"
DS, SEEDS, CAT = R.DS, R.SEEDS, R.CAT
CATNAME = {"beauty": "Hair Loss Products", "sports": "Guns \\& Rifles", "toys": "toy weapons"}
SELECTED = list(csv.DictReader(open(R.OUT_SEL)))
NAU_ARM = next(r["new_config"] for r in SELECTED if r["dataset"] == "beauty" and r["method"] == "NAU")
assert all(r["new_config"] == NAU_ARM for r in SELECTED if r["method"] == "NAU"), "NAU must be one config"
_m = re.match(r"unified_hp2_sepgen_lf0\.1_ls([0-9.]+)_ln(neg)?([0-9.]+)", NAU_ARM)
NAU_LS, NAU_LN = float(_m.group(1)), (-1 if _m.group(2) else 1) * float(_m.group(3))
NAU_TXT = rf"$\lambda_s={NAU_LS:g}$, $\lambda_n={NAU_LN:g}$"
RULE_TXT = {
    # the submitted rule (main protocol since 2026-11-17)
    "uhf": (r"the lowest validation $\mathrm{UHF}@10$ among configurations with validation $\tau_P\geq95\%$ on the cleaned "
            r"validation histories (if none reaches it, the lowest overall)"),
    # alternative (appendix check only)
    "rank": (r"among configurations with validation $\tau_P\geq95\%$ (all, if none reaches it), the best average rank in "
             r"distance to the retrained model on cleaned validation histories, per seed $|\mathrm{UHF}@10-\mathrm{UHF}_R@10|$ "
             r"(requesting users) and $|\mathrm{UHR}@10-\mathrm{UHR}_R@10|$ (other users); ties: smaller $\mathrm{UHF}$ distance, then higher $\tau_P$"),
}[R.RULE]
ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU"]
FLOOR = 0.95
PENDING = []


def nau_arm(ls, ln):
    """NAU grid arm for lambda_s, lambda_n (lambda_s > 0 runs carry _sepck)."""
    lnt = "ln0.0" if ln == 0 else f"lnneg{-ln:.1f}".replace("neg10.0", "neg10.0")
    return f"unified_hp2_sepgen_lf0.1_ls{ls}_{lnt}" + ("_sepck" if ls > 0 else "")


def tag(ds, arm, sd, prefix=None):
    p = prefix or f"sens_{ds}_{CAT[ds]}rm"
    return f"{p}_{arm}_rmsplit_canonck" + ("" if sd == 2 else f"_seed{sd}")


REFS = {}


def refm(view, ds, role, sd):
    k = (view, ds, role, sd)
    if k not in REFS:
        REFS[k] = R.ref(view, ds, role, sd)
    return REFS[k]


def cell(m, retr, d=None):
    if not m or not retr or R.K["f"] not in m:
        return None
    return dict(f=100 * m[R.K["f"]], r=100 * m[R.K["r"]], n=100 * m[R.K["n"]],
                tp=100 * m[R.K["n"]] / retr[R.K["n"]],
                df=100 * abs(m[R.K["f"]] - retr[R.K["f"]]), dr=100 * abs(m[R.K["r"]] - retr[R.K["r"]]), dir=d)


def runs(ds, arm, view="scrub", need=5):
    """{seed: cell} on the cleaned test (scrub) or cleaned validation (scrubval) view."""
    out = {}
    for sd in SEEDS:
        d = R.run_dir(tag(ds, arm, sd))
        if not d:
            continue
        c = cell(R.metrics_csv(f"{R.ROOT}/{view}/runs/{os.path.basename(d)}/eval"), refm(view, ds, "retrain", sd), d)
        if c:
            out[sd] = c
    if need and len(out) != need:
        raise RuntimeError(f"{ds} {arm} {view}: {len(out)} seeds")
    return out


def refcells(ds, role, view="scrub"):
    return {sd: cell(refm(view, ds, role, sd), refm(view, ds, "retrain", sd)) for sd in SEEDS}


def agg(cells):
    a = {}
    for k in ("f", "r", "n", "tp", "df", "dr"):
        v = [c[k] for c in cells.values()]
        a[k] = (S.mean(v), S.stdev(v))
    lat = [m for c in cells.values() if c.get("dir") and (m := h2h.request_latency_minutes(c["dir"])) is not None]
    a["e"] = f"{S.median(lat) * 60:.1f}" if lat else "--"
    return a


def picks(ds):
    sel = {r["method"]: r["new_config"] for r in SELECTED if r["dataset"] == ds}
    return {m: sel[m] for m in ORDER if m in sel}


def retrain_seconds(ds):
    """E(A): retraining time of the retrained references (job id from the evaluated checkpoint)."""
    out = []
    for sd in SEEDS:
        c = json.load(open(f"{R.ROOT}/scrub/reference/v1/{ds}_retrain" + ("" if sd == 2 else f"_seed{sd}") + "/coordinates.json"))
        m = re.search(r"job(\d+)_", c["ckpt"])
        t = h2h.train_minutes(m.group(1)) if m else None
        if t is not None:
            out.append(t * 60)
    return S.mean(out) if out else None


def pm(m, sd, p=3):
    return f"{m:.{p}f}" + ("" if sd is None else rf"\,{{\scriptsize$\pm${sd:.{p}f}}}")


NUMS = []


# ----------------------------------------------------------------------------------------- Tables 3, 22, 23
def method_table(ds, main):
    pk = picks(ds)
    rows = []
    for m, arm in pk.items():
        a = agg(runs(ds, arm))
        rows.append(dict(name=m, arm=arm, a=a, ok=a["tp"][0] >= 100 * FLOOR))
        NUMS.append(dict(table="main" if main else "appendix", dataset=ds, method=m, config=arm,
                         **{k: a[k][0] for k in ("f", "df", "r", "dr", "n", "tp")}, e=a["e"]))
    ro, rr = agg(refcells(ds, "base")), agg(refcells(ds, "retrain"))
    for nm, a in (("Before deletion", ro), ("Retrained", rr)):
        NUMS.append(dict(table="main" if main else "appendix", dataset=ds, method=nm, config="",
                         **{k: a[k][0] for k in ("f", "df", "r", "dr", "n", "tp")}, e=""))
    cols = [("f", "low"), ("df", "low"), ("r", None), ("dr", "low"), ("n", "high"), ("tp", "high")]
    if main:
        cols = [c for c in cols if c[0] != "df"]
    elig = [r["ok"] for r in rows]
    marks = {k: h2h.rank_marks([r["a"][k][0] for r in rows], b, elig) for k, b in cols if b}
    hdr = {"f": r"$\mathrm{UHF}@10{\scriptstyle\,(\%)}\downarrow$",
           "df": r"$\Delta^{\mathrm{retr}}_{\mathrm{UHF}}{\scriptstyle\,(\%)}\downarrow$",
           "r": r"$\mathrm{UHR}@10{\scriptstyle\,(\%)}$",
           "dr": r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}{\scriptstyle\,(\%)}\downarrow$",
           "n": r"$\mathrm{NDCG}@10\uparrow$", "tp": r"$\tau_P$ (\%)"}
    ncol = len(cols) + 2
    label = f"tab:paper-sensitive-{ds}-tiger-rkmeans" if main else f"tab:sensitive-{ds}-tiger-rkmeans"
    cap = [rf"Sensitive item deletion on {ds.capitalize()} ({CATNAME[ds]}) with TIGER + RQ-KMeans, five data seeds, retraining-matched split (each request forgets only the training examples whose next item is a deleted interaction).",
           r"$\mathrm{UHF}@10$ and $\mathrm{UHR}@10$ are the sensitive-item hit rates of the requesting and the remaining users; "
           + (r"$\Delta^{\mathrm{retr}}_{\mathrm{UHF}}$ and $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ are the means over seeds of $|\mathrm{UHF}_U@10-\mathrm{UHF}_R@10|$ and $|\mathrm{UHR}_U@10-\mathrm{UHR}_R@10|$, the distances to the retrained model of the same seed."
              if not main else
              r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ is the mean over seeds of $|\mathrm{UHR}_U@10-\mathrm{UHR}_R@10|$, the distance to the retrained model of the same seed.")]
    if not main:
        cap.append(r"$\Delta^{\mathrm{retr}}_{\mathrm{UHF}}$ is the per-seed distance of the requesting users' exposure to the retrained model's; since sensitive item deletion also asks to stop recommending the category, $\mathrm{UHF}@10$ itself remains the primary metric.")
    cap.append(r"Configurations are selected on the validation split (\Cref{tab:appendix-valsens}): " + RULE_TXT + r"; NAU uses " + NAU_TXT + r", selected once on Beauty. $^{\dagger}$: test $\tau_P<95\%$ on unrounded values, not ranked; bold and underline mark the best and second best among the other methods.")
    cap.append(r"All models are evaluated on cleaned test histories (the requesting users' deleted items removed from their test input histories).")
    if ro["f"][0] <= rr["f"][0]:
        cap.append(r"Retraining does not lower $\mathrm{UHF}@10$ below the original model.")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\renewcommand{\arraystretch}{1.05}",
         r"\caption{", *cap, "}", rf"\label{{{label}}}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{l" + "c" * (ncol - 1) + "}", r"\toprule",
         r"\textbf{Method} & " + " & ".join(hdr[k] for k, _ in cols) + r" & $\mathcal{E}{\scriptstyle\,(s)}\downarrow$ \\", r"\midrule"]

    def line(name, a, e, i=None, sdless=()):
        out = []
        for k, _ in cols:
            p = 1 if k == "tp" else 3
            s = pm(a[k][0], None if k in sdless else a[k][1], p)
            if i is not None and k in marks:
                s = h2h.tex_mark(s, marks[k].get(i))
            out.append(s)
        return f"{name} & " + " & ".join(out) + f" & {e} \\\\"

    for i, r in enumerate(rows):
        if r["name"] == "NAU":
            L.append(rf"\cmidrule(lr){{1-{ncol}}}")
            nm = r"NAU \textit{(Ours)}"
        else:
            nm = r["name"]
        L.append(line(nm + ("" if r["ok"] else r"$^{\dagger}$"), r["a"], r["a"]["e"], i))
    L.append(r"\midrule")
    L.append(line(r"\textit{Original (before deletion)}", ro, "--"))
    ea = retrain_seconds(ds)
    rr2 = dict(rr)
    rr2["df"], rr2["dr"], rr2["tp"] = (0.0, None), (0.0, None), (100.0, None)
    L.append(line(r"\textit{Retrain (reference)}", rr2, rf"\textit{{{ea:,.0f}}}" if ea else "--", sdless=("df", "dr", "tp")))
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    fn = f"{OUT}/{'paper_' if main else ''}sensitive_{ds}_rmsplit.tex"
    # 0.000 for the retrained Deltas without +-: match the old table's plain "0.000"
    open(fn, "w").write("\n".join(L).replace(r"0.000 & 0.000 &", r"0.000 & 0.000 &") + "\n")
    print("wrote", fn)
    return rows, ro, rr


# ----------------------------------------------------------------------------------------- Table 4 (lambda_n)
def lambda_n_table():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\caption{",
         r"Effect of the neighborhood term in sensitive item deletion (TIGER + RQ-KMeans, five data seeds, retraining-matched split, cleaned test histories).",
         r"Each cell gives the mean over seeds for $\lambda_n = 0 \rightarrow -1 \rightarrow -10$ ($\lambda_f=0.1$; all runs share the batch size of 32 of the other unlearning objectives, so $\lambda_s=0$, $\lambda_n=0$ is Forget+Repair with $\lambda_f=0.1$) and, in parentheses, on how many seeds $\lambda_n=-1$ and $\lambda_n=-10$ lower $\mathrm{UHF}$ or $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ or raise $\tau_P$ relative to $\lambda_n=0$.",
         rf"NAU uses $\lambda_s={NAU_LS:g}$ and $\lambda_n={NAU_LN:g}$, selected on the Beauty validation split by the rule of \Cref{{tab:appendix-valsens}}" + ("" if R.NAU_GRID == "submitted" else "; its $\\lambda_s$ is the first row of each dataset, followed by the other separation weights") + rf". Retrained $\mathrm{{UHF}}@10$ and $\mathrm{{UHR}}@10$ are given for reference.", "}",
         r"\label{tab:ablation-unified-sensitive-lambda-n-all}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Dataset & $\lambda_s$ & $\mathrm{UHF}@10{\scriptstyle\,(\%)}{\downarrow}$ & $\mathrm{UHR}@10{\scriptstyle\,(\%)}$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}{\scriptstyle\,(\%)}{\downarrow}$ & $\tau_P{\scriptstyle\,(\%)}{\uparrow}$ & Retrained $\mathrm{UHF}$ / $\mathrm{UHR}$ \\",
         r"\midrule"]
    for di, ds in enumerate(DS):
        rr = agg(refcells(ds, "retrain"))
        reftxt = "%.1f / %.2f" % (rr["f"][0], rr["r"][0])
        lss = [NAU_LS] if R.NAU_GRID == "submitted" else [NAU_LS] + [x for x in (0.1, 0.01, 0.0) if x != NAU_LS]
        for j, ls in enumerate(lss):
            c = [runs(ds, nau_arm(ls, ln)) for ln in (0, -1, -10)]

            def col(k, better, counts=True):
                m = [S.mean(x[k] for x in ci.values()) for ci in c]
                p = 2 if k == "dr" else 1
                s = " $\\rightarrow$ ".join(f"{v:.{p}f}" for v in m)
                if counts:
                    n = []
                    for ci in c[1:]:
                        n.append(sum((ci[sd][k] < c[0][sd][k] - 1e-9) if better == "low" else (ci[sd][k] > c[0][sd][k] + 1e-9) for sd in SEEDS))
                    s += f" ({n[0]}/5, {n[1]}/5)"
                NUMS.append(dict(table="lambda_n", dataset=ds, method=f"NAU ls={ls}", config=k, f=m[0], df=m[1], r=m[2]))
                return s
            L.append(f"{ds.capitalize() if j == 0 else ''} & {ls:g} & {col('f', 'low')} & {col('r', None, False)} & "
                     f"{col('dr', 'low')} & {col('tp', 'high')} & {reftxt if j == 0 else ''} \\\\")
        if di < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\vspace*{-10pt}", r"\end{table}"]
    fn = f"{OUT}/ablation_unified_sensitive_lambda_n_all_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)


# ----------------------------------------------------------------------------------------- user counts
NTEST = {}


# requesting TEST users per (dataset, seed), as in sensitive_counts.py (every UHF@10 value is k/N; asserted below)
N_OLD = {("beauty", 2): 14, ("beauty", 3): 12, ("beauty", 5): 11, ("beauty", 7): 13, ("beauty", 11): 11,
         ("sports", 2): 13, ("sports", 3): 14, ("sports", 5): 14, ("sports", 7): 14, ("sports", 11): 17,
         ("toys", 2): 12, ("toys", 3): 13, ("toys", 5): 13, ("toys", 7): 11, ("toys", 11): 12}


def n_test(ds, sd, fracs):
    n = N_OLD[(ds, sd)]
    assert all(abs(f / 100 * n - round(f / 100 * n)) < 1e-3 for f in fracs), (ds, sd, fracs)
    return n


def counts_table():
    meth = [("Before deletion", None), ("Retrained", None)] + [(m, None) for m in ("Finetune", "Forget+Repair", "Forget only", "SCIF", "NAU")]
    k = {}
    allc = {}
    for ds in DS:
        pk = picks(ds)
        cells = {"Before deletion": refcells(ds, "base"), "Retrained": refcells(ds, "retrain")}
        for m in ("Finetune", "Forget+Repair", "Forget only", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU"):
            if m in pk:
                cells[m] = runs(ds, pk[m])
        for sd in SEEDS:
            NTEST[(ds, sd)] = n_test(ds, sd, [c[sd]["f"] for c in cells.values()])
        for m, c in cells.items():
            for sd in SEEDS:
                k[(ds, m, sd)] = round(c[sd]["f"] / 100 * NTEST[(ds, sd)])
        allc[ds] = cells

    def tot(ds, m):
        return sum(k[(ds, m, sd)] for sd in SEEDS)

    def mcnemar(ds, a, b):
        lo = hi = diff = 0
        for sd in SEEDS:
            ka, kb, n = k[(ds, a, sd)], k[(ds, b, sd)], NTEST[(ds, sd)]
            lo += max(0, ka - kb); hi += min(ka, n - kb); diff += kb - ka
        ps = []
        for n10 in range(lo, hi + 1):
            n01 = n10 + diff
            if n01 < 0:
                continue
            ps.append(1.0 if n10 + n01 == 0 else binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
        return min(ps), max(ps)
    tests = {}
    for ds in DS:
        for a in ("NAU", "Forget only", "Forget+Repair"):
            for b in ("Before deletion", "Retrained", "Forget only", "Forget+Repair", "NAU"):
                if a != b:
                    tests[(ds, a, b)] = mcnemar(ds, a, b)
    ntot = {ds: sum(NTEST[(ds, sd)] for sd in SEEDS) for ds in DS}
    meths = [m for m, _ in meth]
    best = {ds: min(tot(ds, m) for m in meths[2:]) for ds in DS}
    # caption: which NAU vs before-deletion / Forget only comparisons are significant
    rng = lambda t: "$p<0.001$" if t[1] < 0.001 else f"$p$ between ${t[0]:.3f}$ and ${t[1]:.3f}$"
    sent = []
    for nm in ("NAU", "Forget only", "Forget+Repair"):
        s_ = [ds for ds in DS if tests[(ds, nm, "Before deletion")][1] < 0.05]
        if s_:
            sent.append(f"{nm} vs.\\ the model before deletion is significant on " + " and ".join(
                f"{d.capitalize()} ({rng(tests[(d, nm, 'Before deletion')])})" for d in s_))
    pair_min = min(tests[(ds, a, b)][0] for ds in DS for a, b in (("NAU", "Forget only"), ("NAU", "Forget+Repair"), ("Forget only", "Forget+Repair")))
    seif_fan = max(S.mean(allc[ds][m][sd]["tp"] for sd in SEEDS) for ds in DS for m in ("SEIF", "Kookmin", "Fanchuan"))
    sfk = "; ".join(f"{m} " + " / ".join(str(tot(ds, m)) for ds in DS) for m in ("SEIF", "Kookmin", "Fanchuan")) + " users"
    cap = (r"Requesting users still shown the sensitive category, pooled over five data seeds (TIGER + RQ-KMeans, retraining-matched split, deleted interactions removed from the requesting users' test histories). "
           r"Exact McNemar tests on the paired per-user outcomes, taking the range of $p$ over every per-user pairing consistent with the per-seed counts: "
           + "; ".join(sent) + "; no other comparison of these three methods with the model before deletion is significant at every consistent pairing"
           + (", and NAU, Forget only and Forget+Repair do not differ significantly from each other on any dataset ($p \\geq " + f"{pair_min:.2f}" + "$). " if pair_min >= 0.05 else ". ")
           + rf"SEIF, Kookmin and Fanchuan (not shown: {sfk}) keep at most {seif_fan:.0f}\% of retrained utility. Bold: fewest users among the unlearning methods.")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\caption{" + cap + "}", r"\label{tab:sensitive-user-counts}",
         r"\begin{tabular}{lccc}", r"\toprule",
         "Method & " + " & ".join(f"{d.capitalize()} (of {ntot[d]})" for d in DS) + r" \\", r"\midrule"]
    for m in meths:
        cells = []
        for ds in DS:
            t = tot(ds, m)
            cells.append(rf"\textbf{{{t}}}" if m not in ("Before deletion", "Retrained") and t == best[ds] else str(t))
        L.append(f"{m} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fn = f"{OUT}/sensitive_user_counts_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)
    for ds in DS:
        print(ds, {m: tot(ds, m) for m in list(allc[ds])}, "of", ntot[ds])
    for key, v in tests.items():
        print("  McNemar", key, f"[{v[0]:.3f}, {v[1]:.3f}]")
    return k, tests, ntot, {ds: {m: tot(ds, m) for m in allc[ds]} for ds in DS}


# ----------------------------------------------------------------------------------------- selection (validation only)
def valsens_table():
    rows = [r for r in csv.DictReader(open(R.OUT_CSV)) if R.nau_in_grid(r["method"], r["config"])]
    sel = {(r["dataset"], r["method"]): r["new_config"] for r in csv.DictReader(open(R.OUT_SEL))}
    vt = {}                                         # (ds, config) -> validation aggregates incl. Delta_UHF / Delta_UHR
    for r in rows:
        c = runs(r["dataset"], r["config"], "scrubval")
        vt[(r["dataset"], r["config"])] = {k: S.mean(x[k] for x in c.values()) for k in ("f", "df", "dr", "tp")}
        assert abs(vt[(r["dataset"], r["config"])]["f"] - float(r["val_UHF"])) < 1e-6, r
    nau_tp = {ds: vt[(ds, NAU_ARM)]["tp"] for ds in DS}
    fo = {ds: sel[(ds, "Forget only")] for ds in DS}
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{2.5pt}", r"\renewcommand{\arraystretch}{0.92}", r"\caption{",
         r"Hyperparameter selection for sensitive item deletion on the validation split (TIGER + RQ-KMeans, retraining-matched split, cleaned validation histories: the requesting users' deleted items are removed from their validation input histories). "
         r"For every candidate configuration, averaged over five data seeds, on the validation split: $\mathrm{UHF}@10$, the distances to the retrained model of the same seed $\Delta^{\mathrm{retr}}_{\mathrm{UHF}}$ and $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ (in \%), and $\tau_P$ (the test results of the selected configurations are in \Cref{sec:appendix-unwanted}). "
         r"The selected configuration (\checkmark) is chosen " + RULE_TXT + r". "
         + (r"NAU ($\lambda_f=0.1$, $\lambda_s=0$) selects $\lambda_n$ once on the Beauty validation split with the same rule and uses it on all datasets; " if R.NAU_GRID == "submitted" else r"NAU ($\lambda_f=0.1$) selects $\lambda_s$ and $\lambda_n$ once on the Beauty validation split with the same rule and uses them on all datasets; ")
         + r"its $\lambda_n=0$ configuration is Forget+Repair with $\lambda_f=0.1$. Finetune has a single configuration. "
         + rf"On Sports and Toys, NAU's checkmark marks the configuration selected on Beauty; there it has validation $\tau_P$ of {nau_tp['sports']:.1f}\% and {nau_tp['toys']:.1f}\%.", "}",
         r"\label{tab:appendix-valsens}", r"\resizebox{\textwidth}{!}{%", r"\begin{tabular}{ll" + "rrrrc" * 3 + "}", r"\toprule",
         r" & & \multicolumn{5}{c}{Beauty} & \multicolumn{5}{c}{Sports} & \multicolumn{5}{c}{Toys} \\",
         r"\cmidrule(lr){3-7} \cmidrule(lr){8-12} \cmidrule(lr){13-17}",
         r"Method & Configuration" + (r" & $\mathrm{UHF}@10$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHF}}\downarrow$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P\uparrow$ & Sel.") * 3 + r" \\",
         r"\midrule"]

    def lab(l):
        mm = re.match(r"lambda_s=([0-9.]+), lambda_n=(-?[0-9.]+)", l)
        if mm:
            return rf"$\lambda_s={mm.group(1)}$, $\lambda_n={mm.group(2)}$"
        mm = re.match(r"lambda_f=([0-9.]+)", l)
        if mm:
            return rf"$\lambda_f={mm.group(1)}$"
        return l.replace("_", r"\_")

    def key(l):
        mm = re.match(r"lambda_s=([0-9.]+), lambda_n=(-?[0-9.]+)", l)
        if mm:
            return (float(mm.group(1)), -float(mm.group(2)), "")
        mm = re.match(r"lambda_f=([0-9.]+)$", l)
        return (float(mm.group(1)), 0.0, "") if mm else (0.0, 0.0, l)
    for mi, m in enumerate(ORDER):
        labs = sorted({r["label"] for r in rows if r["method"] == m}, key=key)
        if mi and labs:
            L.append(r"\midrule")
        for j, l in enumerate(labs):
            cells = []
            for ds in DS:
                r = next((r for r in rows if r["dataset"] == ds and r["method"] == m and r["label"] == l), None)
                if r is None:
                    cells += ["--", "--", "--", "--", ""]
                    continue
                v = vt[(ds, r["config"])]
                cells += [f"{v['f']:.2f}", f"{v['df']:.2f}", f"{v['dr']:.2f}", f"{v['tp']:.1f}",
                          r"\checkmark" if r["config"] == sel.get((ds, m)) else ""]
            L.append(f"{m if j == 0 else ''} & {lab(l)} & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    fn = f"{OUT}/appendix_valsens_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)


# ----------------------------------------------------------------------------------------- anchor
def anchor_table():
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Optional retain-side anchor for sensitive item deletion (TIGER + RQ-KMeans, five data seeds, retraining-matched split, cleaned evaluation histories). The anchor holds the neighborhood's probability mass under non-requesting users' histories at the original model's level, with weight $\lambda_a$; everything else is NAU's setting for this scenario ($\lambda_f=0.1$, " + NAU_TXT + r"). UHF@10: requesting users still shown the category; UHR@10: all other users; $\Delta^{\mathrm{retr}}_{\mathrm{UHR}} = |\mathrm{UHR}@10 - \mathrm{UHR}@10_{\mathrm{retr}}|$, per seed against that seed's retrained model, then averaged; all in \%. Closer: seeds on which the anchored model's $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ is below NAU's.}",
         r"\label{tab:appendix-sensitive-anchor}", r"\begin{tabular}{llccccc}", r"\toprule",
         r"Dataset & Method & UHF@10 & UHR@10 & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P$ & Closer \\", r"\midrule"]
    missing = []
    for di, ds in enumerate(DS):
        base, retr = refcells(ds, "base"), refcells(ds, "retrain")
        nau = runs(ds, NAU_ARM)
        fmt = lambda c: (S.mean(x["f"] for x in c.values()), S.mean(x["r"] for x in c.values()),
                         S.mean(x["dr"] for x in c.values()), S.mean(x["tp"] for x in c.values()))
        b = fmt(base)
        L.append(f"{ds.capitalize()} & Before deletion & {b[0]:.1f} & {b[1]:.2f} & {b[2]:.3f} & {b[3]:.1f} & \\\\")
        r_ = fmt(retr)
        L.append(f" & Retrained & {r_[0]:.1f} & {r_[1]:.2f} & -- & 100.0 & \\\\")
        n = fmt(nau)
        L.append(f" & NAU & {n[0]:.1f} & {n[1]:.2f} & {n[2]:.3f} & {n[3]:.1f} & \\\\")
        NUMS.append(dict(table="anchor", dataset=ds, method="NAU", config=NAU_ARM, f=n[0], r=n[1], dr=n[2], tp=n[3]))
        for la, t in (("0.1", "anchor0p1"), ("1", "anchor1p0")):
            c = runs(ds, f"{NAU_ARM}_{t}", need=0)
            if len(c) < 5:
                missing.append(f"{ds} anchor {la}: {len(c)}/5")
                PENDING.append(f"{ds} anchor {la}: {len(c)}/5")
                L.append(f" & NAU + anchor, $\\lambda_a={la}$ & \\multicolumn{{5}}{{c}}{{pending ({len(c)}/5 seeds)}} \\\\")
                continue
            a = fmt(c)
            closer = sum(c[sd]["dr"] < nau[sd]["dr"] for sd in SEEDS)
            NUMS.append(dict(table="anchor", dataset=ds, method=f"NAU+anchor {la}", config=t, f=a[0], r=a[1], dr=a[2], tp=a[3]))
            L.append(f" & NAU + anchor, $\\lambda_a={la}$ & {a[0]:.1f} & {a[1]:.2f} & {a[2]:.3f} & {a[3]:.1f} & {closer}/5 \\\\")
        if di < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fn = f"{OUT}/appendix_sensitive_anchor_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn, "MISSING:" if missing else "", missing)
    return missing


# ----------------------------------------------------------------------------------------- brand takedown
TD = "logs/eval/sensitive_scrub/scrub_takedown"


def takedown_table():
    plans = ["logs/eval/sensitive_scrub/plan_takedown_scrub.tsv", "logs/eval/sensitive_scrub/plan_takedown_scrub_followup.tsv",
             "logs/eval/sensitive_scrub/plan_takedown_scrub_rank.tsv", "logs/eval/sensitive_scrub/plan_takedown_scrub_nau0.tsv"]
    rows = [r for p in plans if os.path.isfile(p) for r in csv.DictReader(open(p), delimiter="\t")]
    ref = {}
    arm = {}
    for r in rows:
        m = R.metrics_csv(r["out_dir"])
        if not m:
            continue
        sd = int(r["seed"])
        if r["role"] in ("original", "retrained"):
            ref[(r["role"], sd)] = m
        else:
            a = re.sub(r"_rmsplit_canonck(_seed\d+)?$", "", r["arm"])
            arm.setdefault(a, {})[sd] = m
    def agg_td(ms):
        c = {sd: cell(m, ref[("retrained", sd)]) for sd, m in ms.items()}
        if len(c) < 5:
            return None
        assert all(c.values()), ms.keys()
        return {k: S.mean(x[k] for x in c.values()) for k in ("f", "r", "tp")}
    pk = picks("beauty")

    def td(meth):
        """takedown arm of a Beauty sensitive-deletion pick (the unified family carries _abl_ tags there)."""
        a = pk[meth]
        mm = re.match(r"unified_hp2_(fo|fr|ft)_lf([0-9.]+)_ls0\.0_ln0\.0", a)
        if mm:
            return f"unified_abl_lf{mm.group(2)}_ls0.0_ln0.0" + ("_lr0.0" if mm.group(1) == "fo" else "")
        return a
    spec = [("Retrained", None), ("Before deletion", None), None,
            ("SCIF", td("SCIF")), ("Finetune", td("Finetune")), ("Kookmin", td("Kookmin")),
            ("SEIF", td("SEIF")), ("Forget only", td("Forget only")), ("Fanchuan", td("Fanchuan")),
            ("Forget+Repair", td("Forget+Repair"))]
    if td("Forget+Repair") != "unified_abl_lf0.1_ls0.0_ln0.0":
        spec.append(("\\quad $\\lambda_f=0.1$ (the setting NAU builds on)", "unified_abl_lf0.1_ls0.0_ln0.0"))
    spec += [None, ("NAU \\textit{(ours)}", "unified_frbase_lf0.1_ls0.01_ln0.01"),
             ("\\quad sensitive-deletion setting (" + NAU_TXT + ")", NAU_ARM)]
    vals = []
    for s in spec:
        if s is None:
            vals.append(None); continue
        nm, a = s
        if a is None:
            v = agg_td({sd: ref[("original" if nm.startswith("Before") else "retrained", sd)] for sd in SEEDS})
        else:
            v = agg_td(arm.get(a, {}))
            if v is None:
                PENDING.append(f"takedown {a}: {len(arm.get(a, {}))}/5")
                vals.append((nm, a, None)); continue
        vals.append((nm, a, v))
        NUMS.append(dict(table="takedown", dataset="beauty", method=nm, config=a or "", f=v["f"], r=v["r"], tp=v["tp"]))
    meth = [i for i, x in enumerate(vals) if x and x[1] and x[2]]
    elig = [vals[i][2]["tp"] >= 100 * FLOOR for i in meth]
    mk = {k: h2h.rank_marks([vals[i][2][k] for i in meth], "low", elig) for k in ("f", "r")}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Brand takedown on Beauty (TIGER + RQ-KMeans, five data seeds, retraining-matched split): every interaction with one brand (95 items, 933 users, 0.9\% of interactions) is deleted for every user. The retrained model never recommends the brand, so any remaining exposure is distance to retraining. Buyers: UHF@10 among users who interacted with the brand; others: UHR@10 among all other users; both in \%, on cleaned test histories (the deleted interactions removed from the buyers' test input histories). NAU uses its spam weights ($\lambda_f=0.1$, $\lambda_s=0.01$, $\lambda_n=0.01$) on the Forget+Repair setting with $\lambda_f=0.1$; the baselines use their sensitive item deletion settings selected on Beauty. Bold and underline: best and second best among methods with $\tau_P \geq 95\%$ (unrounded values).}",
         r"\label{tab:appendix-brand-takedown}", r"\begin{tabular}{lccc}", r"\toprule",
         r"Method & Exposure, buyers $\downarrow$ & Exposure, others $\downarrow$ & $\tau_P$ \\", r"\midrule"]
    for i, x in enumerate(vals):
        if x is None:
            L.append(r"\midrule"); continue
        nm, a, v = x
        if v is None:
            L.append(f"{nm} & \\multicolumn{{3}}{{c}}{{pending}} \\\\"); continue
        cf, cr = f"{v['f']:.2f}", f"{v['r']:.2f}"
        if a:
            j = meth.index(i)
            cf, cr = h2h.tex_mark(cf, mk["f"].get(j)), h2h.tex_mark(cr, mk["r"].get(j))
        L.append(f"{nm} & {cf} & {cr} & {v['tp']:.1f} \\\\")
        if nm == "Before deletion":
            L.append(r"\midrule")
    L = [l for k, l in enumerate(L) if not (l == r"\midrule" and L[k - 1] == r"\midrule")]
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fn = f"{OUT}/appendix_brand_takedown_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)
    for x in vals:
        if x and x[2]:
            print(f"   {x[0]:60s} {x[2]['f']:6.2f} {x[2]['r']:5.2f} {x[2]['tp']:6.1f}")


# ----------------------------------------------------------------------------------------- optional: old vs new split
def old_row(fn, name):
    """mean UHF, UHR, dUHR, tau_P of one method row of an old cleaned-test table (paper/tables)."""
    for line in open(fn):
        if line.startswith(name):
            vals = [float(x) for x in re.findall(r"(?<![\\a-z])(\d+\.\d+)\\,", line)]
            return vals  # UHF, UHR, dUHR, NDCG, tau_P
    raise KeyError((fn, name))


def robustness_table():
    old_fn = {"beauty": "paper/tables/paper_sensitive_beauty_scrub_ledger.tex",
              "sports": "paper/tables/sensitive_sports_scrub_ledger.tex", "toys": "paper/tables/sensitive_toys_scrub_ledger.tex"}
    M = [("NAU", "NAU \\textit{(Ours)}", nau_arm(0.0, -1), r"$\lambda_s=0$, $\lambda_n=-1$"),
         ("Forget+Repair", "Forget+Repair", nau_arm(0.0, 0), r"$\lambda_f=0.1$"),
         ("Forget only", "Forget only", "unified_hp2_fo_lf0.1_ls0.0_ln0.0_lr0.0", r"$\lambda_f=0.1$")]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}",
         r"\caption{Robustness to the forget/retain split (TIGER + RQ-KMeans, five data seeds, cleaned test histories), at the configurations of the submitted version. Old split: the requesting users' training sequences form the forget set and are excluded from the retain set; Forget+Repair and Forget only use the baselines' batch size of 256, NAU 32. Retraining-matched split: the forget set holds only the training examples whose next item is a deleted interaction, the retain set is the retrained model's training data (requesting users kept, with the deleted items removed), and all three use a batch size of 32. All values in \%.}",
         r"\label{tab:appendix-sensitive-split-robustness}", r"\begin{tabular}{lllcccc}", r"\toprule",
         r"Dataset & Method & Split & UHF@10$\downarrow$ & UHR@10 & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P$ \\", r"\midrule"]
    for di, ds in enumerate(DS):
        for j, (m, oldname, arm, cfg) in enumerate(M):
            o = old_row(old_fn[ds], oldname)
            n = agg(runs(ds, arm))
            nm = f"{m} ({cfg})"
            L.append(f"{ds.capitalize() if j == 0 else ''} & {nm} & old & {o[0]:.1f} & {o[1]:.2f} & {o[2]:.2f} & {o[4]:.1f} \\\\")
            L.append(f" & & matched & {n['f'][0]:.1f} & {n['r'][0]:.2f} & {n['dr'][0]:.2f} & {n['tp'][0]:.1f} \\\\")
        if di < 2:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fn = f"{OUT}/appendix_sensitive_split_robustness_rmsplit.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn, "(optional)")


def main():
    only = sys.argv[1:]
    want = lambda x: not only or x in only
    if want("main"):
        for ds in DS:
            for main_tab in ((True,) if ds == "beauty" else (False,)):
                rows, ro, rr = method_table(ds, main_tab)
                for r in rows:
                    a = r["a"]
                    print(f"  {ds:7s} {r['name']:14s} UHF {a['f'][0]:6.2f} dUHF {a['df'][0]:6.2f} UHR {a['r'][0]:5.2f} "
                          f"dUHR {a['dr'][0]:5.2f} tauP {a['tp'][0]:6.2f} E {a['e']}{'' if r['ok'] else '  dagger'}")
    if want("lambda_n"):
        lambda_n_table()
    if want("counts"):
        counts_table()
    if want("valsens"):
        valsens_table()
    if want("anchor"):
        anchor_table()
    if want("takedown"):
        takedown_table()
    if want("robust"):
        robustness_table()
    if PENDING:
        print("PENDING:", PENDING)
    if NUMS and not only:
        keys = sorted({k for r in NUMS for k in r}, key=lambda k: ["table", "dataset", "method", "config"].index(k) if k in ("table", "dataset", "method", "config") else 9)
        with open("paper/revision/rmsplit_tables_numbers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(NUMS)


if __name__ == "__main__":
    main()
