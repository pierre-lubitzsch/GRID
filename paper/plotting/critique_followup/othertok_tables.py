"""Sensitive item deletion tables for the OTHER SID tokenizers (RQ-VAE, LETTER, tracerv3), 2026-11-23.

Not for the current paper. Campaign: paper/revision/othertok_sensitive_campaign.md. Retraining-matched split,
cleaned test histories; configurations PINNED to paper/revision/rmsplit_selected.csv (the TIGER + RQ-KMeans picks under
the submitted rule, no selection here); Beauty TRACER uses the Sports/Toys pick tracer_lf1.0_lc1.0_ep4. NAU's lambda_n
ablation at lambda_s=0 (lambda_n 0, -1, -10) in a second table per tokenizer.

Metrics as in rmsplit_tables.py (same helpers from rmsplit_selection.py): per seed UHF@10 (test/SHF@10), UHR@10
(test/SHR@10), Delta^retr_UHF = |UHF - UHF_retr|, Delta^retr_UHR = |UHR - UHR_retr|, tau_P = NDCG@10 / retrained
NDCG@10 of the same seed and view; mean (and std) over the five seeds. References: the tokenizer's own model before
deletion (clean last.ckpt) and retrained model, both on the cleaned test view.

    python paper/plotting/critique_followup/othertok_tables.py
      -> paper/revision/tables/othertok/sensitive_othertok_<track>.tex, ablation_lambda_n_othertok_<track>.tex,
         paper/revision/othertok_tables_numbers.csv, paper/revision/othertok_tables_summary.md
"""
import csv
import importlib.util
import json
import os
import statistics as S

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("rsel", os.path.join(HERE, "rmsplit_selection.py"))
R = importlib.util.module_from_spec(_s)
_s.loader.exec_module(R)

OUT = "paper/revision/tables/othertok"
SEEDS, CAT = R.SEEDS, R.CAT
# track -> (tag prefix, reference token, model, quantizer, datasets, display name)
TRACKS = {
    "rqvae": ("rqvae_", "rqvae", "tiger", "rqvae", ("beauty", "sports", "toys"), "TIGER + RQ-VAE"),
    "letter": ("letter_letter_", "letter_letter", "letter", "letter", ("beauty", "sports", "toys"), "LETTER"),
    "tracerv3": ("tracerv3_", "tracerv3", "tiger", "tracerv3", ("beauty",), "TIGER + TRACER-compatible RQ-KMeans"),
}
ORDER = R.ORDER
TRACER_BEAUTY = "tracer_lf1.0_lc1.0_ep4"
NAU_LN = {0.0: "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0", -1.0: "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0",
          -10.0: "unified_hp2_sepgen_lf0.1_ls0.0_lnneg10.0"}
VIEW = "scrub"


TUNED = os.environ.get("TUNED", "0") == "1"   # 1: per-tokenizer selection on cleaned validation
TUNED_PICKS = {}


def grid_methods(track, ds):
    """{method: [configs]} of the per-tokenizer grid (pinned campaign + plan_othertok_grid)."""
    import re
    out = {}
    for p in (f"logs/rmsplit_sens/plan_othertok_{track}.tsv", f"logs/rmsplit_sens/plan_othertok_grid_{track}.tsv"):
        if not os.path.exists(p):
            continue
        for line in open(p):
            f = line.split("\t")
            if f[1] != ds:
                continue
            arm = re.match(rf"\w*?sens_{ds}_\w+?rm_(.*)_rmsplit_canonck(_seed\d+)?$", f[5]).group(1)
            for m, cfg in R.methods_of(arm):
                if R.nau_in_grid(m, cfg):
                    out.setdefault(m, set()).add(cfg)
    return out


def select(track):
    """lowest cleaned-validation UHF@10 among tau_P >= 95% (else lowest overall); ties: higher tau_P.
    NAU: selected on the tokenizer's Beauty and reused (as for TIGER + RQ-KMeans)."""
    picks_, val = {}, {}
    for ds in TRACKS[track][4]:
        for m, cfgs in grid_methods(track, ds).items():
            score = {}
            for c in sorted(cfgs):
                cells, miss = runs(track, ds, c, "scrubval")
                if len(cells) == 5:
                    score[c] = {k: S.mean(x[k] for x in cells.values()) for k in ("f", "tp")}
                else:
                    MISSING.extend(f"val {x}" for x in miss)
            if len(score) < len(cfgs):
                continue
            pool = [c for c in score if score[c]["tp"] >= 0.95] or list(score)
            best = min(pool, key=lambda c: (round(score[c]["f"], 9), -score[c]["tp"]))
            picks_[(ds, m)] = best
            val[(ds, m)] = (score, bool([c for c in score if score[c]["tp"] >= 0.95]))
    nb = picks_.get(("beauty", "NAU"))
    if nb:
        for ds in TRACKS[track][4]:
            picks_[(ds, "NAU")] = nb
    TUNED_PICKS[track] = (picks_, val)
    return picks_


def picks(ds, track=None):
    if TUNED and track:
        p = TUNED_PICKS[track][0] if track in TUNED_PICKS else select(track)
        return {m: p[(ds, m)] for m in ORDER if (ds, m) in p}
    return picks_pinned(ds)


def picks_pinned(ds):
    sel = {r["method"]: r["new_config"] for r in csv.DictReader(open(R.OUT_SEL)) if r["dataset"] == ds}
    if ds == "beauty" and "TRACER" not in sel:
        sel["TRACER"] = TRACER_BEAUTY
    return {m: sel[m] for m in ORDER if m in sel}


def tag(prefix, ds, arm, sd):
    return f"{prefix}sens_{ds}_{CAT[ds]}rm_{arm}_rmsplit_canonck" + ("" if sd == 2 else f"_seed{sd}")


def ref(track, ds, role, sd, view=None):
    view = view or VIEW
    prefix, tok, model, quant, _, _ = TRACKS[track]
    d = f"{R.ROOT}/{view}/reference/v1/{ds}_{role}_{tok}" + ("" if sd == 2 else f"_seed{sd}")
    c = json.load(open(f"{d}/coordinates.json"))
    assert c["model"] == model and c["quantizer"] == quant, (d, c["model"], c["quantizer"])
    assert c["dataset"] == ds and int(c["seed"]) == sd and c["view"] == view, d
    m = R.metrics_csv(d)
    assert m, f"no metrics in {d}"
    return m


def run_dir(tg, view):
    """latest run dir of a tag that has this view's cleaned eval (works after its checkpoint was deleted)."""
    import glob
    import re
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tg}")
         if glob.glob(f"{R.ROOT}/{view}/runs/{os.path.basename(d)}/eval/csv/version_*/metrics.csv")]
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1))) if c else None


def runs(track, ds, arm, view=None):
    view = view or VIEW
    prefix = TRACKS[track][0]
    out, missing = {}, []
    for sd in SEEDS:
        d = run_dir(tag(prefix, ds, arm, sd), view)
        m = R.metrics_csv(f"{R.ROOT}/{view}/runs/{os.path.basename(d)}/eval") if d else None
        c = R.cell(m, ref(track, ds, "retrain", sd, view))
        if c:
            out[sd] = c
        else:
            missing.append(tag(prefix, ds, arm, sd))
    return out, missing


def agg(cells):
    a = {}
    for k in ("f", "df", "r", "dr", "tp"):
        v = [100 * c[k] for c in cells.values()]
        a[k] = (S.mean(v), S.stdev(v) if len(v) > 1 else 0.0)
    return a


def refagg(track, ds, role):
    return agg({sd: R.cell(ref(track, ds, role, sd), ref(track, ds, "retrain", sd)) for sd in SEEDS})


NUMS, MISSING = [], []


def fmt(a, k, p=2):
    return f"{a[k][0]:.{p}f}\\,{{\\scriptsize$\\pm${a[k][1]:.{p}f}}}"


def method_table(track):
    prefix, tok, model, quant, dss, name = TRACKS[track]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Sensitive item deletion with " + name + r" identifiers (five data seeds, retraining-matched split, "
         r"cleaned test histories). " + (r"Configurations are selected for this tokenizer on its cleaned validation histories "
         r"(lowest $\mathrm{UHF}@10$ among configurations with $\tau_P\geq95\%$, else lowest overall); NAU's $\lambda_n$ "
         r"($\lambda_f=0.1$, $\lambda_s=0$) is selected on Beauty and reused. " if TUNED else
         r"Configurations are those selected for TIGER + RQ-KMeans on the cleaned validation "
         r"histories (no selection for this tokenizer); NAU: $\lambda_f=0.1$, $\lambda_s=0$, $\lambda_n=-10$. ")
         + r"UHF@10 / UHR@10: share of requesting / other users with a sensitive item in their top 10; "
         r"$\Delta^{\mathrm{retr}}$: per-seed distance to the retrained model of the same seed; $\tau_P$: NDCG@10 relative "
         r"to that retrained model; all in \%, mean $\pm$ std over seeds. $^{\dagger}$: $\tau_P<95\%$.}",
         rf"\label{{tab:sensitive-othertok-{track}}}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llccccc}", r"\toprule",
         r"Dataset & Method & $\mathrm{UHF}@10$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHF}}\downarrow$ & $\mathrm{UHR}@10$ & "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P\uparrow$ \\", r"\midrule"]
    for di, ds in enumerate(dss):
        first = True
        for nm, role in (("Before deletion", "base"), ("Retrained", "retrain")):
            a = refagg(track, ds, role)
            L.append(f"{ds.capitalize() if first else ''} & \\textit{{{nm}}} & {fmt(a,'f')} & {fmt(a,'df')} & {fmt(a,'r')} & {fmt(a,'dr')} & {fmt(a,'tp',1)} \\\\")
            NUMS.append(dict(track=track, dataset=ds, method=nm, config="", n=5, **{k: a[k][0] for k in a}))
            first = False
        L.append(r"\cmidrule(lr){2-7}")
        for m, arm in picks(ds, track).items():
            c, miss = runs(track, ds, arm)
            MISSING.extend(miss)
            if len(c) < 5:
                L.append(f" & {m} & \\multicolumn{{5}}{{c}}{{{len(c)}/5 seeds}} \\\\")
                continue
            a = agg(c)
            dag = "" if a["tp"][0] >= 95 else r"$^{\dagger}$"
            nm = r"NAU \textit{(ours)}" if m == "NAU" else m
            L.append(f" & {nm}{dag} & {fmt(a,'f')} & {fmt(a,'df')} & {fmt(a,'r')} & {fmt(a,'dr')} & {fmt(a,'tp',1)} \\\\")
            NUMS.append(dict(track=track, dataset=ds, method=m, config=arm, n=len(c), **{k: a[k][0] for k in a}))
        if di < len(dss) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    fn = f"{OUT}/sensitive_othertok_{track}{'_tuned' if TUNED else ''}.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)


def lambda_n_table(track):
    prefix, tok, model, quant, dss, name = TRACKS[track]
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}",
         r"\caption{Effect of the neighborhood term in sensitive item deletion with " + name + r" identifiers "
         r"(five data seeds, retraining-matched split, cleaned test histories; $\lambda_f=0.1$, $\lambda_s=0$). Each cell "
         r"gives the mean over seeds for $\lambda_n = 0 \rightarrow -1 \rightarrow -10$; $\lambda_n=0$ is Forget+Repair "
         r"with $\lambda_f=0.1$. All in \%.}",
         rf"\label{{tab:ablation-lambda-n-othertok-{track}}}",
         r"\begin{tabular}{lcccc}", r"\toprule",
         r"Dataset & $\mathrm{UHF}@10$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P\uparrow$ & Retrained $\mathrm{UHF}$ \\",
         r"\midrule"]
    for ds in dss:
        cs = []
        for ln, arm in NAU_LN.items():
            c, miss = runs(track, ds, arm)
            MISSING.extend(miss)
            cs.append(agg(c) if len(c) == 5 else None)
            if len(c) == 5:
                NUMS.append(dict(track=track, dataset=ds, method=f"NAU ln={ln:g}", config=arm, n=5,
                                 **{k: cs[-1][k][0] for k in cs[-1]}))
        rr = refagg(track, ds, "retrain")

        def chain(k, p):
            return " $\\rightarrow$ ".join("--" if a is None else f"{a[k][0]:.{p}f}" for a in cs)
        L.append(f"{ds.capitalize()} & {chain('f', 1)} & {chain('dr', 2)} & {chain('tp', 1)} & {rr['f'][0]:.1f} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    fn = f"{OUT}/ablation_lambda_n_othertok_{track}.tex"
    open(fn, "w").write("\n".join(L) + "\n")
    print("wrote", fn)


def main():
    os.makedirs(OUT, exist_ok=True)
    for t in TRACKS:
        method_table(t)
        lambda_n_table(t)
    keys = ["track", "dataset", "method", "config", "n", "f", "df", "r", "dr", "tp"]
    sfx = "_tuned" if TUNED else ""
    with open(f"paper/revision/othertok_tables_numbers{sfx}.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=keys); w.writeheader(); w.writerows(NUMS)
    # short markdown summary: UHF / Delta_UHR / tau_P per method
    L = ["# Sensitive item deletion, other tokenizers (cleaned test, five seeds, %)", "",
         "Generated by `paper/plotting/critique_followup/othertok_tables.py`. " + (
             "Configurations selected per tokenizer on cleaned validation (lowest UHF@10 among tau_P >= 95%, else lowest "
             "overall; NAU's lambda_n selected on the tokenizer's Beauty and reused). " if TUNED else
             "Configurations pinned to the TIGER + RQ-KMeans picks (`paper/revision/rmsplit_selected.csv`). ")
         + "Cells: UHF@10 / Delta^retr_UHR / tau_P. Not in the paper.", ""]
    for t, (_, _, _, _, dss, name) in TRACKS.items():
        L += [f"## {name}", "", "| Method | " + " | ".join(d.capitalize() for d in dss) + " |",
              "|---|" + "---|" * len(dss)]
        meths = [m for m in ["Before deletion", "Retrained"] + ORDER + ["NAU ln=0", "NAU ln=-1"]
                 if any(r["track"] == t and r["method"] == m for r in NUMS)]
        for m in meths:
            cells = []
            for d in dss:
                r = next((r for r in NUMS if r["track"] == t and r["dataset"] == d and r["method"] == m), None)
                cells.append("n/a" if r is None else f"{r['f']:.1f} / {r['dr']:.2f} / {r['tp']:.1f}")
            L.append(f"| {m} | " + " | ".join(cells) + " |")
        L.append("")
    if MISSING:
        L += ["## Missing", ""] + [f"- {m}" for m in sorted(set(MISSING))]
    if TUNED:
        L += ["## Selected configurations (cleaned validation)", "", "| Tokenizer | Dataset | Method | Config | floor met |", "|---|---|---|---|---|"]
        for t, (pk, val) in TUNED_PICKS.items():
            for (ds, m), c in sorted(pk.items()):
                L.append(f"| {t} | {ds} | {m} | {c} | {'yes' if val.get((ds, m), ({}, False))[1] else 'no'} |")
    open(f"paper/revision/othertok_tables_summary{sfx}.md", "w").write("\n".join(L) + "\n")
    print(f"wrote paper/revision/othertok_tables_numbers{sfx}.csv, paper/revision/othertok_tables_summary{sfx}.md;",
          f"missing: {len(set(MISSING))}")


if __name__ == "__main__":
    main()
