"""Cross-dataset effect of the neighborhood term in unwanted-item removal.

Paired per data seed: NAU at lambda_n=-1 vs the same objective at lambda_n=0
(lambda_f=0.1, lambda_s=0, generative separation score), TIGER with RQ-KMeans and
LETTER identifiers. UHF/UHR = SHF/SHR in the code.

    PYTHONPATH=. python paper/plotting/make_unwanted_lambda_n_table.py

--ledger paper/plotting/canonical_runs.tsv (RQ-KMeans rows only): lambda_n=-1 = the canonical
`_mia` execution (as Tables 3/22/23); lambda_n=0 and lambda_n=-10 = the `_valsens` replays (the
same executions Table 8 shows; no checkpoint, so NOT canonical-with-checkpoint). LETTER rows keep
their recorded runs (single source, no checkpoint). The non-canonical cells are printed. Writes
paper/v24_edits/tables/ablation_unified_sensitive_lambda_n_all_ledger.tex.

--view {normal,scrub} (with --ledger): one checkpointed execution per cell for every row:
RQ-KMeans lambda_n=-1 = `_mia`, lambda_n=0 / -10 = the `unwanted_canonck` re-runs (the same runs
Table 8 reads with --view); LETTER lambda_n=0 / -1 / -10 = the `unwanted_canonck` re-runs. scrub
reads the cleaned-test-history evaluations and the retrained reference on the same view
(unwanted_ledger.set_view). Writes ..._lambda_n_all_{normal,scrub}_ledger.tex.
"""
import argparse, importlib.util, os, statistics as S, sys
sys.path.insert(0, "scripts")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
ap = argparse.ArgumentParser()
ap.add_argument("--ledger", default=None, metavar="TSV")
ap.add_argument("--view", choices=("normal", "scrub"), default=None,
                help="with --ledger: all rows from checkpointed executions on this test view")
ARGS = ap.parse_args()
spec = importlib.util.spec_from_file_location("es", "scripts/extract_sensitive.py")
es = importlib.util.module_from_spec(spec); spec.loader.exec_module(es)
if ARGS.view:
    assert ARGS.ledger, "--view needs --ledger"
    import unwanted_ledger as ul
    ul.set_view(ARGS.view, es)

A0, A1 = "unified_hp2_sepgen_lf0.1_ls0.0_ln0.0", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0"
# lambda_n=-10: RQ-KMeans from the validation-selection replays (their test eval), LETTER recorded
A2 = ("unified_hp2_sepgen_lf0.1_ls0.0_lnneg10.0", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg10.0_valsens")
TOK = [("tiger", "rkmeans", "RQ-KMeans"), ("letter", "letter", "LETTER")]


def per_seed(runs, k):
    d = {}
    for r in runs:
        if r[k] is not None:
            d.setdefault(r["seed"], []).append(100 * r[k])
    return {s: S.mean(v) for s, v in d.items()}


NONCANON = []
L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\caption{",
     r"Effect of the neighborhood term in unwanted item removal across datasets and tokenizers: NAU with $\lambda_n=-1$ (and $\lambda_n=-10$) against the same objective without the term ($\lambda_n=0$; $\lambda_f=0.1$, $\lambda_s=0$), paired over five data seeds.",
     r"Each cell gives $\lambda_n=0 \rightarrow -1 \rightarrow -10$ (mean over seeds) and, in parentheses, on how many of the five seeds the value decreases ($\mathrm{UHF}$, $\mathrm{UHR}$) or increases ($\tau_P$) relative to $\lambda_n=0$, for $\lambda_n=-1$ and $\lambda_n=-10$.",
     r"Retrained $\mathrm{UHF}@10$ and $\mathrm{UHR}@10$ are given for reference."
     + (r" All models are evaluated on cleaned test histories (the requesting users' deleted items removed from their test input histories)." if ARGS.view == "scrub" else ""), "}",
     r"\label{tab:ablation-unified-sensitive-lambda-n-all}", r"\resizebox{\textwidth}{!}{%",
     r"\begin{tabular}{llcccc}", r"\toprule",
     r"Dataset & \acp{SID} & $\mathrm{UHF}@10{\scriptstyle\,(\%)}{\downarrow}$ & $\mathrm{UHR}@10{\scriptstyle\,(\%)}$ & $\tau_P{\scriptstyle\,(\%)}{\uparrow}$ & Retrained $\mathrm{UHF}$ / $\mathrm{UHR}$ \\",
     r"\midrule"]
for ds in ("beauty", "sports", "toys"):
    for model, q, tn in TOK:
        if ARGS.view:
            track = f"{model}+{q}"
            cc = ul.canonck_runs(ARGS.ledger, ds, track, "eval")
            r1 = (ul.mia_runs(ARGS.ledger, ds, "eval").get(("unified", A1), []) if q == "rkmeans" else cc.get(A1, []))
            r0, r2 = cc.get(A0, []), cc.get(A2[0], [])
            ok = all(len({r["seed"] for r in x}) == 5 for x in (r0, r1, r2))
            refs = {sd: ul.refs_for(ds, sd, model, q) for sd in sorted({r["seed"] for r in r0 + r1 + r2})}
        else:
            cells, refs, _ = es.collect(ds, model, q, es.CATEGORY[ds], include_valsens=True)
            a2 = next((a for a in A2 if len({r["seed"] for r in cells.get(a, [])}) == 5), None)
        if ARGS.view:
            pass
        elif ARGS.ledger and q == "rkmeans":
            import unwanted_ledger as ul
            mia = ul.mia_runs(ARGS.ledger, ds, "eval").get(("unified", A1), [])
            r0, r1, r2 = cells.get(A0 + "_valsens", []), mia, cells.get(A2[1], [])
            NONCANON.append(f"{ds} {tn} lambda_n=0: `_valsens` replay ({A0}_valsens), no checkpoint")
            NONCANON.append(f"{ds} {tn} lambda_n=-10: `_valsens` replay ({A2[1]}), no checkpoint")
            ok = all(len({r["seed"] for r in x}) == 5 for x in (r0, r1, r2))
        else:
            r0, r1, r2 = cells.get(A0, []), cells.get(A1, []), cells.get(a2, []) if a2 else []
            ok = A0 in cells and A1 in cells and a2 is not None
            if ARGS.ledger:
                # LETTER has no complete lambda_n=-10 arm: keep its 0 -> -1 row, -10 printed as "--"
                ok = A0 in cells and A1 in cells
                NONCANON.append(f"{ds} {tn} lambda_n=0, -1: recorded runs ({A0}, {A1}), no checkpoint; "
                                f"lambda_n=-10: {'recorded ' + a2 if a2 else 'NO complete 5-seed run (shown as --)'}")
        if not ok:
            print(f"  {ds} {tn}: missing an arm" + (f" (seeds: {[len(x) for x in (r0, r1, r2)]})" if ARGS.view else f" (lambda_n=-10: {a2})")); continue
        out = []
        for k, better in (("sensF", "down"), ("sensR", "down"), ("tau_P", "up")):
            p0, p1, p2 = per_seed(r0, k), per_seed(r1, k), per_seed(r2, k)
            if not p2:   # only reachable with --ledger (LETTER without lambda_n=-10)
                sd = sorted(set(p0) & set(p1))
                cnt = lambda p: sum((p[s] < p0[s]) if better == "down" else (p[s] > p0[s]) for s in sd)  # noqa: E731
                m = lambda p: S.mean(p[s] for s in sd)  # noqa: E731
                out.append(f"{m(p0):.1f} $\\rightarrow$ {m(p1):.1f} $\\rightarrow$ -- ({cnt(p1)}, --/{len(sd)})")
                continue
            sd = sorted(set(p0) & set(p1) & set(p2))
            cnt = lambda p: sum((p[s] < p0[s]) if better == "down" else (p[s] > p0[s]) for s in sd)
            m = lambda p: S.mean(p[s] for s in sd)
            out.append(f"{m(p0):.1f} $\\rightarrow$ {m(p1):.1f} $\\rightarrow$ {m(p2):.1f} ({cnt(p1)}, {cnt(p2)}/{len(sd)})")
        retr = es.mean_reference_metrics(refs, 1)
        out.append(f"{100 * retr['test/SHF@10']:.1f} / {100 * retr['test/SHR@10']:.2f}")
        L.append(f"{ds.capitalize()} & {tn} & " + " & ".join(out) + r" \\")
L += [r"\bottomrule", r"\end{tabular}}", r"\vspace*{-10pt}", r"\end{table}"]
OUT = "paper/v24_edits/tables/ablation_unified_sensitive_lambda_n_all" + (
    ul.VIEW_SUFFIX[ARGS.view] if ARGS.view else ("_ledger" if ARGS.ledger else "")) + ".tex"
open(OUT, "w").write("\n".join(L) + "\n")
print("\n".join(L))
print("wrote", OUT)
if ARGS.ledger and not ARGS.view:
    print("NOT canonical-with-checkpoint (reported from their single existing source):")
    for x in NONCANON:
        print("  " + x)
