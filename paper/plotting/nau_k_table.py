"""Appendix table: sensitivity of NAU to the neighborhood size k (Beauty, validation).

NAU at lambda_f=0.1, lambda_s=0.01, lambda_n=0.01 with neighborhood_count k in
{2, 4, 8, 16, 32, 64}; 3 targets x 5 data seeds each (k=8 is the paper's setting, from the
validation grid; the others from logs/valsel_nau_k_plan.tsv, k >= 16 with
unlearning.coherence_checkpoint=true). Whole-set Delta^retr_SH and tau_P on validation and
test, pooled like Table 1.

    PYTHONPATH=. python paper/plotting/nau_k_table.py [--out paper/revision/tables/appendix_nau_k.tex]
"""
import argparse
import glob
import importlib.util
import json
import os
import re
import statistics as S
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("vt", os.path.join(HERE, "valsel_table.py"))
vt = importlib.util.module_from_spec(spec); spec.loader.exec_module(vt)
KS = (2, 4, 8, 16, 32, 64)


def collect():
    R = vt.refs()
    test_ref = {os.path.basename(f)[:-5]: json.load(open(f)) for f in glob.glob("paper/data/spam_references/*.json")}
    rows = defaultdict(lambda: defaultdict(list))
    for d in glob.glob("logs/unlearn/runs/*_bs1_bw_tgt*_valsel_nau_ls0p01_ln0p01*"):
        m = re.search(r"_bs1_bw_tgt(\w+?)_valsel_nau_ls0p01_ln0p01(?:_k(\d+))?(?:_seed(\d+))?$", d)
        if not m:
            continue
        st, k, sd = m.group(1), int(m.group(2) or 8), m.group(3) or "2"
        te, va = vt.metrics(d + "/eval"), vt.metrics(d + "/eval_val")
        cv = R.get(("rkmeans", "beauty", st, sd, "clean"))
        rt = test_ref.get("beauty" + ("" if sd == "2" else f"_seed{sd}"))
        if te and va and cv and rt:
            rows[k][sd].append((abs(va[0] - cv[0]), va[1] / cv[1], abs(te[0] - rt["clean_SH@10"][st]), te[1] / rt["clean_ndcg@10"]))
    return rows


def neighborhood(stem="fig1_scope_k_val", n_boot=3000):
    """k -> (NAU neighborhood NDCG / retrained, gain over Forget+Repair, CI) on the first-token
    neighborhood (Figure 2 evaluation) of logs/diagnose/<stem>: fig1_scope_k_val is the
    validation split, fig1_scope_k the test split (run_scope_k_cell.sh)."""
    import numpy as np
    pool = defaultdict(lambda: {"a": [], "b": [], "ref": []})
    for f in sorted(glob.glob(f"logs/diagnose/{stem}/*.json")):
        r = {(x["method"], x["group"]): x for x in json.load(open(f))["rows"]}
        for k in KS:
            if (f"nau_k{k}", "neighbor") in r and ("forget_repair", "neighbor") in r:
                pool[k]["a"] += r[(f"nau_k{k}", "neighbor")]["ndcg_per_query"]
                pool[k]["b"] += r[("forget_repair", "neighbor")]["ndcg_per_query"]
                pool[k]["ref"] += r[("clean", "neighbor")]["ndcg_per_query"]
    out, rng = {}, np.random.default_rng(0)
    for k, p in pool.items():
        a, b, ref = (np.asarray(p[x]) for x in ("a", "b", "ref"))
        d, s = a - b, 100 / ref.mean()
        bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
        lo, hi = np.percentile(bs, [2.5, 97.5])
        out[k] = (100 * a.mean() / ref.mean(), s * d.mean(), s * lo, s * hi)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="paper/revision/tables/appendix_nau_k.tex")
    a = ap.parse_args()
    rows = collect()
    nb, nb_test = neighborhood("fig1_scope_k_val"), neighborhood("fig1_scope_k")
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{4pt}", r"\caption{",
         r"Sensitivity of NAU to the number of neighbors $k$ in the neighborhood term (Beauty, TIGER + RQ-KMeans, "
         r"$\lambda_f=0.1$, $\lambda_s=0.01$, $\lambda_n=0.01$, three targets $\times$ five data seeds; $k=8$ is the setting used throughout). "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ and $\tau_P$ on the validation and test splits. "
         r"Whole-set values all lie within the retraining noise floor (\Cref{tab:appendix-retrain-floor}). "
         r"Neighborhood (validation split, first-token \ac{SID} neighborhood as in \Cref{fig:scope-locality}): NAU's neighborhood $\tau_P$ and its paired gain over Forget+Repair in points, with a 95\% bootstrap interval.}",
         r"\label{tab:appendix-nau-k}",
         r"\begin{tabular}{rcccccc}", r"\toprule",
         r" & \multicolumn{2}{c}{Validation} & \multicolumn{2}{c}{Test} & \multicolumn{2}{c}{Validation, neighborhood} \\",
         r"\cmidrule(lr){2-3} \cmidrule(lr){4-5} \cmidrule(lr){6-7}",
         r"$k$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{SH}}{\scriptstyle\,(\%)}\downarrow$ & $\tau_P{\scriptstyle\,(\%)}\uparrow$ & "
         r"$\tau_P{\scriptstyle\,(\%)}\uparrow$ & Gain over F+R \\", r"\midrule"]
    for k in KS:
        per = rows[k]
        n = sum(len(v) for v in per.values())
        assert n == 15, (k, n)
        agg = lambda j: 100 * S.mean(S.mean(x[j] for x in v) for v in per.values())
        t, g, lo, hi = nb.get(k, (float("nan"),) * 4)
        vals = f"{agg(0):.3f} & {agg(1):.1f} & {agg(2):.3f} & {agg(3):.1f} & {t:.1f} & {g:+.1f} [{lo:+.1f}, {hi:+.1f}]"
        L.append((rf"\textbf{{{k}}}" if k == 8 else f"{k}") + f" & {vals} \\\\")
        tt = nb_test.get(k, (float("nan"),) * 4)
        print(f"k={k:2d}  {vals.replace(' & ', '  ')}   | test nbr {tt[0]:.1f} {tt[1]:+.1f} [{tt[2]:+.1f}, {tt[3]:+.1f}]")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
