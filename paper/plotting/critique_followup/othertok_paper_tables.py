"""Paper (appendix) versions of the other-tokenizer sensitive item deletion tables (2026-11-25).

Inputs: paper/revision/othertok_tables_numbers_tuned.csv (TUNED=1 othertok_tables.py: per-tokenizer selection on
cleaned validation) and the membership-inference JSONs of scripts/launch_mia_othertok.py
(logs/diagnose/mia_unwanted_othertok/<track>_<ds>_seed<N>.json). The membership bootstrap is the one of
rmsplit_mia_table.py (AUC mean over seeds minus the retrained model's, paired bootstrap 95% intervals, 4000 resamples,
members resampled within seed, non-members jointly, seed 0).

    python paper/plotting/critique_followup/othertok_paper_tables.py
      -> paper/revision/tables/othertok/paper/appendix_sensitive_tokenizers.tex        (summary: NAU vs F+R vs Forget only)
         paper/revision/tables/othertok/paper/appendix_mia_tokenizers.tex             (membership test, all tokenizers)
         paper/revision/tables/othertok/paper/appendix_sensitive_tokenizer_<track>.tex (all methods per tokenizer)
         paper/revision/tables/othertok/paper/appendix_mia_tokenizers.csv
"""
import csv
import importlib.util
import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("MB", os.path.join(HERE, "..", "mia_unwanted_bootstrap_table.py"))
MB = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MB)

OUT = "paper/revision/tables/othertok/paper"
NUMS = "paper/revision/othertok_tables_numbers_tuned.csv"
MIAD = "logs/diagnose/mia_unwanted_othertok"
SEEDS = (2, 3, 5, 7, 11)
TRACKS = [("rqvae", "RQ-VAE", ("beauty", "sports", "toys")), ("letter", "LETTER", ("beauty", "sports", "toys")),
          ("tracerv3", "TRACER-compatible RQ-KMeans", ("beauty",))]
METHODS = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU"]
B = 4000


def load_nums():
    out = {}
    for r in csv.DictReader(open(NUMS)):
        out[(r["track"], r["dataset"], r["method"])] = {k: float(r[k]) for k in ("f", "df", "r", "dr", "tp")} | {"config": r["config"]}
    return out


def nau_ln(cfg):
    v = cfg.split("_ln")[1]
    return "0" if v.startswith("0") else "-" + v[3:].rstrip("0").rstrip(".")


def summary_table(N):
    cols = ["NAU", "Forget+Repair", "Forget only"]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Sensitive item deletion with other \ac{SID} tokenizers (TIGER with RQ-VAE, TRACER-compatible RQ-KMeans "
         r"(Beauty only), and LETTER identifiers; five data seeds, retraining-matched split, cleaned test histories). "
         r"Each tokenizer's configurations are selected on its own cleaned validation histories with the rule of "
         r"\Cref{tab:appendix-valsens}; NAU's $\lambda_n$ ($\lambda_f=0.1$, $\lambda_s=0$) is selected on Beauty and reused. "
         r"$\mathrm{UHF}@10$: requesting users with a sensitive item in their top 10; "
         r"$\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$: other users' distance to the retrained model; $\tau_P$: NDCG@10 relative to "
         r"the retrained model; all in \%, mean over seeds. Bold: smallest $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}$ of the three "
         r"methods. All methods are in \Cref{" + ",".join(f"tab:appendix-sensitive-tokenizer-{t}" for t, _, _ in TRACKS) + r"}.}",
         r"\label{tab:appendix-sensitive-tokenizers}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{llc" + "ccc" * len(cols) + "}", r"\toprule",
         r" & & Retrained & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{c}}}" for c in cols) + r" \\",
         " ".join(rf"\cmidrule(lr){{{4 + 3 * i}-{6 + 3 * i}}}" for i in range(len(cols))),
         r"Tokenizer & Dataset & $\mathrm{UHF}$ & " + " & ".join(
             [r"$\mathrm{UHF}$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P\uparrow$"] * len(cols)) + r" \\", r"\midrule"]
    for ti, (t, name, dss) in enumerate(TRACKS):
        for di, ds in enumerate(dss):
            vals = {c: N.get((t, ds, c)) for c in cols}
            best = min(v["dr"] for v in vals.values() if v)
            cells = []
            for c in cols:
                v = vals[c]
                if not v:
                    cells += ["--"] * 3; continue
                dr = f"{v['dr']:.2f}"
                if abs(v["dr"] - best) < 1e-9:
                    dr = rf"\textbf{{{dr}}}"
                cells += [f"{v['f']:.1f}", dr, f"{v['tp']:.1f}"]
            rr = N[(t, ds, "Retrained")]["f"]
            L.append(f"{name if di == 0 else ''} & {ds.capitalize()} & {rr:.1f} & " + " & ".join(cells) + r" \\")
        if ti < len(TRACKS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return L


def full_table(N, t, name, dss):
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         rf"\caption{{Sensitive item deletion with {name} identifiers, all methods (five data seeds, retraining-matched split, "
         r"cleaned test histories; configurations selected on this tokenizer's cleaned validation histories, NAU: $\lambda_n="
         + nau_ln(N[(t, dss[0], 'NAU')]["config"]) + r"$, selected on Beauty). Columns as in \Cref{tab:appendix-sensitive-tokenizers}; "
         r"all in \%. $^{\dagger}$: $\tau_P<95\%$.}",
         rf"\label{{tab:appendix-sensitive-tokenizer-{t}}}", r"\resizebox{" + (r"\textwidth" if len(dss) > 1 else r"0.6\textwidth") + "}{!}{%",
         r"\begin{tabular}{l" + "ccc" * len(dss) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{d.capitalize()}}}" for d in dss) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(dss))),
         "Method & " + " & ".join([r"$\mathrm{UHF}$ & $\Delta^{\mathrm{retr}}_{\mathrm{UHR}}\downarrow$ & $\tau_P\uparrow$"] * len(dss))
         + r" \\", r"\midrule"]
    for m in ["Before deletion", "Retrained"] + METHODS:
        if m == METHODS[0]:
            L.append(r"\midrule")
        cells = []
        for ds in dss:
            v = N.get((t, ds, m))
            if not v:
                cells += ["--"] * 3; continue
            dag = r"$^{\dagger}$" if m in METHODS and v["tp"] < 95 else ""
            dr = "--" if m == "Retrained" else f"{v['dr']:.2f}"
            cells += [f"{v['f']:.1f}", dr, f"{v['tp']:.1f}{dag}"]
        nm = {"NAU": r"NAU \textit{(ours)}", "Before deletion": r"\textit{Before deletion}", "Retrained": r"\textit{Retrained}"}.get(m, m)
        L.append(nm + " & " + " & ".join(cells) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return L


def mia(N):
    rng = np.random.default_rng(0)
    res, rows = {}, []
    for t, name, dss in TRACKS:
        for ds in dss:
            data = []
            for sd in SEEDS:
                f = f"{MIAD}/{t}_{ds}_seed{sd}.json"
                if not os.path.exists(f):
                    return None, f"missing {f}"
                j = json.load(open(f))
                data.append((j["n_members"], j["n_nonmembers"], {r["method"]: r for r in j["rows"]}))
            nn = data[0][1]
            assert all(d[1] == nn for d in data)
            meths = [m for m, _ in MB.ORDER if all(m in d[2] for d in data)] + ["retrained"]
            point = {m: float(np.mean([d[2][m]["auc"] for d in data])) for m in meths}
            boots = {m: np.zeros(B) for m in meths}
            INN = rng.integers(0, nn, (B, nn))
            for nm, _, r in data:
                im = rng.integers(0, nm, (B, nm))
                for m in meths:
                    boots[m] += MB.auc_b(np.asarray(r[m]["member_logprob"])[im], np.asarray(r[m]["nonmember_logprob"])[INN]) / len(data)
            for m in meths:
                if m == "retrained":
                    continue
                lo, hi = np.percentile(boots[m] - boots["retrained"], [2.5, 97.5])
                res[(t, ds, m)] = (point[m] - point["retrained"], float(lo), float(hi))
                rows.append(dict(track=t, dataset=ds, method=m, auc=round(point[m], 4), auc_retrained=round(point["retrained"], 4),
                                 diff=round(point[m] - point["retrained"], 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                                 includes_zero=bool(lo <= 0 <= hi)))
    return res, rows


def mia_table(res, N):
    cols = [("nau", "NAU"), ("forget_repair", "Forget+Repair"), ("forget_only", "Forget only"), ("finetune", "Finetune"),
            ("scif", "SCIF"), ("tracer", "TRACER"), ("original", "Before deletion")]
    L = [r"\begin{table}[t]", r"\centering", r"\scriptsize", r"\setlength{\tabcolsep}{3pt}",
         r"\caption{Membership inference on the deleted interactions with other \ac{SID} tokenizers (same attack, members, "
         r"non-members, and bootstrap as \Cref{tab:appendix-mia-unwanted-bootstrap}; configurations of "
         r"\Cref{tab:appendix-sensitive-tokenizers}). Entries: AUC minus the retrained model's AUC (mean over five seeds); for NAU "
         r"with the paired bootstrap 95\% interval. Bold: interval includes 0. $^{\dagger}$: $\tau_P<95\%$.}",
         r"\label{tab:appendix-mia-tokenizers}", r"\resizebox{\textwidth}{!}{%",
         r"\begin{tabular}{ll" + "c" * len(cols) + "}", r"\toprule",
         "Tokenizer & Dataset & " + " & ".join(c for _, c in cols) + r" \\", r"\midrule"]
    meth_name = {"nau": "NAU", "forget_repair": "Forget+Repair", "forget_only": "Forget only", "finetune": "Finetune",
                 "scif": "SCIF", "tracer": "TRACER"}
    for ti, (t, name, dss) in enumerate(TRACKS):
        for di, ds in enumerate(dss):
            cells = []
            for k, _ in cols:
                if (t, ds, k) not in res:
                    cells.append("--"); continue
                d, lo, hi = res[(t, ds, k)]
                v = f"{d:+.2f}"
                if lo <= 0 <= hi:
                    v = rf"\textbf{{{v}}}"
                if k == "nau":
                    v += rf" {{\tiny[{lo:+.2f}, {hi:+.2f}]}}"
                n = N.get((t, ds, meth_name.get(k, "")))
                if n and n["tp"] < 95:
                    v += r"$^{\dagger}$"
                cells.append(v)
            L.append(f"{name if di == 0 else ''} & {ds.capitalize()} & " + " & ".join(cells) + r" \\")
        if ti < len(TRACKS) - 1:
            L.append(r"\midrule")
    L += [r"\bottomrule", r"\end{tabular}}", r"\end{table}"]
    return L


def main():
    os.makedirs(OUT, exist_ok=True)
    N = load_nums()
    open(f"{OUT}/appendix_sensitive_tokenizers.tex", "w").write("\n".join(summary_table(N)) + "\n")
    for t, name, dss in TRACKS:
        open(f"{OUT}/appendix_sensitive_tokenizer_{t}.tex", "w").write("\n".join(full_table(N, t, name, dss)) + "\n")
    res, rows = mia(N)
    if res is None:
        print("membership table skipped:", rows)
    else:
        open(f"{OUT}/appendix_mia_tokenizers.tex", "w").write("\n".join(mia_table(res, N)) + "\n")
        with open(f"{OUT}/appendix_mia_tokenizers.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
        for r in rows:
            if r["method"] in ("nau", "forget_repair", "forget_only"):
                print(f"{r['track']:8s} {r['dataset']:6s} {r['method']:14s} {r['diff']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
