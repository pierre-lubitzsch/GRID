"""Membership inference on the deleted interactions, retraining-matched split (2026-11-16).

Same attack, data sets and bootstrap as paper/plotting/mia_unwanted_bootstrap_table.py (AUC minus the
retrained model's AUC, paired bootstrap 95% intervals, 4000 resamples; members within seed, non-members
jointly), on the rmsplit picks (paper/revision/rmsplit_selected.csv): for each method the row whose checkpoint is the
selected configuration, from logs/diagnose/mia_unwanted_rmgrid/<ds>_seed<N>{,_kf,_nfo}.json (submitted
lowest-UHF@10 rule's picks) or _rank.json (rank rule's picks).
tau_P: cleaned test view, as in the _rmsplit method tables.

    python paper/plotting/critique_followup/rmsplit_mia_table.py
Writes paper/revision/tables/appendix_mia_unwanted_bootstrap_rmsplit.{tex,csv}.
"""
import csv
import importlib.util
import json
import os
import re
import statistics as S

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
_s = importlib.util.spec_from_file_location("T", os.path.join(HERE, "rmsplit_tables.py"))
T = importlib.util.module_from_spec(_s)
_s.loader.exec_module(T)
_s = importlib.util.spec_from_file_location("MB", "paper/plotting/mia_unwanted_bootstrap_table.py")
MB = importlib.util.module_from_spec(_s)
_s.loader.exec_module(MB)

D = "logs/diagnose/mia_unwanted_rmgrid"
KEY = {"finetune": "Finetune", "forget_only": "Forget only", "forget_repair": "Forget+Repair", "scif": "SCIF",
       "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan", "tracer": "TRACER", "nau": "NAU"}
B = 4000
USED = {}


def load(ds, sd):
    """Rows of every scoring pass of this cell; per method keep the row whose checkpoint is the selected
    configuration (rmsplit_selected.csv, whichever rule wrote it). Passes: <ds>_seed<N>{,_kf,_nfo}.json scored
    the lowest-UHF@10 (submitted) rule's picks, _rank.json the rank rule's picks."""
    pk = T.picks(ds)
    rows, used, nm = {}, {}, None
    # NAU lambda_s=0 pass (2026-11-17), submitted on three partitions: use the first copy that exists
    nau0 = [x for x in ("_nau0", "_nau0g", "_nau0c", "_nau0l") if os.path.isfile(f"{D}/{ds}_seed{sd}{x}.json")][:1]
    for suf in ("", "_kf", "_nfo", "_rank", *nau0):
        f = f"{D}/{ds}_seed{sd}{suf}.json"
        if not os.path.isfile(f):
            continue
        j = json.load(open(f))
        if nm is None:
            nm, nn = j["n_members"], j["n_nonmembers"]
        assert (j["n_members"], j["n_nonmembers"]) == (nm, nn), f
        for x in j["rows"]:
            m = x["method"]
            if m in ("original", "retrained"):
                if m in rows:
                    assert abs(rows[m]["auc"] - x["auc"]) < 1e-9, (f, m)
                rows[m] = x
                continue
            if m not in KEY or KEY[m] not in pk:
                continue
            run = j["checkpoints"][m].split("/")[-3]
            want = T.tag(ds, pk[KEY[m]], sd)
            if run.endswith("_bs1_" + want):
                if m in rows and m not in used:
                    assert abs(rows[m]["auc"] - x["auc"]) < 1e-9, (f, m)
                rows[m] = x
                used.pop(m, None)
            elif m == "forget_only" and m not in rows:
                # Forget only: lambda_f has no effect (Adam, no retain term), every lambda_f gives the same
                # model behavior; accept another lambda_f only if the selected one was not scored, record it.
                mm = re.search(r"_bs1_sens_\w+?rm_(unified_hp2_fo_lf[0-9.]+_ls0.0_ln0.0_lr0.0)_rmsplit", run)
                assert mm, (ds, sd, m, run)
                rows[m] = x
                used[m] = mm.group(1)
    if used:
        USED.setdefault(ds, {}).update(used)
    return sd, nm, nn, rows


def main():
    rng = np.random.default_rng(0)
    res, counts, rows_csv, missing = {}, {}, [], []
    for ds in T.DS:
        data = [load(ds, sd) for sd in T.SEEDS]
        nn = data[0][2]
        assert all(d[2] == nn for d in data)
        counts[ds] = (sum(d[1] for d in data), nn)
        meths = [m for m, _ in MB.ORDER if all(m in d[3] for d in data)] + ["retrained"]
        missing += [f"{ds}:{m}" for m, _ in MB.ORDER if m not in meths and (m != "tracer" or ds != "beauty")]
        for d in data:
            for m in meths:
                x = d[3][m]
                assert abs(MB.auc(x["member_logprob"], x["nonmember_logprob"]) - x["auc"]) < 1e-6
        point = {m: float(np.mean([d[3][m]["auc"] for d in data])) for m in meths}
        boots = {m: np.zeros(B) for m in meths}
        INN = rng.integers(0, nn, (B, nn))
        for sd, nm, _, r in data:
            im = rng.integers(0, nm, (B, nm))
            for m in meths:
                boots[m] += MB.auc_b(np.asarray(r[m]["member_logprob"])[im], np.asarray(r[m]["nonmember_logprob"])[INN]) / len(data)
        res[ds] = {"retrained_auc": point["retrained"]}
        pk = T.picks(ds)
        for m in meths:
            if m == "retrained":
                continue
            dlt = boots[m] - boots["retrained"]
            lo, hi = np.percentile(dlt, [2.5, 97.5])
            if m == "original":
                tp = S.mean(c["tp"] for c in T.refcells(ds, "base").values())
            else:
                tp = S.mean(c["tp"] for c in T.runs(ds, USED.get(ds, {}).get(m, pk[KEY[m]])).values())
            res[ds][m] = (point[m] - point["retrained"], float(lo), float(hi), point[m], tp)
            rows_csv.append(dict(dataset=ds, method=m, config=USED.get(ds, {}).get(m, pk.get(KEY.get(m), "")) if m != "original" else "",
                                 auc=round(point[m], 4), auc_retrained=round(point["retrained"], 4),
                                 diff=round(point[m] - point["retrained"], 4), lo=round(float(lo), 4), hi=round(float(hi), 4),
                                 includes_zero=bool(lo <= 0 <= hi), tau_P=round(tp, 2)))
        print(ds, "members", counts[ds][0], "non-members", nn)
        for m in meths:
            if m != "retrained":
                v = res[ds][m]
                print(f"  {m:14s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]  auc {v[3]:.3f}  tauP {v[4]:.1f}")
    out = "paper/revision/tables/appendix_mia_unwanted_bootstrap_rmsplit"
    with open(out + ".csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows_csv[0])); w.writeheader(); w.writerows(rows_csv)

    def cell(ds, m):
        if m not in res[ds]:
            return ["--", "--"]
        d, lo, hi, _a, tp = res[ds][m]
        v = f"{d:+.3f}"
        if lo <= 0 <= hi:
            v = rf"\textbf{{{v}}}"
        return [f"{v} {{\\scriptsize[{lo:+.3f}, {hi:+.3f}]}}", f"{tp:.1f}"]
    L = ["\\begin{table}[t]", "\\centering", "\\small", "\\setlength{\\tabcolsep}{2.5pt}",
         "\\caption{Membership inference on the deleted interactions in sensitive item deletion "
         "(TIGER + RQ-KMeans, five data seeds, retraining-matched split; configurations selected on cleaned validation histories, \\Cref{tab:appendix-valsens}). "
         "A loss-based attack scores each interaction by the model's log-probability of the item given the user's preceding history. "
         f"Members are the requesting users' training interactions with the category ({counts['beauty'][0]} / {counts['sports'][0]} / {counts['toys'][0]} over five seeds), "
         f"non-members the {counts['beauty'][1]} / {counts['sports'][1]} / {counts['toys'][1]} held-out cases with the category as next item. "
         "Members and non-members come from different users and positions, so even a model that never saw the members separates them to some degree; "
         f"the reference is therefore the retrained model's AUC ({res['beauty']['retrained_auc']:.3f} / {res['sports']['retrained_auc']:.3f} / {res['toys']['retrained_auc']:.3f}), not 0.5. "
         "Entries: AUC (mean over seeds) minus the retrained model's, with paired bootstrap 95\\% intervals (4000 resamples; members within seed, non-members jointly). "
         "Bold: interval includes 0. Values below 0 mean the deleted interactions became less likely than unseen ones, which itself reveals the deletion. "
         "$\\tau_P$: test utility in \\% of the retrained model (cleaned test histories)."
         + ("" if not USED else " Forget only is scored at $\\lambda_f=" + ", ".join(sorted({u.split("_lf")[1].split("_")[0].rstrip("0").rstrip(".") for d in USED.values() for u in d.values()})) + "$ on " + " and ".join(d.capitalize() for d in USED) + ", which has the same validation results as $\\lambda_f=1$.") + "}",
         "\\label{tab:appendix-mia-unwanted-bootstrap}", "\\begin{tabular}{lcccccc}", "\\toprule",
         " & \\multicolumn{2}{c}{Beauty} & \\multicolumn{2}{c}{Sports} & \\multicolumn{2}{c}{Toys} \\\\",
         "\\cmidrule(lr){2-3} \\cmidrule(lr){4-5} \\cmidrule(lr){6-7}",
         "Method & $\\Delta$AUC & $\\tau_P$ & $\\Delta$AUC & $\\tau_P$ & $\\Delta$AUC & $\\tau_P$ \\\\", "\\midrule"]
    for m, _ in MB.ORDER:
        L.append(MB.TEX_NAME[m] + " & " + " & ".join(c for ds in T.DS for c in cell(ds, m)) + " \\\\")
        if m == "original":
            L.append("\\midrule")
    L += ["\\bottomrule", "\\end{tabular}", "\\end{table}"]
    open(out + ".tex", "w").write("\n".join(L) + "\n")
    print("wrote", out + ".tex", out + ".csv", "MISSING:" if missing else "", missing)


if __name__ == "__main__":
    main()
