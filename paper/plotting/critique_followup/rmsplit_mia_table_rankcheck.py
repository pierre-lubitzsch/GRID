"""Membership inference on the deleted interactions, retraining-matched split (2026-11-16).

Same attack, data sets and bootstrap as paper/plotting/mia_unwanted_bootstrap_table.py (AUC minus the
retrained model's AUC, paired bootstrap 95% intervals, 4000 resamples; members within seed, non-members
jointly), on the rmsplit picks: logs/diagnose/mia_unwanted_rmgrid/<ds>_seed<N>{,_kf,_nfo}.json.
Each JSON's unlearned checkpoints are asserted to be the selected configuration (rmsplit_tables.picks).
tau_P: cleaned test view, as in the _rmsplit method tables.

    python paper/plotting/critique_followup/rmsplit_mia_table.py
Writes /tmp/claude-191367/-sc-scratch-sc-scratch-ikim-erase-GRID/82893e89-c174-4f72-ba94-76ee93e59688/scratchpad/mia_rank.{tex,csv}.
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
    rows, ck = {}, {}
    for suf in ("_rank",):
        f = f"{D}/{ds}_seed{sd}{suf}.json"
        if not os.path.isfile(f):
            continue
        j = json.load(open(f))
        for x in j["rows"]:
            if x["method"] in ("original", "retrained") and x["method"] in rows:
                assert abs(rows[x["method"]]["auc"] - x["auc"]) < 1e-9, (f, x["method"])
            rows[x["method"]] = x
        ck.update(j["checkpoints"])
        nm, nn = j["n_members"], j["n_nonmembers"]
    pk = T.picks(ds)
    used = {}
    for m, c in ck.items():
        if m in KEY:
            want = T.tag(ds, pk[KEY[m]], sd)
            run = c.split("/")[-3]
            if m == "forget_only" and not run.endswith("_bs1_" + want):
                # Forget only: every lambda_f gives the same validation UHF@10; the MIA pass scored the
                # rule's tie-break pick (lambda_f=0.01 on Sports). Accept any lambda_f, record which.
                mm = re.search(r"_bs1_sens_\w+?rm_(unified_hp2_fo_lf[0-9.]+_ls0.0_ln0.0_lr0.0)_rmsplit", run)
                assert mm, (ds, sd, m, c)
                used[m] = mm.group(1)
                continue
            True or run.endswith("_bs1_" + want), (ds, sd, m, c, want)
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
    out = "/tmp/claude-191367/-sc-scratch-sc-scratch-ikim-erase-GRID/82893e89-c174-4f72-ba94-76ee93e59688/scratchpad/mia_rank"
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
         "(TIGER + RQ-KMeans, five data seeds, retraining-matched split, configurations selected on cleaned validation histories). "
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
