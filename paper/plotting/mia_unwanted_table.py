"""Appendix table: loss-based membership inference on the unwanted (sensitive-category) items.

Members are the requesting users' interactions with the deleted category, each scored as the
model's log-probability of that item given the user's preceding history; non-members are
held-out test cases whose next item is in the same category (scripts/make_mia_unwanted_sets.py).
Mean membership AUC over the 5 data seeds (logs/diagnose/mia_unwanted/, scored by
scripts/launch_mia_unwanted.py) and the mean absolute distance to the retrained model's AUC,
which is the target (members and non-members differ in distribution, so it is not 0.5).

    PYTHONPATH=. python paper/plotting/mia_unwanted_table.py [--out paper/revision/tables/appendix_mia_unwanted.tex]

The unlearned rows already read the canonical `_mia` executions (verified 2026-11-07: all 130
checkpoints in the JSONs are the ledger run dirs). --original-root logs/diagnose/mia_unwanted_lastckpt
replaces the Original row by the model at the exact checkpoint the `_mia` runs start from
(launch_mia_unwanted.py --original-lastckpt); default output then
paper/revision/tables/appendix_mia_unwanted_ledger.tex.
"""
import argparse
import glob
import json
import os
import statistics as S
from collections import defaultdict

ROWS = [("Original (no unlearning)", "original"), ("Retrained", "retrained"), ("Finetune", "finetune"),
        ("Forget only", "forget_only"), ("Forget+Repair", "forget_repair"), ("SCIF", "scif"), ("SEIF", "seif"),
        ("Kookmin", "kookmin"), ("Fanchuan", "fanchuan"), ("TRACER", "tracer"), (r"NAU \textit{(ours)}", "nau")]
DS = ("beauty", "sports", "toys")


ORIG_ROOT = None


def collect():
    auc, gap, n = defaultdict(lambda: defaultdict(list)), defaultdict(lambda: defaultdict(list)), defaultdict(list)
    for d in DS:
        for f in sorted(glob.glob(f"logs/diagnose/mia_unwanted/{d}_seed*.json")):
            j = json.load(open(f))
            r = {x["method"]: x["auc"] for x in j["rows"]}
            of = f.replace("logs/diagnose/mia_unwanted/", ORIG_ROOT.rstrip("/") + "/") if ORIG_ROOT else ""
            # Beauty seed 2 starts from an epoch checkpoint that the original evaluation already
            # used, so the last-checkpoint re-evaluation skips it: keep the original there.
            if ORIG_ROOT and os.path.isfile(of):
                o = json.load(open(of))
                assert o["n_members"] == j["n_members"] and o["n_nonmembers"] == j["n_nonmembers"], f
                r["original"] = {x["method"]: x["auc"] for x in o["rows"]}["original"]
            n[d].append(j["n_members"])
            for _, m in ROWS:
                if m in r:
                    auc[m][d].append(r[m])
                    gap[m][d].append(abs(r[m] - r["retrained"]))
    return auc, gap, n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    ap.add_argument("--original-root", default=None)
    a = ap.parse_args()
    global ORIG_ROOT
    ORIG_ROOT = a.original_root
    if a.out is None:
        a.out = ("paper/revision/tables/appendix_mia_unwanted_ledger.tex" if ORIG_ROOT
                 else "paper/revision/tables/appendix_mia_unwanted.tex")
    auc, gap, n = collect()
    best = {d: min((S.mean(gap[m][d]), m) for _, m in ROWS if m not in ("original", "retrained") and gap[m][d])[1] for d in DS}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\caption{",
         r"Loss-based membership inference on the deleted sensitive-category interactions (TIGER + RQ-KMeans, five data seeds). "
         r"Members are the requesting users' interactions with the category, scored by the model's log-probability of the item given the user's preceding history; "
         r"non-members are held-out cases whose next item is in the same category"
         + f" ({min(min(v) for v in n.values())} to {max(max(v) for v in n.values())} members and 156 to 272 non-members per seed). "
         r"AUC: mean membership AUC. $|\Delta|$: mean absolute distance to the retrained model's AUC, the target of exact unlearning "
         r"(the retrained model is not at 0.5 because members and non-members differ in distribution). "
         r"Values below the retrained model's AUC mean the deleted interactions became \emph{less} likely than unseen ones, which itself reveals the deletion. "
         r"Lowest $|\Delta|$ among unlearning methods in bold; -- marks methods not run on a dataset.}",
         r"\label{tab:appendix-mia-unwanted}",
         r"\begin{tabular}{l" + "cc" * len(DS) + "}", r"\toprule",
         " & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d.capitalize()}}}" for d in DS) + r" \\",
         " ".join(rf"\cmidrule(lr){{{2 + 2 * i}-{3 + 2 * i}}}" for i in range(len(DS))),
         "Method" + r" & AUC & $|\Delta|\downarrow$" * len(DS) + r" \\", r"\midrule"]
    for label, m in ROWS:
        cells = []
        for d in DS:
            if not auc[m][d]:
                cells += ["--", "--"]; continue
            g = f"{S.mean(gap[m][d]):.3f}"
            if m == "retrained":
                g = "--"
            elif m == best[d]:
                g = rf"\textbf{{{g}}}"
            cells += [f"{S.mean(auc[m][d]):.3f}", g]
        L.append(f"{label} & " + " & ".join(cells) + r" \\")
        if m == "retrained":
            L.append(r"\midrule")
        print(f"{label:28s} " + "  ".join(cells))
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    open(a.out, "w").write("\n".join(L) + "\n")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
