"""Efficiency E from the dedicated `_timing` campaign (run_timing_launch.sh).

Every row of logs/timing_e/plan.tsv replays one canonical run (Beauty spam at the Table 1
configurations; unwanted items at the `_mia` configurations, all datasets), alone on one
H200 with the fixed unified timer. Per method this prints / writes
  * E_req:  median wall_seconds over ALL requests pooled (Beauty spam: 15 cells x 10 requests;
            unwanted: 5 seeds x their requests), the primary number;
  * E_run:  median over runs of the per-run mean (the definition the older table scripts used,
            extract_sensitive._eu_ea / make_efficiency_table.select), for comparison;
  * the GPU models / hosts the runs ran on (from <run>/timing_host.txt; asserted H200);
  * tau_E = 1 - E_req / E_retrain, E_retrain = mean over the seeds of the retraining wall-clock,
    the same source as Table 9 (spam: h2h.retrain_train_job -> clean fit; TRACER: its tracerv3
    retrain) and Tables 3/22/23 (unwanted: the sensitive retrain reference's training job).
Outputs (new files only): paper/plotting/efficiency_timing.csv,
paper/v24_edits/tables/efficiency_beauty_timing.tex, paper/v24_edits/tables/efficiency_unwanted_timing.tex.

    PYTHONPATH=. python paper/plotting/efficiency_timing_table.py
"""
import argparse
import csv
import glob
import json
import os
import re
import statistics as S
import sys

sys.path.insert(0, "scripts")
import extract_bandwagon_head2head as h2h  # noqa: E402
import extract_sensitive as es  # noqa: E402

PLAN = "logs/timing_e/plan.tsv"
SPAM_ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER",
              "NAU", "NAU (ls0.1)"]
UNW_ORDER = ["Finetune", "Forget only", "Forget+Repair", "SCIF", "SEIF", "Kookmin", "Fanchuan", "TRACER", "NAU"]
TEX_NAME = {"Forget+Repair": "Forget + Repair", "NAU": r"NAU \textit{(ours)}",
            "NAU (ls0.1)": r"\quad NAU, $\lambda_s=0.1$"}


def run_dir(tag):
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}") if d.endswith("_bs1_" + tag)]
    return max(c, key=lambda d: int(re.match(r".*/job(\d+)_", d).group(1))) if c else None


def walls(d):
    for p in glob.glob(os.path.join(d, "*_info.json")):
        try:
            js = json.load(open(p))
        except (OSError, json.JSONDecodeError):
            continue
        w = [float(v["wall_seconds"]) for r in js.get("requests") or [] if isinstance(r, dict)
             for k, v in r.items() if k.endswith("_info") and isinstance(v, dict) and "wall_seconds" in v]
        if w:
            return w
    return None


def host(d):
    f = os.path.join(d, "timing_host.txt")
    if not os.path.isfile(f):
        return None
    return dict(ln.rstrip("\n").split("=", 1) for ln in open(f) if "=" in ln)


def retrain_seconds_spam(sd, quant):
    j = h2h.retrain_train_job("beauty", "tiger", quant, sd)
    t = h2h.train_minutes(j) if j else None
    return 60 * t if t is not None else None


def retrain_seconds_unwanted(ds, sd):
    _m, p = es.reference(ds, "retrain", "tiger", "rkmeans", sd, es.CATEGORY[ds])
    j = h2h.train_job_from_eval_dir(p) if p else None
    t = h2h.train_minutes(j) if j else None
    return 60 * t if t is not None else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", default=PLAN)
    ap.add_argument("--csv", default="paper/plotting/efficiency_timing.csv")
    ap.add_argument("--tex-spam", default="paper/v24_edits/tables/efficiency_beauty_timing.tex")
    ap.add_argument("--tex-unwanted", default="paper/v24_edits/tables/efficiency_unwanted_timing.tex")
    ap.add_argument("--allow-partial", action="store_true", help="report groups with missing runs")
    a = ap.parse_args()
    for p in (a.csv, a.tex_spam, a.tex_unwanted):
        assert "_timing" in os.path.basename(p), f"refusing to write a non-_timing output: {p}"

    plan = list(csv.DictReader(open(a.plan), delimiter="\t"))
    groups = {}
    for r in plan:
        groups.setdefault((r["setting"], r["dataset"], r["method"]), []).append(r)

    out, bad = [], []
    for (setting, ds, method), rs in groups.items():
        req, per_run, gpus, hosts, seeds, missing, eretr = [], [], set(), set(), set(), 0, {}
        for r in rs:
            d = run_dir(r["tag"])
            w = walls(d) if d else None
            if not w:
                missing += 1
                continue
            h = host(d)
            if not h or "H200" not in h.get("gpu", ""):
                bad.append((r["tag"], d, h))
                continue
            gpus.add(h["gpu"]); hosts.add(h["host"])
            req += w; per_run.append(S.mean(w))
            sd = int(r["seed"]); seeds.add(sd)
            if sd not in eretr:
                eretr[sd] = (retrain_seconds_spam(sd, "tracerv3" if method == "TRACER" else "rkmeans")
                             if setting == "spam" else retrain_seconds_unwanted(ds, sd))
        if missing and not a.allow_partial:
            print(f"WAITING {setting} {ds} {method}: {missing} of {len(rs)} runs without timing")
            continue
        if not req:
            continue
        er = [v for v in eretr.values() if v]
        E, Er = S.median(req), (S.mean(er) if er else None)
        out.append(dict(setting=setting, dataset=ds, method=method, config=rs[0]["config"],
                        n_runs=len(per_run), n_planned=len(rs), n_requests=len(req),
                        E_req_median_s=round(E, 4), E_run_median_s=round(S.median(per_run), 4),
                        E_req_min_s=round(min(req), 4), E_req_max_s=round(max(req), 4),
                        E_retrain_s=round(Er, 1) if Er else "", n_retrain_seeds=len(er),
                        tau_E=f"{1 - E / Er:.6f}" if Er else "",
                        gpu=";".join(sorted(gpus)), hosts=";".join(sorted(hosts))))
    if bad:
        for b in bad:
            print("NOT H200 / no host record:", b)
        raise SystemExit("some timing runs did not run on an H200 (or lack timing_host.txt)")

    key = {("spam", m): i for i, m in enumerate(SPAM_ORDER)}
    key.update({("unwanted", m): i for i, m in enumerate(UNW_ORDER)})
    out.sort(key=lambda o: (o["setting"] != "spam", o["dataset"], key.get((o["setting"], o["method"]), 99)))
    cols = list(out[0].keys()) if out else []
    os.makedirs(os.path.dirname(a.csv), exist_ok=True)
    with open(a.csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader(); w.writerows(out)
    for o in out:
        print(f"{o['setting']:8s} {o['dataset']:6s} {o['method']:14s} E_req {o['E_req_median_s']:8.3f}s "
              f"E_run {o['E_run_median_s']:8.3f}s  runs {o['n_runs']}/{o['n_planned']} req {o['n_requests']:4d}  "
              f"E_retr {o['E_retrain_s']}s  tau_E {o['tau_E']}  [{o['gpu']} @ {o['hosts']}]")
    print("wrote", a.csv)

    def fmt_e(e):
        return f"{e:.2f}" if e < 10 else f"{e:.1f}"

    spam = [o for o in out if o["setting"] == "spam"]
    if spam:
        L = [r"\begin{table}[ht]", r"\centering", r"\small", r"\caption{",
             r"Efficiency on Beauty (TIGER + RQ-KMeans), each method at its configuration in "
             r"\Cref{tab:bandwagon-unlearn-all-tiger-rkmeans}. $\mathcal{E}$ is the median wall-clock time per "
             r"unlearning request on one H200 GPU (one run at a time), pooled over three targets $\times$ five data "
             r"seeds; $\tau_E = 1 - \mathcal{E}/\mathcal{E}_{\mathrm{retrain}}$.",
             r"\textsuperscript{\dag}\,TRACER under its own tokenizer.", r"}", r"\label{tab:efficiency-beauty-timing}",
             r"\begin{tabular}{lrr}", r"\toprule", r"Method & $\mathcal{E}$ (s) $\downarrow$ & $\tau_E$ $\uparrow$ \\",
             r"\midrule"]
        for o in spam:
            if o["method"] == "NAU":
                L.append(r"\midrule")
            dag = r"\textsuperscript{\dag}" if o["method"] == "TRACER" else ""
            tau = f"{float(o['tau_E']):.5f}" if o["tau_E"] else "--"
            L.append(f"{TEX_NAME.get(o['method'], o['method'])}{dag} & {fmt_e(o['E_req_median_s'])} & {tau} \\\\")
        er = [o["E_retrain_s"] for o in spam if o["method"] != "TRACER" and o["E_retrain_s"]]
        if er:
            L += [r"\midrule", rf"\textit{{Retrain (reference)}} & \textit{{{er[0]:,.0f}}} & \textit{{0}} \\"]
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        os.makedirs(os.path.dirname(a.tex_spam), exist_ok=True)
        open(a.tex_spam, "w").write("\n".join(L) + "\n")
        print("wrote", a.tex_spam)

    unw = [o for o in out if o["setting"] == "unwanted"]
    if unw:
        dss = [d for d in ("beauty", "sports", "toys") if any(o["dataset"] == d for o in unw)]
        cell = {(o["dataset"], o["method"]): o for o in unw}
        L = [r"\begin{table}[ht]", r"\centering", r"\small", r"\caption{",
             r"Efficiency for unwanted-item removal: $\mathcal{E}$ (s), the median wall-clock time per request on one "
             r"H200 GPU (one run at a time) over five seeds, and $\tau_E$, each method at its configuration in the "
             r"unwanted-item tables.", r"}", r"\label{tab:efficiency-unwanted-timing}",
             r"\begin{tabular}{l" + "rr" * len(dss) + "}", r"\toprule",
             "Method & " + " & ".join(rf"\multicolumn{{2}}{{c}}{{{d.capitalize()}}}" for d in dss) + r" \\",
             " & " + " & ".join([r"$\mathcal{E}$ & $\tau_E$"] * len(dss)) + r" \\", r"\midrule"]
        for m in UNW_ORDER:
            if not any((d, m) in cell for d in dss):
                continue
            vals = []
            for d in dss:
                o = cell.get((d, m))
                vals += ([fmt_e(o["E_req_median_s"]), f"{float(o['tau_E']):.5f}" if o["tau_E"] else "--"]
                         if o else ["--", "--"])
            L.append(f"{TEX_NAME.get(m, m)} & " + " & ".join(vals) + r" \\")
        L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
        os.makedirs(os.path.dirname(a.tex_unwanted), exist_ok=True)
        open(a.tex_unwanted, "w").write("\n".join(L) + "\n")
        print("wrote", a.tex_unwanted)


if __name__ == "__main__":
    main()
