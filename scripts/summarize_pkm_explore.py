"""Summarise the explore/exploit sweep (run_pkm_explore_beauty.sh).

For every logs/train/runs/*/*_EXPLORE_<POS>_<ARM> run: job state, last step,
utilisation trajectory (pkm/util_mean and per-layer live slots), and the best and
last validation metrics. Run from GRID/:  python scripts/summarize_pkm_explore.py
"""

import glob
import os
import re
import subprocess

import pandas as pd

RUNS = sorted(glob.glob("logs/train/runs/*/*_EXPLORE_*"))
# util logged at step s lands in the CSV row of the next log_every_n_steps flush
CKPTS = (200, 500, 1000, 2000, 3000, 4000)


def job_state(job):
    try:
        out = subprocess.run(
            ["sacct", "-j", job, "-X", "-n", "-o", "State,Elapsed"],
            capture_output=True, text=True, timeout=30,
        ).stdout.split()
        return " ".join(out[:2])
    except Exception:
        return "?"


rows, traj = [], []
for run in RUNS:
    m = re.search(r"job(\d+)_.*_EXPLORE_([A-Za-z0-9]+)_([a-z0-9]+)$", os.path.basename(run))
    if not m:
        continue
    job, pos, arm = m.groups()
    csv = os.path.join(run, "csv/version_0/metrics.csv")
    row = {"pos": pos, "arm": arm, "job": job, "state": job_state(job)}
    if os.path.exists(csv):
        d = pd.read_csv(csv)
        row["last_step"] = int(d["step"].max())
        u = d.dropna(subset=["pkm/util_mean"]) if "pkm/util_mean" in d else d.iloc[:0]
        live_cols = [c for c in d.columns if c.startswith("pkm/live_slots/")]
        for s in CKPTS:
            near = u[u["step"] <= s - 1]
            if len(near) and s - 1 <= row["last_step"]:
                row[f"util@{s}"] = round(float(near.iloc[-1]["pkm/util_mean"]), 4)
        if len(u):
            last = u.iloc[-1]
            row["live_last"] = " / ".join(
                f"{c.split('/')[-1].replace('.memory', '')}={int(last[c])}" for c in live_cols
            )
            for _, r in u.iterrows():
                traj.append({"pos": pos, "arm": arm, "step": int(r["step"]),
                             "util": float(r["pkm/util_mean"]),
                             **{c.split("/")[-1]: r[c] for c in live_cols}})
        v = d.dropna(subset=["val/ndcg@10"]) if "val/ndcg@10" in d else d.iloc[:0]
        if len(v):
            best = v.loc[v["val/recall@5"].idxmax()]
            lastv = v.iloc[-1]
            for k, r in (("best", best), ("last", lastv)):
                row[f"{k}_step"] = int(r["step"])
                for met in ("val/ndcg@10", "val/recall@10", "val/SH@10"):
                    if met in r:
                        row[f"{k}_{met.split('/')[1]}"] = round(float(r[met]), 5)
        if "pkm/explore" in d:
            e = d.dropna(subset=["pkm/explore"])
            row["explore_frac_logged"] = round(float(e["pkm/explore"].mean()), 3) if len(e) else None
    rows.append(row)

pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 50)
summary = pd.DataFrame(rows).sort_values(["pos", "arm"])
print(summary.to_string(index=False))
out = "logs/pkm_explore_summary"
os.makedirs(out, exist_ok=True)
summary.to_csv(f"{out}/summary.csv", index=False)
pd.DataFrame(traj).to_csv(f"{out}/util_trajectory.csv", index=False)
print(f"\nwrote {out}/summary.csv and {out}/util_trajectory.csv")
