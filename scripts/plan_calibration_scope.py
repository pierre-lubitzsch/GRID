"""Plan for the calibration check (critique follow-up, addendum 2): score every held-out row
(scope_neighbor_def=none, items_per_query) on Beauty, 15 cells, test and validation, for the
canonical NAU (nau_ls0.01_ln0.01, k=8) and Forget+Repair (forget_repair_lf0.1) executions of
paper/plotting/canonical_runs.tsv, the NAU weight grid with checkpoints (same ledger), and the
k sweep (logs/nau_k_keepck_plan.tsv runs, *_nau_s0p01_n0p01_k{K}_keepck), plus start and clean.
Writes logs/calib_scope/plan.tsv: dataset stratum seed poison arms (arms = 'name=path;name=path...').
Arms whose run started from another checkpoint than the canonical NAU run are reported and kept
(their start is not the scored 'poison'); see the printed warnings.
    python scripts/plan_calibration_scope.py"""
import csv, glob, os, re, sys
led = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
       if r["setting"] == "spam" and r["track"] == "tiger+rkmeans" and r["dataset"] == "beauty"]
GRID = {"forget_repair": "forget_repair_lf0.1", "nau": "nau_ls0.01_ln0.01",
        "g_s0p01_n0": "nau_ls0.01_ln0.0", "g_s0_n0p01": "nau_ls0.0_ln0.01", "g_s0_n0p1": "nau_ls0.0_ln0.1",
        "g_s0p1_n0": "nau_ls0.1_ln0.0", "g_s0p1_n0p01": "nau_ls0.1_ln0.01", "g_s0p1_n0p1": "nau_ls0.1_ln0.1"}
EXTRA = os.environ.get("EXTRA_TAGS", "")   # step 2, e.g. 'ln0p001=n0p001,ln0p003=n0p003': tag = canonical NAU tag with _n0p01_keepck -> _<val>_lnsweep_canonck

def start_of(run_dir):
    ov = dict(o[2:].split("=", 1) for o in open(os.path.join(run_dir, ".hydra/overrides.yaml")).read().splitlines() if o.startswith("- "))
    return os.path.realpath(ov["ckpt_path"].strip("'\""))

rows, warn = [], 0
for st in ("unpopular", "mid", "popular"):
    for sd in (2, 3, 5, 7, 11):
        arms, starts = {}, {}
        for arm, cfg in GRID.items():
            r = [x for x in led if x["stratum"] == st and int(x["seed"]) == sd and x["config"] == cfg]
            assert len(r) == 1 and r[0]["status"] == "ok" and r[0]["has_ckpt"] == "1", (st, sd, cfg, len(r))
            arms[arm] = os.path.realpath(r[0]["ckpt_resolved"]); starts[arm] = start_of(r[0]["run_dir"])
            if arm == "nau":
                nau_tag = r[0]["tag"]
        sfx = "" if sd == 2 else f"_seed{sd}"
        for k in (2, 4, 16, 32, 64):
            runs = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_bw_tgt{st}_nau_s0p01_n0p01_k{k}_keepck{sfx}")
                    if os.path.exists(d + "/checkpoints/unlearned.ckpt")]
            assert runs, (st, sd, k)
            best = max(runs, key=lambda d: int(re.search(r"job(\d+)_", d).group(1)))
            arms[f"nau_k{k}"] = os.path.realpath(best + "/checkpoints/unlearned.ckpt"); starts[f"nau_k{k}"] = start_of(best)
        for item in filter(None, EXTRA.split(",")):
            name, val = item.split("=")
            assert "_n0p01_keepck" in nau_tag, nau_tag
            tag = nau_tag.replace("_n0p01_keepck", f"_{val}_lnsweep2_canonck")
            runs = [d for d in glob.glob(f"logs/unlearn/runs/*_bs1_{tag}")
                    if os.path.exists(d + "/checkpoints/unlearned.ckpt")]
            assert runs, (st, sd, name, tag)
            best = max(runs, key=lambda d: int(re.search(r"job(\d+)_", d).group(1)))
            arms[name] = os.path.realpath(best + "/checkpoints/unlearned.ckpt"); starts[name] = start_of(best)
        poison = starts["nau"]
        for a, s in starts.items():
            if s != poison:
                warn += 1; print(f"WARN {st} s{sd} {a} started from {s} (canonical NAU: {poison})", file=sys.stderr)
        for x in (poison, *arms.values()):
            assert os.path.isfile(x), x
        only = [x for x in os.environ.get("ONLY", "").split(",") if x]   # e.g. ONLY=forget_repair,nau,ln0p001,ln0p003
        rows.append(["beauty", st, str(sd), poison, ";".join(f"{a}={p}" for a, p in arms.items() if not only or a in only)])
out = os.environ.get("PLAN", "logs/calib_scope/plan.tsv")
with open(out, "w") as f:
    f.write("\n".join("\t".join(r) for r in rows) + "\n")
print(len(rows), "rows ->", out, f"({warn} start mismatches)")
