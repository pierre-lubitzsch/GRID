"""Plan for the critique follow-up re-evaluation: every held-out row (scope_neighbor_def=none, items_per_query)
for the unpopular and popular spam cells (3 datasets x 2 strata x 5 seeds), scoring the canonical NAU
(nau_ls0.01_ln0.01) and Forget+Repair (forget_repair_lf0.1) executions of paper/plotting/canonical_runs.tsv,
the exact start checkpoint (from the runs' overrides), and the clean model. The mid stratum is covered by
logs/diagnose/randsid_scope_orig{,_val}/ (run_randsid_eval_seed.sh KIND=orig, same ledger arms).
Writes logs/scope_none/plan.tsv: dataset stratum seed poison nau forget_repair."""
import csv, os
led = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
       if r["setting"] == "spam" and r["track"] == "tiger+rkmeans"]
rows = []
for ds in ("beauty", "sports", "toys"):
    for st in ("unpopular", "popular"):
        for sd in (2, 3, 5, 7, 11):
            out, poison = {}, None
            for arm, cfg in (("nau", "nau_ls0.01_ln0.01"), ("forget_repair", "forget_repair_lf0.1")):
                r = [x for x in led if x["dataset"] == ds and x["stratum"] == st and int(x["seed"]) == sd and x["config"] == cfg]
                assert len(r) == 1 and r[0]["status"] == "ok", (ds, st, sd, cfg, len(r))
                out[arm] = os.path.realpath(r[0]["ckpt_resolved"])
                ov = dict(o[2:].split("=", 1) for o in open(os.path.join(r[0]["run_dir"], ".hydra/overrides.yaml")).read().splitlines() if o.startswith("- "))
                p = os.path.realpath(ov["ckpt_path"].strip("'\""))
                assert poison in (None, p), (ds, st, sd, poison, p)
                poison = p
            for x in (poison, *out.values()):
                assert os.path.isfile(x), x
            rows.append([ds, st, str(sd), poison, out["nau"], out["forget_repair"]])
os.makedirs("logs/scope_none", exist_ok=True)
with open("logs/scope_none/plan.tsv", "w") as f:
    f.write("\n".join("\t".join(r) for r in rows) + "\n")
print(len(rows), "rows -> logs/scope_none/plan.tsv")
