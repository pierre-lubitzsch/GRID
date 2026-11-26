"""Plan: brand takedown (Beauty, e.l.f.) on CLEANED test histories (2026-11-14).

Every brand buyer's deleted interactions are removed from their test history, exactly the items the
takedown retrain builder removes (scripts/build_scrubbed_eval_view.py --category takedown_elf --ratio 0
-> src/data/amazon_data/beauty_sens_takedown_elf_0_seed<N>_scrubtestview). Rows, per data seed:
  original and retrained (takedown) models: logs/eval/takedown_reference/v1/beauty_{base,retrain}_seed<N>
  every retraining-matched takedown run of logs/rmsplit_sens/plan_extra_takedown.tsv with a checkpoint
Output dirs under logs/eval/sensitive_scrub/scrub_takedown/ (no existing dir is reused).

    python scripts/plan_takedown_scrub_eval.py -> logs/eval/sensitive_scrub/plan_takedown_scrub.tsv
    python scripts/plan_takedown_scrub_eval.py --plans logs/rmsplit_sens/plan_followup_takedown.tsv \
        --out logs/eval/sensitive_scrub/plan_takedown_scrub_followup.tsv
        (the takedown rows of the configurations re-selected on cleaned validation; same references and dirs)
    CELL=all PLAN=logs/eval/sensitive_scrub/plan_takedown_scrub.tsv VIEWS=scrub PAR=3 bash run_sensitive_scrub_eval.sh
"""
import csv
import glob
import json
import os

S = "logs/eval/sensitive_scrub"
OUT = f"{S}/plan_takedown_scrub.tsv"
HEAD = ["cell", "dataset", "seed", "role", "method", "arm", "view", "ckpt", "data_dir", "manifest",
        "semantic_id_path", "out_dir", "source", "experiment"]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", nargs="+", default=["logs/rmsplit_sens/plan_extra_takedown.tsv"])
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    rows, miss = [], []
    for sd in (2, 3, 5, 7, 11):
        view = f"src/data/amazon_data/beauty_sens_takedown_elf_0_seed{sd}_scrubtestview"
        man = f"src/data/amazon_data/beauty/forget_manifest_takedown_elf_0_seed{sd}.json"
        for role, tok in (("original", "base"), ("retrained", "retrain")):
            src = f"logs/eval/takedown_reference/v1/beauty_{tok}" + ("" if sd == 2 else f"_seed{sd}")   # seed 2: no suffix
            c = json.load(open(f"{src}/coordinates.json"))
            assert c["manifest"] == man, c
            rows.append(dict(cell=f"beauty_s{sd}", dataset="beauty", seed=str(sd), role=role, method=role, arm=tok,
                             view="scrub", ckpt=c["ckpt"], data_dir=view, manifest=man,
                             semantic_id_path=c["semantic_id_path"],
                             out_dir=f"{S}/scrub_takedown/reference/beauty_{tok}_seed{sd}",
                             source=src, experiment="tiger_train_flat"))
    for line in (l for p in a.plans for l in open(p) if l.strip()):
        f = line.rstrip("\n").split("\t")
        sd, tag, algo, sid = int(f[3]), f[5], f[7], f[8]
        ck = [os.path.join(d, "checkpoints", "unlearned.ckpt") for d in sorted(glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}"))
              if os.path.exists(os.path.join(d, "checkpoints", "unlearned.ckpt"))]
        if not ck:
            miss.append(tag); continue
        run = os.path.basename(os.path.dirname(os.path.dirname(ck[-1])))
        rows.append(dict(cell=f"beauty_s{sd}", dataset="beauty", seed=str(sd), role="unlearned", method=f"{algo}_rm",
                         arm=tag.split("elfrm_", 1)[1], view="scrub", ckpt=ck[-1],
                         data_dir=f"src/data/amazon_data/beauty_sens_takedown_elf_0_seed{sd}_scrubtestview",
                         manifest=f"src/data/amazon_data/beauty/forget_manifest_takedown_elf_0_seed{sd}.json",
                         semantic_id_path=sid, out_dir=f"{S}/scrub_takedown/runs/{run}/eval",
                         source=os.path.dirname(os.path.dirname(ck[-1])), experiment="tiger_train_flat"))
    with open(a.out, "w", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n"); w.writerow(HEAD)
        for r in rows:
            w.writerow([r[h] for h in HEAD])
    print(f"{a.out}: {len(rows)} rows; runs without checkpoint: {len(miss)}")
    for t in miss:
        print("  missing:", t)


if __name__ == "__main__":
    main()
