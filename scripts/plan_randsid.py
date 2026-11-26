"""Plan for the random-SID control's unlearning runs (NAU rebuttal, 2026-11-07).

Four arms x five data seeds on Beauty mid, each replaying the CANONICAL RQ-KMeans run of its
cell (paper/plotting/canonical_runs.tsv) with every unlearning.* override, the same data dir
and unlearning seed; only the SID tensor (embeddings/beauty_randsid) and the starting
checkpoint (the random-SID poisoned model of that seed, last.ckpt, resolved at job start
from its training label) change. Tags `randsid_bw_tgtmid_<arm>[_seedN]` (no _keepck/_valsel/
_mia/_canonck token; run_tags.parse_tag returns None for them and iter_runs skips them).

Columns = run_valsel_batch.sh's, except `ckpt` holds `label:<training label>`.

    python scripts/plan_randsid.py > logs/randsid/plan.tsv
    python scripts/plan_randsid.py --dataset sports > logs/randsid/plan_sports.tsv   # also toys

Sports/Toys (2026-11-08): same design on embeddings/<ds>_randsid; tags
`randsid_<ds>_bw_tgtmid_<arm>[_seedN]`, training labels `<ds>_poison_pct1_n1[_seedN]_tgtmid_randsid`.
Beauty output is unchanged.
"""
import argparse
import csv
import os
import sys

import yaml

ARMS = {"finetune": "finetune", "forget_repair_lf0.1": "fr_lf0p1",
        "nau_ls0.1_ln0.0": "sep_ls0p1", "nau_ls0.01_ln0.01": "nau_s0p01_n0p01"}
POS = ("unlearning.algorithm=", "unlearning.neighborhood_aware=",
       "unlearning.neighborhood_aware_sample_rate=", "unlearning.request_batch_size=")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="beauty", choices=("beauty", "sports", "toys"))
    ds = ap.parse_args().dataset
    sid = f"embeddings/{ds}_randsid/merged_predictions_tensor.pt"
    tag_ds = "" if ds == "beauty" else f"{ds}_"
    rows = [r for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
            if r["setting"] == "spam" and r["dataset"] == ds and r["track"] == "tiger+rkmeans"
            and r["stratum"] == "mid" and r["config"] in ARMS]
    assert len(rows) == 20, len(rows)
    for r in sorted(rows, key=lambda r: (list(ARMS).index(r["config"]), int(r["seed"]))):
        sd, tmpl = int(r["seed"]), r["run_dir"]
        assert r["status"] == "ok" and r["has_ckpt"] == "1", r
        ov = yaml.safe_load(open(os.path.join(tmpl, ".hydra", "overrides.yaml")))
        kv = dict(o.split("=", 1) for o in ov)
        assert kv["experiment"] == "tiger_unlearn_unified_sequential", kv["experiment"]
        # some canonical runs (Sports/Toys fig1js) recorded the absolute path of the same file
        assert os.path.relpath(kv["semantic_id_path"].strip("'\"")) == f"embeddings/{ds}/merged_predictions_tensor.pt"
        assert kv.get("num_hierarchies", "4") == "4"
        assert int(kv["seed"]) == sd, (tmpl, kv["seed"])
        assert kv["data_dir"] == f"src/data/amazon_data/{ds}_spam_tgtmid_seed{sd}_pct1_n1", kv["data_dir"]
        extra = [o for o in ov if o.startswith("unlearning.") and not o.startswith(POS)]
        label = f"{ds}_poison_pct1_n1{'' if sd == 2 else f'_seed{sd}'}_tgtmid_randsid"
        tag = f"randsid_{tag_ds}bw_tgtmid_{ARMS[r['config']]}{'' if sd == 2 else f'_seed{sd}'}"
        print("\t".join([ds, "mid", str(sd), str(sd), tag, f"label:{label}",
                         kv["unlearning.algorithm"], sid,
                         kv.get("unlearning.neighborhood_aware", "false"),
                         kv.get("unlearning.request_batch_size", "1"),
                         kv.get("unlearning.neighborhood_aware_sample_rate", "0.0"),
                         kv["data_dir"], tmpl, " ".join(extra)]))


if __name__ == "__main__":
    sys.exit(main())
