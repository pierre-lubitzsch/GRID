"""Plan: sensitive item deletion with the RETRAINING-MATCHED split for the OTHER SID tokenizers
(RQ-VAE, LETTER, tracerv3), at the configurations pinned for TIGER + RQ-KMeans (2026-11-17).

NOT for the current paper; kept for later. No selection is done here: the configurations are the
per-dataset picks of paper/revision/rmsplit_selected.csv (9 methods; Beauty has no TRACER pick there,
so it uses the Sports/Toys pick tracer_lf1.0_lc1.0_ep4), plus NAU's lambda_n ablation at lambda_s=0
(lambda_n = 0 and -1; -10 is the pinned NAU itself) -> 11 configurations per (dataset, seed).

Each row CLONES the recorded overrides of the finished TIGER + RQ-KMeans rmsplit run of the same
(dataset, configuration, seed) (logs/unlearn/runs/job*_bs1_sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN],
.hydra/overrides.yaml), and swaps ONLY
    ckpt_path               -> the tokenizer's clean model of that seed (last.ckpt; the original model,
                               since <ds>_sens_*/training is a symlink to <ds>/training)
    semantic_id_path        -> the tokenizer's SID tensor
    unlearning.tracer_codebook_ckpt -> the tokenizer's codebook (TRACER rows only)
    run tag                 -> <track prefix>sens_<ds>_<cat>rm_<arm>_rmsplit_canonck[_seedN]
Beauty TRACER has no TIGER template (TRACER never ran on Beauty's RQ-KMeans space): it takes the
Beauty SCIF rmsplit run of the same seed for every shared key and the Sports TRACER rmsplit run of
the same seed for the TRACER keys (n_epochs, lambdas), with the Beauty codebook.

Tracks (model column = MODEL env of the launcher):
    rqvae     MODEL=tiger   beauty/sports/toys   tag prefix rqvae_
    letter    MODEL=letter  beauty/sports/toys   tag prefix letter_letter_  (the runner keeps it as is)
    tracerv3  MODEL=tiger   beauty               tag prefix tracerv3_  (clean models beauty_clean[_seedN]_tracerv3)

    python scripts/plan_rmsplit_othertok.py [--tracks rqvae letter tracerv3]
      -> logs/rmsplit_sens/plan_othertok_<track>.tsv  (same 15 columns as plan_grid_wave*.tsv)
Refuses to write a plan file that already exists unless --overwrite-own (only these new files).
"""
import argparse
import csv
import glob
import json
import os
import re

import yaml

CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
SEEDS = (2, 3, 5, 7, 11)
FIXED = ("experiment", "data_dir", "semantic_id_path", "ckpt_path", "num_hierarchies", "seed",
         "unlearning.algorithm", "unlearning.neighborhood_aware", "unlearning.neighborhood_aware_sample_rate",
         "unlearning.request_batch_size", "unlearning_run_tag")
SELECTED = "paper/revision/rmsplit_selected.csv"
BEAUTY_TRACER = "tracer_lf1.0_lc1.0_ep4"
NAU_ABL = ("unified_hp2_sepgen_lf0.1_ls0.0_ln0.0", "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0")

TRACKS = {
    "rqvae": dict(model="tiger", prefix="rqvae_", datasets=("beauty", "sports", "toys"),
                  sid="embeddings/{ds}_rqvae/merged_predictions_tensor.pt",
                  clean="{ds}_clean{st}_rqvae", retrain="{ds}_clean{st}_rqvae_retrain_{cat}_1e-4",
                  codebook={"beauty": "logs/train/runs/codebook/beauty_rqvae_10451291/checkpoints/checkpoint_000_003000.ckpt",
                            "sports": "logs/train/runs/codebook/sports_rqvae_10462483/checkpoints/checkpoint_000_003000.ckpt",
                            "toys": "logs/train/runs/codebook/toys_rqvae_10462482/checkpoints/checkpoint_000_003000.ckpt"},
                  reftok="rqvae", experiment="tiger_train_flat", quantizer="rqvae"),
    "letter": dict(model="letter", prefix="letter_letter_", datasets=("beauty", "sports", "toys"),
                   sid="embeddings/{ds}_letter/merged_predictions_tensor.pt",
                   clean="letter_{ds}_clean{st}", retrain="letter_{ds}_clean{st}_retrain_{cat}_1e-4",
                   codebook={"beauty": "logs/train/runs/codebook/beauty_letter_10567247/checkpoints/checkpoint_000_060000.ckpt",
                             "sports": "logs/train/runs/codebook/sports_letter_10567248/checkpoints/checkpoint_000_090000.ckpt",
                             "toys": "logs/train/runs/codebook/toys_letter_10567249/checkpoints/checkpoint_000_060000.ckpt"},
                   reftok="letter_letter", experiment="letter_train_flat", quantizer="letter"),
    "tracerv3": dict(model="tiger", prefix="tracerv3_", datasets=("beauty",),
                     sid="embeddings/beauty_tracer_v3/merged_predictions_tensor.pt",
                     clean="{ds}_clean{st}_tracerv3", retrain="{ds}_clean{st}_tracerv3_retrain_{cat}_1e-4",
                     codebook={"beauty": "logs/train/runs/codebook/beauty_rkmeans_10613317/checkpoints/checkpoint_000_000600.ckpt"},
                     reftok="tracerv3", experiment="tiger_train_flat", quantizer="tracerv3"),
}


def sfx(sd):
    return "" if sd == 2 else f"_seed{sd}"


def train_dir(label, required=True):
    """The ONE training run dir whose label is exactly `label`."""
    pat = re.compile(r"^\d{2}-\d{2}-\d{2}_job\d+_" + re.escape(label) + r"$")
    c = [d for d in glob.glob("logs/train/runs/*/*") if pat.match(os.path.basename(d))]
    if not c:
        if required:
            raise SystemExit(f"no training run labelled {label}")
        return None
    assert len(c) == 1, (label, c)
    return c[0]


def clean_ckpt(track, ds, sd):
    t = TRACKS[track]
    d = train_dir(t["clean"].format(ds=ds, st=sfx(sd), cat=CAT[ds]))
    ck = os.path.join(d, "checkpoints", "last.ckpt")
    assert os.path.isfile(ck), ck
    return ck


def retrain_ckpt(track, ds, sd, required=True):
    """best-epoch checkpoint of the retrained model (save_top_k=1: exactly one checkpoint_epoch=*.ckpt)."""
    t = TRACKS[track]
    d = train_dir(t["retrain"].format(ds=ds, st=sfx(sd), cat=CAT[ds]), required)
    if d is None:
        return None
    # never plan a reference against a still-training run (save_top_k rotates the epoch checkpoint)
    import subprocess
    jid = re.search(r"_job(\d+)_", os.path.basename(d)).group(1)
    q = subprocess.run(["squeue", "-h", "-j", jid], capture_output=True, text=True).stdout.strip()
    if q:
        if required:
            raise SystemExit(f"retrain job {jid} still queued/running: {d}")
        return None
    c = sorted(glob.glob(os.path.join(d, "checkpoints", "checkpoint_epoch=*.ckpt")))
    if not (len(c) == 1 and os.path.isfile(os.path.join(d, "checkpoints", "last.ckpt"))):
        if required:
            raise SystemExit(f"retrain run not finished: {d} {c}")
        return None
    return c[0]


def arms_of(ds):
    out = []
    for r in csv.DictReader(open(SELECTED)):
        if r["dataset"] == ds:
            out.append(r["new_config"])
    if ds == "beauty":
        assert not any(a.startswith("tracer_") for a in out)
        out.append(BEAUTY_TRACER)
    out += [a for a in NAU_ABL if a not in out]
    assert len(out) == len(set(out)) == 11, (ds, out)
    return out


def tiger_run(tag):
    """latest finished TIGER rmsplit run dir of a tag (test eval metrics present)."""
    c = [d for d in glob.glob(f"logs/unlearn/runs/job*_bs1_{tag}")
         if os.path.isfile(os.path.join(d, "eval/csv/version_0/metrics.csv"))]
    if not c:
        raise SystemExit(f"no finished TIGER template for {tag}")
    return max(c, key=lambda d: int(re.match(r"job(\d+)_", os.path.basename(d)).group(1)))


def overrides(run):
    ov = yaml.safe_load(open(os.path.join(run, ".hydra", "overrides.yaml")))
    head = [o.split("=", 1)[0] for o in ov[:len(FIXED)]]
    assert tuple(head) == FIXED, (run, head)
    kv = dict(o.split("=", 1) for o in ov[:len(FIXED)])
    extra = [o for o in ov[len(FIXED):] if not o.startswith("unlearning.checkpoint_fractions=")]
    return kv, extra


def template(ds, arm, sd):
    """-> (kv, extra list, template description), Beauty TRACER synthesized."""
    cat = CAT[ds]
    if ds == "beauty" and arm.startswith("tracer_"):
        b = tiger_run(f"sens_beauty_hairlossrm_scif_d0.1_rmsplit_canonck{sfx(sd)}")
        s = tiger_run(f"sens_sports_gunsrm_{arm}_rmsplit_canonck{sfx(sd)}")
        bkv, bex = overrides(b)
        skv, sex = overrides(s)
        bd = dict(o.split("=", 1) for o in bex)
        new = []
        for o in sex:
            k, v = o.split("=", 1)
            if k in bd:                       # shared key: Beauty's value
                v = bd[k]
            elif k == "unlearning.tracer_codebook_ckpt":
                v = "<CODEBOOK>"              # replaced by the track's codebook below
            new.append(f"{k}={v}")
        kv = dict(skv)
        kv.update({k: bkv[k] for k in ("data_dir", "seed")})
        assert bkv["seed"] == skv["seed"] == str(sd)
        return kv, new, f"{b} + TRACER keys of {s}"
    run = tiger_run(f"sens_{ds}_{cat}rm_{arm}_rmsplit_canonck{sfx(sd)}")
    kv, extra = overrides(run)
    return kv, extra, run


def plan_track(track):
    t = TRACKS[track]
    rows = []
    for ds in t["datasets"]:
        cat = CAT[ds]
        sid = t["sid"].format(ds=ds)
        assert os.path.isfile(sid), sid
        cb = t["codebook"][ds]
        assert os.path.isfile(cb), cb
        for arm in arms_of(ds):
            for sd in SEEDS:
                kv, extra, tmpl = template(ds, arm, sd)
                lam = dict(o.split("=", 1) for o in extra)
                sens = f"src/data/amazon_data/{ds}_sens_{cat}_1e-4_seed{sd}"
                man = f"src/data/amazon_data/{ds}/forget_manifest_{cat}_1e-4_seed{sd}.json"
                retr = f"src/data/amazon_data/{ds}_retrain_{cat}_1e-4_seed{sd}"
                # the corrected protocol, verified on the template
                assert int(kv["seed"]) == sd and kv["data_dir"] == sens, tmpl
                assert lam["spam_forget_manifest"] == man, tmpl
                assert lam["unlearning.deletion_spec"] == "item_pairs", tmpl
                assert lam["unlearning.include_context_rows"] == "false", tmpl
                assert lam["unlearning.split_mode"] == "retrain_matched", tmpl
                assert lam["unlearning.split_retain_dir"] == f"{retr}/training", tmpl
                assert os.path.isdir(f"{retr}/training"), retr
                assert (kv["unlearning.algorithm"] == "tracer") == arm.startswith("tracer_"), tmpl
                assert (kv["unlearning.algorithm"] == "tracer") == ("unlearning.tracer_codebook_ckpt" in lam), tmpl
                new = []
                for o in extra:
                    k, v = o.split("=", 1)
                    if k == "unlearning.tracer_codebook_ckpt":
                        v = cb
                    new.append(f"{k}={v}")
                ckpt = clean_ckpt(track, ds, sd)
                tag = f"{t['prefix']}sens_{ds}_{cat}rm_{arm}_rmsplit_canonck{sfx(sd)}"
                assert not glob.glob(f"logs/unlearn/runs/*_bs1_{tag}"), f"tag already used: {tag}"
                rows.append([t["model"], ds, "sens", str(sd), kv["seed"], tag, ckpt, kv["unlearning.algorithm"], sid,
                             kv["unlearning.neighborhood_aware"], kv["unlearning.request_batch_size"],
                             kv["unlearning.neighborhood_aware_sample_rate"], sens, tmpl, " ".join(new)])
    assert len({r[5] for r in rows}) == len(rows)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tracks", nargs="+", default=list(TRACKS))
    ap.add_argument("--overwrite-own", action="store_true")
    a = ap.parse_args()
    for track in a.tracks:
        rows = plan_track(track)
        p = f"logs/rmsplit_sens/plan_othertok_{track}.tsv"
        if os.path.exists(p) and not a.overwrite_own:
            raise SystemExit(f"{p} exists (pass --overwrite-own to rewrite this campaign's own plan)")
        with open(p, "w") as fh:
            for r in rows:
                fh.write("\t".join(r) + "\n")
        print(p, len(rows), "rows")


if __name__ == "__main__":
    main()
