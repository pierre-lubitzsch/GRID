"""Build paper/plotting/canonical_runs.tsv: ONE execution per (configuration, cell).

Implements section 3.2 (spam, TIGER + RQ-KMeans; TRACER on Beauty = tracerv3) and 4.2
(unwanted items: the `_mia` runs for the configurations of Tables 3/22/23) of
paper/revision/canonical_runs_plan.md, plus the Figure 1 fix (a): Beauty mid Forget only
(lambda_f=1, lambda_r=0) and Fanchuan t0.07 come from the `_canonck` re-runs, and the 46 cells
that had no validation eval and no checkpoint (Finetune Sports/Toys unp/pop, SCIF outside mid
except Beauty/Toys seed 2) come from the `_canonck` re-runs of run_canonck_missing_launch.sh.

Rule per (config, cell): the first source in the config's list whose run dir exists with a
finished test eval (latest job id if several). A cell whose designated source does not exist
yet (e.g. a queued `_canonck` run) is written with status=pending and an empty run_dir.

Columns:
  setting dataset track config method stratum seed run_dir tag source
  has_test has_val has_ckpt ckpt_resolved status tables n_candidates

  has_ckpt: checkpoints/unlearned.ckpt resolves (symlinks into /sc-projects followed);
  status: ok | pending | dangling_ckpt | missing.

Read-only on run dirs. Light (globs + stat).

    python scripts/build_canonical_runs.py            # writes the TSV, prints a summary
    python scripts/build_canonical_runs.py --out /tmp/x.tsv
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import re
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = "logs/unlearn/runs"
SEEDS = (2, 3, 5, 7, 11)
STRATA = ("unpopular", "mid", "popular")
DATASETS = ("beauty", "sports", "toys")
OUT = "paper/plotting/canonical_runs.tsv"
FIELDS = ["setting", "dataset", "track", "config", "method", "stratum", "seed", "run_dir", "tag",
          "source", "has_test", "has_val", "has_ckpt", "ckpt_resolved", "status", "tables",
          "n_candidates"]

# Validation-selected baseline configuration per dataset (paper/plotting/valsel_selected.csv, v33).
T1_PICK = {"seif": {"beauty": "seif_std0.06", "sports": "seif_std0.06", "toys": "seif_std0.006"},
           "kookmin": {"beauty": "kookmin_ir0.001", "sports": "kookmin_ir0.001", "toys": "kookmin_ir0.0001"}}


def tag_of(ds, track, st, arm, seed):
    pre = "" if ds == "beauty" else f"{ds}_"
    q = "tracerv3_" if track == "tiger+tracerv3" else ""
    return f"{q}{pre}bw_tgt{st}_{arm}" + ("" if seed == 2 else f"_seed{seed}")


def jid(path):
    return int(re.match(r"job(\d+)_", os.path.basename(path)).group(1))


def flags(path):
    t = os.path.isfile(os.path.join(path, "eval/csv/version_0/metrics.csv"))
    v = os.path.isfile(os.path.join(path, "eval_val/csv/version_0/metrics.csv"))
    ck = os.path.join(path, "checkpoints/unlearned.ckpt")
    c = os.path.exists(ck)
    dangling = os.path.islink(ck) and not c
    return t, v, c, (os.path.realpath(ck) if c else ""), dangling


def find(tag):
    c = glob.glob(os.path.join(RUNS, f"job*_bs1_{tag}"))
    done = [p for p in c if flags(p)[0]]
    return (max(done, key=jid) if done else None), len(c)


# ---------------------------------------------------------------- spam configurations
# Each source: (predicate(ds, st, seed) -> bool, arm, source label). First match wins.
ALL = lambda ds, st, sd: True  # noqa: E731


def on(dss=None, sts=None):
    return lambda ds, st, sd: (dss is None or ds in dss) and (sts is None or st in sts)


BMID = on(("beauty",), ("mid",))
SPAM = []   # (config, method, track_fn, sources, tables_fn)


def add(config, method, sources, tables, datasets=DATASETS, track=lambda ds: "tiger+rkmeans"):
    SPAM.append((config, method, sources, tables, datasets, track))


VS = lambda arm: (ALL, arm + "_valsel", "valsel_replay")  # noqa: E731

add("finetune", "Finetune",
    [(on(("beauty",)), "unified_lf0.0_ls0.0_ac0_ln0.0_sepgen_keepck", "keepck"),
     (on(None, ("mid",)), "unified_lf0.0_ls0.0_ac0_ln0.0_sepgen_keepck", "keepck"),
     # Sports/Toys unp/pop: re-runs with checkpoint (run_canonck_missing_launch.sh)
     (on(("sports", "toys"), ("unpopular", "popular")), "unified_abl_lf0.0_ls0.0_ln0.0_canonck", "canonck"),
     (ALL, "unified_abl_lf0.0_ls0.0_ln0.0", "recorded")],
    "T1;T2;T18;T19;Fig1")
add("forget_only_lf1.0", "Forget only",
    [(BMID, "unified_abl_lf1.0_ls0.0_ln0.0_lr0.0_canonck", "canonck"),
     VS("unified_abl_lf1.0_ls0.0_ln0.0_lr0.0")], "T1;T7;Fig1")
add("forget_only_lf0.1", "Forget only", [VS("unified_abl_lf0.1_ls0.0_ln0.0_lr0.0")], "T7")
add("forget_repair_lf0.1", "Forget+Repair",
    [(on(("beauty",)), "unified_lf0.1_ls0.0_ac0_ln0.0_sepgen_keepck", "keepck"),
     (on(None, ("mid",)), "fig1js_forget_repair", "fig1js"),
     (ALL, "forget_repair_keepck", "keepck")],
    "T1;T7;T6(0,0);T2;T18;T19;Fig1;Fig2;Fig3")
add("forget_repair_lf1.0", "Forget+Repair", [VS("unified_abl_lf1.0_ls0.0_ln0.0")], "T7")
# SCIF outside mid: re-runs with checkpoint (run_canonck_missing_launch.sh), except Beauty/Toys seed 2,
# whose July recorded runs kept their checkpoint (validation re-eval instead).
SCIF_CK = lambda ds, st, sd: st != "mid" and not (ds in ("beauty", "toys") and sd == 2)  # noqa: E731
add("scif", "SCIF", [(on(None, ("mid",)), "fig1js_scif", "fig1js"), (SCIF_CK, "scif_canonck", "canonck"),
                     (ALL, "scif", "recorded")], "T1;Fig1")
for std in ("0.06", "0.006"):
    src = [VS(f"seif_std{std}")]
    if std == "0.06":
        src.insert(0, (BMID, "seif_std0.06_keepck", "keepck"))
    add(f"seif_std{std}", "SEIF", src, "T7;T1(if picked);Fig1(B mid, 0.06)")
for ir in ("0.0001", "0.001", "0.01"):
    src = [VS(f"kookmin_ir{ir}")]
    if ir == "0.001":
        src.insert(0, (BMID, "kookmin_ir0.001_keepck", "keepck"))
    add(f"kookmin_ir{ir}", "Kookmin", src, "T7;T1(if picked);Fig1(B mid, 0.001)")
for t in ("0.007", "0.07"):
    src = [VS(f"fanchuan_t{t}")]
    if t == "0.07":
        src.insert(0, (BMID, "fanchuan_t0.07_canonck", "canonck"))
    add(f"fanchuan_t{t}", "Fanchuan", src, "T7;T1(t0.07);Fig1(B mid, t0.07)")
for tr in ("lf1.0_lc0.1", "lf1.0_lc1.0", "lf0.1_lc1.0", "lf0.1_lc0.1"):
    add(f"tracer_{tr}_plr0.001_ep4", "TRACER", [VS(f"tracer_{tr}_plr0.001_ep4")], "T7;T1(lf1.0_lc0.1)",
        track=lambda ds: "tiger+tracerv3" if ds == "beauty" else "tiger+rkmeans")

# NAU (lambda_s, lambda_n) grid at lambda_f = 0.1. (0, 0) is forget_repair_lf0.1 above.
TOYMID = on(("toys",), ("mid",))
NAU = {
    ("0.01", "0.01"): ([(ALL, "nau_s0p01_n0p01_keepck", "keepck")], "T1 NAU;T6;T2;T18;T19;Fig1;Fig2;Fig3"),
    ("0.1", "0.01"): ([(on(None, ("mid",)), "fig1js_nau", "fig1js"),
                       (on(("beauty",)), "unified_lf0.1_ls0.1_ac0_ln0.01_mass_embedding_keepck", "keepck"),
                       (ALL, "nau_keepck", "keepck")], "T1 NAU(ls0.1);T6;T2;T18;T19;Fig3"),
    ("0.1", "0.0"): ([(on(("beauty",)), "unified_lf0.1_ls0.1_ac0_ln0.0_sepgen_keepck", "keepck"),
                      (ALL, "sep_ln0_keepck", "keepck")], "T6;T2;T18;T19;Fig2;Fig3"),
    ("0.01", "0.0"): ([(on(("beauty",)), "sep_ln0p01_keepck", "keepck"), (TOYMID, "sep_ln0p01_keepck", "keepck"),
                       # Sports all, Toys unp/pop: re-runs with checkpoint of the `_valsel` runs
                       # (run_canonck_sep001_launch.sh; SASRec vs TIGER contrast)
                       (lambda ds, st, sd: ds == "sports" or (ds == "toys" and st != "mid"),
                        "nau_s0p01_n0p0_canonck", "canonck")],
                      "T6;T2;T18;T19;Fig2;SASRec-vs-TIGER"),
    ("0.0", "0.01"): ([(on(("beauty",)), "unified_lf0.1_ls0.0_ac0_ln0.01_mass_embedding_keepck", "keepck"),
                       (TOYMID, "nosep_ln0.01_keepck", "keepck")], "T6;T2;T18;T19;Fig2"),
    ("0.0", "0.1"): ([(on(("beauty",)), "nosep_ln0p1_keepck", "keepck"), (TOYMID, "nosep_ln0.1_keepck", "keepck")],
                     "T6;T2;T18;T19;Fig2"),
    ("0.1", "0.1"): ([(on(("beauty",)), "unified_lf0.1_ls0.1_ac0_ln0.1_sepgen_keepck", "keepck"),
                      (TOYMID, "nau_ln0.1_keepck", "keepck")], "T6;T2;T18;T19;Fig2"),
    ("0.01", "0.1"): ([], "T6"),
}
for (ls, ln), (src, tables) in NAU.items():
    valsel = f"valsel_nau_ls{ls.replace('.', 'p')}_ln{ln.replace('.', 'p')}"
    add(f"nau_ls{ls}_ln{ln}", "NAU", src + [(ALL, valsel, "valsel_replay")], tables)


# ---------------------------------------------------------------- candidates (separate setting)
# Forget+Repair settings tied with the lowest validation exposure on the tuned Beauty grid (R2 Q2,
# run_fr_tiebreak_launch.sh): checkpointed re-runs of the recorded `_valsel` grid runs, scored on the
# validation and test neighborhood (paper/plotting/fr_tiebreak_table.py). Not Table 1/7 rows: written
# with setting=spam_frtie_candidate, which every table reader skips (they filter setting == "spam").
# The tied lambda_f=0.1 ep4 setting is forget_repair_lf0.1 above.
CANDIDATES = []
for lf, ep in (("0.3", "2"), ("0.03", "8")):
    CANDIDATES.append((f"frtie_lf{lf}_ep{ep}", "Forget+Repair (tie-break candidate)",
                       [(ALL, f"unified_abl_lf{lf}_ls0.0_ln0.0_ep{ep}_canonck", "canonck")],
                       "R2Q2 F+R tie-break (appendix)", ("beauty",), lambda ds: "tiger+rkmeans"))


def spam_rows(specs=None, setting="spam"):
    rows = []
    for config, method, sources, tables, datasets, track_fn in (SPAM if specs is None else specs):
        for ds in datasets:
            track = track_fn(ds)
            for st in STRATA:
                for sd in SEEDS:
                    srcs = [(a, s) for pred, a, s in sources if pred(ds, st, sd)]
                    row = dict(setting=setting, dataset=ds, track=track, config=config, method=method,
                               stratum=st, seed=sd, tables=tables)
                    chosen = None
                    for k, (arm, label) in enumerate(srcs):
                        tag = tag_of(ds, track, st, arm, sd)
                        path, n = find(tag)
                        if path:
                            chosen = (path, tag, label, n)
                            break
                        if label == "canonck":     # designated but not run yet: do not fall through
                            chosen = (None, tag, label, n)
                            break
                    if chosen is None:
                        row.update(run_dir="", tag=tag_of(ds, track, st, srcs[0][0], sd) if srcs else "",
                                   source=srcs[0][1] if srcs else "", status="missing", n_candidates=0,
                                   has_test=0, has_val=0, has_ckpt=0, ckpt_resolved="")
                    else:
                        path, tag, label, n = chosen
                        row.update(tag=tag, source=label, n_candidates=n)
                        if path is None:
                            row.update(run_dir="", status="pending", has_test=0, has_val=0, has_ckpt=0,
                                       ckpt_resolved="")
                        else:
                            t, v, c, real, dangling = flags(path)
                            row.update(run_dir=path, has_test=int(t), has_val=int(v), has_ckpt=int(c),
                                       ckpt_resolved=real, status="dangling_ckpt" if dangling else "ok")
                    rows.append(row)
    return rows


# ---------------------------------------------------------------- unwanted items (_mia)
MIA_TABLES = {"unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0": "T3;T22;T23;T4(ln=-1);T8(selected)"}


ALGO_NAME = {"scif": "SCIF", "seif": "SEIF", "kookmin": "Kookmin", "fanchuan": "Fanchuan", "tracer": "TRACER"}


def mia_method(path, arm):
    """Display method of a `_mia` run from its RESOLVED config (.hydra/config.yaml), not the tag:
    run_tags.family_of parses `_mia` tags as negtrain_abl, and algorithm=unified covers four
    methods. Finetune: lf=0, lr=1; Forget only: lf>0, lr=0; Forget+Repair: lf>0, lr=1 (ls=ln=0);
    NAU: lf>0, ln<0."""
    import yaml
    u = yaml.safe_load(open(os.path.join(path, ".hydra", "config.yaml")))["unlearning"]
    algo = u["algorithm"]
    if algo != "unified":
        assert arm.startswith(algo), (path, algo)
        return ALGO_NAME[algo]
    lf, ls, ln = float(u["lambda_f"]), float(u["lambda_s"]), float(u["lambda_n"])
    lr = float(u.get("lambda_r", 1.0))   # absent in older runs = code default 1.0
    if ls == 0 and ln == 0:
        m = "Finetune" if lf == 0 and lr == 1 else "Forget only" if lf > 0 and lr == 0 else \
            "Forget+Repair" if lf > 0 and lr == 1 else None
    elif lf > 0 and ls == 0 and ln < 0 and lr == 1:
        m = "NAU"
    else:
        m = None
    assert m, (path, lf, ls, ln, lr)
    return m


def mia_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(RUNS, "job*_bs1_sens_*_mia*"))):
        tag = re.match(r"job\d+_.*_bs1_(.+)$", os.path.basename(path)).group(1)
        m = re.match(r"sens_(beauty|sports|toys)_([a-z]+)_(.+)_mia(?:_seed(\d+))?$", tag)
        if not m:
            print(f"  unparsed _mia tag: {tag}", file=sys.stderr)
            continue
        ds, cat, arm, sd = m.group(1), m.group(2), m.group(3), int(m.group(4) or 2)
        t, v, c, real, dangling = flags(path)
        rows.append(dict(setting="unwanted", dataset=ds, track="tiger+rkmeans", config=arm,
                         method=mia_method(path, arm), stratum=cat, seed=sd, run_dir=path, tag=tag,
                         source="mia", has_test=int(t), has_val=int(v), has_ckpt=int(c), ckpt_resolved=real,
                         status="dangling_ckpt" if dangling else "ok",
                         tables=MIA_TABLES.get(arm, "T3;T22;T23;T8(selected)"), n_candidates=1))
    # duplicates of one (config, cell) would break "one execution": keep the latest, report
    key = lambda r: (r["dataset"], r["config"], r["stratum"], r["seed"])  # noqa: E731
    cnt = Counter(key(r) for r in rows)
    for k, n in cnt.items():
        if n > 1:
            print(f"  WARNING: {n} _mia runs for {k}; keeping the latest", file=sys.stderr)
    best = {}
    for r in rows:
        if key(r) not in best or jid(r["run_dir"]) > jid(best[key(r)]["run_dir"]):
            best[key(r)] = r
    return list(best.values())


# ---------------------------------------------------------------- unwanted items (_canonck)
# Protocol A (cleaned test histories, paper/revision/sensitive_cleaned_protocol.md): rows shown in
# Table 4 / Table 8 that had no checkpointed execution, re-run with checkpoint by
# run_canonck_sens_launch.sh (plan logs/canonck_sens/plan.tsv). RQ-KMeans NAU lambda_n=0 / -10 replay
# the `_valsens` run of the cell and replace it in Table 8 and Table 4; LETTER NAU lambda_n=0 / -1 /
# -10 replace the recorded LETTER runs of Table 4. Separate setting `unwanted_canonck`: the `_mia`
# readers (setting == "unwanted") are unchanged; the --view readers read both.
# plan_t8.tsv (run_canonck_sens_t8_launch.sh): the other non-selected Table 8 candidates (SCIF / SEIF /
# Kookmin / Fanchuan / TRACER alternatives, NAU lambda_n=-0.1), replacing their `_valsens` replays.
CANONCK_SENS_PLANS = ("logs/canonck_sens/plan.tsv", "logs/canonck_sens/plan_t8.tsv")


def canonck_sens_rows():
    rows = []
    for plan in CANONCK_SENS_PLANS:
        if os.path.isfile(plan):
            rows += _canonck_sens_rows(plan, "T8(candidate)" if plan.endswith("_t8.tsv") else None)
    return rows


def _canonck_sens_rows(plan, tables):
    rows = []
    for f in csv.reader(open(plan), delimiter="\t"):
        model, ds, sd, tag, algo = f[0], f[1], int(f[3]), f[5], f[7]
        m = re.match(r"(?:letter_letter_)?sens_(beauty|sports|toys)_([a-z]+)_(.+)_canonck(?:_seed\d+)?$", tag)
        assert m and m.group(1) == ds, tag
        cat, arm = m.group(2), m.group(3)
        track = "tiger+rkmeans" if model == "tiger" else "letter+letter"
        path, n = find(tag)
        base = dict(setting="unwanted_canonck", dataset=ds, track=track, config=arm,
                    method=ALGO_NAME.get(algo, "NAU"), stratum=cat, seed=sd, tag=tag, source="canonck",
                    n_candidates=n,
                    tables=tables or ("T4;T8(candidate)" if model == "tiger" else "T4(LETTER)"))
        if path is None:
            rows.append(dict(base, run_dir="", has_test=0, has_val=0, has_ckpt=0, ckpt_resolved="",
                             status="pending"))
            continue
        t, v, c, real, dangling = flags(path)
        rows.append(dict(base, run_dir=path, has_test=int(t), has_val=int(v), has_ckpt=int(c), ckpt_resolved=real,
                         status="dangling_ckpt" if dangling else "ok"))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=OUT)
    args = ap.parse_args()
    os.chdir(ROOT)
    rows = spam_rows() + mia_rows() + spam_rows(CANDIDATES, "spam_frtie_candidate") + canonck_sens_rows()
    with open(args.out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {args.out}: {len(rows)} rows")
    c = Counter((r["setting"], r["status"]) for r in rows)
    for k, n in sorted(c.items()):
        print(f"  {k[0]:9s} {k[1]:14s} {n}")
    need = [r for r in rows if r["status"] == "ok" and r["has_ckpt"] and not r["has_val"]]
    print(f"  checkpoint without eval_val (validation re-eval needed): {len(need)}")
    for k, n in sorted(Counter((r["dataset"], r["config"], r["source"]) for r in need).items()):
        print(f"    {k[0]:7s} {k[1]:28s} {k[2]:8s} {n}")
    nov = [r for r in rows if r["status"] == "ok" and not r["has_val"] and not r["has_ckpt"]]
    print(f"  no eval_val and no checkpoint (cannot get val without a re-run): {len(nov)}")
    for k, n in sorted(Counter((r["dataset"], r["config"], r["source"]) for r in nov).items()):
        print(f"    {k[0]:7s} {k[1]:28s} {k[2]:8s} {n}")
    for r in rows:
        if r["status"] in ("missing", "dangling_ckpt", "pending"):
            print(f"  {r['status']}: {r['dataset']} {r['config']} {r['stratum']} s{r['seed']} {r['tag']} {r['run_dir']}")


if __name__ == "__main__":
    main()
