"""Plan: paper runs that depend on the retraining-matched cleaned-validation selection (2026-11-14).

Reads paper/revision/rmsplit_selected.csv (paper/plotting/critique_followup/rmsplit_selection.py).

  anchor    if NAU's Beauty-selected setting is not lambda_s=0, lambda_n=-1: the retain-side anchor runs
            (lambda_a in {0.1, 1}) at the selected setting, Beauty/Sports/Toys x 5 seeds. Template: that seed's
            selected NAU grid row (plan_grid_wave1.tsv) + unlearning.lambda_a; tag
            sens_<ds>_<cat>rm_<NAU arm>_anchor{0p1,1p0}_rmsplit_canonck[_seedN].
  takedown  Beauty brand takedown rows for every Beauty-selected configuration that is not already in
            plan_extra_takedown.tsv. Template: the same algorithm's row of plan_extra_takedown.tsv with only the
            hyperparameter override replaced (SCIF damping, SEIF seif_erase_std, Kookmin kookmin_init_rate,
            Fanchuan fanchuan_contrastive_temperature, Forget only / Forget+Repair lambda_f). NAU: the takedown
            Forget+Repair row (unified_abl_lf0.1_ls0.0_ln0.0) + the selected lambda_s / lambda_n and the generative
            separation score (sep_checkpoint=true when lambda_s > 0; coherence_* / neighborhood_count=8 when lambda_n != 0), as the grid NAU rows add them to Forget+Repair.
            Tag sens_beauty_takedown_elfrm_<arm>_rmsplit_canonck[_seedN].

    python scripts/plan_rmgrid_followup.py -> logs/rmsplit_sens/plan_followup_{anchor,takedown}.tsv
"""
import csv
import re

import os
import sys
SEL = sys.argv[1] if len(sys.argv) > 1 else "paper/revision/rmsplit_selected.csv"
CAT = {"beauty": "hairloss", "sports": "guns", "toys": "toyweapons"}
D = "logs/rmsplit_sens"
OUTD = os.environ.get("OUTD", D)
HP = {"scif": ("scif_d", "unlearning.damping"), "seif": ("seif_std", "unlearning.seif_erase_std"),
      "kookmin": ("kookmin_ir", "unlearning.kookmin_init_rate"),
      "fanchuan": ("fanchuan_t", "unlearning.fanchuan_contrastive_temperature")}
TD_ARM = {"Finetune": "unified_abl_lf0.0_ls0.0_ln0.0", "Forget only": "unified_abl_lf{lf}_ls0.0_ln0.0_lr0.0",
          "Forget+Repair": "unified_abl_lf{lf}_ls0.0_ln0.0"}


def rows(p):
    return [l.rstrip("\n").split("\t") for l in open(p) if l.strip()]


def set_kv(extra, key, val):
    toks = extra.split(" ")
    hit = [i for i, t in enumerate(toks) if t.startswith(key + "=")]
    if hit:
        toks[hit[0]] = f"{key}={val}"
    else:
        toks.append(f"{key}={val}")
    return " ".join(toks)


def main():
    sel = {(r["dataset"], r["method"]): r["new_config"] for r in csv.DictReader(open(SEL))}
    only = [x for x in os.environ.get("ONLY", "").split(",") if x]     # e.g. ONLY=Kookmin,Fanchuan (second pass)
    if only:
        sel = {k: v for k, v in sel.items() if k[1] in only}
    nau = sel.get(("beauty", "NAU"), "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0")   # absent with ONLY: no anchor rows
    out = {"anchor": [], "takedown": []}
    # anchor
    if nau != "unified_hp2_sepgen_lf0.1_ls0.0_lnneg1.0":
        grid = rows(f"{D}/plan_grid_wave1.tsv")
        for ds, cat in CAT.items():
            for r in grid:
                m = re.match(rf"sens_{ds}_{cat}rm_{re.escape(nau)}_rmsplit_canonck(_seed\d+)?$", r[5])
                if not m:
                    continue
                for lab, la in (("0p1", "0.1"), ("1p0", "1.0")):
                    n = list(r)
                    n[5] = f"sens_{ds}_{cat}rm_{nau}_anchor{lab}_rmsplit_canonck{m.group(1) or ''}"
                    n[14] = set_kv(r[14], "unlearning.lambda_a", la)
                    out["anchor"].append(n)
    # takedown
    td = rows(f"{D}/plan_extra_takedown.tsv")
    have = {re.match(r"sens_beauty_takedown_elfrm_(.*)_rmsplit_canonck(?:_seed\d+)?$", r[5]).group(1) for r in td}

    def tmpl(arm):
        return [r for r in td if re.match(rf"sens_beauty_takedown_elfrm_{re.escape(arm)}_rmsplit_canonck(?:_seed\d+)?$", r[5])]

    def add(src_arm, new_arm, kvs):
        if new_arm in have:
            return
        src = tmpl(src_arm)
        assert len(src) == 5, (src_arm, len(src))
        for r in src:
            n = list(r)
            n[5] = r[5].replace(f"_elfrm_{src_arm}_", f"_elfrm_{new_arm}_")
            for k, v in kvs:
                n[14] = set_kv(n[14], k, v)
            out["takedown"].append(n)
        have.add(new_arm)

    for (ds, m), cfg in sel.items():
        if ds != "beauty":
            continue
        fam = cfg.split("_")[0]
        if fam in HP:
            pre, key = HP[fam]
            val = cfg[len(pre):]
            old = next(a for a in have if a.startswith(pre))
            add(old, cfg, [(key, val)])
        elif m in TD_ARM:
            lf = re.search(r"_lf([0-9.]+)", cfg).group(1)
            new = TD_ARM[m].format(lf=lf)
            src = "unified_abl_lf0.1_ls0.0_ln0.0_lr0.0" if m == "Forget only" else "unified_abl_lf0.1_ls0.0_ln0.0"
            add(src, new, [("unlearning.lambda_f", lf)])
        elif m == "NAU":
            mm = re.match(r"unified_hp2_sepgen_lf0\.1_ls([0-9.]+)_ln(neg)?([0-9.]+)", cfg)
            ls, ln = mm.group(1), ("-" if mm.group(2) else "") + mm.group(3)
            kvs = [("unlearning.lambda_s", ls), ("unlearning.lambda_n", ln),
                   ("unlearning.sep_loss_type", "generative"), ("unlearning.sep_gen_temperature", "1.0")]
            if float(ln) != 0:      # the neighbourhood term's settings, as in the grid NAU rows and frbase
                kvs += [("unlearning.coherence_loss_type", "mass"), ("unlearning.coherence_neighbor_method", "embedding"),
                        ("unlearning.coherence_rows", "target_only"), ("unlearning.coherence_embedding_metric", "cosine"),
                        ("unlearning.neighborhood_count", "8")]
            if float(ls) > 0:
                kvs.append(("unlearning.sep_checkpoint", "true"))
            add("unified_abl_lf0.1_ls0.0_ln0.0", cfg, kvs)
    for k, rr in out.items():
        p = f"{OUTD}/plan_followup_{k}{os.environ.get('SFX', '')}.tsv"
        with open(p, "w") as fh:
            for r in rr:
                fh.write("\t".join(r) + "\n")
        print(p, len(rr), "rows:", sorted({re.sub(r"_seed\d+$", "", r[5]) for r in rr}))


if __name__ == "__main__":
    main()
