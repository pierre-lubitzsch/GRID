"""Random-SID control: does the neighborhood damage follow shared codes or semantics?

TIGER on Beauty (mid-popularity target, data seeds 2,3,5,7,11) trained once with the paper's
RQ-KMeans SIDs and once with randomly permuted SIDs (embeddings/beauty_randsid: same prefix-tree
statistics, item-to-tuple assignment random). Four unlearning arms per model, replaying the
canonical runs: Finetune, Forget+Repair (lambda_f=0.1), +L_sep (lambda_s=0.1), NAU
(lambda_s=0.01, lambda_n=0.01, k=8 content-embedding neighbors).

Every held-out row is scored once per checkpoint (run_randsid_eval_seed.sh, neighbor_def=none),
so all neighborhoods are evaluated on the same per-query arrays:
  sem_pfx   next item shares the target's first token under the ORIGINAL RQ-KMeans SIDs
            (the paper's Figure 2 neighborhood; semantically close items)
  sem_knn32 next item among the 32 items closest to the target by content embedding
  rand_pfx  next item shares the target's first token under the RANDOM SIDs
            (code-mates of the target in the random-SID model, semantically arbitrary)
  rest      none of the above (and not the target)
For the RQ-KMeans model, sem_pfx is its code neighborhood and rand_pfx a random item set; for
the random-SID model it is the other way round.

Contrasts (paired per query, pooled over the five seeds, 3000-sample percentile bootstrap,
Appendix G): forget term = F+R - Finetune; separation term = +L_sep - F+R; NAU = NAU - F+R;
in points of the retrained model's NDCG@10 on that neighborhood, and in nats of log p(next item).

    PYTHONPATH=. python paper/plotting/random_sid_table.py [--allow-partial]
    PYTHONPATH=. python paper/plotting/random_sid_table.py --dataset sports   # also toys
    PYTHONPATH=. python paper/plotting/random_sid_table.py --dataset all      # pooled 15 cells

--dataset sports|toys: the same design on embeddings/<ds>_randsid (2026-11-08), outputs
paper/revision/tables/random_sid_control_<ds>.tex and paper/plotting/random_sid_control_<ds>.csv.
--dataset all pools the per-query differences of all three datasets (15 cells); the scale is the
pooled retrained NDCG@10 of the neighborhood, as within one dataset. Beauty (default) unchanged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import statistics as St

import numpy as np
import torch

SEEDS = (2, 3, 5, 7, 11)
ARMS = ("finetune", "forget_repair", "sep_ls0p1", "nau")
ARM_NAME = {"finetune": "Finetune", "forget_repair": "Forget+Repair", "sep_ls0p1": "+L_sep (ls0.1)",
            "nau": "NAU", "clean": "Retrained", "poison": "Poisoned"}
TEX_NAME = {"sep_ls0p1": r"+$\mathcal{L}_{\mathrm{sep}}$ ($\lambda_s=0.1$)"}
CONTRASTS = (("forget term", "forget_repair", "finetune"), ("separation term", "sep_ls0p1", "forget_repair"),
             ("NAU vs F+R", "nau", "forget_repair"), ("+L_sep vs Finetune", "sep_ls0p1", "finetune"))
NBHDS = (("sem_pfx", "first token, RQ-KMeans SIDs"), ("sem_knn32", "32 nearest by content"),
         ("rand_pfx", "first token, random SIDs"), ("rest", "all other cases"))
MODELS = (("rqkmeans", "randsid_scope_orig"), ("random", "randsid_scope"))
DATASETS = ("beauty", "sports", "toys")
DS_NAME = {"beauty": "Beauty", "sports": "Sports", "toys": "Toys"}
EMB = {"beauty": "embeddings/beauty_merged_predictions_tensor_latest.pt",
       "sports": "logs/inference/runs/embeddings/sports_10321537/pickle/merged_predictions_tensor.pt",
       "toys": "logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"}


def orig_sid(ds):
    return f"embeddings/{ds}/merged_predictions_tensor.pt"


def rand_sid(ds):
    return f"embeddings/{ds}_randsid/merged_predictions_tensor.pt"


def load_emb(ds):
    obj = torch.load(EMB[ds], map_location="cpu")
    if isinstance(obj, dict):   # Sports/Toys: {embeddings, item_ids} with item_ids = 0..N-1
        assert torch.equal(obj["item_ids"], torch.arange(len(obj["item_ids"])))
        obj = obj["embeddings"]
    return torch.nn.functional.normalize(obj.float(), dim=1)


POOLED = False   # set by main() for --dataset all


def cell_key(ds, sd):
    """Beauty alone keeps plain seed keys (unchanged output); otherwise (ds, seed)."""
    return sd if ds == "beauty" and not POOLED else (ds, sd)


def neighborhoods(target: int, orig: np.ndarray, rand: np.ndarray, emb: torch.Tensor) -> dict:
    sims = emb @ emb[target]
    sims[target] = float("-inf")
    knn = np.zeros(orig.shape[1], bool)
    knn[torch.topk(sims, 32).indices.numpy()] = True
    sem = orig[0] == orig[0, target]
    rnd = rand[0] == rand[0, target]
    for m in (sem, rnd, knn):
        m[target] = False
    return {"sem_pfx": sem, "sem_knn32": knn, "rand_pfx": rnd, "rest": ~(sem | knn | rnd)}


def load(ds: str, stem: str, split: str, nb_cache: dict, sids, emb, allow_partial: bool):
    """{seed: {"items": array, "target": t, arm: {"ndcg": arr, "logp": arr}}}"""
    out = {}
    s = stem if split == "test" else stem + "_val"
    for sd in SEEDS:
        f = f"logs/diagnose/{s}/{ds}_seed{sd}.json"
        if not os.path.isfile(f):
            if allow_partial:
                continue
            raise FileNotFoundError(f)
        d = json.load(open(f))
        items = np.asarray(d["items_per_query"]["unrelated"])
        cell = {"items": items, "target": d["target_item"],
                "n_target_rows": d["population_counts"].get("target", 0)}
        for r in d["rows"]:
            if r["group"] == "unrelated":
                cell[r["method"]] = {"ndcg": np.asarray(r["ndcg_per_query"], float),
                                     "logp": np.asarray(r["label_logprob_per_query"], float)}
                assert len(cell[r["method"]]["ndcg"]) == len(items)
        key = (ds, d["target_item"])
        if key not in nb_cache:
            nb_cache[key] = neighborhoods(d["target_item"], *sids, emb)
        cell["masks"] = {k: v[items] for k, v in nb_cache[key].items()}
        out[cell_key(ds, sd)] = cell
    return out


def pooled(cells: dict, arm: str, nb: str, field: str) -> np.ndarray:
    return np.concatenate([c[arm][field][c["masks"][nb]] for c in cells.values()]) if cells else np.zeros(0)


def boot(d: np.ndarray, n_boot: int = 3000) -> tuple[float, float]:
    rng = np.random.default_rng(0)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return float(lo), float(hi)


def whole_set(ds: str, model: str, split: str) -> dict:
    """Mean over seeds of |SH - SH_retr| (%) and NDCG / NDCG_retr (%) on the whole split."""
    def last(p):
        val = {}
        for row in csv.DictReader(open(p)):
            for k in ("test/SH@10", "test/ndcg@10"):
                if row.get(k) not in ("", None):
                    val[k] = float(row[k])
        return val["test/SH@10"], val["test/ndcg@10"]
    res = {a: [] for a in ARMS + ("poison",)}
    refs = {}
    rp = "" if ds == "beauty" else f"{ds}_"     # random-SID reference dirs and unlearning tags
    for sd in SEEDS:
        sfx = "" if sd == 2 else f"_seed{sd}"
        try:
            if model == "random":
                cl = last(f"logs/diagnose/randsid_refs/{rp}s{sd}_clean_{split}/csv/version_0/metrics.csv")
                po = last(f"logs/diagnose/randsid_refs/{rp}s{sd}_poison_{split}/csv/version_0/metrics.csv")
                runs = {a: glob.glob(f"logs/unlearn/runs/job*_bs1_randsid_{rp}bw_tgtmid_{t}{sfx}")
                        for a, t in (("finetune", "finetune"), ("forget_repair", "fr_lf0p1"),
                                     ("sep_ls0p1", "sep_ls0p1"), ("nau", "nau_s0p01_n0p01"))}
            else:
                if split == "test":
                    ref = json.load(open(f"paper/data/spam_references/{ds}{sfx}.json"))
                    cl = (ref["clean_SH@10"]["mid"], ref["clean_ndcg@10"])
                else:
                    cl = last(f"logs/diagnose/valsel_refs/rkmeans_{ds}_mid_s{sd}_clean/csv/version_0/metrics.csv")
                po = None
                led = {r["config"]: r["run_dir"] for r in csv.DictReader(open("paper/plotting/canonical_runs.tsv"), delimiter="\t")
                       if r["setting"] == "spam" and r["dataset"] == ds and r["track"] == "tiger+rkmeans"
                       and r["stratum"] == "mid" and int(r["seed"]) == sd}
                runs = {a: [led[c]] for a, c in (("finetune", "finetune"), ("forget_repair", "forget_repair_lf0.1"),
                                                 ("sep_ls0p1", "nau_ls0.1_ln0.0"), ("nau", "nau_ls0.01_ln0.01"))}
        except (FileNotFoundError, KeyError):
            continue
        refs[cell_key(ds, sd)] = cl
        if po:
            res["poison"].append((abs(po[0] - cl[0]), po[1] / cl[1]))
        for a, dirs in runs.items():
            if len(dirs) != 1:
                continue
            p = os.path.join(dirs[0], "eval" if split == "test" else "eval_val", "csv/version_0/metrics.csv")
            if os.path.isfile(p):
                v = last(p)
                res[a].append((abs(v[0] - cl[0]), v[1] / cl[1]))
    out = {a: v for a, v in res.items() if v}
    out["_clean_ndcg"] = {sd: v[1] for sd, v in refs.items()}
    return out


def summarize(raw: dict) -> dict:
    out = {a: (100 * St.mean(x[0] for x in v), 100 * St.mean(x[1] for x in v), len(v))
           for a, v in raw.items() if not a.startswith("_")}
    out["_clean_ndcg"] = raw["_clean_ndcg"]
    return out


def fmt(g, lo, hi, nd=1):
    return f"{g:+.{nd}f} [{lo:+.{nd}f}, {hi:+.{nd}f}]"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-partial", action="store_true")
    ap.add_argument("--dataset", default="beauty", choices=DATASETS + ("all",))
    ap.add_argument("--out", default="", help="default paper/revision/tables/random_sid_control[_<ds>].tex")
    ap.add_argument("--csv", default="", help="default paper/plotting/random_sid_control[_<ds>].csv")
    a = ap.parse_args()
    sfx = "" if a.dataset == "beauty" else f"_{a.dataset}"
    a.out = a.out or f"paper/revision/tables/random_sid_control{sfx}.tex"
    a.csv = a.csv or f"paper/plotting/random_sid_control{sfx}.csv"
    dss = DATASETS if a.dataset == "all" else (a.dataset,)
    global POOLED
    POOLED = a.dataset == "all"
    res = {ds: ((torch.load(orig_sid(ds)).numpy(), torch.load(rand_sid(ds)).numpy()), load_emb(ds)) for ds in dss}
    nb_cache: dict = {}
    rows, data = [], {}
    for split in ("test", "val"):
        for model, stem in MODELS:
            cells = {}
            for ds in dss:
                cells.update(load(ds, stem, split, nb_cache, res[ds][0], res[ds][1], a.allow_partial))
            data[(model, split)] = cells
            print(f"\n== {model} {split}: seeds {sorted(cells)}")
            if not cells:
                continue
            for nb, nb_label in NBHDS:
                ref = pooled(cells, "clean", nb, "ndcg")
                n = len(ref)
                if n == 0:
                    continue
                s = 100 / ref.mean() if ref.mean() > 0 else float("nan")
                rel = {arm: s * pooled(cells, arm, nb, "ndcg").mean() for arm in ("poison",) + ARMS}
                print(f"  {nb:9s} n={n:5d} retr NDCG {ref.mean():.4f} | " +
                      " ".join(f"{ARM_NAME[k]} {v:6.1f}" for k, v in rel.items()))
                for cname, x, y in CONTRASTS:
                    dn = pooled(cells, x, nb, "ndcg") - pooled(cells, y, nb, "ndcg")
                    dl = pooled(cells, x, nb, "logp") - pooled(cells, y, nb, "logp")
                    lo, hi = boot(dn)
                    llo, lhi = boot(dl)
                    per_seed = [float((c[x]["logp"][c["masks"][nb]] - c[y]["logp"][c["masks"][nb]]).mean())
                                for c in cells.values() if c["masks"][nb].sum() > 0]
                    row = dict(split=split, model=model, neighborhood=nb, contrast=cname, n=n,
                               retr_ndcg=ref.mean(), ndcg_pts=s * dn.mean(), ndcg_lo=s * lo, ndcg_hi=s * hi,
                               logp=dl.mean(), logp_lo=llo, logp_hi=lhi,
                               logp_neg_seeds=sum(v < 0 for v in per_seed), n_seeds=len(per_seed),
                               **{f"rel_{k}": v for k, v in rel.items()})
                    rows.append(row)
                    print(f"     {cname:20s} NDCG {fmt(row['ndcg_pts'], row['ndcg_lo'], row['ndcg_hi'])} pts | "
                          f"logp {fmt(row['logp'], llo, lhi, 3)} nats ({row['logp_neg_seeds']}/{row['n_seeds']} seeds < 0)")
    ws = {}
    for m in ("rqkmeans", "random"):
        for sp in ("test", "val"):
            raw = {"_clean_ndcg": {}}
            for ds in dss:
                one = whole_set(ds, m, sp)
                raw["_clean_ndcg"].update(one.pop("_clean_ndcg"))
                for k, v in one.items():
                    raw.setdefault(k, []).extend(v)
            ws[(m, sp)] = summarize(raw)
    print("\n== whole set (mean over seeds): dSH %, tau_P %")
    for (m, sp), w in ws.items():
        print(f"  {m:8s} {sp:4s} " + " | ".join(f"{ARM_NAME[k]} {v[0]:.3f}/{v[1]:.1f} (n={v[2]})"
                                                for k, v in w.items() if not k.startswith("_")))
    cl = {m: ws[(m, "test")]["_clean_ndcg"] for m in ("rqkmeans", "random")}
    common = sorted(set(cl["rqkmeans"]) & set(cl["random"]))
    if common:
        r = [cl["random"][s] / cl["rqkmeans"][s] for s in common]
        print(f"\n== clean test NDCG@10: RQ-KMeans {St.mean(cl['rqkmeans'][s] for s in common):.4f}, "
              f"random {St.mean(cl['random'][s] for s in common):.4f} (random/RQ-KMeans {100 * St.mean(r):.1f}%, "
              f"seeds {common}, per seed {[round(100 * x, 1) for x in r]})")
    if rows:
        with open(a.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print("wrote", a.csv)
    write_tex(a.out, rows, ws, cl, common, a.dataset)


def write_tex(path, rows, ws, cl, common, dataset="beauty"):
    get = {(r["split"], r["model"], r["neighborhood"], r["contrast"]): r for r in rows}
    L = [r"\begin{table}[t]", r"\centering", r"\small", r"\setlength{\tabcolsep}{3.5pt}", r"\caption{",
         (r"Random-SID control (Beauty, mid-popularity target, five data seeds). TIGER" if dataset == "beauty" else
          r"Random-SID control (Beauty, Sports and Toys pooled, mid-popularity target, five data seeds each, 15 cells). TIGER" if dataset == "all" else
          f"Random-SID control ({DS_NAME[dataset]}, mid-popularity target, five data seeds). TIGER") + r" is trained with the RQ-KMeans \acp{SID} and with randomly permuted \acp{SID} "
         r"(same prefix-tree statistics, random assignment of items to tuples), and unlearned with the same four configurations. "
         r"Paired test-set differences in neighborhood $\mathrm{NDCG}@10$, in points of the retrained model's $\mathrm{NDCG}@10$ on that neighborhood "
         r"(95\% bootstrap intervals over the pooled test cases, \Cref{sec:appendix-bootstrap}), and in mean log-probability of the next item (nats). "
         r"Semantic neighborhood: next item shares the target's first RQ-KMeans token; code neighborhood: next item shares the target's first random-SID token. "
         r"Forget term: Forget+Repair minus Finetune; separation term: $\lambda_s=0.1$ minus Forget+Repair.}",
         r"\label{tab:appendix-random-sid" + ("" if dataset == "beauty" else f"-{dataset}") + "}", r"\begin{tabular}{llrrrr}", r"\toprule",
         r"SIDs & Neighborhood & $n$ & Contrast & $\Delta\mathrm{NDCG}@10$ (pts) & $\Delta\log p$ (nats) \\", r"\midrule"]
    for model, mlabel in (("rqkmeans", "RQ-KMeans"), ("random", "Random")):
        for nb, nlabel in (("sem_pfx", "semantic (RQ-KMeans token)"), ("rand_pfx", "random-SID token"),
                           ("sem_knn32", "32 nearest (content)")):
            for cname in ("forget term", "separation term", "NAU vs F+R"):
                r = get.get(("test", model, nb, cname))
                if r is None:
                    continue
                L.append(f"{mlabel} & {nlabel} & {r['n']} & {cname} & "
                         f"{fmt(r['ndcg_pts'], r['ndcg_lo'], r['ndcg_hi'])} & {fmt(r['logp'], r['logp_lo'], r['logp_hi'], 2)} \\\\")
            L.append(r"\addlinespace")
        L.append(r"\midrule")
    if L[-1] == r"\midrule":
        L.pop()
    L += [r"\bottomrule", r"\end{tabular}"]
    # whole-set block
    L += [r"\vspace{4pt}", r"\begin{tabular}{lcccc}", r"\toprule",
          r" & \multicolumn{2}{c}{RQ-KMeans SIDs} & \multicolumn{2}{c}{Random SIDs} \\",
          r"\cmidrule(lr){2-3} \cmidrule(lr){4-5}",
          r"Method (test) & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ & $\tau_P$ & $\Delta^{\mathrm{retr}}_{\mathrm{SH}}$ & $\tau_P$ \\", r"\midrule"]
    for arm in ARMS:
        cells = []
        for m in ("rqkmeans", "random"):
            v = ws[(m, "test")].get(arm)
            cells += [f"{v[0]:.3f}", f"{v[1]:.1f}"] if v else ["--", "--"]
        name = TEX_NAME.get(arm, ARM_NAME[arm])
        L.append(name + " & " + " & ".join(cells) + r" \\")
    if common:
        L.append(r"\midrule")
        L.append(f"Retrained $\\mathrm{{NDCG}}@10$ & \\multicolumn{{2}}{{c}}{{{St.mean(cl['rqkmeans'][s] for s in common):.4f}}} & "
                 f"\\multicolumn{{2}}{{c}}{{{St.mean(cl['random'][s] for s in common):.4f}}} \\\\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "w").write("\n".join(L) + "\n")
    print("wrote", path)


if __name__ == "__main__":
    main()
