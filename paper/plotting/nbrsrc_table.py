"""Neighbor-source comparison (R2 Q1, R3 Q3): NAU with its L_nbr neighbors drawn from content
embeddings (paper, k=8), 8 random items, or the 8 closest items in SID space, vs Forget+Repair,
on the first-SID-token neighborhood (Figure 2 protocol), Beauty, 3 targets x 5 seeds.
Paired bootstrap over pooled test cases (Appendix G), points of the retrained neighborhood NDCG.
Inputs: logs/diagnose/fig1_scope_nbrsrc{,_val}/beauty_<stratum>_seed<N>.json (run_scope_nbrsrc_cell.sh).
    PYTHONPATH=. python paper/plotting/nbrsrc_table.py
"""
import glob, json
from collections import defaultdict
import numpy as np

ARMS = [("nau_k8", "embedding (paper)"), ("nau_nbrrand", "random"), ("nau_nbrpfx", "SID-prefix")]


def pool(stem, group="neighbor"):
    P = defaultdict(list); cells = 0
    for f in sorted(glob.glob(f"logs/diagnose/{stem}/*.json")):
        r = {(x["method"], x["group"]): x for x in json.load(open(f))["rows"]}
        need = [(a, group) for a, _ in ARMS] + [("forget_repair", group), ("clean", group)]
        if not all(k in r and r[k]["n"] for k in need):
            continue
        cells += 1
        for a, _ in ARMS + [("forget_repair", ""), ("clean", "")]:
            P[a] += r[(a, group)]["ndcg_per_query"]
    return {k: np.asarray(v) for k, v in P.items()}, cells


def boot(d, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)])
    return np.percentile(bs, [2.5, 97.5])


def main():
    for stem, name in (("fig1_scope_nbrsrc", "test"), ("fig1_scope_nbrsrc_val", "validation")):
        P, cells = pool(stem)
        if not cells:
            print(name, "no complete cells"); continue
        ref = P["clean"].mean(); s = 100 / ref
        print(f"\n{name}: {cells} cells, {len(P['clean'])} neighborhood test cases")
        print(f"  Forget+Repair nbr tauP {100 * P['forget_repair'].mean() / ref:.1f}")
        for a, lab in ARMS:
            d = P[a] - P["forget_repair"]; lo, hi = boot(d)
            print(f"  {lab:18s} nbr tauP {100 * P[a].mean() / ref:6.1f}   gain over F+R {s * d.mean():+6.1f} [{s * lo:+.1f}, {s * hi:+.1f}]")
        for a, lab in ARMS[1:]:
            d = P[a] - P["nau_k8"]; lo, hi = boot(d)
            print(f"  {lab:18s} minus embedding {s * d.mean():+6.1f} [{s * lo:+.1f}, {s * hi:+.1f}]")


if __name__ == "__main__":
    main()
