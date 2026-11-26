"""Critique follow-up: where is the forgetting damage in the first-token neighborhood?
A = first-token cases whose next item is among the target's 8 nearest embedding items (NAU's L_nbr
training neighbors), B = the other first-token cases. Per dataset (3 strata x 5 seeds), test and val:
paired bootstrap 95% CIs (points of retrained NDCG@10 on the subset) for F+R - start, NAU - start,
NAU - F+R, and pooled NDCG as % of retrained; F+R - Finetune on the mid cells (only those JSONs score
Finetune). Inputs as knn_excluding_trained_all.py.
    PYTHONPATH=. python paper/plotting/critique_followup/damage_split.py"""
import json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from knn_excluding_trained_all import load_codes, load_emb, path, boot, SEEDS

def ci(d, s):
    lo, hi = boot(d); return f'{s*d.mean():+6.1f} [{s*lo:+.1f}, {s*hi:+.1f}]'

for ds in ('beauty', 'sports', 'toys'):
    codes = load_codes(ds); emb = load_emb(ds, codes.shape[0])
    for split in ('test', 'val'):
        P = {}
        for st in ('unpopular', 'mid', 'popular'):
            for sd in SEEDS:
                d = json.load(open(path(ds, st, sd, split))); t = d['target_item']
                items = np.array(d['items_per_query']['unrelated'])
                r = {x['method']: np.array(x['ndcg_per_query']) for x in d['rows'] if x['group'] == 'unrelated'}
                sims = emb @ emb[t]; sims[t] = -1e9
                top8 = set(torch.topk(sims, 8).indices.tolist())
                first = (codes[items, 0] == codes[t, 0]).numpy() & (items != t)
                inA = np.isin(items, list(top8))
                for g, m in (('A', first & inA), ('B', first & ~inA)):
                    for meth in ('clean', 'poison', 'forget_repair', 'nau'):
                        P.setdefault((g, meth), []).extend(r[meth][m].tolist())
                    if 'finetune' in r:
                        for meth in ('clean', 'forget_repair', 'finetune'):
                            P.setdefault((g, 'mid_' + meth), []).extend(r[meth][m].tolist())
        print(f'\n== {ds} {split}')
        for g in ('A', 'B'):
            c = np.array(P[(g, 'clean')]); p = np.array(P[(g, 'poison')]); f = np.array(P[(g, 'forget_repair')]); n = np.array(P[(g, 'nau')])
            if c.mean() == 0:
                print(f'  {g}: n={len(c)} retrained NDCG 0 on this subset; raw NDCG start {p.mean():.4f} F+R {f.mean():.4f} NAU {n.mean():.4f}'); continue
            s = 100 / c.mean()
            print(f'  {g}: n={len(c):5d}  % of retrained: start {s*p.mean():5.1f}  F+R {s*f.mean():5.1f}  NAU {s*n.mean():5.1f}')
            print(f'       F+R - start {ci(f-p, s)}   NAU - start {ci(n-p, s)}   NAU - F+R {ci(n-f, s)}')
            if (g, 'mid_clean') in P:
                mc = np.array(P[(g, 'mid_clean')]); mf = np.array(P[(g, 'mid_forget_repair')]); mt = np.array(P[(g, 'mid_finetune')])
                if mc.mean() > 0:
                    print(f'       mid only (n={len(mc)}): F+R - Finetune {ci(mf-mt, 100/mc.mean())}  (Finetune {100*mt.mean()/mc.mean():.1f}%, F+R {100*mf.mean()/mc.mean():.1f}% of retrained)')
                else:
                    print(f'       mid only (n={len(mc)}): retrained NDCG 0')
