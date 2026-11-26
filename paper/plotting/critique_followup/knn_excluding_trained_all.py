"""Critique follow-up item 2, all cells: NAU minus Forget+Repair (canonical executions) on
  (i)   the first-token neighborhood (next item shares the target's first SID token, target excluded),
  (ii)  first-token cases whose next item is NOT among the target's 8 (or 10) nearest content-embedding
        items (NAU's L_nbr neighbors are the cosine top-8 minus visible forget items, so ranks <= 10
        contain every trained neighbor on the checked cells; the 10-cut is the conservative one),
  (iii) the 32 / 128 nearest items excluding ranks 1 to 10 (and, for reference, all ranks),
per dataset (3 strata x 5 seeds pooled), test and validation, paired bootstrap 95% CI over test cases,
in points of the retrained (clean) model's NDCG@10 on the same subset.
Inputs (scope_neighbor_def=none, items_per_query): mid from logs/diagnose/randsid_scope_orig{,_val}/,
unpopular/popular from logs/diagnose/scope_none_orig{,_val}/ (run_scope_none_batch.sh).
    PYTHONPATH=. python paper/plotting/critique_followup/knn_excluding_trained_all.py [--csv out.csv]"""
import argparse, csv, json, os
import numpy as np, torch
R = '/sc-scratch/sc-scratch-ikim-erase/GRID/'
EMB = {"beauty": "embeddings/beauty_merged_predictions_tensor_latest.pt",
       "sports": "logs/inference/runs/embeddings/sports_10321537/pickle/merged_predictions_tensor.pt",
       "toys": "logs/inference/runs/embeddings/toys_10321536/pickle/merged_predictions_tensor.pt"}
SEEDS = (2, 3, 5, 7, 11)

def load_emb(ds, n_items):
    e = torch.load(R + EMB[ds], map_location='cpu')
    if isinstance(e, dict):
        if 'item_ids' in e:
            row = {int(i): r for r, i in enumerate(e['item_ids'].long().tolist())}
            e = e['embeddings'][torch.tensor([row[i] for i in range(n_items)])]
        else:
            e = e.get('embeddings', e.get('tensor'))
    assert e.shape[0] == n_items, (ds, e.shape, n_items)
    return torch.nn.functional.normalize(e.float(), dim=1)

def load_codes(ds):
    c = torch.load(R + f'embeddings/{ds}/merged_predictions_tensor.pt', map_location='cpu')
    if isinstance(c, dict): c = list(c.values())[0]
    c = c.long()
    return c.T.contiguous() if c.shape[0] < c.shape[1] else c

def path(ds, st, sd, split):
    v = '' if split == 'test' else '_val'
    if st == 'mid':
        return R + f'logs/diagnose/randsid_scope_orig{v}/{ds}_seed{sd}.json'
    return R + f'logs/diagnose/scope_none_orig{v}/{ds}_{st}_seed{sd}.json'

def boot(d, n=3000):
    rng = np.random.default_rng(0)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])

SUBSETS = ['first', 'first_not8', 'first_not10', 'first_in8',
           'knn32', 'knn32_r11', 'knn128', 'knn128_r11']

def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--csv', default=None); a = ap.parse_args()
    out = []
    for ds in ('beauty', 'sports', 'toys'):
        codes = load_codes(ds); emb = load_emb(ds, codes.shape[0])
        for split in ('test', 'val'):
            P = {}; cells = 0; missing = []
            for st in ('unpopular', 'mid', 'popular'):
                for sd in SEEDS:
                    f = path(ds, st, sd, split)
                    if not os.path.isfile(f): missing.append(f'{st}/s{sd}'); continue
                    d = json.load(open(f)); t = d['target_item']
                    items = np.array(d['items_per_query']['unrelated'])
                    r = {x['method']: np.array(x['ndcg_per_query']) for x in d['rows'] if x['group'] == 'unrelated'}
                    sims = emb @ emb[t]; sims[t] = -1e9
                    order = torch.topk(sims, 200).indices.tolist()
                    rank = {it: i + 1 for i, it in enumerate(order)}
                    rk = np.array([rank.get(int(i), 10 ** 6) for i in items])
                    first = (codes[items, 0] == codes[t, 0]).numpy() & (items != t)
                    m = {'first': first, 'first_not8': first & (rk > 8), 'first_not10': first & (rk > 10),
                         'first_in8': first & (rk <= 8), 'knn32': rk <= 32, 'knn32_r11': (rk >= 11) & (rk <= 32),
                         'knn128': rk <= 128, 'knn128_r11': (rk >= 11) & (rk <= 128)}
                    for k, mask in m.items():
                        for meth in ('nau', 'forget_repair', 'clean'):
                            P.setdefault((k, meth), []).extend(r[meth][mask].tolist())
                    cells += 1
            print(f'\n== {ds} {split}: {cells}/15 cells' + (f' (missing {missing})' if missing else ''))
            for k in SUBSETS:
                x = np.array(P.get((k, 'nau'), [])); y = np.array(P.get((k, 'forget_repair'), [])); c = np.array(P.get((k, 'clean'), []))
                if len(x) == 0 or c.mean() == 0: print(f'  {k:12s} n={len(x)}'); continue
                s = 100 / c.mean(); dd = x - y; lo, hi = boot(dd)
                nf = len(P[('first', 'nau')])
                share = f'  ({100*len(x)/nf:.1f}% of first-token cases)' if k.startswith('first') and nf else ''
                print(f'  {k:12s} n={len(x):5d}  gain {s*dd.mean():+6.1f} [{s*lo:+6.1f}, {s*hi:+6.1f}]{share}')
                out.append(dict(dataset=ds, split=split, cells=cells, subset=k, n=len(x), gain=s * dd.mean(), lo=s * lo, hi=s * hi))
    if a.csv:
        with open(a.csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(out[0])); w.writeheader(); w.writerows(out)
        print('wrote', a.csv)

if __name__ == '__main__':
    main()
