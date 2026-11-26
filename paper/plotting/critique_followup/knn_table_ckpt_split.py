"""Critique follow-up item 2: attach next-item ids (from randsid_scope_orig, every test row) to the
knn32/knn128 scope JSON rows of Beauty mid (same test order, verified), then split the table's own
NAU vs Forget+Repair comparison (scope_def_table.py arms) by embedding rank of the next item."""
import json, numpy as np, torch
R = '/sc-scratch/sc-scratch-ikim-erase/GRID/'
emb = torch.load(R + 'embeddings/beauty_merged_predictions_tensor_latest.pt', map_location='cpu')
if isinstance(emb, dict): emb = emb.get('embeddings', emb.get('tensor'))
emb = torch.nn.functional.normalize(emb.float(), dim=1)
def boot(d, n=3000):
    rng = np.random.default_rng(0)
    return np.percentile([d[rng.integers(0, len(d), len(d))].mean() for _ in range(n)], [2.5, 97.5])
for split, st_orig, pre in (('test', 'randsid_scope_orig', 'fig1_scope_def_knn'),):
    for K in (32, 128):
        P = {}
        for s in (2, 3, 5, 7, 11):
            o = json.load(open(R + f'logs/diagnose/{st_orig}/beauty_seed{s}.json'))
            t = o['target_item']; items = np.array(o['items_per_query']['unrelated'])
            sims = emb @ emb[t]; sims[t] = -1e9
            order = torch.argsort(sims, descending=True).tolist(); rank = {it: i + 1 for i, it in enumerate(order[:1000])}
            rk = np.array([rank.get(int(i), 10 ** 6) for i in items])
            onau = np.array([r for r in o['rows'] if r['method'] == 'nau' and r['group'] == 'unrelated'][0]['ndcg_per_query'])
            g = json.load(open(R + f'logs/diagnose/{pre}{K}/beauty_seed{s}.json'))
            gr = {(x['method'], x['group']): x for x in g['rows']}
            gn = np.array(gr[('nau_s0p01_n0p01', 'neighbor')]['ndcg_per_query'])
            sel = rk <= K
            assert np.array_equal(gn, onau[sel]), (K, s)   # same rows, same order
            krk = rk[sel]
            fr = 'forget_repair' if ('forget_repair', 'neighbor') in gr else 'nosep_ln0'
            for lab, m in (('all', krk <= K), ('r9+', krk >= 9), ('r11+', krk >= 11), ('r1-8', krk <= 8)):
                for meth, key in (('nau', 'nau_s0p01_n0p01'), ('fr', fr), ('fr_nosep', 'nosep_ln0'), ('clean', 'clean')):
                    P.setdefault((lab, meth), []).extend(np.array(gr[(key, 'neighbor')]['ndcg_per_query'])[m].tolist())
        print(f'== knn{K} Beauty mid {split} (order check passed for all 5 seeds)')
        for lab in ('all', 'r1-8', 'r9+', 'r11+'):
            a, c = np.array(P[(lab, 'nau')]), np.array(P[(lab, 'clean')])
            for frk in ('fr', 'fr_nosep'):
                b = np.array(P[(lab, frk)]); d = a - b; sc = 100 / c.mean(); lo, hi = boot(d)
                print(f'  ranks {lab:5s} n={len(a):4d} NAU minus {frk:8s} {sc*d.mean():+6.1f} [{sc*lo:+.1f}, {sc*hi:+.1f}]')
