"""Critique follow-up item 1: requesting-user counts for sensitive item deletion (UHF@10 x N per seed),
Fisher exact tests (unpaired) and the feasible range of exact McNemar p-values given per-seed marginals
(per-user hits are not stored). Input: logs/eval/sensitive_scrub/compare.csv. Run from the repo root."""
import csv
from collections import defaultdict
from scipy.stats import fisher_exact, binomtest
rows = list(csv.DictReader(open('logs/eval/sensitive_scrub/compare.csv')))
# requesting test users per (dataset, seed), inferred from the UHF fractions (all values are k/N)
N = {('beauty','2'):14,('beauty','3'):12,('beauty','5'):11,('beauty','7'):13,('beauty','11'):11,
     ('sports','2'):13,('sports','3'):14,('sports','5'):14,('sports','7'):14,('sports','11'):17,
     ('toys','2'):12,('toys','3'):13,('toys','5'):13,('toys','7'):11,('toys','11'):12}
SEEDS = ['2','3','5','7','11']
k = {}
for r in rows:
    v = float(r['UHF']) * N[(r['dataset'], r['seed'])]
    assert abs(v - round(v)) < 1e-3
    k[(r['dataset'], r['method'], r['view'], r['seed'])] = round(v)
def tot(ds, m, v): return sum(k[(ds, m, v, s)] for s in SEEDS)
def mcnemar_range(ds, a, b, v):
    lo = hi = diff = 0
    for s in SEEDS:
        ka, kb, n = k[(ds, a, v, s)], k[(ds, b, v, s)], N[(ds, s)]
        lo += max(0, ka - kb); hi += min(ka, n - kb); diff += kb - ka
    ps = []
    for n10 in range(lo, hi + 1):
        n01 = n10 + diff
        ps.append(1.0 if n10 + n01 == 0 else binomtest(min(n10, n01), n10 + n01, 0.5).pvalue)
    return min(ps), max(ps)
METH = ['original','retrained','nau','forget_only','forget_repair','finetune','scif','seif','kookmin','fanchuan','tracer']
for ds in ['beauty','sports','toys']:
    n = sum(N[(ds, s)] for s in SEEDS)
    for v in ['normal','scrub']:
        print(ds, v, ' '.join(f"{m}={tot(ds,m,v)}/{n}" for m in METH if (ds,m,v,'2') in k))
        for b in ['retrained','original','forget_only','forget_repair','finetune']:
            a_, b_ = tot(ds,'nau',v), tot(ds,b,v)
            pf = fisher_exact([[a_, n-a_], [b_, n-b_]])[1]
            mn, mx = mcnemar_range(ds, 'nau', b, v)
            print(f"   NAU {a_} vs {b} {b_}: Fisher p={pf:.3f}; paired McNemar p in [{mn:.3f}, {mx:.3f}]")
