import json, glob, sys
import numpy as np, torch
R='/sc-scratch/sc-scratch-ikim-erase/GRID/'
emb=torch.load(R+'embeddings/beauty_merged_predictions_tensor_latest.pt',map_location='cpu')
if isinstance(emb,dict): emb=emb.get('embeddings',emb.get('tensor'))
emb=torch.nn.functional.normalize(emb.float(),dim=1)
codes=torch.load(R+'embeddings/beauty/merged_predictions_tensor.pt',map_location='cpu')
if isinstance(codes,dict): codes=list(codes.values())[0]
codes=codes.long()
if codes.shape[0]<codes.shape[1]: codes=codes.T.contiguous()
print('emb',tuple(emb.shape),'codes',tuple(codes.shape))
def boot(d,n=3000):
    rng=np.random.default_rng(0); bs=np.array([d[rng.integers(0,len(d),len(d))].mean() for _ in range(n)]); return np.percentile(bs,[2.5,97.5])
agg={}
for split,stem in (('test','randsid_scope_orig'),('val','randsid_scope_orig_val')):
  P={}
  for f in sorted(glob.glob(R+f'logs/diagnose/{stem}/beauty_seed*.json')):
    d=json.load(open(f)); t=d['target_item']; s=d['seed']
    items=np.array(d['items_per_query']['unrelated'])
    r={x['method']:x for x in d['rows'] if x['group']=='unrelated'}
    sims=emb@emb[t]; sims[t]=-1e9
    order=torch.argsort(sims,descending=True).tolist()
    rank={it:i+1 for i,it in enumerate(order[:600])}
    rk=np.array([rank.get(int(i),10**6) for i in items])
    first=(codes[items,0]==codes[t,0]).numpy() & (items!=t)
    defs={'knn32':(rk<=32),'knn32_r9':(rk>=9)&(rk<=32),'knn32_r17':(rk>=17)&(rk<=32),'knn8':rk<=8,
          'knn128':rk<=128,'knn128_r9':(rk>=9)&(rk<=128),'knn128_r17':(rk>=17)&(rk<=128),
          'knn32_r11':(rk>=11)&(rk<=32),'knn128_r11':(rk>=11)&(rk<=128),'first_notr10':first&(rk>10),'first_tr10':first&(rk<=10),'first':first,'first_notr8':first&(rk>8),'first_tr8':first&(rk<=8)}
    if False:
        # sanity vs knn32 / knn128 JSONs
        for K in (32,128):
            g=json.load(open(R+f'logs/diagnose/fig1_scope_def_knn{K}/beauty_seed{s}.json'))
            gr={(x['method'],x['group']):x for x in g['rows']}
            same_ck=g['checkpoints'].get('nau_s0p01_n0p01')==d['checkpoints']['nau']
            a=sorted(gr[('nau_s0p01_n0p01','neighbor')]['ndcg_per_query']); b=sorted(np.array(r['nau']['ndcg_per_query'])[rk<=K].tolist())
            print(f'seed{s} knn{K}: json n={len(a)} recon n={len(b)} same_ckpt={same_ck} sorted-ndcg equal={np.allclose(a,b) if len(a)==len(b) else False}')
    for k,m in defs.items():
        for meth in ('nau','forget_repair','clean'):
            P.setdefault((k,meth),[]).extend(np.array(r[meth]['ndcg_per_query'])[m].tolist())
        P.setdefault((k,'n_by_seed'),[]).append(int(m.sum()))
  print('\n==',split)
  for k in ['knn32_r11','knn128_r11','first_tr10','first_notr10','knn8','knn32','knn32_r9','knn32_r17','knn128','knn128_r9','knn128_r17','first','first_tr8','first_notr8']:
    a=np.array(P[(k,'nau')]);b=np.array(P[(k,'forget_repair')]);c=np.array(P[(k,'clean')])
    if len(a)==0: print(k,'n=0');continue
    sc=100/c.mean() if c.mean()>0 else float('nan'); dd=a-b; lo,hi=boot(dd)
    print(f'{k:12s} n={len(a):4d} per-seed={P[(k,"n_by_seed")]} clean NDCG={c.mean():.4f} gain NAU-F+R {sc*dd.mean():+.1f} [{sc*lo:+.1f},{sc*hi:+.1f}]  raw dNDCG={dd.mean():+.4f}')
