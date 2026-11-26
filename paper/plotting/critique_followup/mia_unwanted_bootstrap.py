import json, glob, os, numpy as np
os.chdir('/sc-scratch/sc-scratch-ikim-erase/GRID')
def auc(m,n):
    m=np.asarray(m)[:,None]; n=np.asarray(n)[None,:]
    return ((m>n).mean(axis=(-2,-1)) if m.ndim==2 else None)+0.5*(m==n).mean()
def auc_b(M,N):  # M: (B,nm), N: (B,nn)
    g=(M[:,:,None]>N[:,None,:]).mean(axis=(1,2)); e=(M[:,:,None]==N[:,None,:]).mean(axis=(1,2)); return g+0.5*e
rng=np.random.default_rng(0); B=4000
for ds in ['beauty','sports','toys']:
    data=[]
    for sd in [2,3,5,7,11]:
        j=json.load(open(f'logs/diagnose/mia_unwanted/{ds}_seed{sd}.json'))
        r={x['method']:x for x in j['rows']}
        of=f'logs/diagnose/mia_unwanted_lastckpt/{ds}_seed{sd}.json'
        if os.path.isfile(of):
            o={x['method']:x for x in json.load(open(of))['rows']}; r['original']=o['original']
        for k,x in r.items():
            a=auc(x['member_logprob'],x['nonmember_logprob'])
            assert abs(a-x['auc'])<1e-6,(ds,sd,k,a,x['auc'])
        data.append((sd,j['n_members'],j['n_nonmembers'],r)); print(ds,sd,'nonmember dir',j['nonmember_dir'])
    print(f"\n## {ds}: members per seed {[d[1] for d in data]} (total {sum(d[1] for d in data)}), non-members {[d[2] for d in data]} (total {sum(d[2] for d in data)})")
    meths=['original','retrained','nau','finetune','forget_only','forget_repair']
    point={m:np.mean([d[3][m]['auc'] for d in data]) for m in meths}
    print('mean AUC', {m:round(v,3) for m,v in point.items()})
    boots={m:np.zeros(B) for m in meths}
    INN=rng.integers(0,data[0][2],(B,data[0][2]))
    for sd,nm,nn,r in data:
        im=rng.integers(0,nm,(B,nm)); inn=INN
        for m in meths:
            M=np.asarray(r[m]['member_logprob'])[im]; N=np.asarray(r[m]['nonmember_logprob'])[inn]
            boots[m]+=auc_b(M,N)/len(data)
    for a in ['nau','finetune','original','forget_only','forget_repair']:
        d=boots[a]-boots['retrained']; lo,hi=np.percentile(d,[2.5,97.5])
        print(f"  {a:14s} - retrained: {point[a]-point['retrained']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    d=boots['nau']-boots['finetune']; lo,hi=np.percentile(d,[2.5,97.5]); print(f"  nau - finetune: {point['nau']-point['finetune']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
    d=boots['nau']-boots['original']; lo,hi=np.percentile(d,[2.5,97.5]); print(f"  nau - original: {point['nau']-point['original']:+.3f} [{lo:+.3f}, {hi:+.3f}]")
