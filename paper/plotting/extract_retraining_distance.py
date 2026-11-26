import csv, json, statistics as S
from collections import defaultdict
D='/sc-scratch/sc-scratch-ikim-erase/GRID/paper/data/'
_all=[r for r in csv.DictReader(open(D+'runs.csv')) if r['setting']=='spam' and r['done']=='True' and not r['pkm'] and 'pkm' not in r['tag']]
_d={}
for r in _all:
    k=(r['dataset'],r['track'],r['tag'])
    if k not in _d or int(r['jid'])>int(_d[k]['jid']): _d[k]=r
runs=list(_d.values())
SEEDS=['2','3','5','7','11']; STR=['unpopular','mid','popular']
def ref(ds,seed,q):
    t='' if q=='rkmeans' else '_'+q
    return json.load(open(D+f'spam_references/{ds}{t}'+('' if seed=='2' else f'_seed{seed}')+'.json'))
ARMS=[('Finetune','unified_abl_lf0.0_ls0.0_ln0.0'),('Forget only','unified_abl_lf0.1_ls0.0_ln0.0_lr0.0'),
('Forget+Repair','unified_abl_lf0.1_ls0.0_ln0.0'),('SCIF','scif'),('SEIF',{'beauty':'seif_std0.006','sports':'seif_std0.006','toys':'seif_std0.006'}),
('Kookmin','kookmin_ir0.001'),('Fanchuan','fanchuan_t0.007'),('TRACER','tracer_lf1.0_lc0.1_plr0.001_ep4'),
('NAU','unified_lf0.1_ls0.1_ac0_ln0.01_sepgenerative'),('NAU w/o Lsep','unified_lf0.1_ls0.0_ac0_ln0.01_mass_embedding'),('NAU w/o Lnbr','unified_lf0.1_ls0.1_ac0_ln0.0_sepgen')]
def msd(v): return S.mean(v), (S.stdev(v) if len(v)>1 else 0.0)
rows=[]
for ds in ['beauty','sports','toys']:
  for q in ['rkmeans','tracerv3']:
    if q=='tracerv3' and ds!='beauty': continue
    try: R={s:ref(ds,s,q) for s in SEEDS}
    except FileNotFoundError: continue
    shR={s:[R[s]['clean_SH@10'][st] for st in STR] for s in SEEDS}; nR={s:R[s]['clean_ndcg@10'] for s in SEEDS}
    # poisoned
    pd=[S.mean(abs(R[s]['poison_SH@10'][st]-shR[s][i]) for i,st in enumerate(STR)) for s in SEEDS]
    pu=[S.mean(abs(R[s]['poison_ndcg@10'][st]-nR[s])/nR[s] for st in STR) for s in SEEDS]
    psh=[S.mean(R[s]['poison_SH@10'][st] for st in STR) for s in SEEDS]
    pn=[S.mean(R[s]['poison_ndcg@10'][st] for st in STR) for s in SEEDS]
    gap=S.mean(pd)
    if q=='rkmeans':
      rows.append(dict(dataset=ds,track=q,method='Retrain',sh=msd([S.mean(shR[s]) for s in SEEDS]),ndcg=msd(list(nR.values())),tauP=(100,0),dsh=(0,0),d_forget=0.0,d_util=(0,0)))
      rows.append(dict(dataset=ds,track=q,method='Poisoned',sh=msd(psh),ndcg=msd(pn),tauP=msd([pn[i]/nR[s] for i,s in enumerate(SEEDS)]),dsh=msd(pd),d_forget=1.0,d_util=msd(pu)))
    for m,arm in ARMS:
        if isinstance(arm,dict): arm=arm[ds]
        if q=='tracerv3' and m!='TRACER': continue
        tr='tiger+'+q
        per=defaultdict(list)
        for r in runs:
            if r['dataset']==ds and r['arm']==arm and r['track']==tr and r['stratum'] in STR and r['seed'] in SEEDS:
                s=r['seed']; i=STR.index(r['stratum']); sh=float(r['test/SH@10']); n=float(r['test/ndcg@10'])
                per[s].append((sh,n,abs(sh-shR[s][i]),abs(n-nR[s])/nR[s],n/nR[s]))
        if len(per)<5: continue
        agg=lambda j:msd([S.mean(x[j] for x in per[s]) for s in per])
        dsh=agg(2)
        rows.append(dict(dataset=ds,track=q,method=m,sh=agg(0),ndcg=agg(1),tauP=agg(4),dsh=dsh,d_forget=dsh[0]/gap,d_util=agg(3)))
out='/sc-scratch/sc-scratch-ikim-erase/GRID/paper/plotting/retraining_distance.csv'
with open(out,'w',newline='') as f:
    w=csv.writer(f); w.writerow(['dataset','track','method','sh10','sh10_sd','ndcg10','ndcg10_sd','tau_P','tau_P_sd','abs_dsh10','abs_dsh10_sd','d_forget','d_util','d_util_sd'])
    for r in rows:
        w.writerow([r['dataset'],r['track'],r['method'],*[f'{100*x:.4f}' for x in r['sh']],*[f'{100*x:.4f}' for x in r['ndcg']],
          *([f'{x:.2f}' for x in r['tauP']] if r['method']=='Retrain' else [f'{100*x:.2f}' for x in r['tauP']]),
          *[f'{100*x:.4f}' for x in r['dsh']],f"{r['d_forget']:.4f}",*[f'{100*x:.3f}' for x in r['d_util']]])
print(open(out).read())
