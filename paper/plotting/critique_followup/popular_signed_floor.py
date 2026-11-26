import csv, glob, json, os, re, statistics as S
R='/sc-scratch/sc-scratch-ikim-erase/GRID/'; os.chdir(R)
SEEDS=[2,3,5,7,11]
def ref(ds,sd): return json.load(open(f"logs/eval/spam_references/{ds}"+("" if sd==2 else f"_seed{sd}")+".json"))
def last_sh(path):
    rows=[x for x in csv.DictReader(open(path)) if x.get('test/SH@10')]
    return float(rows[-1]['test/SH@10'])
runs=os.listdir('logs/unlearn/runs')
def grid_sh(ds,st,arm,sd):
    pre='' if ds=='beauty' else ds+'_'
    tag=f"bs1_{pre}bw_tgt{st}_valsel_nau_{arm}"+("" if sd==2 else f"_seed{sd}")
    c=sorted([r for r in runs if r.endswith(tag) and os.path.exists(f'logs/unlearn/runs/{r}/eval/csv/version_0/metrics.csv')])
    if not c: return None
    return last_sh(f'logs/unlearn/runs/{c[-1]}/eval/csv/version_0/metrics.csv')
ARMS=[('ls0p0_ln0p0','F+R (0,0)'),('ls0p0_ln0p01','F+R+nbr (0,0.01)'),('ls0p01_ln0p0','F+R+sep (0.01,0)'),('ls0p01_ln0p01','NAU (0.01,0.01)'),('ls0p1_ln0p0','sep0.1 (0.1,0)'),('ls0p1_ln0p01','(0.1,0.01)'),('ls0p0_ln0p1','(0,0.1)'),('ls0p01_ln0p1','(0.01,0.1)'),('ls0p1_ln0p1','(0.1,0.1)')]
for ds in ['beauty','sports','toys']:
  for st in (['popular','mid','unpopular'] if ds=='beauty' else ['popular']):
    Rf={sd:ref(ds,sd) for sd in SEEDS}
    retr={sd:100*Rf[sd]['clean_SH@10'][st] for sd in SEEDS}
    pois={}
    for sd in SEEDS:
        p=f'logs/diagnose/poison_ref_start/{ds}_{st}_s{sd}/csv/version_0/metrics.csv'
        pois[sd]=100*last_sh(p) if os.path.exists(p) else float('nan')
    print(f'\n### {ds} {st} (test, SH@10 in %)')
    print('retrained per seed', {sd:round(retr[sd],3) for sd in SEEDS}, 'mean',round(S.mean(retr.values()),3))
    print('poisoned(start) per seed', {sd:round(pois[sd],3) for sd in SEEDS}, 'mean',round(S.mean(pois.values()),3))
    # floor for this stratum
    fa=[];fs=[]
    for ts in SEEDS:
        a=[];s=[]
        for ms in SEEDS:
            if ms==ts: continue
            v=100*last_sh(f'logs/diagnose/retrain_floor/{ds}_m{ms}_t{ts}_{st}/csv/version_0/metrics.csv')
            a.append(abs(v-retr[ts])); s.append(v-retr[ts])
        fa.append(S.mean(a)); fs.append(S.mean(s))
    print('retrained-vs-retrained floor |d| per target seed',[round(x,3) for x in fa],'mean',round(S.mean(fa),3),' signed mean',round(S.mean(fs),3))
    for arm,lab in ARMS:
        v={sd:grid_sh(ds,st,arm,sd) for sd in SEEDS}
        if any(x is None for x in v.values()): print(lab,'missing',v); continue
        sg={sd:100*v[sd]-retr[sd] for sd in SEEDS}
        print(f"{lab:20s} signed per seed {[round(sg[sd],3) for sd in SEEDS]}  mean signed {S.mean(sg.values()):+.3f}  mean|d| {S.mean(abs(x) for x in sg.values()):.3f}")
