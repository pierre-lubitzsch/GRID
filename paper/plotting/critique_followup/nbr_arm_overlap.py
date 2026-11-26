import json, glob, os, sys
os.environ['CUDA_VISIBLE_DEVICES']=''
R='/sc-scratch/sc-scratch-ikim-erase/GRID/'; sys.path.insert(0,R)
import numpy as np, torch, tensorflow as tf
from src.components.unlearning.neighborhood_sampler import closest_prefix_neighbors, build_sorted_sid_index
emb=torch.load(R+'embeddings/beauty_merged_predictions_tensor_latest.pt',map_location='cpu')
if isinstance(emb,dict): emb=emb.get('embeddings',emb.get('tensor'))
emb=torch.nn.functional.normalize(emb.float(),dim=1)
codes=torch.load(R+'embeddings/beauty/merged_predictions_tensor.pt',map_location='cpu').long()
if codes.shape[0]<codes.shape[1]: codes=codes.T.contiguous()
sids=build_sorted_sid_index(codes); ssids=codes.numpy()[sids]
for split,stem in (('test','randsid_scope_orig'),('val','randsid_scope_orig_val')):
  tot={}
  for s in [2,3,5,7,11]:
    d=json.load(open(R+f'logs/diagnose/{stem}/beauty_seed{s}.json')); t=d['target_item']
    items=np.array(d['items_per_query']['unrelated'])
    dd=R+f'src/data/amazon_data/beauty_spam_tgtmid_seed{s}_pct1_n1'
    S=set()
    for f in glob.glob(dd+'/training_forget/*'):
        for raw in tf.data.TFRecordDataset([f],compression_type='GZIP'):
            ex=tf.train.Example.FromString(raw.numpy())
            for k,v in ex.features.feature.items(): S.update(int(x) for x in v.int64_list.value)
    sims=emb@emb[t]; sims[t]=-1e9; order=torch.argsort(sims,descending=True).tolist()
    emb8_plain=set(order[:8]); emb8_excl=[i for i in order if i not in S][:8]
    pfx_plain=set(closest_prefix_neighbors(codes,t,8,1,sorted_ids=sids,sorted_sids=ssids,exclude_ids={t}))
    pfx_excl=set(closest_prefix_neighbors(codes,t,8,1,sorted_ids=sids,sorted_sids=ssids,exclude_ids=S|{t}))
    first=(codes[items,0]==codes[t,0]).numpy()&(items!=t)
    nf=int(first.sum())
    row={'first':nf}
    for name,st in (('emb_plain',emb8_plain),('emb_exclS',set(emb8_excl)),('emb_union',emb8_plain|set(emb8_excl)),('pfx_plain',pfx_plain),('pfx_exclS',pfx_excl),('pfx_union',pfx_plain|pfx_excl)):
        inset=np.isin(items,list(st))
        row[name]=int((first&inset).sum()); row[name+'_all']=int(inset.sum())
        row[name+'_in_first_region']=sum(int(codes[i,0]==codes[t,0]) for i in st)
    print(split,'seed',s,row)
    for k,v in row.items(): tot[k]=tot.get(k,0)+v
  print(split,'TOTAL',tot, {k:round(100*tot[k]/tot['first'],1) for k in ('emb_plain','emb_exclS','emb_union','pfx_plain','pfx_exclS','pfx_union')})
