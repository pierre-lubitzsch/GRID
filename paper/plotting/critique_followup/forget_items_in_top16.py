import glob, json, os
os.environ['CUDA_VISIBLE_DEVICES']=''
import tensorflow as tf, torch
R='/sc-scratch/sc-scratch-ikim-erase/GRID/'
emb=torch.load(R+'embeddings/beauty_merged_predictions_tensor_latest.pt',map_location='cpu')
if isinstance(emb,dict): emb=emb.get('embeddings',emb.get('tensor'))
emb=torch.nn.functional.normalize(emb.float(),dim=1)
for s in [2,3,5,7,11]:
    dd=R+f'src/data/amazon_data/beauty_spam_tgtmid_seed{s}_pct1_n1' if s!=2 else R+'src/data/amazon_data/beauty_spam_tgtmid_seed2_pct1_n1'
    if not os.path.isdir(dd):
        c=glob.glob(R+f'src/data/amazon_data/beauty_spam_tgtmid*seed{s}*pct1_n1'); dd=c[0] if c else None
    m=json.load(open(dd+'/forget_manifest.json')); t=m['target_items'][0]
    items=set()
    for f in glob.glob(dd+'/training_forget/*'):
        for raw in tf.data.TFRecordDataset([f],compression_type='GZIP'):
            ex=tf.train.Example.FromString(raw.numpy())
            for k,v in ex.features.feature.items():
                if 'item' in k or 'sequence' in k:
                    items.update(int(x) for x in v.int64_list.value)
    sims=emb@emb[t]; sims[t]=-1e9; order=torch.argsort(sims,descending=True).tolist()
    top16=order[:16]
    print(s,os.path.basename(dd),'target',t,'n forget items',len(items),'forget items among top16 ranks:',[i+1 for i,x in enumerate(top16) if x in items])
