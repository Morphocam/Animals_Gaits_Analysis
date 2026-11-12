import sys, numpy as np, pandas as pd, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0,'.')
import evaluate as E
qc=pd.read_csv('results/segmentation_qc.csv')
print('n_det>1 by status:'); print(qc[qc.n_detections>1].assign(sp=lambda d:d.clip_id.str.split('/').str[0]).groupby(['sp','status']).size())
flag=set(qc[qc.n_detections>1].clip_id)
data=E.load()
rng=np.random.default_rng(0)
REPS=["fused","mega_crop","resnet_rgb","vp_ordered","gei","kinematics","resnet_bg"]
def subset(d,keep):
    return dict(meta=d['meta'][keep].reset_index(drop=True), bank={k:v[keep] for k,v in d['bank'].items()})
def maps(d):
    m=d['meta']; ids=m.individual.values; src=m.source_id.values
    sites=m['site'].values if 'site' in m else None
    hs=sites is not None and all(sites!='')
    out={}
    for r in REPS:
        x=E.representation(d['bank'],E.REPS[r][2])
        for p in (("P1","P2","P3") if hs else ("P1","P2")):
            pq=E.retrieval(x,ids,src,p,sites)
            if len(pq): out[(r,p)]=(pq.ap.mean(),(pq['rank']==1).mean(),pq)
    return out
rows=[]
for sp in ['giraffe','lion','polarbear']:
    d=data[sp]; f=d['meta'].clip_id.isin(flag).to_numpy()
    print(sp,'clips',len(f),'flagged',f.sum(), list(d['meta'].clip_id[f]))
    base=maps(d); drop=maps(subset(d,~f))
    # random-drop null: same number of clips removed, from the same individuals
    null={k:[] for k in base}
    inds=d['meta'].individual.values
    for _ in range(200):
        k=np.ones(len(f),bool)
        for ind,c in zip(*np.unique(inds[f],return_counts=True)):
            pool=np.flatnonzero((inds==ind)&~f)
            k[rng.choice(pool,c,replace=False)]=False
        for key,v in maps(subset(d,k)).items(): null[key].append(v[0])
    for key in base:
        pq=base[key][2]; isf=f[pq['query'].values]
        n=np.array(null[key]); dv=drop[key][0] if key in drop else np.nan
        rows.append(dict(species=sp,rep=key[0],proto=key[1],mAP_all=base[key][0],mAP_drop=dv,delta=dv-base[key][0],
            null_lo=np.percentile(n,2.5)-base[key][0],null_hi=np.percentile(n,97.5)-base[key][0],
            r1_all=base[key][1],r1_drop=drop[key][1] if key in drop else np.nan,
            ap_flagged=pq.ap[isf].mean() if isf.any() else np.nan, n_flag_q=int(isf.sum()), ap_rest=pq.ap[~isf].mean()))
df=pd.DataFrame(rows)
pd.set_option('display.width',250); pd.set_option('display.max_rows',200)
print(df.round(3).to_string(index=False))
df.to_csv(sys.argv[1],index=False)
