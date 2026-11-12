import sys, os, cv2, numpy as np
sys.path.insert(0,'.')
from extract_features import load_manifests, read_frames
out=sys.argv[1]
ids="""giraffe/B/B_7 giraffe/B/B_8 lion/C/C_16 lion/D/D_7 polarbear/002/002_C01T093 polarbear/002/002_C02T003 polarbear/004/004_C01T036 polarbear/005/005_C01T057 polarbear/005/005_C02T011 polarbear/007/007_C01T066 polarbear/011/011_C03T057 polarbear/012/012_C03T006""".split()
m=load_manifests().set_index('clip_id')
W=360
for part,(a,b) in enumerate([(0,4),(4,8),(8,12)]):
    rows=[]
    for cid in ids[a:b]:
        fr=read_frames(m.loc[cid,'path'])
        z=np.load(os.path.join('output','masks',cid+'.npz'))
        mk=np.unpackbits(z['masks'],axis=-1)[...,:z['shape'][2]].astype(bool)
        tiles=[]
        for t in np.linspace(0,len(fr)-1,4).astype(int):
            f=cv2.cvtColor(fr[t],cv2.COLOR_RGB2BGR)
            mm=cv2.resize(mk[t].astype(np.uint8),(f.shape[1],f.shape[0]),interpolation=cv2.INTER_NEAREST)
            f[mm>0]=(0.6*f[mm>0]+0.4*np.array([255,0,255])).astype(np.uint8)
            cs,_=cv2.findContours(mm,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(f,cs,-1,(0,255,255),max(1,f.shape[1]//300))
            h=int(W*f.shape[0]/f.shape[1]); f=cv2.resize(f,(W,h))
            f=cv2.resize(f,(W,240)) if h!=240 else f
            cv2.putText(f,f"{cid.split('/',1)[1]} t={t}",(5,16),cv2.FONT_HERSHEY_SIMPLEX,0.45,(255,255,255),1)
            tiles.append(f)
        rows.append(np.hstack(tiles))
    cv2.imwrite(os.path.join(out,f"flag{part}.jpg"),np.vstack(rows),[cv2.IMWRITE_JPEG_QUALITY,85])
