"""Effect of mirroring silhouettes by pose facing direction instead of mask drift,
measured on the gait energy image (needs no encoder): GEIs are rebuilt from the
saved masks with both rules and retrieval is compared.
Writes results/experiments/mirror_fix_gei.csv."""
import os, sys
import cv2, numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from extract_features import GEI_SIZE, MASKS, load_manifests, normalised_silhouettes
import evaluate as E
from extract_pose import facing

def gei(masks, valid, direction):
    s = normalised_silhouettes(masks, valid, direction=direction)
    return np.stack([cv2.resize(x, (GEI_SIZE, GEI_SIZE), interpolation=cv2.INTER_AREA) for x in s]).mean(0).ravel()

data, rows = E.load(), []
for sp, d in data.items():
    m = d["meta"]; old, new, changed = [], [], 0
    for cid in m.clip_id:
        z = np.load(os.path.join(MASKS, *cid.split("/")) + ".npz")
        masks = np.unpackbits(z["masks"], axis=-1)[..., :z["shape"][2]].astype(bool)
        valid = np.where(masks.reshape(len(masks), -1).mean(1) > 0.002)[0]
        face = facing(np.load(os.path.join(ROOT, "output", "pose", *cid.split("/")) + ".npz")["xy"])
        from extract_features import walking_direction
        drift = walking_direction(masks, valid); changed += drift != face
        old.append(gei(masks, valid, drift)); new.append(gei(masks, valid, face))
    sites = m["site"].values if "site" in m and all(m["site"].values != "") else None
    for name, x in (("mask drift (current)", np.stack(old)), ("pose facing (fixed)", np.stack(new))):
        x = E.unit(x - x.mean(0))
        for proto in ("P1", "P2", "P3") if sites is not None else ("P1", "P2"):
            pq = E.retrieval(x, m.individual.values, m.source_id.values, proto, sites)
            if len(pq):
                rows.append(dict(species=sp, mirroring=name, protocol=proto, clips=len(m), clips_changed=int(changed),
                                 mAP=pq.ap.mean(), rank1=(pq["rank"] == 1).mean(), chance=pq.chance_ap.mean()))
out = pd.DataFrame(rows); out.to_csv(os.path.join(ROOT, "results", "experiments", "mirror_fix_gei.csv"), index=False)
print(out.round(3).to_string(index=False))
