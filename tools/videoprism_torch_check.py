"""Check the PyTorch VideoPrism (transformers) against the cached JAX embeddings:
silhouettes are rebuilt from saved masks with the mirroring rule the cache used."""
import os, sys, zlib
import cv2, numpy as np, torch
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, ROOT)
from extract_features import (CACHE, MASKS, N_SAMPLE, SEED, SEG_MAX_SIDE, downscale, load_manifests,
                              normalised_silhouettes, read_frames, uniform_idx)
from transformers import VideoPrismVisionModel
m = VideoPrismVisionModel.from_pretrained("MHRDYN7/videoprism-base-f16r288").cuda().eval()
df = load_manifests(); rows = df[df.exclude == ""].groupby("species").head(3)
emb_t, emb_j = [], []
for r in rows.itertuples():
    mp = os.path.join(MASKS, *r.clip_id.split("/")) + ".npz"
    if not os.path.exists(mp): continue
    c = np.load(os.path.join(CACHE, *r.clip_id.split("/")) + ".npz")
    if "vp_ordered" not in c.files: continue
    fr = downscale(read_frames(os.path.join(ROOT, r.path)), SEG_MAX_SIDE); h, w = fr[0].shape[:2]
    z = np.load(mp); small = np.unpackbits(z["masks"], axis=-1)[..., :z["shape"][2]]
    masks = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in small]).astype(bool)
    valid = np.where(masks.reshape(len(masks), -1).mean(1) > 0.002)[0]
    sils = normalised_silhouettes(masks, valid)               # drift rule, as in the cache
    seq = sils[uniform_idx(len(sils), N_SAMPLE)]
    x = torch.from_numpy(seq[None, :, None].repeat(3, 2)).float().cuda()   # B,T,C,H,W in [0,1]
    with torch.no_grad():
        out = m(pixel_values_videos=x) if "pixel_values_videos" in m.forward.__code__.co_varnames else m(x)
    e = out.last_hidden_state.mean(1)[0].cpu().numpy()
    emb_t.append(e); emb_j.append(c["vp_ordered"])
    print(r.clip_id, out.last_hidden_state.shape, "cos", round(float(e @ c["vp_ordered"] / np.linalg.norm(e) / np.linalg.norm(c["vp_ordered"])), 4))
T, J = np.stack(emb_t), np.stack(emb_j)
T, J = T - T.mean(0), J - J.mean(0)
print("centred cos per clip:", np.round((T * J).sum(1) / np.linalg.norm(T, axis=1) / np.linalg.norm(J, axis=1), 3))
