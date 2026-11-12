"""Masked RGB crops and normalised silhouettes for every segmented clip, rebuilt
from the saved masks (no GPU, no models).

    python tools/export_images.py            # every species
    python tools/export_images.py horse      # selected species

Writes output/images/<species>/<individual>/<clip>/rgb_XX.jpg, sil_XX.png and
overview.jpg: the 16 sampled frames each encoder sees. New runs of
extract_features.py write the same files directly. Masks are stored at half the
segmentation resolution, so images rebuilt here have slightly coarser edges
than the ones the encoders received.
"""
import os
import sys

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extract_features import (MASKS, N_SAMPLE, SEG_MAX_SIDE, bbox, downscale,  # noqa: E402
                              load_manifests, normalised_silhouettes, read_frames,
                              save_clip_images, uniform_idx)


from extract_pose import facing  # noqa: E402


def main():
    df = load_manifests()
    if len(sys.argv) > 1:
        df = df[df.species.isin(sys.argv[1:])]
    done = 0
    for row in df.itertuples():
        mp = os.path.join(MASKS, *row.clip_id.split("/")) + ".npz"
        if not os.path.exists(mp):
            continue
        frames = downscale(read_frames(os.path.join(ROOT, row.path)), SEG_MAX_SIDE)
        z = np.load(mp)
        small = np.unpackbits(z["masks"], axis=-1)[..., :z["shape"][2]]
        h, w = frames[0].shape[:2]
        masks = np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                          for m in small]).astype(bool)
        valid = np.where(masks.reshape(len(masks), -1).mean(1) > 0.002)[0]
        if len(valid) < 8:
            continue
        crops = []
        for t in valid[uniform_idx(len(valid), N_SAMPLE)]:
            y0, y1, x0, x1 = bbox(masks[t])
            crops.append((frames[t] * masks[t][..., None])[y0:y1, x0:x1])
        pp = os.path.join(ROOT, "output", "pose", *row.clip_id.split("/")) + ".npz"
        direction = facing(np.load(pp)["xy"]) if os.path.exists(pp) else None
        sils = normalised_silhouettes(masks, valid, direction=direction)
        save_clip_images(row.clip_id, crops, sils[uniform_idx(len(sils), N_SAMPLE)])
        done += 1
    print(f"Wrote images for {done} clips to output/images/")


if __name__ == "__main__":
    main()
