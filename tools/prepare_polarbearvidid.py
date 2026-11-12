"""Converts a seeded sample of PolarBearVidID tracklets into clips + a manifest.

PolarBearVidID (Zuerl et al. 2023, CC BY 4.0; Zenodo 10.5281/zenodo.7564528)
holds 1431 tracklets of 13 zoo polar bears filmed by several cameras per zoo.
Frames are named <id>C<cam>T<tracklet>F<frame>.jpg.

Each sampled tracklet is written as an mp4 under input/polarbear/ and
gets a manifest row whose source_id is the camera: cross-camera retrieval (P2
in evaluate.py) is then the leakage-free protocol. The zoo is kept as a column
because bears in one zoo share an enclosure background.

    python tools/prepare_polarbearvidid.py external/polarbearvidid/data
"""
import os
import re
import sys
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_VIDEO = os.path.join(ROOT, "input", "polarbear")
OUT_MANIFEST = os.path.join(ROOT, "manifests", "clips_polarbearvidid.csv")
PER_BEAR = 40          # tracklets sampled per individual, stratified by camera
FRAME_STRIDE = 1       # native 12.5 fps (earlier runs used every second frame)
NATIVE_FPS = 12.5
MIN_FRAMES = 40        # tracklets shorter than this (before stride) are skipped
SEED = 42
NAME = re.compile(r"^(\d{3})C(\d{2})T(\d{3})F(\d{3})\.jpg$")


def main(data_dir):
    animals = pd.read_csv(os.path.join(data_dir, "animal_db.csv"))
    tracks = defaultdict(list)
    for bear in sorted(os.listdir(data_dir)):
        d = os.path.join(data_dir, bear)
        if not os.path.isdir(d):
            continue
        for f in os.listdir(d):
            m = NAME.match(f)
            if m:
                tracks[(int(m[1]), int(m[2]), int(m[3]))].append((int(m[4]), os.path.join(d, f)))

    rng = np.random.default_rng(SEED)
    rows = []
    for bear in sorted({k[0] for k in tracks}):
        keys = [k for k in tracks if k[0] == bear and len(tracks[k]) >= MIN_FRAMES]
        by_cam = defaultdict(list)
        for k in sorted(keys):
            by_cam[k[1]].append(k)
        for cam in by_cam:
            rng.shuffle(by_cam[cam])
        picked = []                       # round-robin over cameras
        while len(picked) < PER_BEAR and any(by_cam.values()):
            for cam in sorted(by_cam):
                if by_cam[cam] and len(picked) < PER_BEAR:
                    picked.append(by_cam[cam].pop())
        info = animals.set_index("id").loc[bear]
        ind = f"{bear:03d}"
        os.makedirs(os.path.join(OUT_VIDEO, ind), exist_ok=True)
        for _, cam, tr in picked:
            frames = [p for _, p in sorted(tracks[(bear, cam, tr)])][::FRAME_STRIDE]
            clip = f"{ind}_C{cam:02d}T{tr:03d}.mp4"
            path = os.path.join(OUT_VIDEO, ind, clip)
            first = cv2.imread(frames[0])
            h, w = first.shape[:2]
            vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), NATIVE_FPS / FRAME_STRIDE, (w, h))
            for p in frames:
                img = cv2.imread(p)
                vw.write(img if img.shape[:2] == (h, w) else cv2.resize(img, (w, h)))
            vw.release()
            rows.append(dict(clip_id=f"polarbear/{ind}/{clip[:-4]}", species="polarbear",
                             individual=ind, clip=clip,
                             path=os.path.relpath(path, ROOT).replace("\\", "/"),
                             chunk_index=tr, n_frames=len(frames), fps=NATIVE_FPS / FRAME_STRIDE,
                             width=w, height=h, duration_s=round(len(frames) * FRAME_STRIDE / NATIVE_FPS, 2),
                             source_id=f"polarbear-{ind}-cam{cam}", source_inferred=0,
                             source_type="zoo camera (PolarBearVidID)",
                             source_ref="doi:10.5281/zenodo.7564528",
                             label_basis="dataset annotation", notes=f"{info['name']}; {info['zoo']}",
                             site=info["zoo"]))
    df = pd.DataFrame(rows)
    df.to_csv(OUT_MANIFEST, index=False)
    print(df.groupby("individual").agg(clips=("clip", "size"), cameras=("source_id", "nunique")).T)
    print(f"Wrote {len(df)} clips -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main(sys.argv[1])
