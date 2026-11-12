"""Animal pose for every segmented clip: 17 keypoints per frame.

    python extract_pose.py                 # every clip with a saved mask
    python extract_pose.py horse lion      # selected species
    python extract_pose.py --recompute

extract_features.py runs this for every clip it segments (it needs the facing
direction to mirror the silhouettes); this script recomputes pose from saved
masks without segmenting again. For each frame, the box around the saved SAM 2
mask is passed to ViTPose+ (frozen; AP-10K animal head), so no detector is run
again and the pose is estimated for the same animal that was segmented.

Writes  output/pose/<clip_id>.npz          xy (T,17,2) in pixels of the 640-px
                                           segmentation frame, score (T,17),
                                           frame index, box, fps, walk direction
        output/images/<clip_id>/pose.jpg   skeleton drawn on 8 sampled frames

Locomotion variables are computed from these files by gait_kinematics.py.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from extract_features import (IMAGES, MASKS, ROOT, SEG_MAX_SIDE, cache_path, downscale,
                              load_manifests, read_frames, uniform_idx)

POSE = os.path.join(ROOT, "output", "pose")
VITPOSE_ID = "usyd-community/vitpose-plus-base"
AP10K_HEAD = 3            # ViTPose+ expert index: 0 COCO, 1 AIC, 2 MPII, 3 AP-10K, 4 APT-36K
BOX_PAD = 0.1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Order of the 17 outputs of the animal head, checked by drawing numbered
# keypoints on frames of each species (results/pose_check/). "elbow" and "knee"
# are the AP-10K names for the mid-limb joint; on long-legged species the model
# places them at the carpus and hock.
KEYPOINTS = ["nose", "l_eye", "r_eye", "l_ear", "r_ear",
             "l_shoulder", "r_shoulder", "l_elbow", "r_elbow", "l_front_paw", "r_front_paw",
             "l_hip", "r_hip", "l_knee", "r_knee", "l_hind_paw", "r_hind_paw"]
K = {n: i for i, n in enumerate(KEYPOINTS)}
SKELETON = [("nose", "l_ear"), ("nose", "r_ear"),
            ("l_shoulder", "l_elbow"), ("l_elbow", "l_front_paw"),
            ("r_shoulder", "r_elbow"), ("r_elbow", "r_front_paw"),
            ("l_hip", "l_knee"), ("l_knee", "l_hind_paw"),
            ("r_hip", "r_knee"), ("r_knee", "r_hind_paw"),
            ("l_shoulder", "l_hip"), ("r_shoulder", "r_hip")]


def load_masks(clip_id, h, w):
    z = np.load(os.path.join(MASKS, *clip_id.split("/")) + ".npz")
    small = np.unpackbits(z["masks"], axis=-1)[..., :z["shape"][2]]
    return np.stack([cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
                     for m in small]).astype(bool)


def mask_box(mask, pad=BOX_PAD):
    ys, xs = np.where(mask)
    x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    px, py = pad * (x1 - x0), pad * (y1 - y0)
    H, W = mask.shape
    x0, y0, x1, y1 = max(0, x0 - px), max(0, y0 - py), min(W, x1 + px), min(H, y1 + py)
    return [float(x0), float(y0), float(x1 - x0), float(y1 - y0)]     # COCO x, y, w, h


@torch.no_grad()
def estimate(processor, model, frames, boxes, batch=32):
    xy, score = [], []
    for i in range(0, len(frames), batch):
        fr, bx = frames[i:i + batch], [[b] for b in boxes[i:i + batch]]
        inputs = processor(fr, boxes=bx, return_tensors="pt").to(DEVICE)
        idx = torch.full((len(fr),), AP10K_HEAD, device=DEVICE)
        out = model(**inputs, dataset_index=idx)
        res = processor.post_process_pose_estimation(out, boxes=bx)
        for r in res:
            xy.append(r[0]["keypoints"].cpu().numpy())
            score.append(r[0]["scores"].cpu().numpy())
    return np.stack(xy).astype(np.float32), np.stack(score).astype(np.float32)


def facing(xy, score=None):
    """+1 if the animal faces right in the image, -1 if left: shoulders ahead of
    hips in the median frame. Unlike the drift of the mask centroid, this works on
    a treadmill and when the camera pans with the animal."""
    sh = xy[:, [K["l_shoulder"], K["r_shoulder"]], 0].mean(1)
    hp = xy[:, [K["l_hip"], K["r_hip"]], 0].mean(1)
    return 1 if np.median(sh - hp) >= 0 else -1


def load_pose_model():
    from transformers import VitPoseForPoseEstimation, VitPoseImageProcessor
    return (VitPoseImageProcessor.from_pretrained(VITPOSE_ID),
            VitPoseForPoseEstimation.from_pretrained(VITPOSE_ID).to(DEVICE).eval())


def pose_clip(processor, model, clip_id, frames, masks, valid, fps):
    """Pose for the masked frames of one clip; saves the keypoints and an overview
    image and returns (xy, score)."""
    boxes = [mask_box(masks[t]) for t in valid]
    xy, score = estimate(processor, model, [frames[t] for t in valid], boxes)
    np.savez_compressed(cache_path(POSE, clip_id, ".npz"), xy=xy, score=score,
                        frame=valid, box=np.array(boxes, np.float32), fps=float(fps),
                        direction=facing(xy), size=np.array(masks.shape[1:]),
                        keypoints=np.array(KEYPOINTS))
    overview(clip_id, frames, valid, boxes, xy, score)
    return xy, score


def draw(frame, xy, score, thr=0.3):
    img = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR).copy()
    col = {"l": (60, 140, 255), "r": (255, 170, 40)}
    for a, b in SKELETON:
        if score[K[a]] > thr and score[K[b]] > thr:
            c = col.get(b[0] if b[1] == "_" else "", (230, 230, 230))
            cv2.line(img, tuple(int(v) for v in xy[K[a]]), tuple(int(v) for v in xy[K[b]]), c, 2,
                     cv2.LINE_AA)
    for k, n in enumerate(KEYPOINTS):
        if score[k] > thr:
            c = col.get(n[0] if n[1] == "_" else "", (255, 255, 255))
            cv2.circle(img, tuple(int(v) for v in xy[k]), 3, c, -1, cv2.LINE_AA)
    return img


def overview(clip_id, frames, valid, boxes, xy, score):
    tiles = []
    for j in uniform_idx(len(valid), 8):
        img = draw(frames[valid[j]], xy[j], score[j])
        x, y, w, h = [int(v) for v in boxes[j]]
        crop = img[y:y + h, x:x + w]
        s = 200 / crop.shape[0]
        tiles.append(cv2.resize(crop, (max(1, int(crop.shape[1] * s)), 200)))
    wmax = max(t.shape[1] for t in tiles)
    tiles = [cv2.copyMakeBorder(t, 0, 0, 0, wmax - t.shape[1], cv2.BORDER_CONSTANT) for t in tiles]
    sheet = np.vstack([np.hstack(tiles[:4]), np.hstack(tiles[4:])])
    cv2.imwrite(cache_path(IMAGES, clip_id + "/pose", ".jpg"), sheet)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="*")
    ap.add_argument("--recompute", action="store_true")
    ap.add_argument("--limit", type=int, default=0, help="clips per species (for a quick test)")
    args = ap.parse_args()

    df = load_manifests()
    if args.species:
        df = df[df.species.isin(args.species)]
    if args.limit:
        df = df.groupby("species").head(args.limit)
    todo = [r for r in df.itertuples()
            if os.path.exists(os.path.join(MASKS, *r.clip_id.split("/")) + ".npz")
            and (args.recompute or not os.path.exists(cache_path(POSE, r.clip_id, ".npz")))]
    print(f"{len(todo)} clips to process (device={DEVICE})")
    if not todo:
        return
    processor, model = load_pose_model()

    for row in tqdm(todo):
        frames = downscale(read_frames(os.path.join(ROOT, row.path)), SEG_MAX_SIDE)
        h, w = frames[0].shape[:2]
        masks = load_masks(row.clip_id, h, w)[:len(frames)]
        valid = np.where(masks.reshape(len(masks), -1).mean(1) > 0.002)[0]
        if len(valid) < 8:
            continue
        pose_clip(processor, model, row.clip_id, frames, masks, valid, row.fps)


if __name__ == "__main__":
    main()
