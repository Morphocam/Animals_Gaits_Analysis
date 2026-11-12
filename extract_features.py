"""Segmentation and per-clip embedding extraction for every clip in the manifest.

For each clip listed in manifests/*.csv (own data + external benchmarks):

  1. Grounding DINO (text prompt "a <species>.") finds the focal animal in the
     first frame where it is detected; SAM 2 propagates that box through the
     clip in both directions (prompt-once-then-track).
  2. Masks are saved (output/masks/...) so every downstream number can be
     regenerated without re-running segmentation.
  3. Embedding variants are computed from the same masks:

     appearance (RGB inside the mask, frozen encoders)
       resnet_rgb     ResNet18 (ImageNet) on masked animal crops
       mega_rgb       MegaDescriptor-L-384 on masked animal crops
       mega_crop      MegaDescriptor-L-384 on unmasked crops (standard re-ID use)
     leakage control
       resnet_bg      ResNet18 on the frame with the animal removed (background only)
     silhouette motion (texture-free, size- and position-normalised, walking
     direction normalised to left-to-right)
       vp_ordered     VideoPrism on 16 temporally ordered silhouettes
       vp_shuffled    same 16 silhouettes in random order     (gait control)
       vp_static      one silhouette repeated 16 times         (gait control)
       vp_mean        mean silhouette repeated 16 times        (gait control)
       gei            64x64 gait energy image (Han & Bhanu 2005)
       kinematics     stride frequency, leg-spread amplitude, body aspect,
                      vertical-bob amplitude (interpretable gait descriptors)

Results are cached per clip in output/cache/, so adding species or clips only
processes the new ones. All variants for a species are then written to
results/embeddings/<species>.npz (tracked in git) for evaluate.py.

    python extract_features.py                 # every clip in the manifest
    python extract_features.py --species lion  # one species
"""
import argparse
import json
import os
import shutil
import tempfile
import zlib

import cv2
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

ROOT = os.path.dirname(os.path.abspath(__file__))
MANIFEST_DIR = os.path.join(ROOT, "manifests")
CACHE = os.path.join(ROOT, "output", "cache")
MASKS = os.path.join(ROOT, "output", "masks")
IMAGES = os.path.join(ROOT, "output", "images")   # masked RGB crops and silhouettes
EMB_DIR = os.path.join(ROOT, "results", "embeddings")

SEG_MAX_SIDE = 640        # frames are segmented at this longest side
MIN_PART = 0.1            # mask components smaller than this fraction of the largest are dropped
MAX_FRAMES = 150          # 5 s at 30 fps; longer clips are truncated
N_SAMPLE = 16             # frames per clip for appearance encoders and VideoPrism
VP_SIZE = 288             # VideoPrism input resolution
SIL_HEIGHT = 200          # normalised silhouette height inside the 288 canvas
GEI_SIZE = 64
SEED = 42
# Stored in every per-clip cache file. Clips cached by an older version of the
# pipeline are processed again. 3: silhouettes mirrored by pose facing direction,
# VideoPrism through PyTorch.
CACHE_VERSION = 3

GDINO_ID = "IDEA-Research/grounding-dino-tiny"
SAM2_ID = "facebook/sam2.1-hiera-base-plus"
MEGA_ID = "hf-hub:BVRA/MegaDescriptor-L-384"
VIDEOPRISM_ID = "MHRDYN7/videoprism-base-f16r288"   # transformers port of videoprism_public_v1_base
VIDEOPRISM_REV = "c5bb17adeb575aaf1d46b56c5b9dfa1b00465c80"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Detector text prompt per species folder name (default: the folder name).
PROMPT_NAMES = {"polarbear": "polar bear"}


# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
class Models:
    def __init__(self):
        from sam2.build_sam import build_sam2_video_predictor_hf
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        import timm
        from torchvision.models import ResNet18_Weights, resnet18

        self.gdino_proc = AutoProcessor.from_pretrained(GDINO_ID)
        self.gdino = AutoModelForZeroShotObjectDetection.from_pretrained(GDINO_ID).to(DEVICE).eval()
        self.sam2 = build_sam2_video_predictor_hf(SAM2_ID, device=DEVICE)

        self.resnet = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        self.resnet.fc = torch.nn.Identity()
        self.resnet = self.resnet.to(DEVICE).eval()

        self.mega = timm.create_model(MEGA_ID, num_classes=0, pretrained=True).to(DEVICE).eval()

        # VideoPrism through its PyTorch implementation in transformers. The
        # converted weights reproduce the embeddings of the original JAX release
        # (cosine 0.999 on 21 clips, tools/videoprism_torch_check.py).
        from transformers import VideoPrismVisionModel
        self.videoprism = VideoPrismVisionModel.from_pretrained(
            VIDEOPRISM_ID, revision=VIDEOPRISM_REV).to(DEVICE).eval()
        self.pose = None          # ViTPose+, loaded on first use (extract_pose.py)


# --------------------------------------------------------------------------
# Segmentation
# --------------------------------------------------------------------------
def read_frames(path):
    cap = cv2.VideoCapture(path)
    frames = []
    while len(frames) < MAX_FRAMES:
        ok, f = cap.read()
        if not ok:
            break
        frames.append(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    cap.release()
    return frames


def downscale(frames, max_side):
    h, w = frames[0].shape[:2]
    s = min(1.0, max_side / max(h, w))
    if s == 1.0:
        return frames
    size = (int(round(w * s)), int(round(h * s)))
    return [cv2.resize(f, size, interpolation=cv2.INTER_AREA) for f in frames]


@torch.no_grad()
def detect(models, frame, species):
    """Highest-scoring Grounding DINO box for 'a <species>.' (or None), its score,
    and the number of boxes above threshold (> 1: possibly several animals)."""
    from PIL import Image
    image = Image.fromarray(frame)
    inputs = models.gdino_proc(images=image, text=f"a {PROMPT_NAMES.get(species, species)}.", return_tensors="pt").to(DEVICE)
    outputs = models.gdino(**inputs)
    res = models.gdino_proc.post_process_grounded_object_detection(
        outputs, inputs.input_ids, threshold=0.3, text_threshold=0.25,
        target_sizes=[image.size[::-1]])[0]
    if len(res["scores"]) == 0:
        return None, 0.0, 0
    i = int(torch.argmax(res["scores"]))
    return res["boxes"][i].cpu().numpy(), float(res["scores"][i]), len(res["scores"])


@torch.no_grad()
def segment(models, frames, species):
    """Prompt once on the first detectable frame, propagate both ways.

    Returns (masks [T,H,W] bool at the segmentation resolution, prompt frame,
    detection score, number of detections on the prompt frame, mask parts kept
    per frame).
    """
    prompt_idx, box, score, n_det = None, None, 0.0, 0
    for idx in range(0, len(frames), max(1, len(frames) // 10)):
        box, score, n_det = detect(models, frames[idx], species)
        if box is not None:
            prompt_idx = idx
            break
    T, H, W = len(frames), *frames[0].shape[:2]
    masks = np.zeros((T, H, W), bool)
    parts = np.zeros(T, int)
    if prompt_idx is None:
        return masks, -1, 0.0, 0, parts

    tmp = tempfile.mkdtemp(prefix="sam2_")
    try:
        for i, f in enumerate(frames):
            cv2.imwrite(os.path.join(tmp, f"{i:05d}.jpg"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR),
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
        with torch.autocast(DEVICE, dtype=torch.bfloat16, enabled=DEVICE == "cuda"):
            state = models.sam2.init_state(video_path=tmp)
            models.sam2.add_new_points_or_box(state, frame_idx=prompt_idx, obj_id=1, box=box)
            for reverse in (False, True):
                for fi, _, logits in models.sam2.propagate_in_video(
                        state, start_frame_idx=prompt_idx, reverse=reverse):
                    masks[fi] = (logits[0, 0] > 0).cpu().numpy()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    # Drop specks: keep components with >= 10% of the largest one's area. A thin
    # occluder (e.g. a treadmill handrail) splits one animal into large parts,
    # which must all be kept.
    for t in range(T):
        n, lab, stats, _ = cv2.connectedComponentsWithStats(masks[t].astype(np.uint8), 8)
        parts[t] = n - 1
        if n > 2:
            area = stats[1:, cv2.CC_STAT_AREA]
            keep = 1 + np.flatnonzero(area >= MIN_PART * area.max())
            masks[t] = np.isin(lab, keep)
            parts[t] = len(keep)
    return masks, prompt_idx, score, n_det, parts


# --------------------------------------------------------------------------
# Inputs for the encoders
# --------------------------------------------------------------------------
def bbox(mask, pad=0.1):
    ys, xs = np.where(mask)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
    py, px = int((y1 - y0) * pad), int((x1 - x0) * pad)
    H, W = mask.shape
    return max(0, y0 - py), min(H, y1 + py), max(0, x0 - px), min(W, x1 + px)


def letterbox(img, size):
    h, w = img.shape[:2]
    s = size / max(h, w)
    r = cv2.resize(img, (max(1, int(round(w * s))), max(1, int(round(h * s)))),
                   interpolation=cv2.INTER_AREA)
    out = np.zeros((size, size) + img.shape[2:], img.dtype)
    y, x = (size - r.shape[0]) // 2, (size - r.shape[1]) // 2
    out[y:y + r.shape[0], x:x + r.shape[1]] = r
    return out


def walking_direction(masks, valid):
    xs = [np.where(masks[t])[1].mean() for t in valid]
    return 1 if len(xs) < 2 or np.polyfit(np.arange(len(xs)), xs, 1)[0] >= 0 else -1


def normalised_silhouettes(masks, valid, size=VP_SIZE, height=SIL_HEIGHT, direction=None):
    """Crop, rescale with ONE clip-level factor, centre on the centroid, and flip
    so the animal walks left-to-right. Keeps intra-clip shape change, removes
    camera distance, image position and walking direction."""
    heights = [np.ptp(np.where(masks[t])[0]) + 1 for t in valid]
    widths = [np.ptp(np.where(masks[t])[1]) + 1 for t in valid]
    scale = min(height / np.median(heights), 0.95 * size / max(widths))
    flip = (walking_direction(masks, valid) if direction is None else direction) < 0
    out = []
    for t in valid:
        m = masks[t].astype(np.uint8)
        y0, y1, x0, x1 = bbox(masks[t], pad=0)
        crop = m[y0:y1, x0:x1]
        ch, cw = max(1, int(round(crop.shape[0] * scale))), max(1, int(round(crop.shape[1] * scale)))
        crop = cv2.resize(crop, (cw, ch), interpolation=cv2.INTER_NEAREST)
        if flip:
            crop = crop[:, ::-1]
        cx = int(round(np.where(crop)[1].mean())) if crop.any() else cw // 2
        canvas = np.zeros((size, size), np.float32)
        top = (size - min(ch, size)) // 2
        left = size // 2 - cx
        sx0, dx0 = max(0, -left), max(0, left)
        w = min(cw - sx0, size - dx0)
        h = min(ch, size)
        if w > 0:
            canvas[top:top + h, dx0:dx0 + w] = crop[:h, sx0:sx0 + w]
        out.append(canvas)
    return np.stack(out)


def uniform_idx(n, k):
    return np.linspace(0, n - 1, k).round().astype(int)


def save_clip_images(clip_id, crops_rgb, sils16):
    """Write the encoder inputs of one clip to output/images/<clip_id>/:
    rgb_XX.jpg (masked RGB crop), sil_XX.png (normalised silhouette) for the
    N_SAMPLE sampled frames, and overview.jpg (RGB row above silhouette row)."""
    d = os.path.join(IMAGES, *clip_id.split("/"))
    os.makedirs(d, exist_ok=True)
    tiles_rgb, tiles_sil = [], []
    for i, (c, s) in enumerate(zip(crops_rgb, sils16)):
        bgr = cv2.cvtColor(np.ascontiguousarray(c), cv2.COLOR_RGB2BGR)
        sil = (s > 0.5).astype(np.uint8) * 255
        cv2.imwrite(os.path.join(d, f"rgb_{i:02d}.jpg"), bgr, [cv2.IMWRITE_JPEG_QUALITY, 92])
        cv2.imwrite(os.path.join(d, f"sil_{i:02d}.png"), sil)
        tiles_rgb.append(letterbox(bgr, 144))
        tiles_sil.append(cv2.cvtColor(cv2.resize(sil, (144, 144), interpolation=cv2.INTER_AREA),
                                      cv2.COLOR_GRAY2BGR))
    cv2.imwrite(os.path.join(d, "overview.jpg"),
                np.vstack([np.hstack(tiles_rgb), np.hstack(tiles_sil)]),
                [cv2.IMWRITE_JPEG_QUALITY, 88])


# --------------------------------------------------------------------------
# Encoders
# --------------------------------------------------------------------------
IMNET_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
IMNET_STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def encode_images(model, images, size, mean, std):
    x = np.stack([letterbox(im, size) for im in images]).astype(np.float32) / 255.0
    x = (x - mean) / std
    x = torch.from_numpy(x).permute(0, 3, 1, 2).to(DEVICE)
    return model(x).float().cpu().numpy().mean(0)


def encode_videoprism(models, sils, rng):
    """Four temporal conditions from the same 16 silhouettes, one batch."""
    idx = uniform_idx(len(sils), N_SAMPLE)
    seq = sils[idx]
    shuffled = seq[rng.permutation(N_SAMPLE)]
    static = np.repeat(seq[N_SAMPLE // 2][None], N_SAMPLE, 0)
    mean = np.repeat(sils.mean(0)[None], N_SAMPLE, 0)
    batch = np.stack([seq, shuffled, static, mean])[..., None].repeat(3, -1)
    x = torch.from_numpy(batch.astype(np.float32)).permute(0, 1, 4, 2, 3).to(DEVICE)  # B,T,C,H,W
    with torch.no_grad():
        emb = models.videoprism(pixel_values_videos=x).last_hidden_state.mean(1).cpu().numpy()
    return dict(vp_ordered=emb[0], vp_shuffled=emb[1], vp_static=emb[2], vp_mean=emb[3])


def kinematics(sils, fps):
    """Interpretable gait descriptors from normalised silhouettes at native fps.

    leg spread = horizontal extent of the lowest 25% of the silhouette, which
    oscillates once per stride; its dominant frequency in 0.3-3 Hz estimates
    stride frequency. Amplitudes are relative to the clip mean.
    """
    rows = np.where(sils.any(axis=(0, 2)))[0]
    top, bottom = rows.min(), rows.max()
    leg0 = int(bottom - 0.25 * (bottom - top))
    spread, aspect, cy = [], [], []
    for s in sils:
        ys, xs = np.where(s > 0.5)
        legs = xs[ys >= leg0]
        spread.append(np.ptp(legs) if legs.size else 0.0)
        aspect.append((np.ptp(xs) + 1) / (np.ptp(ys) + 1))
        cy.append(ys.mean())
    spread, cy = np.asarray(spread, float), np.asarray(cy, float)
    x = spread - np.polyval(np.polyfit(np.arange(len(spread)), spread, 1), np.arange(len(spread)))
    n_fft = max(512, len(x))
    power = np.abs(np.fft.rfft(x * np.hanning(len(x)), n_fft)) ** 2
    freqs = np.fft.rfftfreq(n_fft, 1.0 / fps)
    band = (freqs >= 0.3) & (freqs <= 3.0)
    stride_hz = float(freqs[band][np.argmax(power[band])]) if band.any() and len(x) > 8 else np.nan
    # Periodicity check: autocorrelation of the leg-spread signal at one stride
    # period (near 1 for a steady gait, near 0 or negative when the peak is noise),
    # and the number of stride cycles the window spans.
    lag = int(round(fps / stride_hz)) if np.isfinite(stride_hz) else 0
    autocorr = (float(np.corrcoef(x[:-lag], x[lag:])[0, 1])
                if 0 < lag <= len(x) - 4 and x[:-lag].std() > 0 and x[lag:].std() > 0 else np.nan)
    diag = dict(stride_autocorr=round(autocorr, 3) if np.isfinite(autocorr) else None,
                stride_cycles=round(stride_hz * len(x) / fps, 2) if np.isfinite(stride_hz) else None)
    return np.array([stride_hz,
                     float(np.std(spread) / (np.mean(spread) + 1e-6)),
                     float(np.median(aspect)),
                     float(np.std(cy) / SIL_HEIGHT)], np.float32), diag


KIN_NAMES = ["stride_frequency_hz", "leg_spread_cv", "body_aspect", "vertical_bob"]


# --------------------------------------------------------------------------
# Per clip
# --------------------------------------------------------------------------
def process_clip(models, row, rng):
    frames_full = read_frames(os.path.join(ROOT, row.path))
    frames = downscale(frames_full, SEG_MAX_SIDE)
    masks, prompt_idx, det_score, n_det, parts = segment(models, frames, row.species)

    H, W = masks.shape[1:]
    areas = masks.reshape(len(masks), -1).mean(1)
    valid = np.where(areas > 0.002)[0]
    qc = dict(n_frames=len(frames), prompt_frame=int(prompt_idx), det_score=round(det_score, 3),
              n_detections=int(n_det),
              mask_coverage=round(len(valid) / max(1, len(frames)), 3),
              mean_mask_area=round(float(areas[valid].mean()) if len(valid) else 0.0, 4))
    if len(valid):
        # Mask quality, so exclusions can follow stated thresholds:
        #   split_frac   frames whose mask has > 1 part after speck removal (occluders)
        #   border_frac  frames where the mask touches the image edge (animal truncated)
        #   area_cv      variation of mask area through the clip (size change, leaks)
        #   iou_p05      5th percentile of consecutive-frame mask IoU (low: tracking jumps)
        #   gap_frac     frames without a mask between the first and last masked frame
        m = masks[valid]
        edge = m[:, 0].any(1) | m[:, -1].any(1) | m[:, :, 0].any(1) | m[:, :, -1].any(1)
        inter = (m[1:] & m[:-1]).reshape(len(m) - 1, -1).sum(1)
        union = (m[1:] | m[:-1]).reshape(len(m) - 1, -1).sum(1)
        qc.update(split_frac=round(float((parts[valid] > 1).mean()), 3),
                  border_frac=round(float(edge.mean()), 3),
                  area_cv=round(float(areas[valid].std() / areas[valid].mean()), 3),
                  iou_p05=round(float(np.percentile(inter / np.maximum(union, 1), 5)), 3)
                  if len(m) > 1 else None,
                  gap_frac=round(float(1 - len(valid) / (valid[-1] - valid[0] + 1)), 3))
    if len(valid) < 8:
        reason = ("too few frames" if len(frames) < 8 else
                  "no detection" if prompt_idx < 0 else
                  "target too small (<0.2% of frame)" if areas.max() > 0 else "empty mask")
        qc["status"] = f"failed: {reason}"
        return None, masks, qc

    # Truncated-at-border masks are kept; the animal is often partly out of frame
    # at the start or end of a lateral pass.
    pick = valid[uniform_idx(len(valid), N_SAMPLE)]
    crops_rgb, crops_raw, backgrounds = [], [], []
    for t in pick:
        f, m = frames[t], masks[t]
        y0, y1, x0, x1 = bbox(m)
        crops_rgb.append((f * m[..., None])[y0:y1, x0:x1])
        crops_raw.append(f[y0:y1, x0:x1])
        grown = cv2.dilate(m.astype(np.uint8), np.ones((15, 15), np.uint8)).astype(bool)
        backgrounds.append(f * ~grown[..., None])

    feats = {
        "resnet_rgb": encode_images(models.resnet, crops_rgb, 224, IMNET_MEAN, IMNET_STD),
        "resnet_bg": encode_images(models.resnet, backgrounds, 224, IMNET_MEAN, IMNET_STD),
        "mega_rgb": encode_images(models.mega, crops_rgb, 384, 0.5, 0.5),
        "mega_crop": encode_images(models.mega, crops_raw, 384, 0.5, 0.5),
    }
    # Pose keypoints (saved for gait_kinematics.py). The facing direction from the
    # pose decides the mirroring; the drift of the mask centroid (walking_direction)
    # is wrong on a treadmill and when the camera pans with the animal.
    from extract_pose import facing, load_pose_model, pose_clip
    if models.pose is None:
        models.pose = load_pose_model()
    xy, _ = pose_clip(*models.pose, row.clip_id, frames, masks, valid, row.fps)
    direction = facing(xy)
    sils = normalised_silhouettes(masks, valid, direction=direction)
    save_clip_images(row.clip_id, crops_rgb, sils[uniform_idx(len(sils), N_SAMPLE)])
    feats.update(encode_videoprism(models, sils, rng))
    gei = np.stack([cv2.resize(s, (GEI_SIZE, GEI_SIZE), interpolation=cv2.INTER_AREA)
                    for s in sils]).mean(0)
    feats["gei"] = gei.ravel()
    feats["kinematics"], kin_diag = kinematics(sils, row.fps)
    qc.update(kin_diag)
    qc["status"] = "ok"
    qc["walk_direction"] = walking_direction(masks, valid)   # mask drift, for the record
    qc["facing"] = direction
    return feats, masks, qc


def load_manifests():
    """All manifests/*.csv concatenated; clip_ids are <folder>/<individual>/<clip>.

    `species` is the visible species: a curated `visible_species` overrides the
    folder name (kept as `folder_species`) for clips filed under the wrong one.
    """
    df = pd.concat([pd.read_csv(os.path.join(MANIFEST_DIR, f), keep_default_na=False)
                    for f in sorted(os.listdir(MANIFEST_DIR)) if f.endswith(".csv")],
                   ignore_index=True).fillna("")
    for col in ("visible_species", "exclude"):
        if col not in df:
            df[col] = ""
    df["folder_species"] = df["species"]
    df["species"] = df["visible_species"].where(df["visible_species"] != "", df["species"])
    return df


def cache_path(base, clip_id, ext):
    p = os.path.join(base, *clip_id.split("/")) + ext
    os.makedirs(os.path.dirname(p), exist_ok=True)
    return p


def cache_current(clip_id):
    p = cache_path(CACHE, clip_id, ".npz")
    if not os.path.exists(p):
        return False
    try:
        z = np.load(p)
        return "version" in z.files and int(z["version"]) == CACHE_VERSION
    except Exception:
        return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--species", nargs="*")
    ap.add_argument("--recompute", action="store_true", help="ignore the per-clip cache")
    args = ap.parse_args()

    df = load_manifests()
    if args.species:
        df = df[df.species.isin(args.species)]

    todo = [r for r in df.itertuples() if args.recompute or not cache_current(r.clip_id)]
    print(f"{len(df)} clips in scope, {len(todo)} to process (device={DEVICE})")
    if todo:
        models = Models()
        for row in tqdm(todo):
            rng = np.random.default_rng([SEED, zlib.crc32(row.clip_id.encode())])
            try:
                feats, masks, qc = process_clip(models, row, rng)
            except Exception as e:  # keep going; the failure is recorded in QC
                feats, masks, qc = None, None, {"status": f"error: {type(e).__name__}: {e}"}
            if masks is not None:
                small = np.stack([cv2.resize(m.astype(np.uint8), None, fx=0.5, fy=0.5,
                                             interpolation=cv2.INTER_NEAREST) for m in masks])
                np.savez_compressed(cache_path(MASKS, row.clip_id, ".npz"),
                                    masks=np.packbits(small.astype(bool), axis=-1),
                                    shape=np.array(small.shape))
            np.savez(cache_path(CACHE, row.clip_id, ".npz"),
                     qc=json.dumps(qc), version=CACHE_VERSION, **(feats or {}))

    assemble()


def assemble():
    """Per-species embedding files (tracked) and the QC table, from the cache of
    every clip in the manifests (not only those processed in this run)."""
    df = load_manifests()
    df = df[[os.path.exists(cache_path(CACHE, c, ".npz")) for c in df.clip_id]]
    os.makedirs(EMB_DIR, exist_ok=True)
    for f in os.listdir(EMB_DIR):            # drop files of renamed species
        if f.endswith(".npz") and f[:-4] not in set(df.species):
            os.remove(os.path.join(EMB_DIR, f))
    qc_rows = []
    for species, g in df.groupby("species"):
        kept, bank = [], {}
        for r in g.itertuples():
            c = np.load(cache_path(CACHE, r.clip_id, ".npz"))
            qc = json.loads(str(c["qc"]))
            qc_rows.append(dict(clip_id=r.clip_id, **qc))
            if qc.get("status") != "ok":
                continue
            kept.append(r.clip_id)
            for k in c.files:
                if k not in ("qc", "version"):
                    bank.setdefault(k, []).append(c[k])
        np.savez(os.path.join(EMB_DIR, f"{species}.npz"), clip_id=np.array(kept),
                 kinematics_names=np.array(KIN_NAMES),
                 **{k: np.stack(v).astype(np.float32) for k, v in bank.items()})
        print(f"{species}: {len(kept)}/{len(g)} clips embedded")
    pd.DataFrame(qc_rows).to_csv(os.path.join(ROOT, "results", "segmentation_qc.csv"), index=False)
    write_params()


def write_params():
    """results/pipeline_params.json: every setting and library version the
    embeddings depend on, for the Methods and for reproducing them."""
    from importlib.metadata import PackageNotFoundError, version
    import subprocess
    versions = {}
    for pkg in ("torch", "torchvision", "transformers", "sam2", "timm", "jax", "flax",
                "opencv-python", "numpy"):
        try:
            versions[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                                capture_output=True, text=True).stdout.strip()
    except OSError:
        commit = ""
    params = dict(
        models=dict(detector=GDINO_ID, segmenter=SAM2_ID, megadescriptor=MEGA_ID,
                    videoprism=f"{VIDEOPRISM_ID}@{VIDEOPRISM_REV[:8]} (PyTorch)", resnet="torchvision resnet18 IMAGENET1K_V1"),
        detection=dict(prompt="a <species>.", prompt_names=PROMPT_NAMES, box_threshold=0.3,
                       text_threshold=0.25, frames_searched=10),
        segmentation=dict(max_side=SEG_MAX_SIDE, min_part_of_largest=MIN_PART,
                          min_mask_area_of_frame=0.002, min_masked_frames=8),
        sampling=dict(max_frames=MAX_FRAMES, n_sample=N_SAMPLE, background_dilation_px=15,
                      crop_padding=0.1),
        silhouettes=dict(canvas=VP_SIZE, height=SIL_HEIGHT, gei_size=GEI_SIZE,
                         mirrored_by="pose facing direction (shoulders ahead of hips)"),
        pose=dict(model="usyd-community/vitpose-plus-base", head="animal (index 3)",
                  box="SAM 2 mask box, 10% padding", min_keypoint_score=0.3),
        kinematics=dict(names=KIN_NAMES, stride_band_hz=[0.3, 3.0], leg_band="lowest 25%",
                        fft_min_length=512),
        seed=SEED, device=DEVICE, git_commit=commit, libraries=versions,
        written=pd.Timestamp.now().isoformat(timespec="seconds"))
    with open(os.path.join(ROOT, "results", "pipeline_params.json"), "w") as f:
        json.dump(params, f, indent=2)


if __name__ == "__main__":
    main()
