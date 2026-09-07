"""Segmentation and dual-stream feature extraction.

Implements Sections 2.2-2.3 of GaitAnalysis.pdf:

    B = S(I; theta_S, p)              binary silhouette sequence  (Sec 2.2)
    R_t(x) = B_t(x) . I_t(x)          RGB-masked frame            (Sec 2.2)
    s_t = f_R(R_t; theta_R)           spatial stream  -> ResNet18 (Sec 2.3)
    v_t = f_V(B_{t-tau+1:t}; theta_V) temporal stream -> VideoPrism
    h_t = concat(s_t, v_t)            fused per-step feature
    h_bar = (1/T) sum_t h_t           sequence-level embedding    (Sec 2.4)

The key point the paper is explicit about, and which the previous
implementation did not honour, is that the two streams receive *different*
inputs: ResNet18 sees the appearance-preserving RGB mask, VideoPrism sees the
texture-free binary silhouette.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import ResNet18_Weights, resnet18

from src.utils import download_sam_checkpoint

# --- Constants ---
SAM_CHECKPOINT_DIR = "models"
SAM_CHECKPOINT_NAME = "sam_vit_b_01ec64.pth"
SAM_MODEL_TYPE = "vit_b"

VIDEOPRISM_MODEL_NAME = "videoprism_public_v1_base"

# Paper Sec 2.3: "VideoPrism received 16-frame binary-mask clips sampled at 5 fps".
TEMPORAL_WINDOW = 16          # tau
TEMPORAL_SAMPLE_FPS = 5.0
VIDEOPRISM_INPUT_SIZE = 288

SPATIAL_DIM = 512             # d_s, ResNet18 penultimate features
TEMPORAL_DIM = 768            # d_v, VideoPrism base token width

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# --------------------------------------------------------------------------
# Model loading
# --------------------------------------------------------------------------
def get_sam_predictor(model_type=SAM_MODEL_TYPE, checkpoint_name=SAM_CHECKPOINT_NAME,
                      device=DEVICE):
    """Loads a SAM predictor.

    The paper specifies SAM3 (Carion et al., 2025) for prompted video
    segmentation and tracking. Only SAM 1 checkpoints are available locally, so
    this returns a SAM 1 predictor driven by `SamVideoSegmenter` below, which
    reproduces the paper's prompt-once-then-propagate protocol. Swap this
    function (and `SamVideoSegmenter.segment_video`) for a SAM3 video predictor
    to match the paper exactly; the rest of the pipeline is unchanged because it
    only consumes the resulting binary mask sequence.
    """
    from segment_anything import SamPredictor, sam_model_registry

    download_sam_checkpoint(SAM_CHECKPOINT_DIR, checkpoint_name)
    checkpoint_path = os.path.join(SAM_CHECKPOINT_DIR, checkpoint_name)
    sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
    sam.to(device=device)
    return SamPredictor(sam)


def get_resnet_model(checkpoint=None, device=DEVICE):
    """ResNet18 spatial encoder, classification head removed.

    Paper Sec 2.3: the encoders are adapted with a cross-entropy objective and
    "after fine-tuning, the classification heads were removed, and the models
    were used only as feature extractors". Pass `checkpoint` to load a
    fine-tuned backbone; otherwise ImageNet weights are used, which corresponds
    to the pipeline before encoder adaptation.

    Returns a model emitting 512-d features. The previous version appended a
    randomly initialised `Linear(512, 768)` layer, which was never trained and
    therefore applied an arbitrary random projection to every feature; the
    paper's fusion is a concatenation, so no projection is needed at all.
    """
    weights = None if checkpoint else ResNet18_Weights.IMAGENET1K_V1
    backbone = resnet18(weights=weights)
    backbone.fc = nn.Identity()  # drop the classifier, keep global avg pooling

    if checkpoint:
        state = torch.load(checkpoint, map_location="cpu")
        state = state.get("state_dict", state)
        state = {k.replace("backbone.", ""): v for k, v in state.items()}
        missing, unexpected = backbone.load_state_dict(state, strict=False)
        print(f"  loaded ResNet18 checkpoint {checkpoint} "
              f"(missing={len(missing)}, unexpected={len(unexpected)})")

    backbone.to(device)
    backbone.eval()
    return backbone


def get_videoprism_model(model_name=VIDEOPRISM_MODEL_NAME):
    """Loads the VideoPrism temporal encoder and its pretrained weights."""
    import videoprism.models as vp

    model = vp.get_model(model_name)
    params = vp.load_pretrained_weights(model_name)
    return model, params


# --------------------------------------------------------------------------
# Frames
# --------------------------------------------------------------------------
def extract_frames(video_path, max_frames=None):
    """Reads a clip as RGB frames. Returns (frames, native_fps)."""
    frames = []
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if max_frames and len(frames) >= max_frames:
            break
    cap.release()
    return frames, fps


# --------------------------------------------------------------------------
# Segmentation (Sec 2.2)
# --------------------------------------------------------------------------
class SamVideoSegmenter:
    """Prompt-once-then-propagate segmentation over a clip.

    Paper Sec 2.2: "SAM3 was initialised using a positive-point prompt starting
    from the first frame in which the target animal was fully visible ... after
    which the resulting mask was propagated through the remaining frames."

    SAM 1 has no native video tracking, so propagation is approximated by
    prompting each subsequent frame with the previous frame's mask bounding box
    plus its centroid. This preserves the paper's semantics (one initial prompt,
    then tracking) rather than the previous behaviour of re-prompting every
    frame with a fixed image-centre point, which is not tracking at all.
    """

    def __init__(self, predictor, min_area_ratio=0.005, max_side=None):
        self.predictor = predictor
        self.min_area_ratio = min_area_ratio
        # SAM's image encoder runs once per frame and dominates runtime on CPU.
        # Segmenting a downscaled frame and upsampling the mask is far cheaper;
        # masks are always returned at the original frame resolution.
        self.max_side = max_side

    @staticmethod
    def _largest_component(mask):
        """Keeps only the largest connected component, dropping stray blobs."""
        mask_u8 = mask.astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(mask_u8, connectivity=8)
        if n <= 1:
            return mask
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        return labels == largest

    @staticmethod
    def _mask_to_box(mask, pad=8):
        ys, xs = np.where(mask)
        if len(xs) == 0:
            return None
        h, w = mask.shape
        return np.array([max(xs.min() - pad, 0), max(ys.min() - pad, 0),
                         min(xs.max() + pad, w - 1), min(ys.max() + pad, h - 1)],
                        dtype=np.float32)

    def _predict(self, frame, point=None, box=None):
        self.predictor.set_image(frame)
        point_coords = np.array([point], dtype=np.float32) if point is not None else None
        point_labels = np.array([1]) if point is not None else None
        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=True,
        )
        return masks[int(np.argmax(scores))]

    def segment_video(self, frames):
        """Returns a list of boolean masks B_t, one per frame (None if lost).

        Masks are always at the original frame resolution, regardless of
        `max_side`.
        """
        full_h, full_w = frames[0].shape[:2]

        scale = 1.0
        if self.max_side and max(full_h, full_w) > self.max_side:
            scale = self.max_side / max(full_h, full_w)
            frames = [cv2.resize(f, (int(round(full_w * scale)),
                                     int(round(full_h * scale))),
                                 interpolation=cv2.INTER_AREA)
                      for f in frames]

        h, w = frames[0].shape[:2]
        min_area = self.min_area_ratio * h * w

        # Initial positive-point prompt on the first frame.
        init = self._predict(frames[0], point=(w / 2.0, h / 2.0))
        init = self._largest_component(init)
        masks = [init if init.sum() >= min_area else None]

        prev = masks[0]
        for frame in frames[1:]:
            box = self._mask_to_box(prev) if prev is not None else None
            if box is None:
                # Target lost: re-prompt from the centre, as the paper does when
                # propagation fails.
                mask = self._predict(frame, point=(w / 2.0, h / 2.0))
            else:
                cx = (box[0] + box[2]) / 2.0
                cy = (box[1] + box[3]) / 2.0
                mask = self._predict(frame, point=(cx, cy), box=box)
            mask = self._largest_component(mask)
            if mask.sum() < min_area:
                mask = None
            masks.append(mask)
            if mask is not None:
                prev = mask

        if scale != 1.0:
            masks = [None if m is None else
                     cv2.resize(m.astype(np.uint8), (full_w, full_h),
                                interpolation=cv2.INTER_NEAREST).astype(bool)
                     for m in masks]
        return masks


def apply_rgb_mask(frame, mask):
    """R_t(x) = B_t(x) . I_t(x)  -- appearance-preserving RGB mask."""
    return frame * mask[:, :, None].astype(frame.dtype)


def binary_silhouette(mask):
    """B_t rendered as a texture-free silhouette (white animal, black field)."""
    return (mask.astype(np.float32))


# --------------------------------------------------------------------------
# Spatial stream (Sec 2.3)
# --------------------------------------------------------------------------
_SPATIAL_PREPROCESS = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def get_spatial_features(resnet_model, rgb_masked_frames, device=DEVICE, batch_size=16):
    """s_t = f_R(R_t; theta_R) for every RGB-masked frame. Returns (T, 512)."""
    if not rgb_masked_frames:
        return np.zeros((0, SPATIAL_DIM), np.float32)

    feats = []
    for start in range(0, len(rgb_masked_frames), batch_size):
        chunk = rgb_masked_frames[start:start + batch_size]
        batch = torch.stack([_SPATIAL_PREPROCESS(f.astype(np.uint8)) for f in chunk])
        with torch.no_grad():
            out = resnet_model(batch.to(device))
        feats.append(out.cpu().numpy())
    return np.concatenate(feats, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Temporal stream (Sec 2.3)
# --------------------------------------------------------------------------
def build_silhouette_windows(masks, native_fps, window=TEMPORAL_WINDOW,
                             target_fps=TEMPORAL_SAMPLE_FPS, stride=None):
    """Resamples silhouettes to `target_fps` and cuts them into ordered windows.

    Paper Sec 2.3: 16-frame binary-mask clips sampled at 5 fps. Returns a list
    of (window, 288, 288, 3) float arrays in [0, 1].
    """
    valid = [m for m in masks if m is not None]
    if not valid:
        return []

    # Fill dropped frames with the last good mask so temporal order is preserved.
    filled, last = [], valid[0]
    for m in masks:
        if m is not None:
            last = m
        filled.append(last)

    step = max(int(round((native_fps or target_fps) / target_fps)), 1)
    sampled = filled[::step]

    if len(sampled) < window:
        sampled = sampled + [sampled[-1]] * (window - len(sampled))

    rendered = []
    for m in sampled:
        sil = binary_silhouette(m)
        sil = cv2.resize(sil, (VIDEOPRISM_INPUT_SIZE, VIDEOPRISM_INPUT_SIZE),
                         interpolation=cv2.INTER_NEAREST)
        rendered.append(np.repeat(sil[:, :, None], 3, axis=2))

    if stride is None:
        stride = window  # non-overlapping windows
    windows = []
    for start in range(0, len(rendered) - window + 1, stride):
        windows.append(np.stack(rendered[start:start + window], axis=0))
    if not windows:
        windows.append(np.stack(rendered[:window], axis=0))
    return windows


def get_temporal_features(model, params, windows):
    """v = f_V(B_{t-tau+1:t}; theta_V) for each window. Returns (n_windows, 768)."""
    import jax.numpy as jnp

    if not windows:
        return np.zeros((0, TEMPORAL_DIM), np.float32)

    feats = []
    for win in windows:
        batch = jnp.asarray(win[None, ...], dtype=jnp.float32)
        out = model.apply(params, batch, train=False)
        if isinstance(out, tuple):
            out = out[0]
        # Mean over the token axis -> one 768-d vector for the window.
        feats.append(np.asarray(jnp.mean(out, axis=1))[0])
    return np.stack(feats, axis=0).astype(np.float32)


# --------------------------------------------------------------------------
# Sequence-level embedding (Sec 2.4)
# --------------------------------------------------------------------------
def sequence_embedding(spatial, temporal, stream_norm="none"):
    """Mean-pools each stream and concatenates: h_bar = concat(mean s, mean v).

    The paper defines h_t = concat(s_t, v_t) followed by mean pooling over t.
    Because mean pooling is linear, pooling each stream and then concatenating
    is identical to concatenating per step and then pooling, while avoiding one
    VideoPrism forward pass per frame.

    `stream_norm="l2"` L2-normalises each stream before concatenation. This is
    NOT in the paper (Sec 2.3 specifies a plain concatenation) and defaults off,
    but it is exposed because the two streams have very different norms, so an
    unnormalised concatenation lets the larger-norm stream dominate the cosine
    similarity.
    """
    s = np.mean(spatial, axis=0) if len(spatial) else np.zeros(SPATIAL_DIM, np.float32)
    v = np.mean(temporal, axis=0) if len(temporal) else np.zeros(TEMPORAL_DIM, np.float32)

    if stream_norm == "l2":
        s = s / (np.linalg.norm(s) + 1e-12)
        v = v / (np.linalg.norm(v) + 1e-12)
    return s.astype(np.float32), v.astype(np.float32)
