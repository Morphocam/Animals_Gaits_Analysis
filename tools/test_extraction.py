"""Smoke test: one clip through segmentation + both streams, with timings.

Run from the project root:  python tools/test_extraction.py
"""
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.chdir(ROOT)

import numpy as np

from src.feature_extractor import (SamVideoSegmenter, apply_rgb_mask,
                                   build_silhouette_windows, extract_frames,
                                   get_resnet_model, get_sam_predictor,
                                   get_spatial_features, get_temporal_features,
                                   get_videoprism_model, sequence_embedding)

PATH = "input/camel/A/A_1.mp4"

t0 = time.time()
frames, fps = extract_frames(PATH, max_frames=48)
frames = frames[::2]
fps = fps / 2
print(f"frames={len(frames)} fps={fps} shape={frames[0].shape} ({time.time()-t0:.1f}s)")

t0 = time.time()
segmenter = SamVideoSegmenter(get_sam_predictor())
print(f"SAM loaded ({time.time()-t0:.1f}s)")

t0 = time.time()
masks = segmenter.segment_video(frames)
n_valid = sum(m is not None for m in masks)
areas = [float(m.mean()) for m in masks if m is not None]
print(f"segmented {n_valid}/{len(frames)} frames in {time.time()-t0:.1f}s "
      f"({(time.time()-t0)/len(frames):.2f}s/frame), "
      f"mask area frac mean={np.mean(areas):.3f}")

# Verify the two streams really do get different inputs.
rgb_masked = [apply_rgb_mask(f, m) for f, m in zip(frames, masks) if m is not None]
assert rgb_masked[0].shape == frames[0].shape
bg = rgb_masked[0][~masks[0]]
assert bg.max() == 0, "background of the RGB mask must be zeroed"
print("RGB mask R_t = B_t . I_t OK (background zeroed)")

t0 = time.time()
resnet = get_resnet_model()
spatial = get_spatial_features(resnet, rgb_masked)
print(f"spatial {spatial.shape} in {time.time()-t0:.1f}s")
assert spatial.shape[1] == 512, spatial.shape

windows = build_silhouette_windows(masks, fps)
print(f"silhouette windows: {len(windows)} x {windows[0].shape}")
w = windows[0]
assert w.shape[1:] == (288, 288, 3), w.shape
assert set(np.unique(w)) <= {0.0, 1.0}, "silhouettes must be texture-free binary"
assert np.allclose(w[..., 0], w[..., 1]), "channels should be replicated"
print(f"silhouette values are binary {np.unique(w)}, foreground frac={w[..., 0].mean():.3f}")

t0 = time.time()
vp_model, vp_params = get_videoprism_model()
print(f"VideoPrism loaded ({time.time()-t0:.1f}s)")

t0 = time.time()
temporal = get_temporal_features(vp_model, vp_params, windows)
print(f"temporal {temporal.shape} in {time.time()-t0:.1f}s")
assert temporal.shape[1] == 768, temporal.shape

s, v = sequence_embedding(spatial, temporal)
print(f"fused h_bar dim = {len(np.concatenate([s, v]))} "
      f"(spatial 512 + temporal 768)")
print(f"stream norms -> spatial {np.linalg.norm(s):.3f}, temporal {np.linalg.norm(v):.3f}")
print("\nSmoke test passed.")
