"""Locomotion variables per clip from the pose keypoints (output/pose/*.npz).

    python gait_kinematics.py

All variables are dimensionless or in Hz, so they need no camera calibration:
lengths are in trunk lengths (shoulder to hip), and each paw is tracked relative
to the trunk centre, which removes camera panning and makes treadmill and
overground clips comparable.

For each paw, the fore-aft position relative to the trunk is a periodic signal:
the paw moves backward relative to the body while it is on the ground (stance)
and forward while it swings. A harmonic model at the stride frequency is fitted
to each paw, and from the fitted cycle we take

  stride_frequency_hz   common fundamental of the four paw signals
  duty_factor           fraction of the stride a paw moves backward (stance), mean of paws
  step_length           fore-aft excursion of the paw during stance, trunk lengths
  stride_length         step_length / duty_factor, trunk lengths
  speed                 stride_length x stride_frequency, trunk lengths per second
  limb_phase            Hildebrand's limb phase: delay of the fore footfall after the
                        same-side hind footfall, fraction of the stride (0 pace,
                        0.25 lateral-sequence singlefoot walk, 0.5 trot)
  fore_pair_phase       delay between the two fore footfalls (0.5 = symmetrical gait)
  hind_pair_phase       same for the hind pair
  fore_rom_deg          range of the shoulder-to-paw angle (protraction-retraction)
  hind_rom_deg          range of the hip-to-paw angle
  head_height           nose above the shoulder, trunk lengths (clip median)
  head_bob              SD of that height after removing its trend

Quality per clip: fit_r2 (variance of the paw signals explained by the harmonic
model, worst paw), n_cycles, paw_score (mean keypoint confidence of the paws).
A clip has a "resolved gait" when fit_r2 >= MIN_R2 and n_cycles >= MIN_CYCLES.

Writes results/kinematics/pose_gait.csv, traces.npz (fitted cycles, for
figures), and results/pose_check/<species>_traces.jpg.
"""
import os

import numpy as np
import pandas as pd

from extract_features import ROOT, load_manifests
from extract_pose import K, POSE, facing

OUT = os.path.join(ROOT, "results", "kinematics")
CHECK = os.path.join(ROOT, "results", "pose_check")
MIN_SCORE = 0.3        # keypoints below this confidence are treated as missing
MAX_MISSING = 0.3      # a paw missing in more than this fraction of frames is unusable
MIN_R2 = 0.5           # harmonic fit, worst paw
MIN_CYCLES = 1.5
BAND = (0.3, 3.0)      # Hz
PAWS = ["l_front_paw", "r_front_paw", "l_hind_paw", "r_hind_paw"]
GAIT_VARS = ["stride_frequency_hz", "duty_factor", "step_length", "stride_length", "speed",
             "limb_phase", "fore_pair_phase", "hind_pair_phase", "fore_rom_deg",
             "hind_rom_deg", "head_height", "head_bob"]
CIRCULAR = ["limb_phase", "fore_pair_phase", "hind_pair_phase"]


def fill(v, ok):
    """Linear interpolation over missing frames."""
    t = np.arange(len(v))
    return np.interp(t, t[ok], v[ok]) if ok.sum() >= 2 else np.full(len(v), np.nan)


def detrend(v):
    t = np.arange(len(v))
    return v - np.polyval(np.polyfit(t, v, 1), t)


def harmonic_fit(v, f0, fps, n_harm):
    """Least-squares fit of n_harm harmonics of f0. Returns coefficients and R2."""
    t = np.arange(len(v)) / fps
    cols = [np.ones_like(t)]
    for h in range(1, n_harm + 1):
        cols += [np.cos(2 * np.pi * h * f0 * t), np.sin(2 * np.pi * h * f0 * t)]
    A = np.stack(cols, 1)
    coef = np.linalg.lstsq(A, v, rcond=None)[0]
    res = v - A @ coef
    return coef, 1 - res.var() / (v.var() + 1e-12)


def cycle(coef, n=200):
    """One fitted stride, phase 0..1."""
    ph = np.arange(n) / n
    y = np.full(n, coef[0])
    for h in range(1, (len(coef) - 1) // 2 + 1):
        y += coef[2 * h - 1] * np.cos(2 * np.pi * h * ph) + coef[2 * h] * np.sin(2 * np.pi * h * ph)
    return y


def circ_mean(p):
    p = np.asarray(p, float)
    p = p[np.isfinite(p)]
    if not len(p):
        return np.nan
    return float(np.angle(np.exp(2j * np.pi * p).mean()) / (2 * np.pi) % 1)


def median3(v):
    w = np.stack([np.r_[v[0], v[:-1]], v, np.r_[v[1:], v[-1]]])
    return np.median(w, 0)


def clip_gait(z):
    xy, sc, fps = z["xy"].astype(float), z["score"].copy(), float(z["fps"])
    T = len(xy)
    out = dict(n_frames=T, fps=fps, paw_score=float(sc[:, [K[p] for p in PAWS]].mean()))
    if T < 8:
        return out, None
    # Facing direction from the pose, not from the drift of the mask, which is
    # zero on a treadmill and unreliable when the camera pans.
    direction = facing(xy, sc)
    out["facing"] = direction
    x, y = xy[..., 0] * direction, xy[..., 1]            # +x forward, +y down

    def joint(names):
        """Confidence-weighted mean of one or two keypoints, gaps interpolated."""
        idx = [K[n] for n in names]
        w = np.where(sc[:, idx] >= MIN_SCORE, sc[:, idx], 0.0)
        ok = w.sum(1) > 0
        if ok.mean() < 1 - MAX_MISSING:
            return None, None
        ws = np.maximum(w.sum(1), 1e-9)
        return fill((x[:, idx] * w).sum(1) / ws, ok), fill((y[:, idx] * w).sum(1) / ws, ok)

    sx, sy = joint(["l_shoulder", "r_shoulder"])
    hx, hy = joint(["l_hip", "r_hip"])
    if sx is None or hx is None:
        return out, None
    trunk = float(np.median(np.hypot(sx - hx, sy - hy)))
    if trunk < 5:
        return out, None
    cx = (sx + hx) / 2
    out["trunk_px"] = trunk

    rel, ang = {}, {}
    for p in PAWS:
        px, py = joint([p])
        if px is None:
            return out, None
        px, py = median3(px), median3(py)
        rel[p] = detrend((px - cx) / trunk)
        ox, oy = (sx, sy) if "front" in p else (hx, hy)
        ang[p] = np.degrees(np.arctan2(px - ox, py - oy))    # 0 = vertical, + = forward

    # Stride frequency: peak of the summed spectrum of the four paw signals
    n_fft = max(1024, T)
    freqs = np.fft.rfftfreq(n_fft, 1 / fps)
    power = sum(np.abs(np.fft.rfft(rel[p] * np.hanning(T), n_fft)) ** 2 / (rel[p].var() + 1e-12)
                for p in PAWS)
    band = (freqs >= BAND[0]) & (freqs <= min(BAND[1], 0.45 * fps))
    f0 = float(freqs[band][np.argmax(power[band])])
    n_cycles = f0 * T / fps
    n_harm = 3 if fps / f0 >= 12 else 2

    duty, step, touchdown, r2, cycles = {}, {}, {}, {}, {}
    for p in PAWS:
        coef, r2[p] = harmonic_fit(rel[p], f0, fps, n_harm)
        c = cycle(coef)
        v = np.gradient(c)
        duty[p] = float((v < 0).mean())                       # paw moving backward = stance
        step[p] = float(np.ptp(c))
        touchdown[p] = float(np.argmax(c) / len(c))           # most forward point
        cycles[p] = c
    ph = lambda a, b: (touchdown[a] - touchdown[b]) % 1.0
    nose_ok = sc[:, K["nose"]] >= MIN_SCORE
    head = (sy - fill(y[:, K["nose"]], nose_ok)) / trunk if nose_ok.mean() >= 1 - MAX_MISSING \
        else np.full(T, np.nan)
    rom = lambda names: float(np.mean([np.percentile(ang[n], 95) - np.percentile(ang[n], 5)
                                       for n in names]))
    d = float(np.mean(list(duty.values())))
    s = float(np.mean(list(step.values())))
    out.update(stride_frequency_hz=f0, duty_factor=d, step_length=s,
               stride_length=s / d if d > 0 else np.nan, speed=s / d * f0 if d > 0 else np.nan,
               limb_phase=circ_mean([ph("l_front_paw", "l_hind_paw"),
                                     ph("r_front_paw", "r_hind_paw")]),
               fore_pair_phase=ph("r_front_paw", "l_front_paw"),
               hind_pair_phase=ph("r_hind_paw", "l_hind_paw"),
               fore_rom_deg=rom(PAWS[:2]), hind_rom_deg=rom(PAWS[2:]),
               head_height=float(np.nanmedian(head)),
               head_bob=float(np.std(detrend(head))) if np.isfinite(head).all() else np.nan,
               fit_r2=float(min(r2.values())), fit_r2_mean=float(np.mean(list(r2.values()))),
               n_cycles=float(n_cycles))
    for p in PAWS:                       # per paw, for footfall diagrams
        out["td_" + p[:-4]] = ph(p, "l_hind_paw")
        out["duty_" + p[:-4]] = duty[p]
    out["resolved"] = bool(out["fit_r2"] >= MIN_R2 and n_cycles >= MIN_CYCLES)
    trace = dict(rel=np.stack([rel[p] for p in PAWS]), cycle=np.stack([cycles[p] for p in PAWS]),
                 touchdown=np.array([touchdown[p] for p in PAWS]),
                 duty=np.array([duty[p] for p in PAWS]), f0=f0, fps=fps)
    return out, trace


def trace_sheet(species, items):
    """Paw traces of up to 8 clips per species, to check the signals by eye."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    items = items[:8]
    fig, axes = plt.subplots(len(items), 1, figsize=(8, 1.5 * len(items)), squeeze=False)
    col = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    for ax, (cid, row, tr) in zip(axes[:, 0], items):
        t = np.arange(tr["rel"].shape[1]) / tr["fps"]
        for j, p in enumerate(PAWS):
            ax.plot(t, tr["rel"][j], color=col[j], lw=1, label=p.replace("_paw", ""))
        ax.set_title(f"{cid}  f={row['stride_frequency_hz']:.2f} Hz  duty={row['duty_factor']:.2f}  "
                     f"limb phase={row['limb_phase']:.2f}  R2={row['fit_r2']:.2f}  "
                     f"{'resolved' if row['resolved'] else 'not resolved'}", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0, 0].legend(fontsize=6, ncol=4, loc="upper right")
    axes[-1, 0].set_xlabel("time (s)", fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(CHECK, f"{species}_traces.jpg"), dpi=110)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(CHECK, exist_ok=True)
    df = load_manifests()
    rows, traces, sheets = [], {}, {}
    for r in df.itertuples():
        p = os.path.join(POSE, *r.clip_id.split("/")) + ".npz"
        if not os.path.exists(p):
            continue
        row, tr = clip_gait(np.load(p))
        row = dict(clip_id=r.clip_id, species=r.species, individual=r.individual, **row)
        row.setdefault("resolved", False)
        rows.append(row)
        if tr is not None:
            for k, v in tr.items():
                traces[f"{r.clip_id}|{k}"] = v
            if r.exclude == "":
                sheets.setdefault(r.species, []).append((r.clip_id, row, tr))
    out = pd.DataFrame(rows)
    for c in GAIT_VARS + ["fit_r2", "fit_r2_mean", "n_cycles", "facing", "trunk_px"] +             [f"{a}_{p[:-4]}" for a in ("td", "duty") for p in PAWS]:
        if c not in out:
            out[c] = np.nan
    cols = ["clip_id", "species", "individual", "resolved"] + GAIT_VARS + \
           ["fit_r2", "fit_r2_mean", "n_cycles", "paw_score", "facing", "trunk_px",
            "n_frames", "fps"] + [f"{a}_{p[:-4]}" for a in ("td", "duty") for p in PAWS]
    out = out[cols]
    out.to_csv(os.path.join(OUT, "pose_gait.csv"), index=False)
    np.savez_compressed(os.path.join(OUT, "traces.npz"), **traces)
    for sp, items in sheets.items():
        step = max(1, len(items) // 8)
        trace_sheet(sp, items[::step])
    ok = out[out.resolved]
    print(out.groupby("species").resolved.agg(["size", "sum"]).to_string())
    print(ok.groupby("species")[GAIT_VARS].median().round(2).T.to_string())


if __name__ == "__main__":
    main()
