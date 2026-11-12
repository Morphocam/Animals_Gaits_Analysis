"""Cuts the Franches-Montagnes stallion treadmill videos into side-view clips.

Source: Gmel, Maninchedda, Cruz & Weishaupt (2023), "Video clips of
Franches-Montagnes stallions at walk and trot on a treadmill during an
incremental speed test", Mendeley Data, V2, doi:10.17632/bwfkyvy593.2 (CC BY 4.0).
24 stallions, one video each (1920x1080, 25 fps). After a 10 s title, each
video shows four conditions in order: walk at 1.7 m/s, peak walk, trot at
4.5 m/s, peak trot; within each, about 20 s from behind, 20 s from the left
side and 20 s from the front. Some videos lack a view of a condition.

    python tools/prepare_fm_stallions.py --download     # fetch the 24 videos (7 GB)
    python tools/prepare_fm_stallions.py                # cut clips, write the manifest

Steps: (1) shot boundaries from frame differences at 1 s steps; (2) a horse is
detected on the middle frame of every shot with Grounding DINO and the shot is
called a side view when the box is wider than tall; (3) the condition of a
side view is the number of "from behind" shots before it (each condition starts
from behind); (4) three 5 s clips are cut from each side-view shot, away from
the shot boundaries, and written at 960x540.

Writes  input/horse/<FM id>/<FM id>_<condition>_<n>.mp4
        manifests/clips_fm_stallions.csv     source_id = stallion x condition
        results/dataset_check/fm_stallions_shots.csv and fm_stallions_side_views.jpg
"""
import json
import os
import sys
import urllib.request

import cv2
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "external", "fm_stallions")
OUT_VIDEO = os.path.join(ROOT, "input", "horse")
OUT_MANIFEST = os.path.join(ROOT, "manifests", "clips_fm_stallions.csv")
CHECK = os.path.join(ROOT, "results", "dataset_check")
RECORD = "https://data.mendeley.com/public-api/datasets/bwfkyvy593"
DOI = "doi:10.17632/bwfkyvy593.2"
CONDITIONS = ["walk17", "walkmax", "trot45", "trotmax"]
SPEED = {"walk17": "walk at 1.7 m/s", "walkmax": "peak walk (up to 2.0 m/s)",
         "trot45": "trot at 4.5 m/s", "trotmax": "peak trot (up to 6.5 m/s)"}
CLIPS_PER_SHOT, CLIP_SECONDS, EDGE = 3, 5.0, 2.5
CUT_THRESHOLD = 40.0        # mean absolute grey-level change between frames 1 s apart
SIZE = (960, 540)


def download():
    os.makedirs(RAW, exist_ok=True)
    req = urllib.request.Request(RECORD, headers={"Accept": "application/vnd.mendeley-public-dataset.1+json"})
    rec = json.load(urllib.request.urlopen(req))
    with open(os.path.join(RAW, "mendeley_record.json"), "w", encoding="utf-8") as f:
        json.dump(rec, f)
    for x in rec["files"]:
        name, dst = x["filename"], os.path.join(RAW, x["filename"])
        if not name.startswith("FM") or (os.path.exists(dst) and os.path.getsize(dst) == x["size"]):
            continue
        print("downloading", name, flush=True)
        urllib.request.urlretrieve(x["content_details"]["download_url"], dst)


def shots(path):
    """[(start_s, end_s)] of every shot after the title screen."""
    cap = cv2.VideoCapture(path)
    fps, n = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    prev, cuts, dark = None, [], []
    for s in range(int(n / fps)):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(s * fps))
        ok, f = cap.read()
        if not ok:
            break
        g = cv2.resize(cv2.cvtColor(f, cv2.COLOR_BGR2GRAY), (64, 36)).astype(float)
        dark.append(g.mean() < 20)
        if prev is not None and np.abs(g - prev).mean() > CUT_THRESHOLD:
            cuts.append(s)
        prev = g
    cap.release()
    bounds = cuts + [int(n / fps)]
    out = [(a, b) for a, b in zip(bounds[:-1], bounds[1:]) if not dark[min(a + 1, len(dark) - 1)]]
    return out, fps


def load_detector():
    import torch
    from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = AutoProcessor.from_pretrained("IDEA-Research/grounding-dino-tiny")
    model = AutoModelForZeroShotObjectDetection.from_pretrained("IDEA-Research/grounding-dino-tiny").to(dev).eval()

    def detect(frame_rgb):
        inputs = proc(images=frame_rgb, text="a horse.", return_tensors="pt").to(dev)
        with torch.no_grad():
            out = model(**inputs)
        res = proc.post_process_grounded_object_detection(out, inputs.input_ids, threshold=0.3,
                                                          text_threshold=0.25,
                                                          target_sizes=[frame_rgb.shape[:2]])[0]
        if not len(res["boxes"]):
            return None
        x0, y0, x1, y1 = res["boxes"][int(res["scores"].argmax())].tolist()
        return x0, y0, x1, y1
    return detect


def frame_at(path, second):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(second * cap.get(cv2.CAP_PROP_FPS)))
    ok, f = cap.read()
    cap.release()
    return f if ok else None


def cut(path, start, seconds, dst, fps):
    cap = cv2.VideoCapture(path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(start * fps)))
    vw = cv2.VideoWriter(dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, SIZE)
    n = 0
    for _ in range(int(round(seconds * fps))):
        ok, f = cap.read()
        if not ok:
            break
        vw.write(cv2.resize(f, SIZE, interpolation=cv2.INTER_AREA))
        n += 1
    vw.release()
    cap.release()
    return n


def main():
    if "--download" in sys.argv:
        download()
        return
    os.makedirs(CHECK, exist_ok=True)
    detect = load_detector()
    shot_rows, rows, thumbs = [], [], []
    only = [a for a in sys.argv[1:] if a.startswith("FM")]     # optional: redo selected stallions
    for name in sorted(f for f in os.listdir(RAW) if f.startswith("FM") and f.endswith(".mp4")):
        if only and name[:-4] not in only:
            continue
        for old in os.listdir(os.path.join(OUT_VIDEO, name[:-4])) if os.path.isdir(os.path.join(OUT_VIDEO, name[:-4])) else []:
            if any(f"_{c}_" in old for c in CONDITIONS):
                os.remove(os.path.join(OUT_VIDEO, name[:-4], old))
        horse, path = name[:-4], os.path.join(RAW, name)
        sh, fps = shots(path)
        group = -1                               # index of the current condition
        for a, b in sh:
            mid = frame_at(path, (a + b) / 2)
            box = detect(cv2.cvtColor(mid, cv2.COLOR_BGR2RGB)) if mid is not None else None
            aspect = (box[2] - box[0]) / max(1.0, box[3] - box[1]) if box else np.nan
            view = "side" if aspect > 1.15 else "front_or_behind" if aspect == aspect else "none"
            shot_rows.append(dict(horse=horse, start_s=a, end_s=b, aspect=round(aspect, 2), view=view))
        # Conditions: split the shot list into groups that begin "from behind".
        # Views alternate behind, (side), front; a front/behind shot directly
        # after a front shot begins the next condition.
        mine = [r for r in shot_rows if r["horse"] == horse]
        pos = 0                                  # 0 = expect behind, 1 = expect side/front, 2 = expect front
        for r in mine:
            if r["view"] == "side":
                if pos == 0:                     # the view from behind is missing
                    group += 1
                r["condition_index"] = group
                pos = 2
            else:
                if pos == 0:
                    group += 1
                    pos = 1
                else:                            # front view closes the condition
                    pos = 0
                r["condition_index"] = group
        for r in mine:
            if r["view"] != "side" or not 0 <= r["condition_index"] < len(CONDITIONS):
                continue
            cond = CONDITIONS[r["condition_index"]]
            a, b = r["start_s"], r["end_s"]
            starts = np.linspace(a + EDGE, b - EDGE - CLIP_SECONDS, CLIPS_PER_SHOT)
            if b - a < CLIPS_PER_SHOT * CLIP_SECONDS + 2 * EDGE - 1e-6:
                starts = np.arange(a + EDGE, b - EDGE - CLIP_SECONDS + 1e-6, CLIP_SECONDS)
            os.makedirs(os.path.join(OUT_VIDEO, horse), exist_ok=True)
            for k, s in enumerate(starts, 1):
                clip = f"{horse}_{cond}_{k}.mp4"
                dst = os.path.join(OUT_VIDEO, horse, clip)
                n = cut(path, float(s), CLIP_SECONDS, dst, fps)
                rows.append(dict(
                    clip_id=f"horse/{horse}/{clip[:-4]}", species="horse", individual=horse, clip=clip,
                    path=os.path.relpath(dst, ROOT).replace("\\", "/"), chunk_index=k, n_frames=n,
                    fps=fps, width=SIZE[0], height=SIZE[1], duration_s=round(n / fps, 2),
                    source_id=f"horse-{horse}-{cond}", source_inferred=0,
                    source_type="laboratory treadmill, fixed side camera (Mendeley Data)",
                    source_ref=DOI, label_basis="dataset annotation (stallion code in the video title)",
                    notes=f"{SPEED[cond]}; {name} {s:.1f}-{s + CLIP_SECONDS:.1f} s",
                    visible_species="", exclude="", condition=cond,
                    gait="walk" if cond.startswith("walk") else "trot"))
            t = cv2.resize(frame_at(path, (a + b) / 2), (240, 135))
            thumbs.append(cv2.putText(t, f"{horse} {cond}", (4, 16), 0, 0.45, (0, 255, 255), 1))
        print(horse, [(r["view"][0], r.get("condition_index")) for r in mine], flush=True)
    shots_df, df = pd.DataFrame(shot_rows), pd.DataFrame(rows)
    if only:                                     # merge with the rows of the other stallions
        old_s = pd.read_csv(os.path.join(CHECK, "fm_stallions_shots.csv"))
        shots_df = pd.concat([old_s[~old_s.horse.isin(only)], shots_df]).sort_values(["horse", "start_s"])
        old_m = pd.read_csv(OUT_MANIFEST, keep_default_na=False)
        df = pd.concat([old_m[~old_m.individual.isin(only)], df]).sort_values("clip_id")
    shots_df.to_csv(os.path.join(CHECK, "fm_stallions_shots.csv"), index=False)
    df.to_csv(OUT_MANIFEST, index=False)
    if only:
        print(df.groupby(["individual", "condition"]).size().unstack(fill_value=0).loc[only].to_string())
        return
    thumbs += [np.zeros_like(thumbs[0])] * (-len(thumbs) % 8)
    cv2.imwrite(os.path.join(CHECK, "fm_stallions_side_views.jpg"),
                np.vstack([np.hstack(thumbs[i:i + 8]) for i in range(0, len(thumbs), 8)]))
    print(df.groupby(["individual", "condition"]).size().unstack(fill_value=0).to_string())
    print(f"Wrote {len(df)} clips of {df.individual.nunique()} stallions -> {OUT_MANIFEST}")


if __name__ == "__main__":
    main()
