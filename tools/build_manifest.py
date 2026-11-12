"""Builds manifests/clips.csv: one row per clip with provenance columns.

Every clip under input/<species>/<individual>/<clip>.mp4 gets a row. The
`source_id` column groups clips cut from the same original recording; it is the
unit used for leakage-free (source-grouped) evaluation in evaluate.py.

Source ids are *inferred* conservatively when no curated value exists: clips of
one individual that share resolution and frame rate are assumed to come from
one recording (most clips are consecutive 3-5 s cuts of a longer video). Merging
too much only makes the cross-source evaluation stricter, never leakier.

Curated values are preserved: if manifests/clips.csv already exists, its
curated columns are
kept for every clip still present, and `source_inferred` is set to 0 for rows
whose source_id was edited by hand. `visible_species` overrides the folder name
when a clip was filed under the wrong species; a non-empty `exclude` (the
reason) drops the clip from evaluation. Re-run after adding species or clips:

    python tools/build_manifest.py
"""
import os
import re
import sys

import cv2
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT = os.path.join(ROOT, "input")
OUT = os.path.join(ROOT, "manifests", "clips.csv")
CURATED = ["source_id", "source_type", "source_ref", "label_basis", "notes",
           "visible_species", "exclude"]


def clip_index(name):
    """Numeric chunk index from <individual>_<n>.mp4 (used to order clips)."""
    m = re.search(r"_(\d+)\.mp4$", name)
    return int(m.group(1)) if m else -1


def probe(path):
    cap = cv2.VideoCapture(path)
    info = dict(n_frames=int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                fps=round(cap.get(cv2.CAP_PROP_FPS) or 0.0, 2),
                width=int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                height=int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    cap.release()
    info["duration_s"] = round(info["n_frames"] / info["fps"], 2) if info["fps"] else 0.0
    return info


def owned_elsewhere():
    """Clips listed in another manifest (manifests/clips_*.csv, written by the
    tools/prepare_*.py scripts for public datasets) are not indexed again here."""
    d, paths = os.path.dirname(OUT), set()
    for f in os.listdir(d) if os.path.isdir(d) else []:
        if f.endswith(".csv") and f != os.path.basename(OUT):
            paths |= set(pd.read_csv(os.path.join(d, f), usecols=["path"]).path)
    return paths


def main():
    rows = []
    skip = owned_elsewhere()
    for species in sorted(os.listdir(INPUT)):
        sp_dir = os.path.join(INPUT, species)
        if not os.path.isdir(sp_dir):
            continue
        for individual in sorted(os.listdir(sp_dir)):
            ind_dir = os.path.join(sp_dir, individual)
            if not os.path.isdir(ind_dir):
                continue
            for clip in sorted(os.listdir(ind_dir), key=clip_index):
                if not clip.lower().endswith(".mp4"):
                    continue
                rel = os.path.relpath(os.path.join(ind_dir, clip), ROOT).replace("\\", "/")
                if rel in skip:
                    continue
                rows.append(dict(clip_id=f"{species}/{individual}/{clip[:-4]}",
                                 species=species, individual=individual,
                                 clip=clip, path=rel, chunk_index=clip_index(clip),
                                 **probe(os.path.join(ind_dir, clip))))
    if not rows:
        sys.exit(f"No clips found under {INPUT}")
    df = pd.DataFrame(rows)

    # Conservative inference: one source per (individual, resolution, fps).
    fmt = (df["width"].astype(str) + "x" + df["height"].astype(str)
           + "@" + df["fps"].astype(str))
    df["inferred_source"] = ""
    for (sp, ind), g in df.groupby(["species", "individual"], sort=False):
        for k, key in enumerate(dict.fromkeys(fmt[g.index])):
            df.loc[g.index[fmt[g.index] == key], "inferred_source"] = f"{sp}-{ind}-s{k + 1}"

    df["source_id"] = df["inferred_source"]
    df["source_inferred"] = 1
    for col in CURATED[1:]:
        df[col] = ""

    if os.path.exists(OUT):
        old = pd.read_csv(OUT, dtype=str, keep_default_na=False).set_index("clip_id")
        for col in CURATED:
            if col in old.columns:
                keep = df["clip_id"].isin(old.index)
                df.loc[keep, col] = old.loc[df.loc[keep, "clip_id"], col].values
        df["source_inferred"] = (df["source_id"] == df["inferred_source"]).astype(int)

    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    df.drop(columns="inferred_source").to_csv(OUT, index=False)

    summary = df.groupby("species").agg(individuals=("individual", "nunique"),
                                        clips=("clip", "size"),
                                        sources=("source_id", "nunique"))
    print(summary.to_string())
    multi = df.groupby(["species", "individual"])["source_id"].nunique()
    print(f"\nIndividuals with >=2 source recordings: {int((multi >= 2).sum())} / {len(multi)}")
    print(f"Wrote {OUT} ({len(df)} clips)")


if __name__ == "__main__":
    main()
