"""Export per-clip embeddings from <emb>/<species>.npz to CSV.

One file per representation, <out>/<species>/<rep>.csv, with the clip's
manifest columns (individual, source_id, site, exclude) followed by the vector
(d0, d1, ...). Kinematic descriptors keep their names, plus `stride_resolved`:
0 where the analysed window spans < 1.5 cycles and the spectral peak sits on the
band edge (evaluate.py sets those stride frequencies to NaN), and the stride
periodicity checks from the QC table when present (stride_autocorr,
stride_cycles). <out>/<species>/qc.csv has the segmentation QC of every clip,
including those that failed. Excluded clips are kept and flagged, so the
analysis can decide.

    python tools/export_csv.py                    # every species -> results/csv/
    python tools/export_csv.py horse              # selected species
    python tools/export_csv.py --emb results/v1_largest_component/embeddings \\
        --qc results/v1_largest_component/segmentation_qc.csv \\
        --out results/v1_largest_component/csv
"""
import argparse
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from extract_features import MAX_FRAMES, load_manifests  # noqa: E402

META = ["clip_id", "species", "folder_species", "individual", "source_id", "site",
        "n_frames", "fps", "duration_s", "exclude"]
KIN_QC = ["stride_autocorr", "stride_cycles"]


def export(species, manifest, qc, emb, out_dir):
    z = np.load(os.path.join(emb, f"{species}.npz"))
    meta = pd.DataFrame({"clip_id": z["clip_id"]}).merge(manifest, on="clip_id", how="left")
    meta = meta[[c for c in META if c in meta]]
    os.makedirs(os.path.join(out_dir, species), exist_ok=True)
    for rep in z.files:
        if rep in ("clip_id", "kinematics_names"):
            continue
        x = z[rep]
        cols = list(z["kinematics_names"]) if rep == "kinematics" else \
            [f"d{i}" for i in range(x.shape[1])]
        out = pd.concat([meta, pd.DataFrame(x, columns=cols)], axis=1)
        if rep == "kinematics":
            seconds = np.minimum(out.n_frames.astype(float), MAX_FRAMES) / out.fps.astype(float)
            floor = np.maximum(0.3, 1.5 / seconds) + 0.02
            out["stride_resolved"] = (out.stride_frequency_hz > floor).astype(int)
            extra = [c for c in KIN_QC if c in qc]
            if extra:
                out = out.merge(qc[["clip_id"] + extra], on="clip_id", how="left")
        out.to_csv(os.path.join(out_dir, species, f"{rep}.csv"), index=False)
    sp_qc = manifest[manifest.species == species][["clip_id", "individual", "source_id",
                                                     "exclude"]].merge(qc, on="clip_id")
    sp_qc.to_csv(os.path.join(out_dir, species, "qc.csv"), index=False)
    print(f"{species}: {len(meta)} clips -> {os.path.relpath(out_dir, ROOT)}/{species}/")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="*")
    ap.add_argument("--emb", default=os.path.join(ROOT, "results", "embeddings"))
    ap.add_argument("--qc", default=os.path.join(ROOT, "results", "segmentation_qc.csv"))
    ap.add_argument("--out", default=os.path.join(ROOT, "results", "csv"))
    args = ap.parse_args()
    manifest = load_manifests()
    qc = pd.read_csv(args.qc).drop(columns=["n_frames"], errors="ignore")
    species = args.species or sorted(f[:-4] for f in os.listdir(args.emb) if f.endswith(".npz"))
    for sp in species:
        export(sp, manifest, qc, args.emb, args.out)


if __name__ == "__main__":
    main()
