"""Recompute the paper's Table 2 from the published CSVs in CSVs/.

Table 2 caption: diagonal self-comparisons excluded, each unordered pair counted
once after averaging the upper- and lower-triangular values.

Usage:  python tools/verify_table2.py
"""
import os

import numpy as np
import pandas as pd

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

PAPER = {  # (intra_mean, intra_sd, inter_mean, inter_sd)
    "camel": (0.921, 0.076, 0.858, 0.065),
    "giraffe": (0.948, 0.037, 0.800, 0.051),
    "hyena": (0.976, 0.009, 0.747, 0.061),
    "lion": (0.965, 0.026, 0.913, 0.056),
    "zebra": (0.961, 0.025, 0.777, 0.051),
}

print(f"{'species':9s} {'n':>4s}  {'computed intra':>18s} {'paper intra':>16s}   "
      f"{'computed inter':>18s} {'paper inter':>16s}")
for sp, (pim, pis, pem, pes) in PAPER.items():
    # utf-8-sig strips the BOM present in the zebra file.
    df = pd.read_csv(f"CSVs/similarity_{sp}_v3.csv", encoding="utf-8-sig")
    label_col = df.columns[1]

    ids = df["cluster_name"].astype(str).str.strip().values
    clips = df[label_col].astype(str).str.strip().values

    mat = df.drop(columns=["cluster_name", label_col])
    mat.columns = [str(c).strip() for c in mat.columns]
    mat.index = clips
    mat = mat.apply(pd.to_numeric, errors="coerce")

    common = [c for c in mat.index if c in set(mat.columns)]
    A = mat.loc[common, common].values.astype(float)
    ids = np.array([ids[list(clips).index(c)] for c in common])

    A = (A + A.T) / 2.0                      # average the two triangles
    iu = np.triu_indices_from(A, k=1)        # exclude the diagonal
    vals, same = A[iu], ids[iu[0]] == ids[iu[1]]
    intra, inter = vals[~np.isnan(vals) & same], vals[~np.isnan(vals) & ~same]

    print(f"{sp:9s} {len(common):4d}  {intra.mean():10.3f} +/- {intra.std():.3f} "
          f"{pim:10.3f} +/- {pis:.3f}   "
          f"{inter.mean():10.3f} +/- {inter.std():.3f} {pem:10.3f} +/- {pes:.3f}")
