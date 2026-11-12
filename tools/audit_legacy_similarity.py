"""Audit a legacy CSVs/similarity_<species>_v3.csv matrix: can it be a cosine
similarity matrix, and can its source clips be traced?  Writes
results/legacy_audit/<species>_audit.txt."""
import os, sys, glob, collections
import numpy as np, pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sp = sys.argv[1] if len(sys.argv) > 1 else "zebra"
out_dir = os.path.join(ROOT, "results", "legacy_audit"); os.makedirs(out_dir, exist_ok=True)


def read(path):
    df = pd.read_csv(path, encoding="utf-8-sig")
    labels = df.iloc[:, 1].astype(str).str.strip().tolist()
    return labels, df.iloc[:, 2:].to_numpy(float)


labels, m = read(os.path.join(ROOT, "CSVs", f"similarity_{sp}_v3.csv"))
n = len(labels); ids = np.array([l.split("_")[0] for l in labels])
L = []
L.append(f"{sp}: {n} clips, individuals {dict(collections.Counter(ids))}")
d = np.diag(m)
L.append(f"diagonal (self-similarity, must be 1): min {d.min():.6f} max {d.max():.6f}; "
         f"{(np.abs(d - 1) > 1e-4).sum()} of {n} differ from 1 by > 1e-4: {np.round(d, 6).tolist()}")
asym = np.abs(m - m.T); iu = np.triu_indices(n, 1)
L.append(f"asymmetry |m[i,j]-m[j,i]|: {(asym[iu] > 1e-6).sum()} of {len(iu[0])} pairs differ; "
         f"max {asym.max():.3f}, mean {asym[iu].mean():.3f}")
ev = np.linalg.eigvalsh((m + m.T) / 2)
L.append(f"eigenvalues of symmetrised matrix: min {ev.min():.4f} ({(ev < -1e-6).sum()} negative)")
off = m[~np.eye(n, dtype=bool)]
cnt = collections.Counter(np.round(off, 8).tolist())
rep = [(v, c) for v, c in cnt.most_common() if c > 2]
L.append(f"off-diagonal cells: {len(off)}, distinct values: {len(cnt)}")
L.append("values occurring in more than one symmetric pair (count > 2): "
         + ", ".join(f"{v} x{c}" for v, c in rep[:15]))
# fractional-digit tails shared between different values (e.g. .x41553, .x0635)
tails = collections.Counter(f"{v:.8f}"[4:] for v in cnt)
L.append("shared decimal tails (digits after the 2nd decimal) among distinct values: "
         + ", ".join(f"..{t} x{c}" for t, c in tails.most_common(8) if c > 1))
same = ids[:, None] == ids[None, :]
intra = m[same & ~np.eye(n, dtype=bool)]; inter = m[~same]
L.append(f"intra-individual mean {intra.mean():.3f} sd {intra.std():.3f}; "
         f"inter {inter.mean():.3f} sd {inter.std():.3f}")
# values shared with the other species' files
mine = set(cnt)
for f in sorted(glob.glob(os.path.join(ROOT, "CSVs", "similarity_*_v3.csv"))):
    o = os.path.basename(f).split("_")[1]
    if o.lower() == sp.lower():
        continue
    _, mo = read(f)
    oo = set(np.round(mo[~np.eye(len(mo), dtype=bool)], 8).tolist())
    L.append(f"distinct values shared with {o}: {len(mine & oo)} of {len(mine)}")
man = pd.read_csv(os.path.join(ROOT, "manifests", "clips.csv"))
z = man[man.astype(str).apply(lambda c: c.str.contains(sp, case=False)).any(axis=1)]
L.append(f"clips in manifests/clips.csv mentioning '{sp}': {len(z)} "
         f"({z.individual.nunique() if len(z) else 0} individual folders: "
         f"{sorted((z.folder + '/' + z.individual).unique()) if 'folder' in z else sorted(z.individual.unique())})")
txt = "\n".join(L); print(txt)
open(os.path.join(out_dir, f"{sp}_audit.txt"), "w", encoding="utf-8").write(txt + "\n")
