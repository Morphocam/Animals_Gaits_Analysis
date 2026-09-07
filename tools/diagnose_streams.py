"""Diagnose how much of each stream's embedding is a shared constant offset.

Usage:  python tools/diagnose_streams.py [output_dir]
"""
import os
import sys

import numpy as np

os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
out_dir = sys.argv[1] if len(sys.argv) > 1 else "output_subset_camel"

d = np.load(os.path.join(out_dir, "embeddings.npz"), allow_pickle=True)
sp, tp = d["spatial"], d["temporal"]
names = [str(n) for n in d["names"]]
ids = [str(i) for i in d["individuals"]]
print("names:", names)
print("ids  :", ids)

for label, X in (("spatial", sp), ("temporal", tp)):
    mu = X.mean(0)
    # How much of each vector is the shared mean component?
    frac = np.linalg.norm(mu) / np.linalg.norm(X, axis=1).mean()
    resid = X - mu
    print(f"\n--- {label} ({X.shape}) ---")
    print(f"  ||mean|| / mean||x||        = {frac:.4f}   "
          f"(1.0 => all clips share one direction)")
    print(f"  mean residual norm / norm   = "
          f"{(np.linalg.norm(resid, axis=1) / np.linalg.norm(X, axis=1)).mean():.4f}")

    def cos(M):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
        return Mn @ Mn.T

    for tag, M in (("raw", X), ("mean-centred", resid)):
        C = cos(M)
        iu = np.triu_indices(len(M), 1)
        same = np.array(ids)[iu[0]] == np.array(ids)[iu[1]]
        intra, inter = C[iu][same], C[iu][~same]
        print(f"  {tag:13s}: intra {intra.mean():+.4f}  inter {inter.mean():+.4f}  "
              f"separation {intra.mean() - inter.mean():+.4f}")

# Distance spread vs the paper's 0.15 dendrogram cut.
print("\n--- cosine distance spread vs dendrogram cut 0.15 ---")
for label, X in (("spatial", sp), ("temporal", tp),
                 ("fused", np.concatenate([sp, tp], 1))):
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    D = 1 - Xn @ Xn.T
    iu = np.triu_indices(len(X), 1)
    print(f"  {label:9s}: d in [{D[iu].min():.4f}, {D[iu].max():.4f}]  "
          f"{'ALL BELOW 0.15 -> single cluster' if D[iu].max() < 0.15 else 'spans the cut'}")
