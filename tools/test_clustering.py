"""Fast synthetic checks of the Sec 2.5 clustering maths (no models needed).

Run from anywhere:  python tools/test_clustering.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from sklearn.manifold import MDS

from src import clustering as clu

rng = np.random.default_rng(0)

# 3 individuals x 5 clips, tight within-identity clusters.
centres = rng.normal(size=(3, 64))
feats, ids = [], []
for k in range(3):
    for _ in range(5):
        feats.append(centres[k] + 0.05 * rng.normal(size=64))
        ids.append("ABC"[k])
feats = np.stack(feats)

A = clu.similarity_matrix(feats)
assert np.allclose(np.diag(A), 1.0, atol=1e-5), "diagonal must be 1"
assert np.allclose(A, A.T, atol=1e-6), "similarity must be symmetric"
print("similarity OK", A.shape)

D = clu.dissimilarity_matrix(A)
assert np.allclose(np.diag(D), 0.0), "distance diagonal must be 0"
assert (D >= 0).all(), "distances must be non-negative"
print("dissimilarity OK  range [%.4f, %.4f]" % (D.min(), D.max()))

# Classical MDS must preserve pairwise distances well for a near-Euclidean D.
X = clu.classical_mds(D, 2)
assert X.shape == (15, 2), X.shape
from scipy.spatial.distance import pdist, squareform
recon = squareform(pdist(X))
corr = np.corrcoef(squareform(recon), squareform(D))[0, 1]
print("classical MDS OK  distance correlation = %.4f" % corr)
assert corr > 0.9, "MDS should approximately preserve distances"

# Cross-check against sklearn's classical (non-metric=False) MDS embedding.
sk = MDS(n_components=2, dissimilarity="precomputed", random_state=0,
         normalized_stress=False).fit_transform(D)
sk_corr = np.corrcoef(squareform(squareform(pdist(sk))), squareform(D))[0, 1]
print("sklearn MDS distance correlation = %.4f (ours %.4f)" % (sk_corr, corr))

km = clu.kmeans_on_mds(X, 3)
print("kmeans labels:", km)
agree = clu.cluster_identity_agreement(km, ids)
assert agree["adjusted_rand_index"] > 0.99, agree
print("kmeans recovers identities, ARI = %.3f" % agree["adjusted_rand_index"])

Z, hier = clu.hierarchical_clusters(D, cut=0.15)
print("hierarchical labels:", hier, "n =", len(set(hier)))

sil = clu.silhouette_on_distances(D, km)
print("silhouette (precomputed) = %.4f" % sil)
assert 0.5 < sil <= 1.0, sil

stats = clu.intra_inter_similarity(A, ids)
n = len(ids)
assert stats["intra_pairs"] + stats["inter_pairs"] == n * (n - 1) // 2, stats
assert stats["intra_pairs"] == 3 * (5 * 4 // 2), stats
print("Table 2 stats:", {k: round(v, 4) if isinstance(v, float) else v
                         for k, v in stats.items()})
assert stats["intra_mean"] > stats["inter_mean"]

# Degenerate case: a single cluster must yield NaN rather than crashing.
assert np.isnan(clu.silhouette_on_distances(D, np.zeros(15, int)))
print("degenerate silhouette -> NaN OK")

print("\nAll clustering checks passed.")
