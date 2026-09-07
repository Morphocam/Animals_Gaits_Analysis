"""Similarity and clustering analysis.

Implements Sections 2.4 and 2.5 of GaitAnalysis.pdf:

    A_ij = <h_i, h_j> / (||h_i|| ||h_j||)     cosine similarity   (Sec 2.4)
    d_ij = 1 - A_ij                           cosine dissimilarity (Sec 2.5)
    G = -1/2 J D^(2) J,  X = Q_2 Lambda_2^0.5 classical MDS        (Sec 2.5)
    k-means on X, K = #annotated individuals
    average-linkage hierarchical clustering on d_ij, cut at 0.15
    silhouette coefficient on the ORIGINAL d_ij (not the MDS projection)
"""

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from scipy.spatial.distance import squareform
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import cosine_similarity

# Paper Sec 2.5 experimental settings.
LINKAGE_METHOD = "average"
DENDROGRAM_CUT = 0.15
KMEANS_N_INIT = 50
RANDOM_SEED = 42


def similarity_matrix(features):
    """A_ij, clipped to [-1, 1] to absorb floating-point overshoot."""
    return np.clip(cosine_similarity(features), -1.0, 1.0)


def dissimilarity_matrix(similarity):
    """d_ij = 1 - A_ij, forced symmetric with an exact zero diagonal."""
    d = 1.0 - similarity
    d = (d + d.T) / 2.0
    np.fill_diagonal(d, 0.0)
    return np.maximum(d, 0.0)


def classical_mds(dissimilarity, n_components=2):
    """Classical (Torgerson) MDS, following the paper's formulation directly.

    G = -1/2 J D^(2) J with J = I - (1/N) 11^T; coordinates from the largest
    positive eigenvalues, X = Q_2 Lambda_2^(1/2).
    """
    n = dissimilarity.shape[0]
    d_squared = dissimilarity ** 2
    j = np.eye(n) - np.ones((n, n)) / n
    gram = -0.5 * j @ d_squared @ j
    gram = (gram + gram.T) / 2.0  # enforce symmetry before eigendecomposition

    eigvals, eigvecs = np.linalg.eigh(gram)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]

    keep = eigvals[:n_components].clip(min=0.0)
    return eigvecs[:, :n_components] * np.sqrt(keep)


def kmeans_on_mds(coords, n_clusters, seed=RANDOM_SEED, n_init=KMEANS_N_INIT):
    """k-means++ on the MDS coordinates, K = number of annotated individuals."""
    n_clusters = min(n_clusters, len(coords))
    km = KMeans(n_clusters=n_clusters, init="k-means++",
                n_init=n_init, random_state=seed)
    return km.fit_predict(coords)


def hierarchical_clusters(dissimilarity, cut=DENDROGRAM_CUT, method=LINKAGE_METHOD):
    """Average-linkage clustering on the cosine dissimilarities. Returns (Z, labels)."""
    condensed = squareform(dissimilarity, checks=False)
    z = linkage(condensed, method=method)
    labels = fcluster(z, t=cut, criterion="distance")
    return z, labels


def silhouette_on_distances(dissimilarity, labels):
    """Silhouette coefficient computed on the original cosine-distance matrix.

    Paper Sec 2.5: "The reported silhouette coefficient was calculated using the
    resulting cluster labels and the original pairwise cosine-distance matrix,
    rather than distances in the two-dimensional MDS projection."
    Returns NaN when the score is undefined (fewer than 2 populated clusters).
    """
    n_labels = len(set(labels))
    if n_labels < 2 or n_labels >= len(labels):
        return float("nan")
    return float(silhouette_score(dissimilarity, labels, metric="precomputed"))


def intra_inter_similarity(similarity, individuals):
    """Table 2 statistics: mean +/- sd of within- and between-individual similarity.

    Paper Table 2 caption: "Diagonal self-comparisons were excluded, and each
    unordered video pair was counted once after averaging the corresponding
    upper- and lower-triangular similarity values." Taking the upper triangle of
    the symmetrised matrix does exactly that.
    """
    individuals = np.asarray(individuals)
    sym = (similarity + similarity.T) / 2.0
    iu = np.triu_indices_from(sym, k=1)          # excludes the diagonal
    pair_vals = sym[iu]
    same = individuals[iu[0]] == individuals[iu[1]]

    intra, inter = pair_vals[same], pair_vals[~same]
    return {
        "intra_mean": float(np.mean(intra)) if intra.size else float("nan"),
        "intra_std": float(np.std(intra)) if intra.size else float("nan"),
        "intra_pairs": int(intra.size),
        "inter_mean": float(np.mean(inter)) if inter.size else float("nan"),
        "inter_std": float(np.std(inter)) if inter.size else float("nan"),
        "inter_pairs": int(inter.size),
    }


def shared_mean_fraction(features):
    """||mean(X)|| / mean(||x||): how much of every embedding is a common offset.

    Values near 1.0 mean all clips point in almost the same direction, which
    compresses every cosine similarity towards 1 and shrinks the usable range of
    the dissimilarity matrix. Frozen (non-fine-tuned) encoders show this
    strongly; the paper's Sec 2.3 encoder adaptation is what pulls the
    embeddings apart. Reported as a diagnostic so a compressed similarity range
    is attributable rather than mysterious.
    """
    norms = np.linalg.norm(features, axis=1)
    if not norms.mean():
        return float("nan")
    return float(np.linalg.norm(features.mean(axis=0)) / norms.mean())


def cluster_identity_agreement(labels, individuals):
    """Descriptive agreement between clusters and annotated identities.

    Reported as supporting descriptives only. The paper is explicit that this is
    exploratory clustering, not a re-identification benchmark (Sec 3, Sec 4.1).
    """
    from sklearn.metrics import (adjusted_rand_score,
                                 normalized_mutual_info_score)

    return {
        "adjusted_rand_index": float(adjusted_rand_score(individuals, labels)),
        "normalized_mutual_info": float(normalized_mutual_info_score(individuals, labels)),
        "n_clusters_found": int(len(set(labels))),
    }


def similarity_dataframe(similarity, labels):
    return pd.DataFrame(similarity, index=labels, columns=labels)


__all__ = [
    "similarity_matrix", "dissimilarity_matrix", "classical_mds",
    "kmeans_on_mds", "hierarchical_clusters", "silhouette_on_distances",
    "intra_inter_similarity", "cluster_identity_agreement",
    "similarity_dataframe", "shared_mean_fraction", "dendrogram",
    "DENDROGRAM_CUT", "RANDOM_SEED",
]
