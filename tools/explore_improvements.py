"""Exploratory: do cheap, label-free changes separate individuals better?

    python tools/explore_improvements.py

Works on results/embeddings only (no GPU). Writes results/experiments/*.csv.
These are exploratory comparisons on the evaluation data itself; anything
adopted for the paper should be fixed in advance and reported as such.
"""
import os
import sys
import warnings

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering, KMeans
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import adjusted_rand_score

warnings.filterwarnings("ignore")
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
import evaluate as E  # noqa: E402

OUT = os.path.join(ROOT, "results", "experiments")
os.makedirs(OUT, exist_ok=True)
unit = E.unit


def whiten(x, var=0.95):
    """PCA-whitening: unsupervised, equalises the dominant shared directions."""
    k = min(x.shape[0] - 1, x.shape[1])
    p = PCA(k, whiten=True, random_state=0).fit(x)
    keep = np.searchsorted(np.cumsum(p.explained_variance_ratio_), var) + 1
    return unit(p.transform(x)[:, :keep])


def smooth(x, k=3):
    """Neighbour averaging (query expansion): each clip becomes the mean of
    itself and its k nearest clips. No labels used."""
    s = x @ x.T
    nn = np.argsort(-s, 1)[:, :k + 1]
    return unit(x[nn].mean(1))


def variants(bank):
    P = lambda k: E.prepare(bank, k)
    fuse = lambda *xs: unit(np.hstack(xs))
    v = {
        "A0 fused (paper): ResNet18 + VP": fuse(P("resnet_rgb"), P("vp_ordered")),
        "A1 MegaDescriptor RGB mask": P("mega_rgb"),
        "A2 MegaDescriptor crop": P("mega_crop"),
        "A3 VP ordered": P("vp_ordered"),
        "B1 fused: Mega RGB + VP": fuse(P("mega_rgb"), P("vp_ordered")),
        "B2 fused: Mega RGB + VP + GEI + kin": fuse(P("mega_rgb"), P("vp_ordered"), P("gei"),
                                                   P("kinematics")),
        "C1 whitened ResNet18 + VP": fuse(whiten(P("resnet_rgb")), whiten(P("vp_ordered"))),
        "C2 whitened Mega RGB": whiten(P("mega_rgb")),
        "C3 whitened Mega RGB + VP": fuse(whiten(P("mega_rgb")), whiten(P("vp_ordered"))),
        "C4 whitened VP ordered": whiten(P("vp_ordered")),
    }
    v["D1 A0 + neighbour smoothing"] = smooth(v["A0 fused (paper): ResNet18 + VP"])
    v["D2 C3 + neighbour smoothing"] = smooth(v["C3 whitened Mega RGB + VP"])
    return v


def score(x, m):
    ids, src = m.individual.values, m.source_id.values
    sites = m["site"].values if "site" in m and all(m["site"] != "") else None
    out = {}
    for p in ("P1", "P2", "P3") if sites is not None else ("P1", "P2"):
        pq = E.retrieval(x, ids, src, p, sites)
        if len(pq):
            out[f"mAP_{p}"] = pq.ap.mean()
            out[f"rank1_{p}"] = (pq["rank"] == 1).mean()
    k = len(np.unique(ids))
    out["ARI_kmeans"] = adjusted_rand_score(ids, KMeans(k, n_init=20, random_state=0).fit_predict(x))
    d = np.clip(1 - x @ x.T, 0, 2)
    out["ARI_average"] = adjusted_rand_score(ids, AgglomerativeClustering(
        k, metric="precomputed", linkage="average").fit_predict(d))
    return out


def source_level(x, m):
    """Pool all clips of one recording into one embedding (a recording shows one
    animal), then retrieve recordings across recordings."""
    g = m.groupby("source_id").indices
    xs = unit(np.stack([x[i].mean(0) for i in g.values()]))
    ids = np.array([m.individual.values[i[0]] for i in g.values()])
    sites = (np.array([m["site"].values[i[0]] for i in g.values()])
             if "site" in m and all(m["site"] != "") else None)
    srcs = np.array(list(g))
    out = {}
    for p in ("P2", "P3") if sites is not None else ("P2",):
        pq = E.retrieval(xs, ids, srcs, p, sites)
        if len(pq):
            out[f"src_mAP_{p}"] = pq.ap.mean()
            out[f"src_rank1_{p}"] = (pq["rank"] == 1).mean()
            out["src_queries"] = len(pq)
    return out


def supervised(x, m):
    """Leave-one-recording-out: train on the other recordings, classify the
    held-out one. Only for individuals with >= 2 recordings."""
    ids, src = m.individual.values, m.source_id.values
    ids = ids.astype(str)
    multi = pd.Series(src).groupby(ids).nunique()
    ok = np.isin(ids, multi[multi >= 2].index)
    res = {"lda": [], "logreg": [], "knn": []}
    for s in np.unique(src[ok]):
        te, tr = src == s, (src != s)
        xtr, xte = x[tr], x[te]
        p = PCA(min(64, xtr.shape[0] - 1), random_state=0).fit(xtr)
        a, b = p.transform(xtr), p.transform(xte)
        res["lda"] += list(LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto")
                           .fit(a, ids[tr]).predict(b) == ids[te])
        res["logreg"] += list(LogisticRegression(C=1.0, max_iter=2000).fit(a, ids[tr]).predict(b)
                              == ids[te])
        res["knn"] += list(ids[tr][np.argmax(xte @ xtr.T, 1)] == ids[te])
    return {f"acc_{k}": float(np.mean(v)) for k, v in res.items()} | {"n_test": int(ok.sum())}


def main():
    data = E.load()
    qc = pd.read_csv(os.path.join(ROOT, "results", "segmentation_qc.csv")).set_index("clip_id")
    rows, sup, filt = [], [], []
    for sp, d in data.items():
        m = d["meta"]
        V = variants(d["bank"])
        for name, x in V.items():
            rows.append(dict(species=sp, variant=name, n=len(m), **score(x, m),
                             **source_level(x, m)))
        if m.source_id.nunique() > m.individual.nunique():
            for name in ("A0 fused (paper): ResNet18 + VP", "A1 MegaDescriptor RGB mask",
                         "B1 fused: Mega RGB + VP", "A3 VP ordered"):
                sup.append(dict(species=sp, variant=name, **supervised(V[name], m)))
        # Quality filter for the silhouette streams
        q = qc.loc[m.clip_id]
        rules = {"all clips": np.ones(len(m), bool),
                 "border_frac <= 0.5": (q.border_frac <= 0.5).to_numpy(),
                 "stride_autocorr >= 0.3": (q.stride_autocorr >= 0.3).to_numpy(),
                 "both": ((q.border_frac <= 0.5) & (q.stride_autocorr >= 0.3)).to_numpy()}
        for rule, keep in rules.items():
            mk = m[keep].reset_index(drop=True)
            if mk.individual.nunique() < 2 or mk.groupby("individual").size().min() < 2:
                continue
            for rep in ("vp_ordered", "gei", "kinematics", "fused"):
                bank = {k: v[keep] for k, v in d["bank"].items()}
                x = E.representation(bank, E.REPS[rep][2])
                s = score(x, mk)
                filt.append(dict(species=sp, rule=rule, rep=rep, n=len(mk),
                                 individuals=mk.individual.nunique(),
                                 **{k: v for k, v in s.items() if k.startswith("mAP")}))
    for name, r in (("variants", rows), ("supervised", sup), ("quality_filter", filt)):
        df = pd.DataFrame(r)
        df.to_csv(os.path.join(OUT, f"{name}.csv"), index=False)
        print(f"\n== {name} ==")
        print(df.round(3).to_string(index=False))


if __name__ == "__main__":
    pd.set_option("display.width", 260)
    pd.set_option("display.max_rows", 400)
    pd.set_option("display.max_columns", 40)
    main()
