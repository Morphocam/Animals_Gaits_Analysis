"""Every table and figure in the manuscript, from results/embeddings/*.npz.

    python evaluate.py

Inputs : manifests/*.csv, results/embeddings/<species>.npz,
         results/segmentation_qc.csv, output/masks (Fig. 2 only)

The appearance stream is MegaDescriptor on the masked RGB crop; the fused
representation joins it with VideoPrism on ordered silhouettes at equal weight
(no tuning). Gait analyses use only periodic clips (GAIT_MIN_AUTOCORR below).
Outputs: results/tables/*.csv and *.tex, paper/figures/fig*.pdf|png

Evaluation is retrieval-based (leave-one-clip-out within a species):

  P1  same-source allowed   gallery = every other clip of the species. Clips
                            cut from one recording share background, light and
                            camera, so P1 is an optimistic upper bound.
  P2  cross-source          gallery excludes every clip from the query's own
                            source recording; only individuals filmed in >= 2
                            source recordings can be queried. This is the
                            leakage-free estimate of re-identification.
  P3  cross-source, same site
                            as P2, but the gallery is also restricted to the
                            query's site (e.g. zoo enclosure), so a shared
                            background cannot separate animals. Only computed
                            where the manifest has a `site` column.

Embeddings are centred on the species mean (unsupervised; removes the shared
offset that otherwise compresses cosine similarities towards 1) and
L2-normalised. Fused embeddings concatenate equal-weighted normalised streams.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap
from sklearn.cluster import KMeans
from sklearn.metrics import (adjusted_rand_score, normalized_mutual_info_score,
                             roc_auc_score, silhouette_score)

ROOT = os.path.dirname(os.path.abspath(__file__))
EMB = os.path.join(ROOT, "results", "embeddings")
TAB = os.path.join(ROOT, "results", "tables")
FIG = os.path.join(ROOT, "paper", "figures")
SEED = 42
N_BOOT = 2000

# name -> (label, family, recipe). A recipe is one stored key or a tuple of keys
# to fuse; "raw:" marks the uncentred concatenation used in the earlier submission.
REPS = {
    "fused":        ("Fused: MegaDescriptor + VideoPrism", "proposed", ("mega_rgb", "vp_ordered")),
    "fused_resnet": ("Fused: ResNet18 + VideoPrism (previous)", "proposed", ("resnet_rgb", "vp_ordered")),
    "fused_raw":    ("Fused, uncentred (earlier submission)", "proposed", "raw:resnet_rgb+vp_ordered"),
    "mega_rgb":     ("MegaDescriptor, RGB mask", "appearance", "mega_rgb"),
    "resnet_rgb":   ("ResNet18, RGB mask", "appearance", "resnet_rgb"),
    "mega_crop":    ("MegaDescriptor, crop (baseline)", "appearance", "mega_crop"),
    "vp_ordered":   ("VideoPrism, ordered silhouettes", "silhouette", "vp_ordered"),
    "vp_shuffled":  ("VideoPrism, shuffled silhouettes", "silhouette", "vp_shuffled"),
    "vp_static":    ("VideoPrism, single static silhouette", "silhouette", "vp_static"),
    "vp_mean":      ("VideoPrism, mean silhouette", "silhouette", "vp_mean"),
    "gei":          ("Gait energy image", "silhouette", "gei"),
    "kinematics":   ("Kinematic descriptors", "silhouette", "kinematics"),
    "gait_pose":    ("Gait variables (pose)", "gait", "gait_pose"),
    "fused_gait":   ("Fused: MegaDescriptor + gait variables", "gait", ("mega_rgb", "gait_pose")),
    "resnet_bg":    ("Background only (leakage control)", "control", "resnet_bg"),
}
MAIN = ["fused", "mega_rgb", "mega_crop", "resnet_rgb", "vp_ordered", "gei", "resnet_bg"]
GAIT_CONTROLS = ["vp_ordered", "vp_shuffled", "vp_static", "vp_mean"]

# Stride-quality gate for the gait analyses. A clip counts as "periodic" when the
# leg-spread signal repeats at its estimated stride period: autocorrelation at
# that lag >= GAIT_MIN_AUTOCORR (stride_autocorr in results/segmentation_qc.csv).
# Clips below it contain no measurable stride (standing, turning, truncated).
# The main retrieval tables use every clip; the gate defines the kinematics
# table and a second retrieval table restricted to periodic clips.
GAIT_MIN_AUTOCORR = 0.3
GATE_SWEEP = [0.0, 0.2, 0.3, 0.4, 0.5]
GATED_TEX = ["mega_rgb", "vp_ordered", "vp_shuffled", "kinematics", "resnet_bg"]
GATED_REPS = ["fused", "mega_rgb", "vp_ordered", "vp_shuffled", "gei", "kinematics", "resnet_bg"]

# Reference palette (validated; see the dataviz skill). <=3 hues per scatter.
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED = (
    "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948")
INK, INK2, MUTED, GRID = "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
FAMILY_COLOR = {"proposed": BLUE, "appearance": ORANGE, "silhouette": AQUA, "gait": VIOLET,
                "control": MUTED}
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#f4f8fd", "#cde2fb", "#86b6ef",
                                                     "#3987e5", "#1c5cab", "#0d366b"])

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.titlesize": 9,
    "axes.labelsize": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "legend.frameon": False,
    "savefig.dpi": 300, "savefig.bbox": "tight", "lines.linewidth": 2,
})


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def nice(species):
    """Display name: polarbear -> Polar bear."""
    if "_" in species:                           # horse_walk -> Horse, walk (see split_by_gait)
        base, gait = species.split("_", 1)
        return f"{nice(base)}, {gait}"
    return {"polarbear": "Polar bear"}.get(species, species.capitalize())


def split_by_gait(data):
    """Species whose manifest names the gait of each clip (column `gait`: walk or
    trot) are split into <species>_<gait>, so that gait variables are never
    pooled over different gaits."""
    out = {}
    for sp, d in data.items():
        g = d["meta"]["gait"].fillna("").astype(str) if "gait" in d["meta"] else None
        if g is None or not (g != "").any():
            out[sp] = d
            continue
        for name in [x for x in dict.fromkeys(g) if x]:
            keep = (g == name).to_numpy()
            out[f"{sp}_{name}"] = dict(d, meta=d["meta"][keep].reset_index(drop=True),
                                       bank={k: v[keep] for k, v in d["bank"].items()})
    return out

# Pose-based locomotion variables (gait_kinematics.py). Phases are circular.
GAIT_VARS = ["stride_frequency_hz", "duty_factor", "step_length", "stride_length", "speed",
             "limb_phase", "fore_pair_phase", "hind_pair_phase", "fore_rom_deg",
             "hind_rom_deg", "head_height", "head_bob"]
GAIT_CIRCULAR = ["limb_phase", "fore_pair_phase", "hind_pair_phase"]
GAIT_LABEL = {"stride_frequency_hz": "Stride frequency (Hz)", "duty_factor": "Duty factor",
              "step_length": "Step length (trunk lengths)",
              "stride_length": "Stride length (trunk lengths)",
              "speed": "Speed (trunk lengths s$^{-1}$)", "limb_phase": "Limb phase",
              "fore_pair_phase": "Fore pair phase", "hind_pair_phase": "Hind pair phase",
              "fore_rom_deg": "Forelimb range (deg)", "hind_rom_deg": "Hindlimb range (deg)",
              "head_height": "Head height (trunk lengths)", "head_bob": "Head bob (trunk lengths)"}
GAIT_REPS = ["gait_pose", "kinematics", "vp_ordered", "mega_rgb", "fused_gait", "resnet_bg"]


def circ_centre(v):
    """Phases (0-1) as signed distance from their circular mean, so that values
    either side of 0/1 are not split."""
    ok = np.isfinite(v)
    if not ok.any():
        return v
    m = np.angle(np.exp(2j * np.pi * v[ok]).mean()) / (2 * np.pi)
    return (v - m + 0.5) % 1 - 0.5


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-12)


def prepare(bank, key):
    x = bank[key].astype(np.float64)
    if key == "gait_pose":                       # phases centred, then as kinematics
        x = x.copy()
        for c in GAIT_CIRCULAR:
            j = GAIT_VARS.index(c)
            x[:, j] = circ_centre(x[:, j])
    if key in ("kinematics", "gait_pose"):       # interpretable scalars: z-score
        x = np.nan_to_num((x - np.nanmean(x, 0)) / (np.nanstd(x, 0) + 1e-9))
        return unit(x)
    return unit(x - x.mean(0))


def representation(bank, recipe):
    if isinstance(recipe, tuple):
        return unit(np.hstack([prepare(bank, k) for k in recipe]))
    if recipe.startswith("raw:"):
        return unit(np.hstack([bank[k] for k in recipe[4:].split("+")]))
    return prepare(bank, recipe)


def gated(data, threshold=GAIT_MIN_AUTOCORR, column="stride_autocorr"):
    """The same data restricted to periodic clips (or, with column="gait_resolved",
    to clips whose pose-based gait was resolved). Species left with fewer than
    two individuals are dropped."""
    out = {}
    for sp, d in data.items():
        if column not in d["meta"]:
            continue
        keep = (d["meta"][column] >= threshold).to_numpy()
        meta = d["meta"][keep].reset_index(drop=True)
        if meta.individual.nunique() < 2:
            continue
        out[sp] = dict(meta=meta, bank={k: v[keep] for k, v in d["bank"].items()},
                       kin_names=d["kin_names"])
    return out


def load():
    from extract_features import MAX_FRAMES, load_manifests
    man = load_manifests()
    qc = pd.read_csv(os.path.join(ROOT, "results", "segmentation_qc.csv"))
    man = man.merge(qc[["clip_id", "stride_autocorr"]], on="clip_id", how="left")
    gp = os.path.join(ROOT, "results", "kinematics", "pose_gait.csv")
    gait = pd.read_csv(gp).set_index("clip_id") if os.path.exists(gp) else None
    data = {}
    for f in sorted(os.listdir(EMB)):
        if not f.endswith(".npz"):
            continue
        z = np.load(os.path.join(EMB, f))
        bank = {k: z[k] for k in z.files if k not in ("clip_id", "kinematics_names")}
        meta = man.set_index("clip_id").loc[z["clip_id"]].reset_index()
        keep = (meta["exclude"] == "").to_numpy()
        meta = meta[keep].reset_index(drop=True)
        bank = {k: v[keep] for k, v in bank.items()}
        if meta["individual"].nunique() < 2:
            continue
        # A stride frequency is only resolved if the analysed window spans >= 1.5
        # strides; otherwise the FFT peak sits on the lower band edge (0.3 Hz).
        names = [str(n) for n in z["kinematics_names"]]
        j = names.index("stride_frequency_hz")
        seconds = np.minimum(meta.n_frames.astype(float), MAX_FRAMES) / meta.fps.astype(float)
        floor = np.maximum(0.3, 1.5 / seconds.to_numpy()) + 0.02
        kin = bank["kinematics"].astype(np.float64)
        kin[kin[:, j] <= floor, j] = np.nan
        bank["kinematics"] = kin
        if gait is not None:
            g = gait.reindex(meta.clip_id)
            ok = (g.resolved == True).to_numpy()                      # noqa: E712
            v = g[GAIT_VARS].to_numpy(float)
            v[~ok] = np.nan
            bank["gait_pose"] = v
            meta["gait_resolved"] = ok
            for c in [c for c in g.columns if c.startswith(("td_", "duty_"))]:
                meta[c] = np.where(ok, g[c].to_numpy(float), np.nan)
        data[f[:-4]] = dict(meta=meta, bank=bank, kin_names=names)
    return data


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def average_precision(scores, relevant):
    order = np.argsort(-scores)
    rel = relevant[order]
    if not rel.any():
        return np.nan
    hits = np.cumsum(rel)
    return float((hits[rel] / (np.flatnonzero(rel) + 1)).mean())


def retrieval(x, ids, sources, protocol, sites=None):
    """Per-query rank of first correct match and AP. Returns a DataFrame."""
    sim = x @ x.T
    rows = []
    for q in range(len(x)):
        gallery = np.ones(len(x), bool)
        gallery[q] = False
        if protocol in ("P2", "P3"):
            gallery &= sources != sources[q]
        if protocol == "P3":
            gallery &= sites == sites[q]
        rel = ids[gallery] == ids[q]
        if not rel.any():
            continue                      # no valid positive in this protocol
        s = sim[q, gallery]
        first = int(np.argmax(rel[np.argsort(-s)])) + 1
        rows.append(dict(query=q, individual=ids[q], rank=first,
                         ap=average_precision(s, rel), chance_ap=rel.mean(),
                         n_gallery=int(gallery.sum())))
    return pd.DataFrame(rows)


def cluster_bootstrap(per_query, stat, rng, n=N_BOOT):
    """95% CI resampling individuals (clips within an individual are not independent)."""
    groups = [g for _, g in per_query.groupby("individual")]
    if len(groups) < 2:
        return np.nan, np.nan
    vals = []
    for _ in range(n):
        pick = rng.integers(0, len(groups), len(groups))
        vals.append(stat(pd.concat([groups[i] for i in pick])))
    return tuple(np.percentile(vals, [2.5, 97.5]))


def open_set_auroc(x, ids):
    """Leave-one-individual-out: can the best gallery match reject a new animal?
    Known queries keep their same-source gallery clips (P1 conditions)."""
    sim = x @ x.T
    np.fill_diagonal(sim, -np.inf)
    aucs = []
    for u in np.unique(ids):
        unknown, known = ids == u, ids != u
        score_unknown = sim[np.ix_(unknown, known)].max(1)
        score_known = sim[np.ix_(known, known)].max(1)
        y = np.r_[np.zeros(unknown.sum()), np.ones(known.sum())]
        aucs.append(roc_auc_score(y, np.r_[score_unknown, score_known]))
    return float(np.mean(aucs))


def clustering_scores(x, ids, seed=SEED):
    k = len(np.unique(ids))
    labels = KMeans(k, n_init=50, random_state=seed).fit_predict(x)
    d = np.clip(1 - x @ x.T, 0, 2)
    np.fill_diagonal(d, 0)
    return dict(ari=adjusted_rand_score(ids, labels),
                nmi=normalized_mutual_info_score(ids, labels),
                silhouette_gt=silhouette_score(d, ids, metric="precomputed"))


def repeatability(values, ids, alpha=0.05):
    """ICC(1) from one-way ANOVA with its exact F-based confidence interval
    (Nakagawa & Schielzeth 2010). Returns (R, lo, hi). A bootstrap over
    individuals is unusable here because species have only 2-13 individuals."""
    from scipy.stats import f as f_dist
    df = pd.DataFrame(dict(v=values, g=ids)).dropna()
    groups = [g.v.values for _, g in df.groupby("g")]
    k, N = len(groups), len(df)
    if k < 2 or N <= k:
        return np.nan, np.nan, np.nan
    n0 = (N - sum(len(g) ** 2 for g in groups) / N) / (k - 1)
    grand = df.v.mean()
    msb = sum(len(g) * (g.mean() - grand) ** 2 for g in groups) / (k - 1)
    msw = sum(((g - g.mean()) ** 2).sum() for g in groups) / (N - k)
    F = msb / msw
    fl = F / f_dist.ppf(1 - alpha / 2, k - 1, N - k)
    fu = F * f_dist.ppf(1 - alpha / 2, N - k, k - 1)
    icc = lambda x: (x - 1) / (x + n0 - 1)
    return icc(F), icc(fl), icc(fu)


# --------------------------------------------------------------------------
# Analyses
# --------------------------------------------------------------------------
def dataset_table(data, qc):
    rows = []
    for sp, d in data.items():
        m = d["meta"]
        per = m.groupby("individual").size()
        src = m.groupby("individual")["source_id"].nunique()
        q = qc[qc.species == sp]
        rows.append(dict(species=sp, individuals=m.individual.nunique(),
                         clips_embedded=len(m), clips_total=len(q),
                         clips_excluded=int((q.exclude != "").sum()),
                         clips_failed_qc=int(((q.status != "ok") & (q.exclude == "")).sum()),
                         source_recordings=m.source_id.nunique(),
                         clips_per_individual=f"{per.min()}-{per.max()} (median {int(per.median())})",
                         individuals_multi_source=int((src >= 2).sum()),
                         median_clip_s=float(m.duration_s.median())))
    return pd.DataFrame(rows)


def run_retrieval(data, rng, reps=None):
    per_query, summary = [], []
    for sp, d in data.items():
        ids = d["meta"].individual.values
        srcs = d["meta"].source_id.values
        sites = d["meta"]["site"].values if "site" in d["meta"] else None
        has_sites = sites is not None and all(sites != "")
        for name in reps or REPS:
            label, family, recipe = REPS[name]
            x = representation(d["bank"], recipe)
            for proto in ("P1", "P2", "P3") if has_sites else ("P1", "P2"):
                pq = retrieval(x, ids, srcs, proto, sites)
                if pq.empty:
                    continue
                pq = pq.assign(species=sp, rep=name, protocol=proto)
                per_query.append(pq)
                lo, hi = cluster_bootstrap(pq, lambda g: g.ap.mean(), rng)
                summary.append(dict(
                    species=sp, rep=name, label=label, family=family, protocol=proto,
                    n_queries=len(pq), n_individuals=pq.individual.nunique(),
                    rank1=(pq["rank"] == 1).mean(), rank5=(pq["rank"] <= 5).mean(),
                    mAP=pq.ap.mean(), mAP_lo=lo, mAP_hi=hi, chance_mAP=pq.chance_ap.mean(),
                    open_set_auroc=open_set_auroc(x, ids) if proto == "P1" else np.nan,
                    **(clustering_scores(x, ids) if proto == "P1" else {})))
    return pd.concat(per_query, ignore_index=True), pd.DataFrame(summary)


def gate_sensitivity(data, reps=("vp_ordered", "kinematics", "gei", "fused", "mega_rgb")):
    """mAP on the clips that pass the gate, for a range of thresholds, so the
    choice of GAIT_MIN_AUTOCORR can be judged."""
    rows = []
    for thr in [None] + GATE_SWEEP:
        sub = data if thr is None else gated(data, thr)
        for sp, d in sub.items():
            m = d["meta"]
            ids, srcs = m.individual.values, m.source_id.values
            sites = m["site"].values if "site" in m else None
            has_sites = sites is not None and all(sites != "")
            for name in reps:
                x = representation(d["bank"], REPS[name][2])
                for proto in ("P1", "P2", "P3") if has_sites else ("P1", "P2"):
                    pq = retrieval(x, ids, srcs, proto, sites)
                    if pq.empty:
                        continue
                    rows.append(dict(species=sp, threshold="all clips" if thr is None else str(thr),
                                     rep=name, protocol=proto, clips=len(m),
                                     n_queries=len(pq), n_individuals=pq.individual.nunique(),
                                     rank1=(pq["rank"] == 1).mean(), mAP=pq.ap.mean(),
                                     chance_mAP=pq.chance_ap.mean()))
    return pd.DataFrame(rows)


def strictest(df, sp):
    """The most leakage-resistant protocol evaluated for a species."""
    protos = set(df[df.species == sp].protocol)
    return next(p for p in ("P3", "P2", "P1") if p in protos)


def paired_delta(g, rep, ref, rng):
    """Mean per-query AP of `rep` minus `ref` on the same queries, with a cluster bootstrap."""
    a = g[g.rep == rep].set_index("query")
    b = g[g.rep == ref].set_index("query")
    diff = (a.ap - b.ap.reindex(a.index)).to_frame("d")
    diff["individual"] = a.individual
    lo, hi = cluster_bootstrap(diff, lambda h: h.d.mean(), rng)
    return diff.d.mean(), lo, hi


def gait_control_tests(per_query, rng):
    """Paired per-query AP differences: ordered silhouettes minus each control."""
    rows = []
    for (sp, proto), g in per_query.groupby(["species", "protocol"], sort=False):
        for ctrl in GAIT_CONTROLS[1:]:
            d, lo, hi = paired_delta(g, "vp_ordered", ctrl, rng)
            rows.append(dict(species=sp, protocol=proto, control=ctrl, delta_mAP=d,
                             ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def background_tests(per_query, rng):
    """Leakage-resistant protocols: each representation minus the background-only control.

    A representation carries identity information beyond the scene only if it beats the
    background on the same queries."""
    rows = []
    for (sp, proto), g in per_query[per_query.protocol != "P1"].groupby(
            ["species", "protocol"], sort=False):
        for rep in REPS:
            if rep == "resnet_bg":
                continue
            d, lo, hi = paired_delta(g, rep, "resnet_bg", rng)
            rows.append(dict(species=sp, protocol=proto, rep=rep, delta_mAP=d,
                             ci_lo=lo, ci_hi=hi))
    return pd.DataFrame(rows)


def order_sensitivity(data):
    """How far each control moves a clip's VideoPrism embedding, in the centred
    space used for retrieval (all variants centred on the ordered-clip mean).

    self_similarity = mean cos(ordered_i, control_i). For scale it is reported
    next to the mean similarity of two *different* clips of the same animal and
    of two different animals: a control that stays closer to its source clip
    than other clips of that animal do has removed little information.
    """
    rows = []
    for sp, d in data.items():
        b, ids = d["bank"], d["meta"].individual.values
        mu = b["vp_ordered"].astype(np.float64).mean(0)
        o = unit(b["vp_ordered"] - mu)
        sim = o @ o.T
        same = (ids[:, None] == ids[None]) & ~np.eye(len(ids), dtype=bool)
        diff = ids[:, None] != ids[None]
        for ctrl in GAIT_CONTROLS[1:]:
            c = unit(b[ctrl] - mu)
            rows.append(dict(species=sp, control=ctrl,
                             self_similarity=float(np.mean(np.sum(o * c, 1))),
                             same_animal_other_clip=float(sim[same].mean()),
                             different_animals=float(sim[diff].mean())))
    return pd.DataFrame(rows)


def pair_types(data, reps=("fused", "mega_rgb", "mega_crop", "vp_ordered", "resnet_bg")):
    """Similarity by pair type, for individuals recorded in >= 2 sources."""
    rows = []
    for sp, d in data.items():
        m = d["meta"]
        ids, srcs = m.individual.values, m.source_id.values
        multi = set(m.groupby("individual").source_id.nunique().loc[lambda s: s >= 2].index)
        if not multi:
            continue
        for name in reps:
            x = representation(d["bank"], REPS[name][2])
            sim = x @ x.T
            iu = np.triu_indices(len(x), 1)
            for i, j in zip(*iu):
                if ids[i] not in multi and ids[j] not in multi:
                    continue
                if ids[i] == ids[j]:
                    kind = "same animal, same recording" if srcs[i] == srcs[j] else \
                           "same animal, different recording"
                else:
                    kind = "different animals"
                rows.append(dict(species=sp, rep=name, kind=kind, sim=sim[i, j]))
    return pd.DataFrame(rows)


def kinematics_table(data, clips="all"):
    rows = []
    for sp, d in data.items():
        ids = d["meta"].individual.values
        k = d["bank"]["kinematics"]
        for j, name in enumerate(d["kin_names"]):
            v = k[:, j]
            r, lo, hi = repeatability(v, ids)
            rows.append(dict(species=sp, clips=clips, metric=name,
                             n_resolved=int(np.isfinite(v).sum()),
                             mean=np.nanmean(v), sd=np.nanstd(v), R=r, R_lo=lo, R_hi=hi))
    return pd.DataFrame(rows)


# Fine-tuned encoders (train_encoders.py): name -> (label, kind, stored key or frozen key)
FT_REPS = [
    ("resnet_ft", "ResNet18, fine-tuned", "ft"),
    ("resnet_rgb", "ResNet18, frozen", "frozen"),
    ("vp_ft", "VideoPrism, fine-tuned", "ft"),
    ("vp_ft_shuffled", "VideoPrism, fine-tuned, shuffled at test", "ft"),
    ("vp_ft_trained_shuffled", "VideoPrism, fine-tuned on shuffled frames", "ft"),
    ("vp_ordered", "VideoPrism, frozen", "frozen"),
    ("fused_ft", "Fused: fine-tuned ResNet18 + VideoPrism", "ft"),
    ("mega_rgb", "MegaDescriptor, frozen", "frozen"),
    ("gait_pose", "Gait variables (pose)", "frozen"),
    ("resnet_bg", "Background only (leakage control)", "frozen"),
]


def finetuned_eval(data, rng):
    """Retrieval with the encoders of train_encoders.py under its source-grouped
    folds. For fold f the query set is the held-out clips (recordings the fold-f
    model never saw) and the gallery is every other clip. Frozen representations
    are scored on exactly the same queries and galleries."""
    d_ft = os.path.join(ROOT, "results", "embeddings_finetuned")
    per_query, rows = [], []
    for f_name in sorted(os.listdir(d_ft)) if os.path.isdir(d_ft) else []:
        sp = f_name[:-4]
        if sp not in data or not f_name.endswith(".npz"):
            continue
        z = np.load(os.path.join(d_ft, f_name))
        meta, bank = data[sp]["meta"], data[sp]["bank"]
        if list(z["clip_id"]) != list(meta.clip_id):
            print(f"finetuned_eval: {sp} skipped, embeddings are out of date (re-run train_encoders.py)")
            continue
        fold, ids = z["fold"], meta.individual.values
        for name, label, kind in FT_REPS:
            if kind == "frozen" and name not in bank:
                continue
            if name == "gait_pose" and meta.gait_resolved.mean() < 0.9:
                continue
            for f in sorted(set(fold[fold >= 0])):
                q, g = np.where(fold == f)[0], np.where(fold != f)[0]
                if kind == "frozen":
                    x = bank[name].astype(np.float64)
                    if name == "gait_pose":
                        x = prepare(bank, name)
                elif name == "fused_ft":
                    parts = []
                    for k in ("resnet_ft", "vp_ft"):
                        v = z[k][f].astype(np.float64)
                        parts.append(unit(v - v[g].mean(0)))
                    x = np.hstack(parts)
                else:
                    x = z[name][f].astype(np.float64)
                x = unit(x - x[g].mean(0))
                sim = x[q] @ x[g].T
                for i, qi in enumerate(q):
                    rel = ids[g] == ids[qi]
                    if not rel.any():
                        continue
                    order = np.argsort(-sim[i])
                    per_query.append(dict(species=sp, rep=name, fold=int(f), clip_id=meta.clip_id[qi],
                                          individual=ids[qi], ap=average_precision(sim[i], rel),
                                          rank=int(np.flatnonzero(rel[order])[0]) + 1,
                                          chance_ap=rel.mean()))
    pq = pd.DataFrame(per_query)
    if pq.empty:
        return pq, pd.DataFrame()
    for (sp, name), g in pq.groupby(["species", "rep"], sort=False):
        lo, hi = cluster_bootstrap(g, lambda t: t.ap.mean(), rng)
        rows.append(dict(species=sp, rep=name, label=dict((n, l) for n, l, _ in FT_REPS)[name],
                         n_queries=len(g), n_individuals=g.individual.nunique(),
                         rank1=(g["rank"] == 1).mean(), mAP=g.ap.mean(), mAP_lo=lo, mAP_hi=hi,
                         chance_mAP=g.chance_ap.mean()))
    summ = pd.DataFrame(rows)
    # paired differences on the same queries
    deltas = []
    for sp, g in pq.groupby("species", sort=False):
        w = g.pivot_table(index=["clip_id", "individual"], columns="rep", values="ap").reset_index()
        for a, b in (("resnet_ft", "resnet_rgb"), ("vp_ft", "vp_ordered"), ("vp_ft", "vp_ft_shuffled"),
                     ("vp_ft", "vp_ft_trained_shuffled"), ("resnet_ft", "resnet_bg"),
                     ("vp_ft", "resnet_bg"), ("fused_ft", "resnet_ft"), ("resnet_ft", "mega_rgb")):
            if a in w and b in w:
                t = w.assign(ap=w[a] - w[b])
                lo, hi = cluster_bootstrap(t, lambda u: u.ap.mean(), rng)
                deltas.append(dict(species=sp, rep=a, minus=b, delta_mAP=t.ap.mean(), lo=lo, hi=hi))
    pd.DataFrame(deltas).to_csv(os.path.join(TAB, "table_finetuned_deltas.csv"), index=False)
    return pq, summ


def finetuned_tex(summ):
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Species & Representation & Queries (animals) & Rank-1 & mAP [95\% CI] \\", r"\midrule"]
    for sp, g in summ.groupby("species", sort=False):
        for _, r in g.iterrows():
            lines.append(f"{nice(sp)} & {r.label} & {r.n_queries} ({r.n_individuals}) & {fmt(r.rank1)} & "
                         f"{fmt(r.mAP)} [{fmt(r.mAP_lo)}, {fmt(r.mAP_hi)}]" + r" \\")
        lines.append(f"{nice(sp)} & Chance & & & {fmt(g.chance_mAP.iloc[0])}" + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_finetuned.tex"), "w") as f:
        f.write("\n".join(lines))


def fig_finetuned(summ):
    species = list(dict.fromkeys(summ.species))
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 3.2), sharey=True, squeeze=False)
    color = {"resnet_ft": ORANGE, "resnet_rgb": ORANGE, "mega_rgb": ORANGE, "vp_ft": AQUA,
             "vp_ft_shuffled": AQUA, "vp_ft_trained_shuffled": AQUA, "vp_ordered": AQUA,
             "fused_ft": BLUE, "gait_pose": VIOLET, "resnet_bg": MUTED}
    rows = [(n, l) for n, l, _ in FT_REPS if n in set(summ.rep)][::-1]
    for ax, sp in zip(axes[0], species):
        g = summ[summ.species == sp].set_index("rep")
        for i, (name, _) in enumerate(rows):
            if name not in g.index:
                continue
            r = g.loc[name]
            frozen = name in ("resnet_rgb", "vp_ordered", "mega_rgb", "gait_pose", "resnet_bg")
            ax.barh(i, r.mAP, height=0.62, color=color[name], alpha=0.45 if frozen else 1.0)
            ax.plot([r.mAP_lo, r.mAP_hi], [i, i], color=INK, lw=0.9)
        ax.axvline(g.chance_mAP.iloc[0], color=INK2, lw=0.8, ls=":")
        ax.set_yticks(range(len(rows)))
        ax.set_yticklabels([l for _, l in rows], fontsize=6)
        ax.set_xlim(0, 1)
        ax.set_xlabel("mAP, held-out recordings")
        ax.set_title(f"{nice(sp)} ({int(g.n_queries.max())} queries, {int(g.n_individuals.max())} animals)",
                     fontsize=8, color=INK)
        ax.grid(axis="y", visible=False)
    save(fig, "fig_finetuned")


def gait_tables(data, resolved):
    """Species-level locomotion variables and their repeatability within individuals,
    on clips whose gait was resolved."""
    summ, reps_ = [], []
    for sp, d in data.items():
        n_all = len(d["meta"])
        if sp not in resolved:
            summ.append(dict(species=sp, clips=n_all, resolved=int(d["meta"].gait_resolved.sum())))
            continue
        r = resolved[sp]
        ids, v = r["meta"].individual.values, r["bank"]["gait_pose"]
        row = dict(species=sp, clips=n_all, resolved=len(ids),
                   individuals=len(set(ids)))
        for j, name in enumerate(GAIT_VARS):
            x = v[:, j]
            if name in GAIT_CIRCULAR:
                m = np.angle(np.exp(2j * np.pi * x).mean()) / (2 * np.pi) % 1
                c = circ_centre(x)
                row[name], row[name + "_sd"] = m, c.std()
                x = c
            else:
                row[name], row[name + "_sd"] = np.median(x), x.std()
            R, lo, hi = repeatability(x, ids)
            reps_.append(dict(species=sp, metric=name, n=len(x), individuals=len(set(ids)),
                              R=R, R_lo=lo, R_hi=hi))
        summ.append(row)
    return pd.DataFrame(summ), pd.DataFrame(reps_)


def gait_by_condition(resolved):
    """Gait variables per recording condition (manifest column `condition`, e.g.
    treadmill walk at a fixed belt speed). Within a condition every individual
    moved at the same or a similar speed, so repeatability there is not driven
    by differences in speed between individuals' clips."""
    rows = []
    for sp, d in resolved.items():
        m = d["meta"]
        if "condition" not in m or not (m["condition"] != "").any():
            continue
        for cond in [c for c in dict.fromkeys(m["condition"]) if c]:
            keep = (m["condition"] == cond).to_numpy()
            ids, v = m.individual.values[keep], d["bank"]["gait_pose"][keep]
            for j, name in enumerate(GAIT_VARS):
                x = v[:, j]
                if name in GAIT_CIRCULAR:
                    centre = np.angle(np.exp(2j * np.pi * x).mean()) / (2 * np.pi) % 1
                    x = circ_centre(x)
                    sd = x.std()
                else:
                    centre, sd = np.median(x), x.std()
                R, lo, hi = repeatability(x, ids)
                rows.append(dict(species=sp, condition=cond, metric=name, clips=int(keep.sum()),
                                 individuals=len(set(ids)), value=centre, sd=sd, R=R, R_lo=lo, R_hi=hi))
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    for sp, g in out.groupby("species", sort=False):
        conds = list(dict.fromkeys(g.condition))
        lines = [r"\begin{tabular}{l" + "cc" * len(conds) + "}", r"\toprule",
                 " & " + " & ".join(r"\multicolumn{2}{c}{" + c + "}" for c in conds) + r" \\",
                 "Variable & " + " & ".join([r"Median (SD) & $R$ [95\% CI]"] * len(conds)) + r" \\",
                 r"\midrule",
                 "Clips (animals) & " + " & ".join(
                     r"\multicolumn{2}{c}{" + f"{int(g[g.condition == c].clips.iloc[0])} "
                     f"({int(g[g.condition == c].individuals.iloc[0])})" + "}" for c in conds) + r" \\",
                 r"\addlinespace"]
        for name in GAIT_VARS:
            cells = []
            for c in conds:
                r = g[(g.condition == c) & (g.metric == name)].iloc[0]
                dgt = 1 if name.endswith("_deg") else 2
                cells += [f"{fmt(r.value, dgt)} ({fmt(r.sd, dgt)})",
                          "--" if not np.isfinite(r.R) else f"{fmt(r.R)} [{fmt(r.R_lo)}, {fmt(r.R_hi)}]"]
            lines.append(GAIT_LABEL[name].replace(" (trunk lengths)", "").replace(" (trunk lengths s$^{-1}$)", "")
                         + " & " + " & ".join(cells) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}"]
        with open(os.path.join(TAB, f"table_gait_conditions_{sp}.tex"), "w") as f:
            f.write("\n".join(lines))
    return out


CROSS_PAIRS = [("walk17", "walkmax", "Walk: 1.7 m/s vs peak"), ("trot45", "trotmax", "Trot: 4.5 m/s vs peak"),
               ("walk17", "trot45", "Walk 1.7 m/s vs trot 4.5 m/s")]
CROSS_REPS = [("gait_pose", "Gait variables (pose)"), ("vp_ordered", "VideoPrism, ordered silhouettes"),
              ("vp_shuffled", "VideoPrism, shuffled silhouettes"), ("gei", "Gait energy image"),
              ("mega_rgb", "MegaDescriptor, RGB mask"), ("resnet_bg", "Background only (leakage control)")]


def gait_cross_condition(resolved, rng):
    """Identification across recording conditions (manifest column `condition`).
    Queries are the clips of one condition and the gallery the clips of the
    other (both directions pooled), so query and gallery come from different
    bouts at different belt speeds. Every representation is centred (scalars:
    z-scored) within each condition, which removes the effect of the condition
    itself. Also: agreement of each individual's median gait variable between
    the two conditions (Pearson r and repeatability R of the centred medians)."""
    rows, agree = [], []
    for sp, d in resolved.items():
        m, bank = d["meta"], d["bank"]
        if "condition" not in m or not (m["condition"] != "").any():
            continue
        cond, ids = m["condition"].to_numpy(), m.individual.values
        for a, b, label in CROSS_PAIRS:
            ia, ib = np.where(cond == a)[0], np.where(cond == b)[0]
            if not len(ia) or not len(ib):
                continue
            for name, rlabel in CROSS_REPS:
                if name not in bank:
                    continue
                xa, xb = (prepare({name: bank[name][i]}, name) for i in (ia, ib))
                pq = []
                for xq, xg, iq, ig in ((xa, xb, ia, ib), (xb, xa, ib, ia)):
                    sim = xq @ xg.T
                    for k, qi in enumerate(iq):
                        rel = ids[ig] == ids[qi]
                        if rel.any():
                            order = np.argsort(-sim[k])
                            pq.append(dict(individual=ids[qi], ap=average_precision(sim[k], rel),
                                           rank=int(np.flatnonzero(rel[order])[0]) + 1, chance=rel.mean()))
                pq = pd.DataFrame(pq)
                lo, hi = cluster_bootstrap(pq, lambda t: t.ap.mean(), rng)
                rows.append(dict(species=sp, pair=label, rep=name, label=rlabel, n_queries=len(pq),
                                 n_individuals=pq.individual.nunique(), rank1=(pq["rank"] == 1).mean(),
                                 mAP=pq.ap.mean(), mAP_lo=lo, mAP_hi=hi, chance_mAP=pq.chance.mean()))
            # agreement of per-individual medians between the two conditions
            v = bank["gait_pose"]
            for j, var in enumerate(GAIT_VARS):
                med = {}
                for c, idx in ((a, ia), (b, ib)):
                    x = circ_centre(v[idx, j]) if var in GAIT_CIRCULAR else v[idx, j] - np.median(v[idx, j])
                    med[c] = pd.Series(x).groupby(ids[idx]).median()
                both = pd.concat(med, axis=1).dropna()
                R, lo, hi = repeatability(both.to_numpy().ravel(), np.repeat(both.index.to_numpy(), 2))
                agree.append(dict(species=sp, pair=label, metric=var, individuals=len(both),
                                  pearson_r=np.corrcoef(both[a], both[b])[0, 1], R=R, R_lo=lo, R_hi=hi))
    out, agree = pd.DataFrame(rows), pd.DataFrame(agree)
    if out.empty:
        return out, agree
    pairs = list(dict.fromkeys(out.pair))
    lines = [r"\begin{tabular}{l" + "cc" * len(pairs) + "}", r"\toprule",
             " & " + " & ".join(r"\multicolumn{2}{c}{" + q + "}" for q in pairs) + r" \\",
             "Representation & " + " & ".join([r"Rank-1 & mAP [95\% CI]"] * len(pairs)) + r" \\", r"\midrule"]
    for name, rlabel in CROSS_REPS:
        cells = []
        for q in pairs:
            r = out[(out.pair == q) & (out.rep == name)]
            if r.empty:
                cells += ["--", "--"]
            else:
                r = r.iloc[0]
                cells += [fmt(r.rank1), f"{fmt(r.mAP)} [{fmt(r.mAP_lo)}, {fmt(r.mAP_hi)}]"]
        lines.append(rlabel + " & " + " & ".join(cells) + r" \\")
    lines.append("Chance & " + " & ".join(
        f"& {fmt(out[out.pair == q].chance_mAP.iloc[0])}" for q in pairs) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_gait_cross_condition.tex"), "w") as f:
        f.write("\n".join(lines))
    lines = [r"\begin{tabular}{l" + "c" * len(pairs) + "}", r"\toprule",
             "Variable & " + " & ".join(pairs) + r" \\", r"\midrule"]
    for var in GAIT_VARS:
        cells = []
        for q in pairs:
            r = agree[(agree.pair == q) & (agree.metric == var)].iloc[0]
            cells.append(f"{fmt(r.R)} [{fmt(r.R_lo)}, {fmt(r.R_hi)}]")
        lines.append(GAIT_LABEL[var].replace(" (trunk lengths)", "").replace(" (trunk lengths s$^{-1}$)", "")
                     + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_gait_between_conditions.tex"), "w") as f:
        f.write("\n".join(lines))
    return out, agree


def gait_tex(summ, rept, g_summary):
    order = [sp for sp in summ.species if sp in set(rept.species)]
    lines = [r"\begin{tabular}{l" + "c" * len(order) + "}", r"\toprule",
             "Variable & " + " & ".join(nice(sp) for sp in order) + r" \\", r"\midrule",
             "Clips with resolved gait & " + " & ".join(
                 f"{int(summ[summ.species == sp].resolved.iloc[0])}/"
                 f"{int(summ[summ.species == sp].clips.iloc[0])}" for sp in order) + r" \\",
             r"\addlinespace"]
    for name in GAIT_VARS:
        cells = []
        for sp in order:
            r = summ[summ.species == sp].iloc[0]
            d = 1 if name.endswith("_deg") else 2
            cells.append(f"{fmt(r[name], d)} ({fmt(r[name + '_sd'], d)})")
        lines.append(GAIT_LABEL[name] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_gait.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{tabular}{l" + "c" * len(order) + "}", r"\toprule",
             "Variable & " + " & ".join(nice(sp) for sp in order) + r" \\", r"\midrule"]
    for name in GAIT_VARS:
        cells = []
        for sp in order:
            r = rept[(rept.species == sp) & (rept.metric == name)].iloc[0]
            cells.append("--" if not np.isfinite(r.R) else
                         f"{fmt(r.R)} [{fmt(r.R_lo)}, {fmt(r.R_hi)}]")
        lines.append(GAIT_LABEL[name] + " & " + " & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_gait_repeatability.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{tabular}{lllccc}", r"\toprule",
             r"Species & Protocol & Representation & Queries (animals) & Rank-1 & "
             r"mAP [95\% CI] \\", r"\midrule"]
    for (sp, proto), g in g_summary.groupby(["species", "protocol"], sort=False):
        if g.n_individuals.max() < 2:
            continue
        for r in GAIT_REPS:
            x = g[g.rep == r]
            if x.empty:
                continue
            x = x.iloc[0]
            lines.append(f"{nice(sp)} & {proto} & {REPS[r][0]} & {x.n_queries} "
                         f"({x.n_individuals}) & {fmt(x.rank1)} & "
                         f"{fmt(x.mAP)} [{fmt(x.mAP_lo)}, {fmt(x.mAP_hi)}]" + r" \\")
        lines.append(f"{nice(sp)} & {proto} & Chance & & & {fmt(g.chance_mAP.iloc[0])}" + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_retrieval_gait.tex"), "w") as f:
        f.write("\n".join(lines))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
LIMBS = [("l_hind", "LH"), ("l_front", "LF"), ("r_front", "RF"), ("r_hind", "RH")]


def fig_gait(resolved):
    """(a) Hildebrand gait graph: limb phase against duty factor, one point per
    clip. (b) Footfall diagram per species: stance of each limb over one stride,
    from the circular-mean touchdown phase and mean duty factor."""
    species = list(resolved)
    palette = [BLUE, ORANGE, AQUA, VIOLET, MAGENTA, GREEN]
    fig = plt.figure(figsize=(7.2, 3.4))
    gs = fig.add_gridspec(len(species), 2, width_ratios=[1.15, 1], wspace=0.3, hspace=0.55)
    ax = fig.add_subplot(gs[:, 0])
    for i, sp in enumerate(species):
        v = resolved[sp]["bank"]["gait_pose"]
        duty = v[:, GAIT_VARS.index("duty_factor")]
        ph = v[:, GAIT_VARS.index("limb_phase")]
        ax.plot(duty, ph, "o", ms=3.5, color=palette[i % len(palette)], alpha=0.75,
                markeredgecolor="white", markeredgewidth=0.4, label=f"{nice(sp)} (n={len(ph)})")
    for y, name in ((0.0, "pace"), (0.25, "singlefoot"), (0.5, "trot")):
        ax.axhline(y, color=MUTED, lw=0.6, ls=":")
        ax.text(0.995, y + 0.012, name, color=MUTED, fontsize=6, ha="right",
                transform=ax.get_yaxis_transform())
    ax.axvline(0.5, color=MUTED, lw=0.6, ls=":")
    ax.set_xlabel("duty factor (fraction of stride in stance)")
    ax.set_ylabel("limb phase (fore after same-side hind)")
    ax.set_ylim(-0.03, 1.0)
    ax.legend(fontsize=6, loc="upper left", frameon=False)
    ax.set_title("(a) Gait graph", fontsize=8, color=INK, loc="left")
    for i, sp in enumerate(species):
        a = fig.add_subplot(gs[i, 1])
        m = resolved[sp]["meta"]
        for k, (limb, lab) in enumerate(LIMBS):
            td = m["td_" + limb].to_numpy(float)
            start = np.angle(np.exp(2j * np.pi * td).mean()) / (2 * np.pi) % 1
            width = float(np.nanmean(m["duty_" + limb]))
            for s0 in (start - 1, start):
                a.barh(k, width, left=s0, height=0.7, color=palette[i % len(palette)])
        a.set_xlim(0, 1)
        a.set_ylim(len(LIMBS) - 0.4, -0.6)
        a.set_yticks(range(len(LIMBS)))
        a.set_yticklabels([lab for _, lab in LIMBS], fontsize=6)
        a.set_xticks([0, 0.25, 0.5, 0.75, 1])
        a.tick_params(axis="x", labelsize=6, labelbottom=i == len(species) - 1)
        a.grid(axis="y", visible=False)
        a.set_ylabel(nice(sp), fontsize=6, rotation=0, ha="right", va="center", labelpad=14)
        if i == 0:
            a.set_title("(b) Stance of each limb", fontsize=8, color=INK, loc="left")
    a.set_xlabel("fraction of stride (0 = left hind touchdown)")
    save(fig, "fig_gait")


def fig_gait_individuals(resolved, rept, variables=("stride_frequency_hz", "duty_factor",
                                                    "stride_length", "limb_phase")):
    """Each locomotion variable by individual, resolved clips only."""
    species = list(resolved)
    widths = [min(14, max(4, resolved[sp]["meta"].individual.nunique())) for sp in species]
    fig, axes = plt.subplots(len(variables), len(species), figsize=(10.5, 1.9 * len(variables)),
                             squeeze=False, gridspec_kw=dict(width_ratios=widths, wspace=0.45,
                                                             hspace=0.45))
    for r, name in enumerate(variables):
        j = GAIT_VARS.index(name)
        for c, sp in enumerate(species):
            ax, m = axes[r, c], resolved[sp]["meta"]
            v = resolved[sp]["bank"]["gait_pose"][:, j]
            if name in GAIT_CIRCULAR:
                mean = np.angle(np.exp(2j * np.pi * v).mean()) / (2 * np.pi) % 1
                v = circ_centre(v) + mean
            inds = sorted(m.individual.unique())
            for i, ind in enumerate(inds):
                vals = v[m.individual.values == ind]
                jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
                ax.plot(i + jitter, vals, "o", ms=3, color=BLUE, alpha=0.7,
                        markeredgecolor="white", markeredgewidth=0.4)
                if len(vals):
                    ax.plot([i - 0.3, i + 0.3], [np.median(vals)] * 2, color=INK, lw=1)
            q = rept[(rept.species == sp) & (rept.metric == name)].iloc[0]
            head = f"{nice(sp)}\n" if r == 0 else ""
            ax.set_title(f"{head}R = {q.R:.2f}\n[{q.R_lo:.2f}, {q.R_hi:.2f}]", color=INK, fontsize=6.5)
            ax.set_xticks(range(len(inds)))
            long = max(len(str(i)) for i in inds) > 3
            if r == len(variables) - 1:
                ax.set_xticklabels([str(i).replace("FM", "") for i in inds],
                                   fontsize=4.5 if long or len(inds) > 6 else 7,
                                   rotation=60 if long else 0, ha="right" if long else "center")
            else:
                ax.set_xticklabels([])
            ax.tick_params(axis="y", labelsize=6)
            ax.set_xlim(-0.6, len(inds) - 0.4)
            ax.grid(axis="x", visible=False)
        axes[r, 0].set_ylabel(GAIT_LABEL[name].replace(" (trunk lengths)", "\n(trunk lengths)"),
                              fontsize=7)
    save(fig, "fig_gait_individuals")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, name + ".pdf"))
    fig.savefig(os.path.join(FIG, name + ".png"))
    plt.close(fig)


def fig_dataset(data):
    """Clips per individual, segmented by source recording."""
    species = list(data)
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 2.4), sharey=True,
                             gridspec_kw=dict(width_ratios=[d["meta"].individual.nunique()
                                                            for d in data.values()]))
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, species):
        m = data[sp]["meta"]
        for xi, (ind, g) in enumerate(m.groupby("individual")):
            bottom = 0
            for k, (_, s) in enumerate(g.groupby("source_id", sort=False)):
                ax.bar(xi, len(s), bottom=bottom, width=0.7, color=[BLUE, ORANGE, AQUA][k % 3],
                       edgecolor="white", linewidth=1.5)
                bottom += len(s)
        ax.set_xticks(range(m.individual.nunique()))
        labels = sorted(m.individual.unique())
        long = max(len(str(l)) for l in labels) > 3
        ax.set_xticklabels(labels, rotation=60 if long else 0, ha="right" if long else "center",
                           fontsize=6 if long or len(labels) > 6 else 8)
        ax.set_title(nice(sp), color=INK)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("clips")
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in (BLUE, ORANGE, AQUA)]
    axes[-1].legend(handles, ["recording 1", "recording 2", "recording 3"],
                    loc="upper right", fontsize=7)
    save(fig, "fig_dataset")


def fig_masks(data, rng):
    """One seeded random clip per species: the frame zoomed to the animal with the
    SAM 2 mask outlined, and eight silhouettes spread over the clip."""
    import cv2
    rows = [(sp, d["meta"].clip_id.values[rng.integers(len(d["meta"]))]) for sp, d in data.items()]
    cell = 96
    fig, axes = plt.subplots(len(rows), 2, figsize=(7.2, 1.25 * len(rows)),
                             gridspec_kw=dict(width_ratios=[1.3, 8 * 1.06]), squeeze=False)
    for r, (sp, cid) in enumerate(rows):
        z = np.load(os.path.join(ROOT, "output", "masks", *cid.split("/")) + ".npz")
        shape = tuple(z["shape"])
        masks = np.unpackbits(z["masks"], axis=-1)[..., :shape[2]].astype(bool)
        valid = np.flatnonzero(masks.reshape(len(masks), -1).any(1))
        t = valid[len(valid) // 2]
        cap = cv2.VideoCapture(os.path.join(ROOT, data[sp]["meta"].set_index("clip_id").loc[cid, "path"]))
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t))
        ok, frame = cap.read()
        cap.release()
        frame = cv2.resize(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), (shape[2], shape[1]))
        m = masks[t].astype(np.uint8)
        contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(frame, contours, -1, (230, 90, 40), max(1, shape[2] // 300))
        ys, xs = np.where(m)
        cy, cx = (ys.min() + ys.max()) // 2, (xs.min() + xs.max()) // 2
        half = int(max(np.ptp(ys), np.ptp(xs)) * 0.8) + 8
        y0, x0 = max(0, cy - half), max(0, cx - int(half * 1.3))
        axes[r, 0].imshow(frame[y0:cy + half, x0:cx + int(half * 1.3)])
        strip = []
        for i in valid[np.linspace(0, len(valid) - 1, 8).round().astype(int)]:
            yy, xx = np.where(masks[i])
            c = masks[i][yy.min():yy.max() + 1, xx.min():xx.max() + 1].astype(np.float32)
            s = (cell - 8) / max(c.shape)
            c = cv2.resize(c, (max(1, round(c.shape[1] * s)), max(1, round(c.shape[0] * s))),
                           interpolation=cv2.INTER_AREA)
            box = np.zeros((cell, cell), np.float32)
            oy, ox = (cell - c.shape[0]) // 2, (cell - c.shape[1]) // 2
            box[oy:oy + c.shape[0], ox:ox + c.shape[1]] = c
            strip += [box, np.zeros((cell, 10), np.float32)]
        axes[r, 1].imshow(np.hstack(strip[:-1]), cmap="gray_r", vmin=0, vmax=1)
        axes[r, 0].set_ylabel(nice(sp), fontsize=8, color=INK)
        axes[r, 1].set_ylabel(cid.split("/", 1)[1],
                              fontsize=6.5, color=MUTED, rotation=0, ha="right", va="center")
        for ax in axes[r]:
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for sp_ in ax.spines.values():
                sp_.set_visible(False)
    axes[0, 0].set_title("frame + SAM 2 mask", fontsize=8, color=INK)
    axes[0, 1].set_title("silhouettes sampled across the clip", fontsize=8, color=INK)
    fig.tight_layout(w_pad=4)
    save(fig, "fig_masks")


def fig_retrieval(summary):
    """mAP per representation and species: bars = P1 (same recording allowed),
    markers = leakage-resistant protocols; dotted line = P1 chance."""
    species = list(summary.species.unique())
    reps = [r for r in REPS if r != "fused_raw"]
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 3.9), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, species):
        s = summary[(summary.species == sp) & (summary.protocol == "P1")].set_index("rep")
        y = np.arange(len(reps))[::-1]
        for yi, r in zip(y, reps):
            row = s.loc[r]
            ax.barh(yi, row.mAP, height=0.62, color=FAMILY_COLOR[row.family])
            ax.plot([row.mAP_lo, row.mAP_hi], [yi, yi], color=INK, lw=1)
        ax.axvline(s.chance_mAP.mean(), color=INK2, ls=":", lw=1)
        ax.text(s.chance_mAP.mean(), len(reps) - 0.3, " chance", fontsize=6.5, color=INK2)
        for proto, mk, face in (("P2", "D", "white"), ("P3", "s", INK)):
            pp = summary[(summary.species == sp) & (summary.protocol == proto)].set_index("rep")
            for yi, r in zip(y, reps):
                if r in pp.index:
                    ax.plot(pp.loc[r, "mAP"], yi, mk, ms=4.5, color=face,
                            markeredgecolor=INK, markeredgewidth=1)
        ax.set_xlim(0, 1)
        ax.set_title(f"{nice(sp)}", color=INK)
        ax.set_xlabel("mAP")
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(np.arange(len(reps))[::-1])
    axes[0].set_yticklabels([REPS[r][0] for r in reps])
    handles = [plt.Rectangle((0, 0), 1, 1, color=FAMILY_COLOR[f]) for f in FAMILY_COLOR] + \
              [plt.Line2D([], [], marker="D", ls="", color="white", markeredgecolor=INK),
               plt.Line2D([], [], marker="s", ls="", color=INK, markeredgecolor=INK)]
    labels = ["proposed fusion", "appearance", "silhouette / gait", "background control",
              "cross-recording (P2)", "cross-camera, same site (P3)"]
    if "P3" not in set(summary.protocol):
        handles, labels = handles[:-1], labels[:-1]
    fig.legend(handles, labels, loc="lower center", ncol=3, fontsize=7,
               bbox_to_anchor=(0.55, -0.12))
    save(fig, "fig_retrieval")


def fig_cmc(per_query):
    """CMC under each species' most leakage-resistant protocol."""
    species = list(per_query.species.unique())
    colors = {"fused": BLUE, "mega_rgb": ORANGE, "vp_ordered": AQUA}
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 2.3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, species):
        proto = strictest(per_query, sp)
        g = per_query[(per_query.species == sp) & (per_query.protocol == proto)]
        for r, c in colors.items():
            ranks = g[g.rep == r]["rank"].values
            ks = np.arange(1, 11)
            ax.plot(ks, [(ranks <= k).mean() for k in ks], color=c, marker="o", ms=4,
                    label=REPS[r][0])
        ax.set_title(f"{nice(sp)} ({proto})", color=INK)
        ax.set_xlabel("rank k")
        ax.set_xticks([1, 5, 10])
        ax.set_ylim(0, 1.02)
    axes[0].set_ylabel("identification rate (CMC)")
    axes[-1].legend(fontsize=6.5, loc="lower right")
    save(fig, "fig_cmc")


def fig_gait_controls(summary, sens):
    species = list(summary.species.unique())
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.2, 2.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    w = 0.2
    shades = ["#0d3f80", "#2a78d6", "#7fb0ee", "#b9d5f7"]
    for i, r in enumerate(GAIT_CONTROLS):
        vals = [summary[(summary.species == sp) & (summary.rep == r) &
                        (summary.protocol == "P1")].mAP.item() for sp in species]
        a.bar(np.arange(len(species)) + (i - 1.5) * w, vals, w * 0.9, color=shades[i],
              label=REPS[r][0].replace("VideoPrism, ", ""))
    a.set_xticks(range(len(species)))
    a.set_xticklabels([nice(s) for s in species])
    a.set_ylabel("mAP (P1)")
    a.set_ylim(0, 1)
    a.legend(fontsize=6.5, ncol=2, loc="upper left")
    a.grid(axis="x", visible=False)
    a.set_title("Identity retrieval from silhouettes", color=INK)
    marks = ["o", "s", "^"]
    for i, ctrl in enumerate(GAIT_CONTROLS[1:]):
        v = [sens[(sens.species == sp) & (sens.control == ctrl)].self_similarity.item()
             for sp in species]
        b.plot(v, np.arange(len(species)) + (i - 1) * 0.2, marks[i], ms=5.5, color=shades[i + 1],
               markeredgecolor=INK2, markeredgewidth=0.5, label=REPS[ctrl][0].split(", ")[1])
    ref = sens.groupby("species", sort=False).first()
    for j, sp in enumerate(species):
        b.plot([ref.loc[sp, "same_animal_other_clip"]] * 2, [j - 0.38, j + 0.38], color=INK, lw=1.2)
        b.plot([ref.loc[sp, "different_animals"]] * 2, [j - 0.38, j + 0.38], color=MUTED, lw=1.2,
               ls=(0, (2, 1.5)))
    b.plot([], [], color=INK, lw=1.2, label="other clips, same animal")
    b.plot([], [], color=MUTED, lw=1.2, ls=(0, (2, 1.5)), label="different animals")
    b.set_yticks(range(len(species)))
    b.set_yticklabels([nice(s) for s in species])
    b.set_xlabel("cosine to the ordered clip (centred)")
    b.set_xlim(None, 1.02)
    b.set_title("How far each control moves a clip", color=INK)
    b.legend(fontsize=6, loc="upper center", bbox_to_anchor=(0.45, -0.28), ncol=2)
    save(fig, "fig_gait_controls")


def fig_pair_types(pairs):
    """Similarity distributions by pair type; one row per species that has
    individuals filmed in more than one recording (or camera)."""
    kinds = ["same animal, same recording", "same animal, different recording", "different animals"]
    colors = [BLUE, ORANGE, AQUA]
    reps = list(pairs.rep.unique())
    species = list(pairs.species.unique())
    fig, axes = plt.subplots(len(species), len(reps), figsize=(7.2, 2.1 * len(species) + 0.5),
                             sharey=True, squeeze=False)
    for r, sp in enumerate(species):
        for c, rep in enumerate(reps):
            ax = axes[r, c]
            g = pairs[(pairs.rep == rep) & (pairs.species == sp)]
            vals = [g[g.kind == k].sim.values for k in kinds]
            parts = ax.violinplot([v if len(v) else [np.nan] for v in vals], showextrema=False,
                                  widths=0.8)
            for body, col in zip(parts["bodies"], colors):
                body.set_facecolor(col); body.set_alpha(0.8); body.set_edgecolor("white")
            for i, v in enumerate(vals):
                if len(v):
                    ax.plot([i + 0.7, i + 1.3], [np.median(v)] * 2, color=INK, lw=1.5)
            ax.set_xticks([1, 2, 3])
            ax.set_xticklabels(["same\nrec.", "other\nrec.", "other\nanimal"]
                               if r == len(species) - 1 else [], fontsize=6.5)
            ax.grid(axis="x", visible=False)
            if r == 0:
                ax.set_title(REPS[rep][0].split(" (")[0].replace(", ", "\n", 1)
                             .replace(": ", ":\n").replace(" + ", "\n+ "), fontsize=7, color=INK)
        axes[r, 0].set_ylabel(f"{nice(sp)}\ncosine similarity", fontsize=7.5)
    fig.tight_layout()
    save(fig, "fig_pair_types")


def fig_kinematics(data, kin):
    species = list(data)
    widths = [max(3, data[sp]["meta"].individual.nunique()) for sp in species]
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 2.7), sharey=False,
                             gridspec_kw=dict(width_ratios=widths, wspace=0.45))
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, species):
        m = data[sp]["meta"]
        v = data[sp]["bank"]["kinematics"][:, 0]
        inds = sorted(m.individual.unique())
        for i, ind in enumerate(inds):
            vals = v[m.individual.values == ind]
            vals = vals[~np.isnan(vals)]
            if not len(vals):
                continue
            ax.boxplot([vals], positions=[i], widths=0.55, showfliers=False,
                       medianprops=dict(color=INK), boxprops=dict(color=MUTED),
                       whiskerprops=dict(color=MUTED), capprops=dict(color=MUTED))
            jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
            ax.plot(i + jitter, vals, "o", ms=3.5, color=BLUE, alpha=0.7,
                    markeredgecolor="white", markeredgewidth=0.4)
        r = kin[(kin.species == sp) & (kin.metric == "stride_frequency_hz")].iloc[0]
        ax.set_title(f"{nice(sp)}\nR = {r.R:.2f}\n[{r.R_lo:.2f}, {r.R_hi:.2f}]",
                     color=INK, fontsize=8)
        ax.set_xticks(range(len(inds)))
        long = max(len(str(i)) for i in inds) > 3
        ax.set_xticklabels(inds, fontsize=6 if long or len(inds) > 6 else 8,
                           rotation=60 if long else 0, ha="right" if long else "center")
        ax.set_xlim(-0.6, len(inds) - 0.4)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("stride frequency (Hz)")
    save(fig, "fig_kinematics")


def fig_heatmaps(data):
    species = list(data)
    fig, axes = plt.subplots(1, len(species), figsize=(7.2, 2.6),
                             gridspec_kw=dict(wspace=0.35))
    axes = np.atleast_1d(axes)
    for ax, sp in zip(axes, species):
        m = data[sp]["meta"]
        order = np.lexsort((m.chunk_index.values, m.source_id.values, m.individual.values))
        x = representation(data[sp]["bank"], REPS["fused"][2])[order]
        im = ax.imshow(x @ x.T, cmap=SEQ, vmin=-0.2, vmax=1, interpolation="nearest")
        ids = m.individual.values[order]
        edges = np.flatnonzero(ids[1:] != ids[:-1]) + 0.5
        lw, fs = (0.4, 5) if len(edges) > 6 else (1, 8)
        for e in edges:
            ax.axhline(e, color="white", lw=lw); ax.axvline(e, color="white", lw=lw)
        centers = [np.mean(np.flatnonzero(ids == u)) for u in pd.unique(ids)]
        ax.set_xticks(centers); ax.set_xticklabels(pd.unique(ids), fontsize=fs)
        ax.set_yticks(centers); ax.set_yticklabels(pd.unique(ids), fontsize=fs)
        ax.grid(False)
        ax.set_title(nice(sp), color=INK)
    fig.colorbar(im, ax=axes, shrink=0.8, label="cosine similarity (fused, centred)")
    save(fig, "fig_heatmaps")


def fig_embedding_maps(data, reps=("mega_crop", "vp_ordered"), max_ids=8):
    """Classical MDS of two representations per species: colour = individual,
    marker = source recording within that individual. Clusters that follow
    marker rather than colour are recordings, not animals."""
    species = [sp for sp in data if data[sp]["meta"].individual.nunique() <= max_ids]
    markers = "os^Dv"
    palette = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]
    fig, axes = plt.subplots(len(reps), len(species), figsize=(7.2, 2.3 * len(reps) + 0.4),
                             squeeze=False)
    for c, sp in enumerate(species):
        m = data[sp]["meta"]
        ids = m.individual.values
        src_rank = m.groupby("individual").source_id.transform(
            lambda s: pd.factorize(s)[0]).to_numpy()
        for r, rep in enumerate(reps):
            x = representation(data[sp]["bank"], REPS[rep][2])
            d = np.clip(1 - x @ x.T, 0, 2)
            n = len(d)
            J = np.eye(n) - 1 / n
            w, v = np.linalg.eigh(-0.5 * J @ (d ** 2) @ J)
            xy = v[:, ::-1][:, :2] * np.sqrt(np.clip(w[::-1][:2], 0, None))
            ax = axes[r, c]
            for i, ind in enumerate(sorted(np.unique(ids))):
                for k in np.unique(src_rank[ids == ind]):
                    sel = (ids == ind) & (src_rank == k)
                    ax.scatter(xy[sel, 0], xy[sel, 1], color=palette[i % 8], s=20,
                               marker=markers[k % len(markers)], edgecolors="white",
                               linewidths=0.6, label=ind if (r == 0 and k == 0) else None)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(nice(sp), color=INK)
                if src_rank.max() > 0:
                    ax.scatter([], [], color=MUTED, marker="s", s=20, label="2nd recording")
                ax.legend(fontsize=6, ncol=len(np.unique(ids)) + 1, loc="upper center",
                          bbox_to_anchor=(0.5, -0.02), handletextpad=0.05, columnspacing=0.5,
                          frameon=False, markerscale=0.9)
            if c == 0:
                ax.set_ylabel(REPS[rep][0].split(" (")[0].replace(", ", "\n", 1), fontsize=7.5)
    fig.tight_layout(h_pad=2.2)
    save(fig, "fig_embedding_maps")


# --------------------------------------------------------------------------
# Tables
# --------------------------------------------------------------------------
def fmt(v, d=2):
    return "--" if pd.isna(v) else f"{v:.{d}f}"


def gated_tex(g_summary, n_all):
    """Retrieval restricted to periodic clips (stride gate)."""
    lines = [r"\begin{tabular}{lllcccc}", r"\toprule",
             r"Species & Protocol & Representation & Clips kept & Queries (animals) & Rank-1 & "
             r"mAP [95\% CI] \\", r"\midrule"]
    for (sp, proto), g in g_summary.groupby(["species", "protocol"], sort=False):
        if g.n_individuals.max() < 2:        # one queried animal: no interval, not informative
            continue
        for r in GATED_TEX:
            s = g[g.rep == r]
            if s.empty:
                continue
            s = s.iloc[0]
            lines.append(f"{nice(sp)} & {proto} & {REPS[r][0]} & {s.clips}/{n_all[sp]} & "
                         f"{s.n_queries} ({s.n_individuals}) & {fmt(s.rank1)} & "
                         f"{fmt(s.mAP)} [{fmt(s.mAP_lo)}, {fmt(s.mAP_hi)}]" + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_retrieval_gated.tex"), "w") as f:
        f.write("\n".join(lines))


def write_tables(ds, summary, controls, sens, kin):
    os.makedirs(TAB, exist_ok=True)
    ds.to_csv(os.path.join(TAB, "table_dataset.csv"), index=False)
    summary.to_csv(os.path.join(TAB, "table_retrieval_full.csv"), index=False)
    controls.to_csv(os.path.join(TAB, "table_gait_controls.csv"), index=False)
    sens.to_csv(os.path.join(TAB, "table_order_sensitivity.csv"), index=False)
    kin.to_csv(os.path.join(TAB, "table_kinematics.csv"), index=False)

    # LaTeX: main retrieval table (rows = representation, cols = species x metric)
    species = list(summary.species.unique())
    lines = [r"\begin{tabular}{l" + "ccc" * len(species) + "}", r"\toprule",
             " & " + " & ".join(rf"\multicolumn{{3}}{{c}}{{{nice(s)}}}" for s in species)
             + r" \\",
             " ".join(rf"\cmidrule(lr){{{2 + 3 * i}-{4 + 3 * i}}}" for i in range(len(species))),
             "Representation & " + " & ".join(["R-1", "mAP", "AUROC"] * len(species)) + r" \\",
             r"\midrule"]
    for r in REPS:
        cells = []
        for sp in species:
            s = summary[(summary.species == sp) & (summary.rep == r) & (summary.protocol == "P1")]
            cells += [fmt(s.rank1.item()), fmt(s.mAP.item()), fmt(s.open_set_auroc.item())]
        lines.append(REPS[r][0] + " & " + " & ".join(cells) + r" \\")
    chance = [fmt(summary[(summary.species == sp) & (summary.protocol == "P1")].chance_mAP.mean())
              for sp in species]
    lines += [r"\midrule", "Chance & " + " & ".join(f"-- & {c} & 0.50" for c in chance) + r" \\",
              r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_retrieval_p1.tex"), "w") as f:
        f.write("\n".join(lines))

    p2 = summary[summary.protocol.isin(["P2", "P3"])]
    lines = [r"\begin{tabular}{lllcccc}", r"\toprule",
             r"Species & Protocol & Representation & Queries (animals) & Rank-1 & mAP [95\% CI] & Chance mAP \\",
             r"\midrule"]
    for (sp, proto), g in p2.groupby(["species", "protocol"], sort=False):
        for r in MAIN:
            s = g[g.rep == r]
            if s.empty:
                continue
            s = s.iloc[0]
            lines.append(f"{nice(sp)} & {proto} & {REPS[r][0]} & {s.n_queries} ({s.n_individuals}) & "
                         f"{fmt(s.rank1)} & {fmt(s.mAP)} [{fmt(s.mAP_lo)}, {fmt(s.mAP_hi)}] & "
                         f"{fmt(s.chance_mAP)}" + r" \\")
        lines.append(r"\addlinespace")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_retrieval_p2.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{tabular}{lcccccc}", r"\toprule",
             r"Species & Individuals & Clips used & Excluded & Failed QC & Source recordings & Animals in $\geq$2 recordings \\",
             r"\midrule"]
    for _, r in ds.iterrows():
        lines.append(f"{nice(r.species)} & {r.individuals} & {r.clips_embedded} & "
                     f"{r.clips_excluded} & {r.clips_failed_qc} & "
                     f"{r.source_recordings} & {r.individuals_multi_source}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_dataset.tex"), "w") as f:
        f.write("\n".join(lines))

    lines = [r"\begin{tabular}{llcccc}", r"\toprule",
             r"Species & Descriptor & Clips & Mean $\pm$ SD & Repeatability $R$ & 95\% CI \\", r"\midrule"]
    for _, r in kin[kin.clips == "periodic"].iterrows():
        lines.append(f"{nice(r.species)} & {r.metric.replace('_', ' ')} & {r.n_resolved} & "
                     f"{r['mean']:.3f} $\\pm$ {r.sd:.3f} & {fmt(r.R)} & [{fmt(r.R_lo)}, {fmt(r.R_hi)}]"
                     + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_kinematics.tex"), "w") as f:
        f.write("\n".join(lines))

    ref = sens.set_index(["species", "control"])
    lines = [r"\begin{tabular}{llccc}", r"\toprule",
             r"Species & Protocol & Control & $\Delta$mAP (ordered $-$ control) [95\% CI] & "
             r"cos(ordered, control) \\", r"\midrule"]
    for _, r in controls.iterrows():
        lines.append(f"{nice(r.species)} & {r.protocol} & "
                     f"{REPS[r.control][0].split(', ')[1]} & {r.delta_mAP:+.3f} "
                     f"[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}] & "
                     f"{ref.loc[(r.species, r.control), 'self_similarity']:.2f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    with open(os.path.join(TAB, "table_gait_controls.tex"), "w") as f:
        f.write("\n".join(lines))


def main():
    rng = np.random.default_rng(SEED)
    data = load()
    from extract_features import load_manifests
    qc = pd.read_csv(os.path.join(ROOT, "results", "segmentation_qc.csv")).merge(
        load_manifests()[["clip_id", "species", "exclude"]], on="clip_id")
    ds = dataset_table(data, qc)
    per_query, summary = run_retrieval(data, rng)
    controls = gait_control_tests(per_query, rng)
    vs_bg = background_tests(per_query, rng)
    vs_bg.to_csv(os.path.join(TAB, "table_vs_background.csv"), index=False)
    sens = order_sensitivity(data)
    pairs = pair_types(data)
    periodic = gated(data)
    kin = pd.concat([kinematics_table(periodic, "periodic"), kinematics_table(data, "all")],
                    ignore_index=True)
    write_tables(ds, summary, controls, sens, kin)
    per_query.to_csv(os.path.join(TAB, "per_query.csv"), index=False)

    # Stride gate: retrieval on periodic clips only, and its threshold sensitivity
    _, g_summary = run_retrieval(periodic, rng, GATED_REPS)
    g_summary["clips"] = g_summary.species.map({sp: len(d["meta"]) for sp, d in periodic.items()})
    g_summary.to_csv(os.path.join(TAB, "table_retrieval_gated.csv"), index=False)
    gated_tex(g_summary, {sp: len(d["meta"]) for sp, d in data.items()})
    sweep = gate_sensitivity(data)
    sweep.to_csv(os.path.join(TAB, "table_gate_sensitivity.csv"), index=False)

    # Pose-based gait analysis: species-level variables, repeatability, retrieval
    gait_summary = None
    if all("gait_resolved" in d["meta"] for d in data.values()):
        resolved = gated(data, True, "gait_resolved")
        by_gait = split_by_gait(resolved)        # walk and trot are described separately
        gait_summ, gait_rep = gait_tables(split_by_gait(data), by_gait)
        _, gait_summary = run_retrieval(resolved, rng, GAIT_REPS)
        gait_summ.to_csv(os.path.join(TAB, "table_gait.csv"), index=False)
        gait_rep.to_csv(os.path.join(TAB, "table_gait_repeatability.csv"), index=False)
        gait_summary.to_csv(os.path.join(TAB, "table_retrieval_gait.csv"), index=False)
        gait_tex(gait_summ, gait_rep, gait_summary)
        gait_cond = gait_by_condition(resolved)
        if not gait_cond.empty:
            gait_cond.to_csv(os.path.join(TAB, "table_gait_conditions.csv"), index=False)
        cross, agree = gait_cross_condition(resolved, rng)
        if not cross.empty:
            cross.to_csv(os.path.join(TAB, "table_gait_cross_condition.csv"), index=False)
            agree.to_csv(os.path.join(TAB, "table_gait_between_conditions.csv"), index=False)
        fig_gait(by_gait)
        fig_gait_individuals(by_gait, gait_rep)

    ft_pq, ft_summ = finetuned_eval(data, rng)
    if not ft_summ.empty:
        ft_pq.to_csv(os.path.join(TAB, "per_query_finetuned.csv"), index=False)
        ft_summ.to_csv(os.path.join(TAB, "table_finetuned.csv"), index=False)
        finetuned_tex(ft_summ)
        fig_finetuned(ft_summ)

    fig_dataset(data)
    fig_masks(data, np.random.default_rng(SEED))
    fig_retrieval(summary)
    fig_cmc(per_query)
    fig_gait_controls(summary, sens)
    if not pairs.empty:
        fig_pair_types(pairs)
    fig_kinematics(periodic, kin[kin.clips == "periodic"])
    fig_heatmaps(data)
    fig_embedding_maps(data)

    pd.set_option("display.width", 200)
    print(ds.to_string(index=False))
    cols = ["species", "rep", "protocol", "n_queries", "rank1", "mAP", "mAP_lo", "mAP_hi",
            "chance_mAP", "open_set_auroc", "ari", "silhouette_gt"]
    print(summary[cols].round(3).to_string(index=False))
    print(controls.round(3).to_string(index=False))
    print(vs_bg.round(3).to_string(index=False))
    print(sens.round(4).to_string(index=False))
    print(kin.round(3).to_string(index=False))
    print(f"\nStride gate (stride_autocorr >= {GAIT_MIN_AUTOCORR}), periodic clips only")
    print(g_summary[["species", "rep", "protocol", "clips", "n_queries", "rank1", "mAP",
                     "mAP_lo", "mAP_hi", "chance_mAP"]].round(3).to_string(index=False))
    print(sweep.pivot_table(index=["species", "protocol", "rep"], columns="threshold",
                            values="mAP", sort=False).round(3).to_string())
    if not ft_summ.empty:
        print("\nFine-tuned encoders, held-out recordings (source-grouped folds)")
        print(ft_summ.drop(columns="label").round(3).to_string(index=False))
        print(pd.read_csv(os.path.join(TAB, "table_finetuned_deltas.csv")).round(3).to_string(index=False))
    if gait_summary is not None:
        print("\nPose-based gait, clips with a resolved gait")
        print(gait_summ.round(3).T.to_string())
        print(gait_rep.round(3).to_string(index=False))
        print(gait_summary[["species", "rep", "protocol", "n_queries", "n_individuals", "rank1",
                            "mAP", "mAP_lo", "mAP_hi", "chance_mAP"]].round(3).to_string(index=False))
    if not pairs.empty:
        print(pairs.groupby(["species", "rep", "kind"]).sim.agg(["median", "size"]).round(3))
    qc_fail = qc[qc.status != "ok"]
    print(f"\nSegmentation QC: {len(qc) - len(qc_fail)}/{len(qc)} ok")
    with open(os.path.join(TAB, "qc_failures.json"), "w") as f:
        json.dump(qc_fail.to_dict("records"), f, indent=1)


if __name__ == "__main__":
    main()
