"""Wildlife gait analysis pipeline.

Implements the pipeline described in GaitAnalysis.pdf:

  SAM segmentation  -> RGB mask R_t + binary silhouette B_t         (Sec 2.2)
  ResNet18(R_t) = s_t, VideoPrism(B_{t-tau+1:t}) = v_t              (Sec 2.3)
  h_bar = mean_t concat(s_t, v_t)                                   (Sec 2.3/2.4)
  A = cosine(h_bar), d = 1 - A                                      (Sec 2.4/2.5)
  classical MDS -> k-means; average-linkage dendrogram cut at 0.15;
  silhouette on the original cosine distances; Table 2 statistics   (Sec 2.5)

All analysis is performed *within* a species (Sec 2.1), and the fused condition
is reported alongside the spatial-only and temporal-only ablations (Sec 3.3).

Example:
    python run_pipeline.py --species giraffe
    python run_pipeline.py --species camel --individuals A B --max-videos 3
"""

import argparse
import json
import multiprocessing
import os

# Set before sklearn/joblib import so the thread limits actually take effect.
os.environ.setdefault("OMP_NUM_THREADS", "1")
try:
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", str(multiprocessing.cpu_count()))
except NotImplementedError:
    pass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from src import clustering as clu
from src.feature_extractor import (SamVideoSegmenter, apply_rgb_mask,
                                   build_silhouette_windows, extract_frames,
                                   get_resnet_model, get_sam_predictor,
                                   get_spatial_features, get_temporal_features,
                                   get_videoprism_model, sequence_embedding)

# Sec 3.3 ablations: spatial-only, temporal-only, and the fused representation.
CONDITIONS = ("fused", "spatial_only", "temporal_only")


# --------------------------------------------------------------------------
# Data discovery
# --------------------------------------------------------------------------
def discover_videos(input_dir, individuals=None, max_videos=None):
    """Finds clips laid out as <input_dir>/<individual>/<clip>.mp4."""
    found = sorted(d.name for d in os.scandir(input_dir) if d.is_dir())
    if individuals:
        missing = set(individuals) - set(found)
        if missing:
            raise SystemExit(f"Individuals not found in {input_dir}: {sorted(missing)}")
        found = [i for i in found if i in individuals]

    videos = []
    for individual in found:
        directory = os.path.join(input_dir, individual)
        clips = sorted(f for f in os.listdir(directory) if f.endswith(".mp4"))
        if max_videos:
            clips = clips[:max_videos]
        for clip in clips:
            videos.append({
                "path": os.path.join(directory, clip),
                "individual": individual,
                "filename": clip,
                # Sec 2.1: each snippet retains a source-video / session id so
                # clips from one recording can be identified. Filenames here are
                # <individual>_<n>.mp4, so the individual is the coarsest
                # provenance key available; refine if a manifest is added.
                "source_group": f"{individual}",
            })
    return found, videos


# --------------------------------------------------------------------------
# Feature extraction
# --------------------------------------------------------------------------
def extract_video_features(video, segmenter, resnet_model, videoprism, args):
    """Returns (spatial 512-d, temporal 768-d, diagnostics) for one clip."""
    frames, fps = extract_frames(video["path"], max_frames=args.max_frames)
    if not frames:
        return None, None, {"error": "no frames"}

    if args.frame_stride > 1:
        frames = frames[::args.frame_stride]
        fps = fps / args.frame_stride

    # Sec 2.2 -- one segmentation pass yields both streams' inputs.
    masks = segmenter.segment_video(frames)
    n_valid = sum(m is not None for m in masks)
    if n_valid == 0:
        return None, None, {"error": "segmentation lost target in all frames"}

    # Spatial stream: RGB-masked frames.
    rgb_masked = [apply_rgb_mask(f, m) for f, m in zip(frames, masks) if m is not None]
    spatial = get_spatial_features(resnet_model, rgb_masked, device=args.device)

    # Temporal stream: ordered binary silhouettes, 5 fps, 16-frame windows.
    windows = build_silhouette_windows(masks, fps)
    temporal = get_temporal_features(videoprism[0], videoprism[1], windows)

    diagnostics = {
        "n_frames": len(frames),
        "n_masked_frames": n_valid,
        "mask_coverage": round(n_valid / len(frames), 3),
        "n_temporal_windows": len(windows),
        "native_fps": round(fps, 2),
    }
    return spatial, temporal, diagnostics


def build_embeddings(videos, args):
    """Extracts (or loads cached) per-clip stream embeddings."""
    cache_path = os.path.join(args.output_dir, "embeddings.npz")
    if args.use_cache and os.path.exists(cache_path):
        print(f"Loading cached embeddings from {cache_path}")
        cached = np.load(cache_path, allow_pickle=True)
        cached_names = [str(n) for n in cached["names"]]
        by_name = {v["filename"]: v for v in videos}
        # The cache stores only clips that yielded features, so match against
        # the requested set rather than requiring equality.
        if set(cached_names) <= set(by_name):
            kept = [by_name[n] for n in cached_names]
            return (cached["spatial"], cached["temporal"], kept,
                    json.loads(cached["diagnostics"].item()))
        print("  cache does not match the requested clip set; re-extracting.")

    print("Loading models...")
    segmenter = SamVideoSegmenter(get_sam_predictor(device=args.device),
                                  max_side=args.seg_max_side)
    resnet_model = get_resnet_model(checkpoint=args.resnet_checkpoint, device=args.device)
    videoprism = get_videoprism_model()
    print("Models loaded.")

    spatial_list, temporal_list, kept, diagnostics = [], [], [], {}
    for video in tqdm(videos, desc="Processing videos"):
        spatial, temporal, diag = extract_video_features(
            video, segmenter, resnet_model, videoprism, args)
        diagnostics[video["filename"]] = diag
        if spatial is None or len(spatial) == 0 or len(temporal) == 0:
            print(f"  skipping {video['filename']}: {diag.get('error', 'empty features')}")
            continue
        s, v = sequence_embedding(spatial, temporal, stream_norm=args.stream_norm)
        spatial_list.append(s)
        temporal_list.append(v)
        kept.append(video)

    if not kept:
        raise SystemExit("No clips produced usable features.")

    spatial_arr = np.stack(spatial_list)
    temporal_arr = np.stack(temporal_list)

    os.makedirs(args.output_dir, exist_ok=True)
    np.savez(cache_path,
             names=np.array([v["filename"] for v in kept]),
             individuals=np.array([v["individual"] for v in kept]),
             spatial=spatial_arr, temporal=temporal_arr,
             diagnostics=json.dumps(diagnostics))
    print(f"Cached embeddings to {cache_path}")
    return spatial_arr, temporal_arr, kept, diagnostics


def compose_features(spatial, temporal, condition):
    """Sec 2.3 fusion (concatenation) and the Sec 3.3 single-stream ablations."""
    if condition == "spatial_only":
        return spatial
    if condition == "temporal_only":
        return temporal
    return np.concatenate([spatial, temporal], axis=1)


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------
def plot_similarity_heatmap(similarity, names, individuals, path, title):
    """Heatmap ordered by individual, with block boundaries drawn (Figs 5-9)."""
    order = np.argsort(individuals, kind="stable")
    sim = similarity[np.ix_(order, order)]
    ordered_names = [f"{individuals[i]}:{names[i]}" for i in order]

    fig, ax = plt.subplots(figsize=(max(8, len(names) * 0.42),
                                    max(6.5, len(names) * 0.38)))
    im = ax.imshow(sim, cmap="viridis", vmin=np.min(sim), vmax=1.0)
    fig.colorbar(im, ax=ax, label="cosine similarity")

    # Separators between annotated individuals.
    sorted_individuals = np.asarray(individuals)[order]
    boundaries = np.where(sorted_individuals[1:] != sorted_individuals[:-1])[0] + 0.5
    for b in boundaries:
        ax.axhline(b, color="w", lw=1.2)
        ax.axvline(b, color="w", lw=1.2)

    ax.set_xticks(range(len(ordered_names)))
    ax.set_xticklabels(ordered_names, rotation=90, fontsize=7)
    ax.set_yticks(range(len(ordered_names)))
    ax.set_yticklabels(ordered_names, fontsize=7)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_mds_kmeans(coords, clusters, individuals, path, title):
    """k-means assignments on the MDS coordinates (Fig 3)."""
    fig, ax = plt.subplots(figsize=(9, 7.5))
    scatter = ax.scatter(coords[:, 0], coords[:, 1], c=clusters,
                         cmap="viridis", s=90, edgecolors="k", linewidths=0.5)
    for i, individual in enumerate(individuals):
        ax.annotate(individual, (coords[i, 0], coords[i, 1]),
                    textcoords="offset points", xytext=(0, 9),
                    ha="center", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("MDS dimension 1")
    ax.set_ylabel("MDS dimension 2")
    ax.legend(*scatter.legend_elements(), title="k-means cluster",
              loc="best", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_dendrogram(z, names, individuals, path, title, cut=clu.DENDROGRAM_CUT):
    """Average-linkage dendrogram with the cut threshold drawn (Fig 4)."""
    labels = [f"{ind}:{n}" for ind, n in zip(individuals, names)]
    fig, ax = plt.subplots(figsize=(max(9, len(names) * 0.4), 6.5))
    clu.dendrogram(z, labels=labels, ax=ax, color_threshold=cut,
                   leaf_font_size=7)
    ax.axhline(cut, color="r", ls="--", lw=1,
               label=f"cut = {cut}")
    ax.set_ylabel("cosine distance (average linkage)")
    ax.set_title(title)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Analysis
# --------------------------------------------------------------------------
def analyse_condition(features, videos, condition, args):
    """Runs Sec 2.4-2.5 for one feature condition and writes its outputs."""
    out_dir = os.path.join(args.output_dir, condition)
    os.makedirs(out_dir, exist_ok=True)

    names = [v["filename"] for v in videos]
    individuals = [v["individual"] for v in videos]
    n_individuals = len(set(individuals))

    similarity = clu.similarity_matrix(features)
    dissimilarity = clu.dissimilarity_matrix(similarity)

    clu.similarity_dataframe(similarity, names).to_csv(
        os.path.join(out_dir, "similarity.csv"))
    plot_similarity_heatmap(
        similarity, names, individuals,
        os.path.join(out_dir, "similarity_heatmap.png"),
        f"{args.species} cosine similarity ({condition})")

    # Classical MDS -> k-means (Sec 2.5).
    coords = clu.classical_mds(dissimilarity, n_components=2)
    kmeans_labels = clu.kmeans_on_mds(coords, n_individuals, seed=args.seed)
    plot_mds_kmeans(coords, kmeans_labels, individuals,
                    os.path.join(out_dir, "mds_kmeans.png"),
                    f"{args.species} k-means on MDS coordinates ({condition})")

    # Average-linkage hierarchical clustering on d_ij (Sec 2.5).
    z, hier_labels = clu.hierarchical_clusters(dissimilarity, cut=args.dendrogram_cut)

    # The paper's fixed 0.15 cut assumes the wider embedding spread produced by
    # fine-tuned encoders. With frozen encoders every distance can fall below
    # it, collapsing the dendrogram to one cluster; say so rather than silently
    # reporting n_clusters = 1.
    max_distance = float(dissimilarity[np.triu_indices_from(dissimilarity, 1)].max())
    if max_distance < args.dendrogram_cut:
        print(f"  NOTE: all pairwise cosine distances (max {max_distance:.4f}) are "
              f"below the {args.dendrogram_cut} dendrogram cut, so hierarchical "
              f"clustering returns a single cluster.")

    plot_dendrogram(z, names, individuals,
                    os.path.join(out_dir, "dendrogram.png"),
                    f"{args.species} hierarchical clustering ({condition})",
                    cut=args.dendrogram_cut)

    pd.DataFrame({
        "video": names,
        "individual": individuals,
        "source_group": [v["source_group"] for v in videos],
        "kmeans_cluster": kmeans_labels,
        "hierarchical_cluster": hier_labels,
        "mds_1": coords[:, 0],
        "mds_2": coords[:, 1],
    }).to_csv(os.path.join(out_dir, "clustering_results.csv"), index=False)

    # Silhouette on the ORIGINAL cosine distances, not the MDS projection.
    metrics = {
        "species": args.species,
        "condition": condition,
        "n_videos": len(names),
        "n_individuals": n_individuals,
        "feature_dim": int(features.shape[1]),
        "silhouette_kmeans": clu.silhouette_on_distances(dissimilarity, kmeans_labels),
        "silhouette_hierarchical": clu.silhouette_on_distances(dissimilarity, hier_labels),
        "silhouette_ground_truth": clu.silhouette_on_distances(dissimilarity, individuals),
        "shared_mean_fraction": clu.shared_mean_fraction(features),
        "max_pairwise_distance": max_distance,
        "dendrogram_cut": args.dendrogram_cut,
    }
    metrics.update(clu.intra_inter_similarity(similarity, individuals))
    metrics.update({f"kmeans_{k}": v for k, v in
                    clu.cluster_identity_agreement(kmeans_labels, individuals).items()})
    metrics.update({f"hierarchical_{k}": v for k, v in
                    clu.cluster_identity_agreement(hier_labels, individuals).items()})

    with open(os.path.join(out_dir, "metrics.json"), "w") as f:
        # NaN is not valid JSON; emit null so the files parse with strict readers.
        json.dump({k: (None if isinstance(v, float) and np.isnan(v) else v)
                   for k, v in metrics.items()}, f, indent=2)
    return metrics


def main():
    parser = argparse.ArgumentParser(
        description="Wildlife gait analysis pipeline (GaitAnalysis.pdf).")
    parser.add_argument("--species", default="giraffe",
                        help="Species folder under --input-root.")
    parser.add_argument("--input-root", default="input")
    parser.add_argument("--input-dir", default=None,
                        help="Overrides <input-root>/<species>.")
    parser.add_argument("--output-dir", default=None,
                        help="Defaults to output_<species>.")
    parser.add_argument("--individuals", nargs="*", default=None,
                        help="Restrict to these individuals (for subset runs).")
    parser.add_argument("--max-videos", type=int, default=None,
                        help="Max clips per individual.")
    parser.add_argument("--max-frames", type=int, default=None,
                        help="Max frames read per clip.")
    parser.add_argument("--frame-stride", type=int, default=1,
                        help="Keep every Nth frame before segmentation.")
    parser.add_argument("--seg-max-side", type=int, default=None,
                        help="Downscale frames to this longest side for SAM "
                             "(masks are upsampled back). Large CPU speedup.")
    parser.add_argument("--conditions", nargs="*", default=list(CONDITIONS),
                        choices=list(CONDITIONS))
    parser.add_argument("--stream-norm", default="none", choices=["none", "l2"],
                        help="L2-normalise each stream before concatenation. "
                             "Not in the paper; default 'none' matches Sec 2.3.")
    parser.add_argument("--resnet-checkpoint", default=None,
                        help="Fine-tuned ResNet18 backbone (Sec 2.3). "
                             "Omitted = ImageNet weights.")
    parser.add_argument("--dendrogram-cut", type=float, default=clu.DENDROGRAM_CUT)
    parser.add_argument("--seed", type=int, default=clu.RANDOM_SEED)
    parser.add_argument("--device", default=None, choices=[None, "cpu", "cuda"])
    parser.add_argument("--use-cache", action="store_true",
                        help="Reuse embeddings.npz if it matches the clip set.")
    args = parser.parse_args()

    if args.device is None:
        import torch
        args.device = "cuda" if torch.cuda.is_available() else "cpu"
    args.input_dir = args.input_dir or os.path.join(args.input_root, args.species)
    args.output_dir = args.output_dir or f"output_{args.species}"

    if not os.path.isdir(args.input_dir):
        raise SystemExit(f"Input directory not found: {args.input_dir}")

    individuals, videos = discover_videos(args.input_dir, args.individuals,
                                          args.max_videos)
    if not videos:
        raise SystemExit(f"No .mp4 clips found under {args.input_dir}")

    print(f"Species '{args.species}': {len(videos)} clips from "
          f"{len(individuals)} individuals {individuals} (device={args.device})")

    os.makedirs(args.output_dir, exist_ok=True)
    spatial, temporal, kept, diagnostics = build_embeddings(videos, args)
    print(f"Extracted features for {len(kept)}/{len(videos)} clips "
          f"(spatial {spatial.shape}, temporal {temporal.shape})")

    with open(os.path.join(args.output_dir, "extraction_diagnostics.json"), "w") as f:
        json.dump(diagnostics, f, indent=2)

    # Stream norms drive how much each stream influences the fused cosine
    # similarity; report them so an imbalanced fusion is visible.
    print(f"Mean stream norms -- spatial: {np.linalg.norm(spatial, axis=1).mean():.3f}, "
          f"temporal: {np.linalg.norm(temporal, axis=1).mean():.3f} "
          f"(stream_norm={args.stream_norm})")

    summary = []
    for condition in args.conditions:
        print(f"\n--- {condition} ---")
        features = compose_features(spatial, temporal, condition)
        metrics = analyse_condition(features, kept, condition, args)
        summary.append(metrics)
        print(f"  intra-animal similarity: {metrics['intra_mean']:.3f} "
              f"+/- {metrics['intra_std']:.3f} ({metrics['intra_pairs']} pairs)")
        print(f"  inter-animal similarity: {metrics['inter_mean']:.3f} "
              f"+/- {metrics['inter_std']:.3f} ({metrics['inter_pairs']} pairs)")
        print(f"  silhouette (k-means, cosine distances): "
              f"{metrics['silhouette_kmeans']:.3f}")
        print(f"  k-means ARI vs annotated identity: "
              f"{metrics['kmeans_adjusted_rand_index']:.3f}")
        print(f"  shared mean component: {metrics['shared_mean_fraction']:.4f} "
              f"(near 1.0 compresses all similarities towards 1)")

    summary_df = pd.DataFrame(summary)
    summary_path = os.path.join(args.output_dir, "summary.csv")
    summary_df.to_csv(summary_path, index=False)

    # Table 2 layout: species x (intra, inter) similarity.
    table2 = summary_df[["condition", "intra_mean", "intra_std",
                         "inter_mean", "inter_std"]].copy()
    table2["intra_animal_similarity"] = table2.apply(
        lambda r: f"{r['intra_mean']:.3f} +/- {r['intra_std']:.3f}", axis=1)
    table2["inter_animal_similarity"] = table2.apply(
        lambda r: f"{r['inter_mean']:.3f} +/- {r['inter_std']:.3f}", axis=1)
    table2[["condition", "intra_animal_similarity", "inter_animal_similarity"]].to_csv(
        os.path.join(args.output_dir, "table2_similarity.csv"), index=False)

    print(f"\nWrote summary to {summary_path}")
    print("Pipeline finished successfully.")


if __name__ == "__main__":
    main()
