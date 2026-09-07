# Animals_Gaits_Analysis

A video-based pipeline for **identifying individual animals from how they move**.

Animals are segmented out of natural footage, encoded through complementary appearance and motion
streams, and grouped by cosine similarity — no collars, tags, or physical handling required.

The method is described in `GaitAnalysis.pdf`.

> **Status:** research code. Works end to end, but see [Limitations](#limitations) before relying
> on the output.

---

## Quick start

```bash
conda create -n gaits python=3.11
conda activate gaits
pip install -r requirements.txt
pip install --no-deps -e vendor/videoprism    # VideoPrism is vendored

# Verify the clustering maths (fast, no models or data needed)
python tools/test_clustering.py

# Run one species
python run_pipeline.py --species giraffe
```

Model weights download automatically on first run: SAM via `src/utils.py`, ResNet18 via
torchvision, VideoPrism from Hugging Face.

> ⚠️ The editable install records an **absolute path**. If you move or rename this directory,
> re-run `pip install --no-deps -e vendor/videoprism` or `import videoprism` will break.

---

## How it works

| Stage | Operation | Model |
|---|---|---|
| 1. Segmentation | `B = S(I; θ_S, p)` — prompt once, propagate through the clip | SAM |
| 2. Dual masks | `R_t(x) = B_t(x) ⊙ I_t(x)` (appearance) and `B_t` (texture-free silhouette) | — |
| 3. Spatial stream | `s_t = f_R(R_t; θ_R)` on RGB-masked frames → 512-d | ResNet18 |
| 4. Temporal stream | `v_t = f_V(B_{t−τ+1:t}; θ_V)` on ordered silhouettes, 16 frames @ 5 fps → 768-d | VideoPrism |
| 5. Fusion | `h̄ = mean_t concat(s_t, v_t)` → 1280-d per clip | — |
| 6. Similarity | `A_ij = ⟨h̄_i, h̄_j⟩ / (‖h̄_i‖‖h̄_j‖)`, `d_ij = 1 − A_ij` | — |
| 7. Clustering | classical MDS → k-means; average-linkage dendrogram; silhouette score | — |

The two streams deliberately receive **different inputs**: ResNet18 sees colour and coat texture,
while VideoPrism sees only the silhouette outline changing over time. Clustering runs **within a
single species** — species classification is not part of this pipeline.

---

## Data layout

Clips are organised by species, then by individual. The folder name is the identity label and
defines the expected clusters.

```
input/
  camel/
    A/  A_1.mp4  A_2.mp4  ...
    B/  B_1.mp4  ...
  giraffe/
  lion/
```

Each clip should show a single animal walking laterally across the frame.

`input/` and `models/` are **not tracked by git** — see `.gitignore`.

---

## Usage

```bash
# Full run for one species
python run_pipeline.py --species giraffe

# Quick subset while iterating
python run_pipeline.py --species camel --individuals A B C --max-videos 2 \
                       --max-frames 48 --frame-stride 2 --seg-max-side 512

# Re-run clustering only, reusing cached embeddings
python run_pipeline.py --species camel --use-cache

# Fused representation only, skipping the ablations
python run_pipeline.py --species lion --conditions fused
```

### Options

| Flag | Purpose |
|---|---|
| `--species` | Species folder under `--input-root` |
| `--input-dir`, `--output-dir` | Override the default paths |
| `--individuals`, `--max-videos` | Restrict the clip set for subset runs |
| `--max-frames`, `--frame-stride` | Limit frames read / kept per clip |
| `--seg-max-side` | Segment at reduced resolution — large CPU speedup |
| `--conditions` | Any of `fused`, `spatial_only`, `temporal_only` |
| `--use-cache` | Reuse `embeddings.npz` instead of re-extracting |
| `--resnet-checkpoint` | Load a fine-tuned ResNet18 backbone |
| `--stream-norm` | `none` (default) or `l2` — see [Limitations](#limitations) |
| `--dendrogram-cut` | Linkage cut distance (default `0.15`) |
| `--device`, `--seed` | Force cpu/cuda; set the random seed (default `42`) |

### Outputs

Written to `output_<species>/<condition>/`:

| File | Contents |
|---|---|
| `similarity.csv` | Pairwise cosine similarity matrix |
| `similarity_heatmap.png` | Heatmap ordered by individual, with block boundaries |
| `mds_kmeans.png` | k-means assignments on classical-MDS coordinates |
| `dendrogram.png` | Average-linkage dendrogram with the cut threshold drawn |
| `clustering_results.csv` | Per-clip cluster assignments and MDS coordinates |
| `metrics.json` | Silhouette scores, intra/inter similarity, ARI, NMI, diagnostics |

At the species level: `summary.csv`, `table2_similarity.csv` (intra/inter similarity ± sd),
`extraction_diagnostics.json` (per-clip mask coverage and frame counts), and `embeddings.npz`
(cached embeddings for `--use-cache`).

**Check `extraction_diagnostics.json` first.** Low `mask_coverage` means segmentation lost the
animal, and those clips will cluster poorly regardless of anything downstream.

---

## Performance

Segmentation dominates runtime — SAM's image encoder runs once per frame, at roughly **5 s/frame on
CPU**. A GPU is strongly recommended for full runs.

If you are CPU-bound, these cut the cost significantly:

- `--seg-max-side 512` — segment downscaled frames; masks are upsampled back to full resolution
- `--frame-stride 2` — keep every Nth frame
- `--max-frames 48` — cap frames read per clip
- `--use-cache` — never re-extract just to re-cluster

---

## Repository layout

```
run_pipeline.py            main pipeline (segmentation → features → clustering)
src/
  feature_extractor.py     segmentation + spatial/temporal encoders
  clustering.py            cosine similarity, classical MDS, k-means, dendrogram, silhouette
  utils.py                 SAM checkpoint download
tools/
  test_clustering.py       synthetic checks of the clustering maths (fast, no models)
  test_extraction.py       single-clip smoke test with timings
  diagnose_streams.py      reports how much of each embedding is a shared constant offset
  fetch_resnet_weights.py  ResNet18 weight download with a mirror fallback
  verify_table2.py         recompute intra/inter similarity stats from CSVs/
CSVs/                      per-species similarity matrices (camel, giraffe, hyena, lion, zebra)
gaits_clusters.py          standalone figure script for the CSVs
vendor/videoprism/         vendored VideoPrism (google-deepmind/videoprism)
GaitAnalysis.pdf           method write-up
```

### `CSVs/` and `gaits_clusters.py`

`CSVs/` holds per-species cosine-similarity matrices, each row labelled with its annotated
individual. Two quirks when parsing them: the zebra file carries a UTF-8 BOM, and the matrices are
stored **asymmetrically** (`A_1→A_2` need not equal `A_2→A_1`), so average the two triangles before
computing statistics. `tools/verify_table2.py` does this correctly and is the better starting point
for new analysis code.

`gaits_clusters.py` renders dendrograms, MDS plots, and heatmaps from those matrices:

```bash
pip install seaborn      # not needed by the main pipeline
mkdir vis_output
python gaits_clusters.py
```

It uses Ward linkage, whereas `run_pipeline.py` uses average linkage on cosine distance. Edit the
`similarity_*_v3.csv` filename near the top to switch species.

---

## Limitations

**Encoders run frozen.** The method fine-tunes both encoders on identity labels and then discards
the classification heads. That training stage is **not implemented here** — `run_pipeline.py` uses
pretrained ImageNet and VideoPrism weights as-is.

This has a visible effect on output. With frozen weights ~99.9% of each VideoPrism embedding is a
shared constant direction, so raw cosine similarities compress towards 1 and the temporal stream
separates individuals by only ~0.002. Mean-centre the same embeddings and separation jumps to ~0.84
— the signal is there, masked by an offset that fine-tuning is what removes. `metrics.json` reports
`shared_mean_fraction` (near 1.0 means heavy compression) and `tools/diagnose_streams.py` breaks it
down per stream. Pass `--resnet-checkpoint` to load adapted spatial weights; there is no equivalent
hook for VideoPrism yet.

**Segmentation uses SAM 1, not SAM3.** SAM 1 has no native video tracking, so `SamVideoSegmenter`
approximates prompt-once-then-propagate by seeding each frame with the previous frame's mask box and
centroid. It sits behind an interface, so a SAM3 predictor can be substituted without touching
anything downstream.

**The 0.15 dendrogram cut assumes fine-tuned embeddings.** With frozen encoders every pairwise
distance can fall below it, collapsing the dendrogram to a single cluster. The pipeline prints an
explicit warning when this happens; adjust with `--dendrogram-cut`.

**Fusion is an unnormalised concatenation.** In practice the spatial stream carries a noticeably
larger norm than the temporal one, so it dominates the fused cosine similarity. `--stream-norm l2`
normalises each stream first.

**Input assumptions.** Clips are expected to show a single animal walking laterally with adequate
duration and resolution. Frontal and rear views, infrared footage, occlusion, multiple animals, and
interrupted tracks are not handled.

---

## Citation

If you use this code, please cite the accompanying write-up (`GaitAnalysis.pdf`). The pipeline
builds on [SAM](https://github.com/facebookresearch/segment-anything),
[ResNet](https://arxiv.org/abs/1512.03385), and
[VideoPrism](https://github.com/google-deepmind/videoprism).

## License

MIT — see [LICENSE](LICENSE). Vendored VideoPrism code retains its original Apache 2.0 license.
