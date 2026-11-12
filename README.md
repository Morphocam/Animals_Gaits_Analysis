# Animals_Gaits_Analysis

Can individual animals be re-identified from **how they move** in ordinary video, separately from
how they look? This repository contains a pipeline that segments the animal, builds
appearance, silhouette, pose and gait representations of each clip, fine-tunes the two encoders
on identity labels, and evaluates everything with **leakage-aware retrieval protocols and gait
controls**. Every number and figure in the manuscript (`paper/manuscript_v6.tex`) is regenerated
by one script. The repository includes the training code, the fine-tuned weights, the clip
manifests, the per-clip embeddings and the openly licensed video clips.

> **Status:** research code accompanying a manuscript in revision. Read
> [What the evaluation shows](#what-the-evaluation-shows) and [Data caveats](#data-caveats)
> before reusing the data.

---

## Quick start

```bash
conda create -n gaits python=3.11 && conda activate gaits
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130   # or CPU wheels
pip install -r requirements.txt

# 1. Index the study clips (input/<species>/<individual>/<clip>.mp4) -> manifests/clips.csv
python tools/build_manifest.py

# 2. Public datasets. The cut clips are already in input/horse and input/polarbear with their
#    manifests; these commands rebuild them from the original downloads (see "Datasets").
python tools/prepare_fm_stallions.py --download && python tools/prepare_fm_stallions.py
python tools/prepare_polarbearvidid.py external/polarbearvidid/data

# 3. Segment, estimate pose and embed every clip in manifests/*.csv (cached per clip; GPU)
python extract_features.py

# 4. Locomotion variables from the pose keypoints -> results/kinematics/pose_gait.csv
python gait_kinematics.py

# 5. Fine-tune ResNet18 (masked RGB) and VideoPrism (binary silhouettes) on identity labels,
#    source-grouped five-fold cross-validation -> output/trained_models/, results/training/
python train_encoders.py horse polarbear hyena

# 6. Every table and figure, from results/embeddings*/ (no video or GPU needed)
python evaluate.py

# Optional: per-clip vectors and QC as CSV for analysis elsewhere -> results/csv/<species>/
python tools/export_csv.py
```

Steps 3 to 6 in one go: `sh tools/run_after_extraction.sh python`.

**Segmentation QC columns** (`results/segmentation_qc.csv`, and `qc.csv` per species):
`n_detections` (> 1: several animals on the prompt frame), `split_frac` (frames whose mask an
occluder split into parts), `border_frac` (mask touches the image edge), `area_cv`, `iou_p05`
(low: tracking jumped), `gap_frac` (unmasked frames inside the track), and for the stride
estimate `stride_autocorr` (near 1: steady periodic gait; ≤ 0: no stride) and `stride_cycles`.

Model weights download on first run from Hugging Face: Grounding DINO
(`IDEA-Research/grounding-dino-tiny`), SAM 2 (`facebook/sam2.1-hiera-base-plus`), MegaDescriptor
(`BVRA/MegaDescriptor-L-384`), ViTPose+ (`usyd-community/vitpose-plus-base`) and VideoPrism;
ResNet18 from torchvision.

> **VideoPrism runs in PyTorch.** The pipeline uses `transformers.VideoPrismVisionModel` with
> the PyTorch conversion `MHRDYN7/videoprism-base-f16r288` (revision pinned in
> `extract_features.py`) of Google's `videoprism_public_v1_base`. JAX is no longer needed.
> `tools/videoprism_torch_check.py` compares it with embeddings from the original JAX model
> (cosine 0.999 on 21 clips). `vendor/videoprism` is kept only for that comparison.

---

## Datasets

All clips used in the study are under `input/`. Clips cut from openly licensed datasets are
tracked by git; third-party footage is not (see [Data caveats](#data-caveats)).

| Folder | Animals | Clips | Source | Licence |
|---|---|---|---|---|
| `input/horse/` | 24 Franches-Montagnes stallions on a treadmill, side view, at four conditions (walk at 1.7 m/s, peak walk, trot at 4.5 m/s, peak trot) | 285 clips of 5 s, 95 recordings (stallion x condition) | Gmel, Maninchedda, Cruz & Weishaupt (2023), *Video clips of Franches-Montagnes stallions at walk and trot on a treadmill during an incremental speed test*, Mendeley Data, V2. <https://doi.org/10.17632/bwfkyvy593.2> | CC BY 4.0 |
| `input/polarbear/` | 13 polar bears in six zoos | 520 tracklets (40 per bear) at the native 12.5 frames per second | Zuerl et al. (2023), *PolarBearVidID*, Zenodo. <https://doi.org/10.5281/zenodo.7564528> | CC BY 4.0 |
| `input/camel/`, `input/giraffe/`, `input/lion/` | spotted hyenas, giraffes, lions (plus one zebra and one elephant, not evaluated) | 171 clips | third-party footage collected by the authors; provenance per clip in `manifests/clips.csv` | not redistributed |

`tools/prepare_fm_stallions.py` documents how the stallion clips are cut: shot boundaries from
frame differences, side views found with Grounding DINO, the condition from the order of the
shots, three 5 s clips per side-view shot. `results/dataset_check/` holds the shot list and a
contact sheet of every side view. The manifests carry `condition` and `gait` (walk or trot) for
the stallions and the zoo and camera for the polar bears.

---

## Pipeline

| Stage | What happens | Model |
|---|---|---|
| Detect | text prompt `"a <species>."` on the first frame where the animal is found | Grounding DINO |
| Track | box prompt, mask propagated forward and backward through the clip | SAM 2 |
| Appearance | RGB pixels inside the mask, 16 frames averaged | MegaDescriptor (main), ResNet18 (comparison) |
| Silhouette | binary masks, one scale per clip, centred, mirrored to walk left→right | VideoPrism, GEI |
| Kinematics | stride frequency, leg-spread CV, body aspect, vertical bob | — |
| Pose | 17 keypoints per masked frame, box from the SAM 2 mask (`extract_pose.py`) | ViTPose+ (animal head) |
| Gait variables | stride frequency, duty factor, step and stride length, speed, limb phase, fore/hind pair phase, limb angular range, head height and bob (`gait_kinematics.py`) | — |
| Fine-tuning | ResNet18 on masked RGB frames and VideoPrism on binary silhouettes, cross-entropy on identity, classification heads discarded (`train_encoders.py`) | ResNet18, VideoPrism |
| Fusion | MegaDescriptor ⊕ VideoPrism, each centred on the species mean and L2-normalised, equal weight (not tuned). The earlier ResNet18 ⊕ VideoPrism fusion is kept as `fused_resnet` | — |

**Controls** computed for every clip:

- *Gait controls*: VideoPrism on shuffled silhouettes, one repeated static silhouette, and the
  mean silhouette. If ordered silhouettes do not beat these, the "gait" stream is using body shape.
- *Leakage control*: ResNet18 on the frame with the animal blacked out. A background-only
  representation should not identify animals unless the protocol rewards matching the scene.

**Retrieval protocols** (`evaluate.py` docstring has the details):

| | Gallery | Measures |
|---|---|---|
| P1 | every other clip of the species | optimistic upper bound (same recording allowed) |
| P2 | excludes the query's own source recording | re-identification in an independent recording |
| P3 | as P2, restricted to the query's site (zoo) | as P2, with no background shortcut between sites |

Metrics: Rank-1, CMC, mAP with chance level, open-set AUROC, ARI/NMI, silhouette coefficient,
ICC repeatability; 95% CIs from a cluster bootstrap over individuals.

**Stride gate.** A clip is *periodic* when `stride_autocorr >= 0.3` (`GAIT_MIN_AUTOCORR` in
`evaluate.py`). The main retrieval tables use every clip. The kinematics table and figure use
periodic clips only, and `table_retrieval_gated.*` repeats retrieval on periodic clips for all
representations. `table_gate_sensitivity.csv` gives mAP for thresholds 0 to 0.5.

**Gait variables.** Each paw is tracked relative to the trunk centre in trunk lengths, so no
calibration is needed and treadmill and panning clips are comparable. A harmonic model at the
stride frequency is fitted to each paw; a clip has a *resolved gait* when the model explains at
least half the variance of every paw and the clip spans at least 1.5 strides. `evaluate.py`
writes `table_gait.*`, `table_gait_repeatability.*`, `table_retrieval_gait.*`, `fig_gait` and
`fig_gait_individuals` from resolved clips.

**Fine-tuning** (`train_encoders.py`; settings in `results/training/protocol.json`). Per
species, both encoders are trained with cross-entropy on the individual identity through a
temporary linear head, which is then discarded. Every clip of a source recording is in the same
partition; the recordings of each individual are spread over five folds, so held-out clips come
from recordings the model never saw. Individuals with one recording are only ever trained on.
10% of the development clips are used for validation and the checkpoint with the lowest
validation loss is kept. AdamW, learning rate 1e-4, weight decay 1e-4, batch of 8 clips, 30
epochs. ResNet18: conv1 to layer2 frozen, layer3 and layer4 trained. VideoPrism: tubelet
embedding and the 12 spatial layers frozen, the four temporal layers and final layer norm
trained. Two gait controls: the fine-tuned VideoPrism is tested on shuffled silhouettes, and a
second VideoPrism is trained and tested on shuffled silhouettes.

The weights are in `output/trained_models/<species>/fold<k>_{resnet18,videoprism}.safetensors`
(half precision, with a `.json` description each). Load one with

```python
from train_encoders import load_finetuned
resnet = load_finetuned("horse", 0)                       # masked RGB frames -> 512-d
temporal, base = load_finetuned("horse", 0, "videoprism") # silhouettes -> 768-d
```

**Silhouette mirroring.** Silhouettes are mirrored to a common walking direction using the
facing direction from the pose (shoulders ahead of hips). An earlier rule (drift of the mask
centroid) mirrored about half the treadmill and panning clips the wrong way;
`results/experiments/mirror_fix_gei.csv` shows the effect.

---

## Data layout and the manifest

```
input/<folder>/<individual>/<clip>.mp4       every clip used (horse and polarbear tracked)
manifests/clips.csv                          one row per study clip, provenance columns (tracked)
manifests/clips_fm_stallions.csv             stallion clips: condition, gait, source (tracked)
manifests/clips_polarbearvidid.csv           polar bear tracklets: zoo, camera (tracked)
output/trained_models/<species>/             fine-tuned encoder weights per fold (tracked)
results/training/                            folds per clip, loss per epoch, protocol.json
results/embeddings_finetuned/<species>.npz   embeddings from every fold model
results/dataset_check/                       shot list and side-view sheet of the stallion videos
external/                                    raw dataset downloads (not tracked)
results/embeddings/<species>.npz             all representations per clip (tracked)
results/segmentation_qc.csv                  per-clip segmentation outcome and mask-quality metrics (tracked)
results/pipeline_params.json                 every setting, model id and library version used
results/csv/<species>/<rep>.csv              per-clip vectors as CSV (tools/export_csv.py), plus qc.csv
results/v1_largest_component/                archived outputs of the earlier mask rule (largest
                                             connected component only), for comparison
results/tables/, paper/figures/fig_*         generated by evaluate.py
results/v2_resnet_fused/                     tables, figures and evaluate.py from before the switch
                                             to MegaDescriptor fusion and the stride gate
results/kinematics/pose_gait.csv             locomotion variables and fit quality per clip
results/pose_check/                          paw traces and keypoint/direction check sheets
results/v3_before_pose_gait/                 tables, figures, embeddings and evaluate.py from before
                                             the pose-based gait analysis
output/pose/<clip>.npz, output/masks/        keypoints and masks per clip (tracked)
results/experiments/                         exploratory comparisons (tools/explore_improvements.py)
results/review/                              mask overlays and with/without comparison for the 12
                                             clips with more than one detection
output/images/<species>/<individual>/<clip>/ rgb_XX.jpg, sil_XX.png, overview.jpg: the 16 masked RGB
                                             crops and silhouettes each encoder sees (not tracked;
                                             written by extract_features.py, or rebuilt from saved
                                             masks with tools/export_images.py)
```

Curated manifest columns survive re-running `build_manifest.py`:

| Column | Meaning |
|---|---|
| `source_id` | original recording the clip was cut from; the unit for P2. Inferred as one recording per (individual, resolution, fps) when not curated, which can only make P2 stricter |
| `visible_species` | overrides the folder name when a clip was filed under the wrong species |
| `exclude` | non-empty (the reason) drops the clip from evaluation |
| `source_type`, `source_ref`, `label_basis`, `notes` | provenance, licence and how the identity was established |

**Adding species:** drop clips into `input/<species>/<individual>/`, run `build_manifest.py`,
then `extract_features.py` (only new clips are processed) and `evaluate.py`. A species
contributes to the leakage-free protocol only if some individuals appear in **two or more
independent recordings** (different days or cameras), recorded in `source_id`.

---

## What the evaluation shows

Full numbers are in `results/tables/`; the manuscript is `paper/manuscript_v6.tex` and the
status of every reviewer comment is in `paper/reviewer_comments_2_status.md`. In short
(922 clips, 47 individuals):

- **Within a recording the scene identifies the animal.** With the query's own recording in the
  gallery (P1), the background-only control scores mAP 0.98 / 0.75 / 0.94 for giraffe / hyena /
  lion, as high as any representation.
- **Frozen appearance models carry identity across recordings; frozen silhouettes do not.**
  Stallions across treadmill conditions: MegaDescriptor mAP 0.72 (Rank-1 0.99), background 0.42,
  VideoPrism silhouettes 0.11. Polar bears across cameras within a zoo (P3): MegaDescriptor +0.11
  mAP over the background [0.06, 0.16], VideoPrism +0.02 [0.00, 0.04].
- **Fine-tuning helps appearance most** (`table_finetuned.*`, held-out recordings): ResNet18
  mAP 1.00 in stallions (frozen 0.49) and 0.52 in polar bears (frozen 0.19, background 0.14).
- **The silhouette encoder does not use gait dynamics, frozen or fine-tuned.** Shuffling frames
  changes the frozen VideoPrism by at most 0.02 mAP. Fine-tuned VideoPrism reaches 0.33
  (stallions) and 0.27 (polar bears), but a copy trained on shuffled frames reaches 0.34 and 0.33.
- **Measured gait is biologically consistent and repeatable.** Stallions: walk with limb phase
  0.22 and duty factor 0.58, trot with limb phase 0.46 to 0.49; stride frequency 0.90, 0.95, 1.51
  and 1.59 Hz over the four belt speeds (`table_gait_conditions.*`). Stride frequency
  repeatability: 0.74 (stallions at walk), 0.90 (lions), 0.68 (polar bears).
- **Gait variables identify stallions across bouts, well below appearance**
  (`table_gait_cross_condition.*`): mAP 0.50 between the two walking speeds and 0.52 between the
  two trotting speeds (chance 0.04), against 0.99 for MegaDescriptor. The background reaches
  0.91 and 0.74 there, because each stallion was filmed in its own session.
- **Gait is rarely measurable in zoo footage.** A gait is resolved in 282/285 stallion clips,
  56/64 lion clips, 16/27 hyena clips, 45/520 polar bear tracklets and 1/26 giraffe clips.

`fig_heatmaps` (clip x clip similarity of the fused embedding) is generated as a supplementary
figure.

## Data caveats

An audit of `input/` (contact sheet of every individual) found that the folder `camel/` contains
spotted-hyena footage, `giraffe/C` a zebra and `giraffe/E` an elephant. These are corrected in
`manifests/clips.csv` via `visible_species`; species with a single individual are dropped from
evaluation. Herd footage in which the focal individual cannot be identified (`giraffe/D`,
`giraffe/H`, `lion/E`) is excluded via `exclude`. Some third-party clips carry stock-footage
watermarks and must not be redistributed.

---

## Legacy code

`run_pipeline.py`, `src/`, `gaits_clusters.py` and `CSVs/` produced the similarity matrices in
earlier versions of the manuscript (SAM 1, frozen encoders, unnormalised fusion, average-linkage
cut at 0.15). They are kept for the record only. The stored `CSVs/similarity_*_v3.csv` matrices
are asymmetric, not positive semi-definite, and do not reproduce the earlier Table 2 for three
of five species (`tools/verify_table2.py`); they should not be used for new analyses.

## Citation

Please cite the accompanying manuscript. The pipeline builds on
[Grounding DINO](https://github.com/IDEA-Research/GroundingDINO),
[SAM 2](https://github.com/facebookresearch/sam2),
[MegaDescriptor](https://huggingface.co/BVRA/MegaDescriptor-L-384),
[ResNet](https://arxiv.org/abs/1512.03385) and
[VideoPrism](https://github.com/google-deepmind/videoprism).

## License

MIT — see [LICENSE](LICENSE). Vendored VideoPrism code retains its Apache 2.0 license.
