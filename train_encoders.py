"""Supervised fine-tuning of the two encoders on identity labels, with
source-grouped five-fold cross-validation.

    python train_encoders.py                    # every species that can be evaluated
    python train_encoders.py horse polarbear
    python train_encoders.py horse --epochs 30 --folds 5

Runs after extract_features.py and reads its outputs:
    output/images/<clip_id>/rgb_XX.jpg   16 RGB-masked frames per clip   -> ResNet18
    output/images/<clip_id>/sil_XX.png   16 binary silhouettes per clip  -> VideoPrism

Protocol (per species)
    * Objective: cross-entropy on the individual identity, through a temporary
      linear classification head. After training the head is discarded and the
      encoder is used as a feature extractor.
    * Split: every clip of one source recording (manifest column source_id) is in
      the same partition. The recordings of each individual are spread over the
      five folds, so the held-out clips of a fold come from recordings the model
      never saw, of individuals it did see in other recordings. Individuals with a
      single recording are always in the training partition and are never
      evaluated: with one recording, identity cannot be separated from recording.
    * Validation: 10% of the development clips, stratified by individual; the
      checkpoint with the lowest validation loss is kept.
    * Optimiser: AdamW, learning rate 1e-4, weight decay 1e-4, batch of 8 clips,
      30 epochs.
    * ResNet18 (ImageNet weights): conv1, layer1 and layer2 frozen; layer3 and
      layer4 trained. Two random frames per clip per step. Augmentation:
      horizontal flip, random resized crop, rotation up to 10 degrees,
      brightness/contrast jitter. Clip embedding: mean of the 16 frames (512-d).
    * VideoPrism-B: tubelet embedding and the 12 spatial layers frozen; the four
      temporal layers and the final layer norm trained (25% of the parameters).
      Input: the 16 silhouettes in order. Clip embedding: mean of all tokens
      (768-d). No augmentation (silhouettes are already normalised for position,
      scale and facing direction).
    * Gait controls: the fine-tuned VideoPrism is also evaluated on silhouettes
      in shuffled order, and a second VideoPrism is trained and evaluated on
      shuffled silhouettes. If frame order carries identity, both must do worse.

Writes
    output/trained_models/<species>/fold<k>_resnet18.safetensors     encoder weights (no head)
    output/trained_models/<species>/fold<k>_videoprism.safetensors   trained VideoPrism layers (no head)
    output/trained_models/<species>/fold<k>_videoprism_shuffled.safetensors   (control, not tracked by git)
        half precision, each with a .json description; load with load_finetuned()
    results/training/<species>_folds.csv                  clip -> fold, role per fold
    results/training/<species>_log.csv                    loss/accuracy per epoch
    results/training/<species>_summary.csv                held-out classifier accuracy per fold
    results/embeddings_finetuned/<species>.npz            embeddings of every clip from every
                                                          fold model (evaluate.py reads these)
"""
import argparse
import json
import os
import time

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from extract_features import (DEVICE, EMB_DIR, IMAGES, IMNET_MEAN, IMNET_STD, N_SAMPLE, ROOT, SEED,
                              VIDEOPRISM_ID, VIDEOPRISM_REV, VP_SIZE, letterbox, load_manifests)

MODELS = os.path.join(ROOT, "output", "trained_models")     # tracked by git (see .gitignore)
TRAIN = os.path.join(ROOT, "results", "training")
EMB_FT = os.path.join(ROOT, "results", "embeddings_finetuned")
LR, WEIGHT_DECAY, BATCH, EPOCHS, FOLDS, VAL_FRAC = 1e-4, 1e-4, 8, 30, 5, 0.1
RGB_SIZE, FRAMES_PER_STEP = 224, 2


# --------------------------------------------------------------------------
# Saving and loading the fine-tuned weights

def to_half_precision(module):
    """Rounds every floating-point weight to 16 bits, in place. The weights are
    stored in half precision (files below GitHub's 100 MB limit); rounding
    before the embeddings are computed makes the stored file reproduce them."""
    with torch.no_grad():
        for v in module.state_dict().values():
            if v.is_floating_point():
                v.copy_(v.half().float())
    return module


def save_weights(module, path, **info):
    """<path>.safetensors (half precision) and <path>.json (what the file is)."""
    from safetensors.torch import save_file
    state = {k: (v.half() if v.is_floating_point() else v).cpu().contiguous()
             for k, v in module.state_dict().items()}
    save_file(state, path + ".safetensors")
    with open(path + ".json", "w") as fh:
        json.dump(info, fh, indent=2)


def load_finetuned(species, fold, which="resnet18"):
    """The fine-tuned encoder of one fold, ready for inference.
    which: "resnet18" -> ResNetEncoder (uint8 masked RGB frames -> 512-d);
           "videoprism" or "videoprism_shuffled" -> (TemporalEncoder, base model):
           clip embedding = encoder(spatial_features(base, silhouettes))."""
    from safetensors.torch import load_file
    state = {k: (v.float() if v.is_floating_point() else v) for k, v in
             load_file(os.path.join(MODELS, species, f"fold{fold}_{which}.safetensors")).items()}
    if which == "resnet18":
        enc = ResNetEncoder().to(DEVICE)
        enc.net.load_state_dict(state)
        return enc.eval()
    vp = load_videoprism()
    enc = TemporalEncoder(vp).to(DEVICE)
    enc.load_state_dict(state)
    return enc.eval(), vp


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------
def species_table(species):
    """Analysed clips of one species (segmented, not excluded)."""
    man = load_manifests()
    emb = np.load(os.path.join(EMB_DIR, f"{species}.npz"))
    m = man.set_index("clip_id").loc[emb["clip_id"]].reset_index()
    return m[m.exclude == ""].reset_index(drop=True)


def assign_folds(meta, k=FOLDS, seed=SEED):
    """Fold of each clip. The recordings of one individual go to different folds
    (round-robin from a random start); clips of single-recording individuals get
    fold -1 and are never held out."""
    rng = np.random.default_rng(seed)
    fold = np.full(len(meta), -1)
    for _, g in meta.groupby("individual"):
        sources = sorted(g.source_id.unique())
        if len(sources) < 2:
            continue
        order = rng.permutation(len(sources))
        start = rng.integers(k)
        for j, s in enumerate(np.array(sources)[order]):
            fold[(meta.source_id == s).to_numpy()] = (start + j) % k
    return fold


def load_rgb(meta):
    """(N, 16, 3, 256, 256) uint8, letterboxed masked crops."""
    out = np.zeros((len(meta), N_SAMPLE, 256, 256, 3), np.uint8)
    for i, cid in enumerate(meta.clip_id):
        for j in range(N_SAMPLE):
            img = cv2.imread(os.path.join(IMAGES, *cid.split("/"), f"rgb_{j:02d}.jpg"))
            out[i, j] = letterbox(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), 256)
    return torch.from_numpy(out).permute(0, 1, 4, 2, 3).contiguous()


def load_sils(meta):
    """(N, 16, 288, 288) uint8 in {0, 1}."""
    out = np.zeros((len(meta), N_SAMPLE, VP_SIZE, VP_SIZE), np.uint8)
    for i, cid in enumerate(meta.clip_id):
        for j in range(N_SAMPLE):
            s = cv2.imread(os.path.join(IMAGES, *cid.split("/"), f"sil_{j:02d}.png"), 0)
            out[i, j] = s > 127
    return torch.from_numpy(out)


def split_dev(dev_idx, labels, rng):
    """10% of the development clips for validation, stratified by individual."""
    val = []
    for c in np.unique(labels[dev_idx]):
        idx = dev_idx[labels[dev_idx] == c]
        n = int(round(VAL_FRAC * len(idx)))
        if len(idx) >= 4:
            n = max(1, n)
        val += list(rng.choice(idx, n, replace=False))
    val = np.array(sorted(val), int)
    return np.setdiff1d(dev_idx, val), val


# --------------------------------------------------------------------------
# ResNet18 on RGB-masked frames
# --------------------------------------------------------------------------
class ResNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        from torchvision.models import ResNet18_Weights, resnet18
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
        net.fc = nn.Identity()
        self.net = net
        self.frozen = [net.conv1, net.bn1, net.layer1, net.layer2]
        for mod in self.frozen:
            for p in mod.parameters():
                p.requires_grad = False
        self.register_buffer("mean", torch.tensor(IMNET_MEAN).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(IMNET_STD).view(1, 3, 1, 1))

    def train(self, mode=True):
        super().train(mode)
        for mod in self.frozen:           # frozen batch-norm keeps its ImageNet statistics
            mod.eval()
        return self

    def forward(self, frames):            # (B, 3, H, W) uint8
        x = (frames.float() / 255 - self.mean) / self.std
        return self.net(x)


def augment_rgb(x):
    """Flip, random resized crop, rotation <= 10 deg, brightness/contrast. x: (B,3,256,256) uint8."""
    from torchvision.transforms import v2
    B = x.shape[0]
    x = x.float() / 255
    flip = torch.rand(B, device=x.device) < 0.5
    x = torch.where(flip.view(B, 1, 1, 1), x.flip(-1), x)
    out = []
    for img in x:
        img = v2.functional.rotate(img, float(np.random.uniform(-10, 10)))
        i, j, h, w = v2.RandomResizedCrop.get_params(img, scale=(0.7, 1.0), ratio=(0.85, 1.18))
        img = v2.functional.resized_crop(img, i, j, h, w, [RGB_SIZE, RGB_SIZE], antialias=True)
        img = img * np.random.uniform(0.8, 1.2)                               # brightness
        img = (img - img.mean()) * np.random.uniform(0.8, 1.2) + img.mean()   # contrast
        out.append(img.clamp(0, 1))
    return (torch.stack(out) * 255).to(torch.uint8)


def centre_rgb(x):
    return F.interpolate(x.float(), size=RGB_SIZE, mode="bilinear", antialias=True,
                         align_corners=False).round().clamp(0, 255).to(torch.uint8)


@torch.no_grad()
def resnet_embed(enc, rgb, head=None):
    """Clip embeddings (mean over 16 frames) and, with a head, clip logits."""
    enc.eval()
    embs, logits = [], []
    for i in range(0, len(rgb), 16):
        x = rgb[i:i + 16].to(DEVICE)
        B = x.shape[0]
        f = enc(centre_rgb(x.flatten(0, 1))).view(B, N_SAMPLE, -1)
        embs.append(f.mean(1).cpu())
        if head is not None:
            logits.append(head(f).mean(1).cpu())
    return torch.cat(embs).numpy(), (torch.cat(logits) if logits else None)


def train_resnet(rgb, labels, tr, va, n_classes, epochs, rng, log, tag):
    enc, head = ResNetEncoder().to(DEVICE), nn.Linear(512, n_classes).to(DEVICE)
    params = [p for p in enc.parameters() if p.requires_grad] + list(head.parameters())
    opt = torch.optim.AdamW(params, lr=LR, weight_decay=WEIGHT_DECAY)
    y = torch.from_numpy(labels)
    best = (np.inf, None, None)
    for ep in range(epochs):
        enc.train()
        head.train()
        order, losses = rng.permutation(tr), []
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            fr = torch.from_numpy(rng.integers(0, N_SAMPLE, (len(b), FRAMES_PER_STEP)))
            x = rgb[torch.from_numpy(b)[:, None], fr].flatten(0, 1).to(DEVICE)
            yy = y[b].repeat_interleave(FRAMES_PER_STEP).to(DEVICE)
            loss = F.cross_entropy(head(enc(augment_rgb(x))), yy)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        _, lg = resnet_embed(enc, rgb[va], head)
        vloss = F.cross_entropy(lg, y[va]).item()
        vacc = (lg.argmax(1) == y[va]).float().mean().item()
        log.append(dict(**tag, model="resnet18", epoch=ep + 1, train_loss=np.mean(losses),
                        val_loss=vloss, val_acc=vacc))
        if vloss < best[0]:
            best = (vloss, {k: v.detach().cpu().clone() for k, v in enc.net.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    enc.net.load_state_dict(best[1])
    head.load_state_dict(best[2])
    to_half_precision(enc.net)
    return enc, head


# --------------------------------------------------------------------------
# VideoPrism on binary silhouettes
# --------------------------------------------------------------------------
def load_videoprism():
    from transformers import VideoPrismVisionModel
    return VideoPrismVisionModel.from_pretrained(VIDEOPRISM_ID, revision=VIDEOPRISM_REV).to(DEVICE)


@torch.no_grad()
def spatial_features(vp, sils):
    """Output of the frozen spatial encoder for every clip: (N, 16, P, 768) float16.
    Computed once per species, because these layers are not trained."""
    vp.eval()
    out = []
    for i in range(0, len(sils), 4):
        x = sils[i:i + 4].to(DEVICE).float()[:, :, None].repeat(1, 1, 3, 1, 1)   # B,T,C,H,W
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            h = vp.spatial_embeddings(x, False)
            for layer in vp.spatial_layers:
                h = layer(h)
            h = vp.layernorm1(h)
        out.append(h.view(x.shape[0], N_SAMPLE, -1, h.shape[-1]).half().cpu())
    return torch.cat(out)


class TemporalEncoder(nn.Module):
    """The trainable part of VideoPrism: temporal embeddings, four temporal layers
    and the final layer norm, applied to cached spatial features."""

    def __init__(self, vp):
        super().__init__()
        import copy
        self.embed = copy.deepcopy(vp.temporal_embeddings)
        self.layers = copy.deepcopy(vp.temporal_layers)
        self.norm = copy.deepcopy(vp.layernorm2)
        for p in self.parameters():
            p.requires_grad = True

    def forward(self, feats):             # (B, T, P, D) -> clip embedding (B, D)
        B, T, P, D = feats.shape
        shape = (B, T, 3, VP_SIZE, VP_SIZE)
        h = self.embed(feats.reshape(B * T, P, D), shape, False)
        for layer in self.layers:
            h = layer(h)
        return self.norm(h).reshape(B, -1, D).mean(1)


def permute_frames(feats, perms):
    return torch.stack([f[p] for f, p in zip(feats, perms)])


@torch.no_grad()
def vp_embed(enc, feats, head=None, perms=None):
    enc.eval()
    embs, logits = [], []
    for i in range(0, len(feats), 16):
        x = feats[i:i + 16]
        if perms is not None:
            x = permute_frames(x, perms[i:i + 16])
        with torch.autocast("cuda", enabled=DEVICE == "cuda"):
            e = enc(x.to(DEVICE).float())
        embs.append(e.float().cpu())
        if head is not None:
            logits.append(head(e.float()).cpu())
    return torch.cat(embs).numpy(), (torch.cat(logits) if logits else None)


def train_videoprism(vp, feats, labels, tr, va, n_classes, epochs, rng, log, tag, perms=None):
    """perms: fixed frame permutation per clip; given, the model is trained and
    validated on shuffled silhouettes (control)."""
    enc, head = TemporalEncoder(vp).to(DEVICE), nn.Linear(768, n_classes).to(DEVICE)
    opt = torch.optim.AdamW(list(enc.parameters()) + list(head.parameters()), lr=LR,
                            weight_decay=WEIGHT_DECAY)
    scaler = torch.amp.GradScaler(enabled=DEVICE == "cuda")
    y = torch.from_numpy(labels)
    best = (np.inf, None, None)
    name = "videoprism" if perms is None else "videoprism_shuffled"
    for ep in range(epochs):
        enc.train()
        head.train()
        order, losses = rng.permutation(tr), []
        for i in range(0, len(order), BATCH):
            b = order[i:i + BATCH]
            x = feats[b]
            if perms is not None:         # a new order every step
                x = permute_frames(x, [torch.from_numpy(rng.permutation(N_SAMPLE)) for _ in b])
            with torch.autocast("cuda", enabled=DEVICE == "cuda"):
                loss = F.cross_entropy(head(enc(x.to(DEVICE).float()).float()), y[b].to(DEVICE))
            opt.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            losses.append(loss.item())
        _, lg = vp_embed(enc, feats[va], head, None if perms is None else perms[va])
        vloss = F.cross_entropy(lg, y[va]).item()
        vacc = (lg.argmax(1) == y[va]).float().mean().item()
        log.append(dict(**tag, model=name, epoch=ep + 1, train_loss=np.mean(losses),
                        val_loss=vloss, val_acc=vacc))
        if vloss < best[0]:
            best = (vloss, {k: v.detach().cpu().clone() for k, v in enc.state_dict().items()},
                    {k: v.detach().cpu().clone() for k, v in head.state_dict().items()})
    enc.load_state_dict(best[1])
    head.load_state_dict(best[2])
    to_half_precision(enc)
    return enc, head


# --------------------------------------------------------------------------
# Per species
# --------------------------------------------------------------------------
def run_species(species, epochs, k):
    meta = species_table(species)
    fold = assign_folds(meta, k)
    evaluable = meta.individual[fold >= 0].nunique()
    if evaluable < 2:
        print(f"{species}: skipped. {evaluable} individual(s) with two or more recordings; "
              f"a source-grouped split needs at least two.")
        return
    classes = sorted(meta.individual.unique())
    labels = np.array([classes.index(i) for i in meta.individual], np.int64)
    print(f"{species}: {len(meta)} clips, {len(classes)} individuals "
          f"({evaluable} with >= 2 recordings), {meta.source_id.nunique()} recordings")
    for d in (os.path.join(MODELS, species), TRAIN, EMB_FT):
        os.makedirs(d, exist_ok=True)

    rgb, sils = load_rgb(meta), load_sils(meta)
    vp = load_videoprism()
    feats = spatial_features(vp, sils)
    del sils
    perm_rng = np.random.default_rng([SEED, 7])
    perms = torch.stack([torch.from_numpy(perm_rng.permutation(N_SAMPLE)) for _ in range(len(meta))])

    N = len(meta)
    bank = {n: np.zeros((k, N, d), np.float16) for n, d in
            (("resnet_ft", 512), ("vp_ft", 768), ("vp_ft_shuffled", 768), ("vp_ft_trained_shuffled", 768))}
    roles = pd.DataFrame(dict(clip_id=meta.clip_id, individual=meta.individual,
                              source_id=meta.source_id, fold=fold))
    log, summary = [], []
    y = torch.from_numpy(labels)
    for f in range(k):
        t0 = time.time()
        test = np.where(fold == f)[0]
        if not len(test):                 # no recording was assigned to this fold
            continue
        rng = np.random.default_rng([SEED, f])
        tr, va = split_dev(np.where(fold != f)[0], labels, rng)
        role = np.full(N, "train", object)
        role[va], role[test] = "val", "test"
        roles[f"role_fold{f}"] = role
        tag = dict(species=species, fold=f)
        row = dict(**tag, n_train=len(tr), n_val=len(va), n_test=len(test),
                   test_individuals=len(set(labels[test])))

        enc, head = train_resnet(rgb, labels, tr, va, len(classes), epochs, rng, log, tag)
        bank["resnet_ft"][f], lg = resnet_embed(enc, rgb, head)
        row["resnet18_test_acc"] = (lg[test].argmax(1) == y[test]).float().mean().item() if len(test) else np.nan
        save_weights(enc.net, os.path.join(MODELS, species, f"fold{f}_resnet18"),
                     classes=classes, species=species, fold=f, input="16 RGB-masked frames per clip",
                     note="torchvision resnet18 with fc removed; conv1-layer2 unchanged from ImageNet; "
                          "classification head discarded")
        del enc, head

        enc, head = train_videoprism(vp, feats, labels, tr, va, len(classes), epochs, rng, log, tag)
        bank["vp_ft"][f], lg = vp_embed(enc, feats, head)
        bank["vp_ft_shuffled"][f], lgs = vp_embed(enc, feats, head, perms)
        if len(test):
            row["videoprism_test_acc"] = (lg[test].argmax(1) == y[test]).float().mean().item()
            row["videoprism_test_acc_shuffled_input"] = (lgs[test].argmax(1) == y[test]).float().mean().item()
        save_weights(enc, os.path.join(MODELS, species, f"fold{f}_videoprism"),
                     base=f"{VIDEOPRISM_ID}@{VIDEOPRISM_REV}", classes=classes, species=species, fold=f,
                     input="16 binary silhouettes per clip, in order",
                     note="temporal_embeddings (embed), temporal_layers (layers) and layernorm2 (norm) "
                          "of VideoPrismVisionModel; all other weights are those of the base model; "
                          "classification head discarded")
        del enc, head

        enc, head = train_videoprism(vp, feats, labels, tr, va, len(classes), epochs, rng, log, tag, perms)
        bank["vp_ft_trained_shuffled"][f], lg = vp_embed(enc, feats, head, perms)
        if len(test):
            row["videoprism_shuffled_test_acc"] = (lg[test].argmax(1) == y[test]).float().mean().item()
        save_weights(enc, os.path.join(MODELS, species, f"fold{f}_videoprism_shuffled"),
                     base=f"{VIDEOPRISM_ID}@{VIDEOPRISM_REV}", classes=classes, species=species, fold=f,
                     input="16 binary silhouettes per clip, in shuffled order",
                     note="control: trained on silhouettes in shuffled order")
        del enc, head
        row["chance_acc"] = 1 / len(classes)
        row["minutes"] = round((time.time() - t0) / 60, 1)
        summary.append(row)
        print({k_: (round(v, 3) if isinstance(v, float) else v) for k_, v in row.items()}, flush=True)
        pd.DataFrame(log).to_csv(os.path.join(TRAIN, f"{species}_log.csv"), index=False)
        pd.DataFrame(summary).to_csv(os.path.join(TRAIN, f"{species}_summary.csv"), index=False)

    roles.to_csv(os.path.join(TRAIN, f"{species}_folds.csv"), index=False)
    np.savez_compressed(os.path.join(EMB_FT, f"{species}.npz"), clip_id=meta.clip_id.to_numpy(str),
                        fold=fold, **bank)
    with open(os.path.join(TRAIN, "protocol.json"), "w") as fh:
        json.dump(dict(lr=LR, weight_decay=WEIGHT_DECAY, batch_clips=BATCH, epochs=epochs, folds=k,
                       val_fraction=VAL_FRAC, frames_per_clip_per_step=FRAMES_PER_STEP,
                       resnet_trainable="layer3, layer4", videoprism_trainable=
                       "temporal embeddings, 4 temporal layers, final layer norm",
                       videoprism_base=f"{VIDEOPRISM_ID}@{VIDEOPRISM_REV}", seed=SEED), fh, indent=2)
    del vp, feats, rgb
    torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("species", nargs="*")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--folds", type=int, default=FOLDS)
    args = ap.parse_args()
    species = args.species or sorted(f[:-4] for f in os.listdir(EMB_DIR) if f.endswith(".npz"))
    torch.manual_seed(SEED)
    np.random.seed(SEED)
    for sp in species:
        run_species(sp, args.epochs, args.folds)


if __name__ == "__main__":
    main()
