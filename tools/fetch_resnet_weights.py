"""Fetch ResNet18 ImageNet weights into the torch hub cache, with fallbacks."""
import hashlib
import os

URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
HF_URL = ("https://huggingface.co/timm/resnet18.tv_in1k/resolve/main/"
          "pytorch_model.bin")
CACHE = os.path.expanduser("~/.cache/torch/hub/checkpoints")
DEST = os.path.join(CACHE, "resnet18-f37072fd.pth")
os.makedirs(CACHE, exist_ok=True)

if os.path.exists(DEST) and os.path.getsize(DEST) > 1_000_000:
    print("already cached:", DEST, os.path.getsize(DEST))
    raise SystemExit(0)


def try_requests(url, dest):
    import requests
    with requests.get(url, stream=True, timeout=60) as r:
        r.raise_for_status()
        with open(dest, "wb") as f:
            for chunk in r.iter_content(1 << 20):
                f.write(chunk)
    return os.path.getsize(dest)


for name, url in (("download.pytorch.org", URL), ("huggingface", HF_URL)):
    try:
        size = try_requests(url, DEST)
        print(f"downloaded from {name}: {size} bytes")
        break
    except Exception as e:
        print(f"{name} failed: {type(e).__name__}: {e}")
        if os.path.exists(DEST):
            os.remove(DEST)
else:
    raise SystemExit("all sources failed")

import torch
sd = torch.load(DEST, map_location="cpu")
sd = sd.get("state_dict", sd)
print("keys:", len(sd), "sample:", list(sd)[:3])

from torchvision.models import resnet18
net = resnet18()
missing, unexpected = net.load_state_dict(sd, strict=False)
print("missing:", len(missing), "unexpected:", len(unexpected))
assert len(missing) == 0, missing[:5]
print("ResNet18 ImageNet weights OK")
