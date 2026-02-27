"""JTP PILOT2 SigLIP tagger — matches the reference HuggingFace Space implementation."""

from __future__ import annotations

import json
import os
import threading
from typing import TYPE_CHECKING

import torch
import torch.nn as nn
from PIL import Image
from torchvision.transforms import transforms, InterpolationMode
import torchvision.transforms.functional as TF

import log
from gpu.pool import fix_meta_tensors, repair_accelerate_leak

if TYPE_CHECKING:
    from gpu.pool import GpuInstance

_model_path: str | None = None
_transform = None

# Per-GPU tagger state: maps torch.device → (model, tags)
_tagger_instances: dict[torch.device, tuple[nn.Module, list[str]]] = {}
_tagger_lock = threading.Lock()
# Separate lock for timm.create_model() which is NOT thread-safe
_model_create_lock = threading.Lock()



# ====================================================================
# Preprocessing (exact match to reference app.py)
# ====================================================================

class _Fit(nn.Module):
    """Resize image to fit within bounds, preserving aspect ratio. No padding."""

    def __init__(self, bounds: tuple[int, int]):
        super().__init__()
        self.bounds = bounds

    def forward(self, img: Image.Image) -> Image.Image:
        wimg, himg = img.size
        hbound, wbound = self.bounds
        hscale = hbound / himg
        wscale = wbound / wimg
        scale = min(hscale, wscale)
        if scale == 1.0:
            return img
        hnew = min(round(himg * scale), hbound)
        wnew = min(round(wimg * scale), wbound)
        return TF.resize(img, (hnew, wnew), InterpolationMode.LANCZOS)


class _CompositeAlpha(nn.Module):
    """Composite alpha channel over a solid background color (on tensor, not PIL)."""

    def __init__(self, background: float):
        super().__init__()
        self.background = torch.tensor([background, background, background]).unsqueeze(1).unsqueeze(2)

    def forward(self, img: torch.Tensor) -> torch.Tensor:
        if img.shape[-3] == 3:
            return img
        alpha = img[..., 3, None, :, :]
        img[..., :3, :, :] *= alpha
        background = self.background.expand(-1, img.shape[-2], img.shape[-1])
        img[..., :3, :, :] += (1.0 - alpha) * background
        return img[..., :3, :, :]


def _build_transform():
    return transforms.Compose([
        _Fit((384, 384)),
        transforms.ToTensor(),
        _CompositeAlpha(0.5),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True),
        transforms.CenterCrop((384, 384)),
    ])


# ====================================================================
# Gated classification head (exact match to reference)
# ====================================================================

class _GatedHead(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.linear = nn.Linear(num_features, num_classes * 2)
        self.act = nn.Sigmoid()
        self.gate = nn.Sigmoid()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.linear(x)
        return self.act(x[:, :self.num_classes]) * self.gate(x[:, self.num_classes:])


# ====================================================================
# Public API
# ====================================================================

def set_model_path(path: str) -> None:
    global _model_path
    _model_path = path




def init_tagger(device: torch.device) -> None:
    """Load tagger model and tags for a specific GPU device.

    Thread-safe: multiple GPUs can load in parallel since the heavy work
    (file I/O, model construction, CUDA transfer) runs outside the lock.
    """
    global _transform

    # Fast check — already loaded?
    if device in _tagger_instances:
        return

    if _model_path is None:
        raise RuntimeError("Tagger model path not configured.")

    model_dir = _model_path

    # Load tags
    tags_file = os.path.join(model_dir, "tags.json")
    if not os.path.isfile(tags_file):
        raise RuntimeError(f"Tags file not found: {tags_file}")

    with open(tags_file) as f:
        tag_dict: dict[str, int] = json.load(f)

    # Build ordered tag list (index -> tag name) matching reference
    allowed_tags = [""] * len(tag_dict)
    for tag_name, idx in tag_dict.items():
        allowed_tags[idx] = tag_name.replace("_", " ")

    # Load model via timm (exact same architecture as reference)
    import timm
    import safetensors.torch

    safetensors_path = os.path.join(model_dir, "model.safetensors")
    if not os.path.isfile(safetensors_path):
        raise RuntimeError(f"Model file not found: {safetensors_path}")

    num_tags = len(tag_dict)
    # timm.create_model() is NOT thread-safe — concurrent calls for the same
    # architecture race on internal model factory state, producing meta-device
    # tensors that crash on .to(). Serialize creation; CUDA transfer stays parallel.
    with _model_create_lock:
        model = timm.create_model(
            "vit_so400m_patch14_siglip_384.webli",
            pretrained=False,
            num_classes=num_tags,
        )
        model.head = _GatedHead(min(model.head.weight.shape), num_tags)

    # timm may leak accelerate's init_empty_weights() context, which
    # monkey-patches nn.Module globally to create meta tensors.  Detect
    # and repair this BEFORE doing anything else with nn.Module.
    repair_accelerate_leak()

    # Fix any meta tensors left by timm/accelerate in the model.
    n = fix_meta_tensors(model)
    if n:
        log.info(f"  Tagger: Fixed {n} meta tensor(s)")

    safetensors.torch.load_model(model, safetensors_path)
    model.to(device=device, dtype=torch.float16).eval()

    # Only hold the lock for the dict write — never during heavy loading.
    with _tagger_lock:
        if device not in _tagger_instances:
            _tagger_instances[device] = (model, allowed_tags)
        if _transform is None:
            _transform = _build_transform()

    log.info(f"  Tagger: Loaded JTP PILOT2 SigLIP ({num_tags} tags) to {device}")


def is_loaded_on(device: torch.device) -> bool:
    """Check if the tagger is loaded on the given device."""
    return device in _tagger_instances


def unload_tagger(device: torch.device) -> None:
    """Unload the tagger from a specific GPU device (called on cache eviction)."""
    with _tagger_lock:
        if device in _tagger_instances:
            del _tagger_instances[device]
            log.info(f"  Tagger: Unloaded from {device}")


def process_image(image: Image.Image, gpu: "GpuInstance",
                  threshold: float = 0.2) -> dict[str, float]:
    """Tag an image using the tagger loaded on the given GPU.
    Returns {tag: score} dict sorted by score descending."""
    device = gpu.device
    instance = _tagger_instances.get(device)
    if instance is None:
        raise RuntimeError(f"Tagger not initialized on {device}. Call init_tagger() first.")
    model, tags = instance

    if _transform is None:
        raise RuntimeError("Tagger transform not initialized.")

    img = image.convert("RGBA")
    tensor = _transform(img).unsqueeze(0).to(device=device, dtype=torch.float16)

    with torch.no_grad():
        probs = model(tensor)[0]

    values, indices = probs.cpu().topk(250)

    matches = {}
    for idx, val in zip(indices, values):
        score = val.item()
        if score >= threshold:
            tag = tags[idx.item()]
            matches[tag] = score

    return dict(sorted(matches.items(), key=lambda x: x[1], reverse=True))
