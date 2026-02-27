"""Background removal handler using RMBG-2.0 (BRIA AI) via PyTorch."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import torch
from PIL import Image

import log

if TYPE_CHECKING:
    from gpu.pool import GpuInstance
    from scheduling.job import InferenceJob

MODEL_W = 1024
MODEL_H = 1024

_model_path: str | None = None


def set_model_path(path: str) -> None:
    global _model_path
    _model_path = path


def load_model(device: torch.device) -> torch.nn.Module:
    """Load the RMBG-2.0 background removal model."""
    if _model_path is None:
        raise RuntimeError("BGRemove model path not configured.")

    try:
        from transformers import AutoModelForImageSegmentation, PreTrainedModel

        # Newer transformers unconditionally uses torch.device("meta") context
        # during from_pretrained, which breaks BiRefNet's SwinTransformer init
        # (it calls torch.linspace().item() which fails on meta tensors).
        # Temporarily patch get_init_context to exclude the meta device.
        _orig_get_init_context = PreTrainedModel.get_init_context

        @classmethod
        def _cpu_instead_of_meta(cls, *args, **kwargs):
            contexts = _orig_get_init_context.__func__(cls, *args, **kwargs)
            return [torch.device("cpu") if (isinstance(c, torch.device) and c.type == "meta") else c
                    for c in contexts]

        PreTrainedModel.get_init_context = _cpu_instead_of_meta
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                _model_path, trust_remote_code=True)
        finally:
            PreTrainedModel.get_init_context = _orig_get_init_context
        model.to(device=device, dtype=torch.float16).eval()
        log.info(f"  BGRemove: Loaded RMBG-2.0 to {device}")
        return model
    except Exception as e:
        log.warning(f"  BGRemove: Failed to load RMBG-2.0: {e}")
        raise


def execute(job: "InferenceJob", gpu: "GpuInstance") -> Image.Image:
    """Pipeline adapter: execute background removal job."""
    if job.input_image is None:
        raise RuntimeError("InputImage is required for background removal.")
    return remove_background(job.input_image, gpu)


def remove_background(source: Image.Image, gpu: "GpuInstance") -> Image.Image:
    """Remove background from a PIL image. Returns RGBA image."""
    # Get model from cache
    model = None
    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == "bgremove":
                model = cached.model
                break

    if model is None:
        raise RuntimeError("BGRemove model not loaded on this GPU.")

    device = gpu.device
    orig_w, orig_h = source.size

    # Resize to model input size
    resized = source.convert("RGB").resize((MODEL_W, MODEL_H), Image.BICUBIC)

    # Convert to tensor [1, 3, H, W] in [0, 1]
    arr = np.array(resized, dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = t.to(device=device, dtype=torch.float16)

    # Normalize using ImageNet mean/std for RMBG-2.0
    mean = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)
    t = (t - mean) / std

    with torch.no_grad():
        result = model(t)

    # Extract alpha mask
    if isinstance(result, (list, tuple)):
        alpha_tensor = result[-1]
    elif hasattr(result, "logits"):
        alpha_tensor = result.logits
    else:
        alpha_tensor = result

    # Sigmoid if needed (outputs may be logits)
    if alpha_tensor.min() < 0 or alpha_tensor.max() > 1:
        alpha_tensor = torch.sigmoid(alpha_tensor)

    # Handle multi-channel output — take first channel
    if alpha_tensor.dim() == 4 and alpha_tensor.shape[1] > 1:
        alpha_tensor = alpha_tensor[:, 0:1, :, :]

    # Convert to numpy alpha map
    alpha = alpha_tensor.squeeze().cpu().float().numpy()
    alpha = np.clip(alpha, 0, 1)
    alpha_u8 = (alpha * 255).astype(np.uint8)

    # Create mask image and resize to original dimensions
    mask = Image.fromarray(alpha_u8, "L")
    mask = mask.resize((orig_w, orig_h), Image.BICUBIC)

    # Apply alpha to original image
    result_img = source.convert("RGBA")
    result_img.putalpha(mask)

    return result_img
