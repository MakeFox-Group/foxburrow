"""Tensor <-> PIL Image conversion helpers."""

import numpy as np
import torch
from PIL import Image


def tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert a [1, 3, H, W] tensor in [-1, 1] range to a PIL RGB image."""
    # Remove batch dim: [3, H, W]
    t = tensor.squeeze(0).clamp(-1, 1)
    # Map [-1, 1] -> [0, 255]
    t = ((t + 1) / 2 * 255).byte()
    # CHW -> HWC
    arr = t.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def tensor_to_pil_rgba(tensor: torch.Tensor, alpha: np.ndarray | None = None) -> Image.Image:
    """Convert a [1, 3, H, W] tensor in [-1, 1] range to a PIL RGBA image."""
    t = tensor.squeeze(0).clamp(-1, 1)
    t = ((t + 1) / 2 * 255).byte()
    arr = t.permute(1, 2, 0).cpu().numpy()
    img = Image.fromarray(arr, "RGB").convert("RGBA")
    if alpha is not None:
        alpha_img = Image.fromarray(alpha, "L")
        if alpha_img.size != img.size:
            alpha_img = alpha_img.resize(img.size, Image.BICUBIC)
        img.putalpha(alpha_img)
    return img


def pil_to_tensor(image: Image.Image, normalize: bool = True) -> torch.Tensor:
    """Convert a PIL image to a [1, 3, H, W] float32 tensor.

    If normalize=True, maps [0,255] -> [-1,1].
    If normalize=False, maps [0,255] -> [0,1].
    """
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)
    # HWC -> CHW
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    if normalize:
        t = t / 127.5 - 1.0
    else:
        t = t / 255.0
    return t


def pil_to_tensor_01(image: Image.Image) -> torch.Tensor:
    """Convert a PIL image to [1, 3, H, W] tensor in [0, 1] range."""
    return pil_to_tensor(image, normalize=False)
