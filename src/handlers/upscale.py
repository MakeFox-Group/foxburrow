"""RealESRGAN 2x upscaler handler using PyTorch.

Includes inlined RRDBNet architecture (from Real-ESRGAN/BasicSR) to avoid
external dependencies that don't build on Python 3.14.
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

import log
from gpu.pool import fix_meta_tensors, repair_accelerate_leak

if TYPE_CHECKING:
    from gpu.pool import GpuInstance
    from scheduling.job import InferenceJob

# Module-level model path (set by main.py during init)
_model_path: str | None = None

# Tile size for large images (0 = no tiling)
TILE_SIZE = 512
TILE_PAD = 10


# ====================================================================
# Inlined RRDBNet architecture (matches basicsr.archs.rrdbnet_arch)
# ====================================================================

class ResidualDenseBlock(nn.Module):
    """Residual Dense Block with 5 convolutions."""

    def __init__(self, num_feat: int = 64, num_grow_ch: int = 32):
        super().__init__()
        self.conv1 = nn.Conv2d(num_feat, num_grow_ch, 3, 1, 1)
        self.conv2 = nn.Conv2d(num_feat + num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv3 = nn.Conv2d(num_feat + 2 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv4 = nn.Conv2d(num_feat + 3 * num_grow_ch, num_grow_ch, 3, 1, 1)
        self.conv5 = nn.Conv2d(num_feat + 4 * num_grow_ch, num_feat, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return x5 * 0.2 + x


class RRDB(nn.Module):
    """Residual in Residual Dense Block (3 × RDB)."""

    def __init__(self, num_feat: int, num_grow_ch: int = 32):
        super().__init__()
        self.rdb1 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb2 = ResidualDenseBlock(num_feat, num_grow_ch)
        self.rdb3 = ResidualDenseBlock(num_feat, num_grow_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.rdb1(x)
        out = self.rdb2(out)
        out = self.rdb3(out)
        return out * 0.2 + x


def _pixel_unshuffle(x: torch.Tensor, scale: int) -> torch.Tensor:
    """Rearrange spatial pixels into channel dimension (inverse of pixel shuffle)."""
    b, c, h, w = x.shape
    out_c = c * (scale ** 2)
    out_h = h // scale
    out_w = w // scale
    x = x.view(b, c, out_h, scale, out_w, scale)
    return x.permute(0, 1, 3, 5, 2, 4).reshape(b, out_c, out_h, out_w)


class RRDBNet(nn.Module):
    """RRDBNet architecture for RealESRGAN (matches basicsr exactly)."""

    def __init__(self, num_in_ch: int, num_out_ch: int, scale: int = 4,
                 num_feat: int = 64, num_block: int = 23, num_grow_ch: int = 32):
        super().__init__()
        self.scale = scale

        # For scale=2, pixel unshuffle expands channels 4x and halves spatial
        if scale == 2:
            num_in_ch = num_in_ch * 4
        elif scale == 1:
            num_in_ch = num_in_ch * 16

        self.conv_first = nn.Conv2d(num_in_ch, num_feat, 3, 1, 1)
        self.body = nn.Sequential(
            *[RRDB(num_feat=num_feat, num_grow_ch=num_grow_ch) for _ in range(num_block)]
        )
        self.conv_body = nn.Conv2d(num_feat, num_feat, 3, 1, 1)

        # Upsampling (always 2 layers)
        self.conv_up1 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(num_feat, num_feat, 3, 1, 1)
        self.conv_last = nn.Conv2d(num_feat, num_out_ch, 3, 1, 1)
        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.scale == 2:
            feat = _pixel_unshuffle(x, scale=2)
        elif self.scale == 1:
            feat = _pixel_unshuffle(x, scale=4)
        else:
            feat = x

        feat = self.conv_first(feat)
        body_feat = self.conv_body(self.body(feat))
        feat = feat + body_feat

        feat = self.lrelu(self.conv_up1(F.interpolate(feat, scale_factor=2, mode="nearest")))
        feat = self.lrelu(self.conv_up2(F.interpolate(feat, scale_factor=2, mode="nearest")))
        out = self.conv_last(self.lrelu(self.conv_hr(feat)))
        return out


# ====================================================================
# Public API
# ====================================================================

def set_model_path(path: str) -> None:
    global _model_path
    _model_path = path


def load_model(device: torch.device) -> nn.Module:
    """Load RealESRGAN model weights."""
    if _model_path is None:
        raise RuntimeError("Upscale model path not configured.")

    # Repair any leaked accelerate init_empty_weights() context from concurrent
    # SDXL extractions — otherwise RRDBNet() params land on the meta device.
    repair_accelerate_leak()

    model = RRDBNet(
        num_in_ch=3, num_out_ch=3, num_feat=64,
        num_block=23, num_grow_ch=32, scale=2,
    )

    state = torch.load(_model_path, map_location="cpu", weights_only=True)
    if "params_ema" in state:
        state = state["params_ema"]
    elif "params" in state:
        state = state["params"]

    # Fix any meta tensors left by accelerate's init_empty_weights() leak.
    n = fix_meta_tensors(model)
    if n:
        log.debug(f"  Upscale: Fixed {n} meta tensor(s)")
    model.load_state_dict(state, strict=True)
    model.to(device).half().eval()

    log.debug(f"  Upscale: Loaded RealESRGAN x2plus to {device}")
    return model


def execute(job: "InferenceJob", gpu: "GpuInstance") -> Image.Image:
    """Pipeline adapter: execute upscale job."""
    if job.input_image is None:
        raise RuntimeError("InputImage is required for upscaling.")
    return upscale_image(job.input_image, gpu, job=job)


def upscale_image(image: Image.Image, gpu: "GpuInstance", *,
                   job: "InferenceJob | None" = None) -> Image.Image:
    """2x upscale a PIL image using the cached RealESRGAN model."""
    # Get model from cache
    model = None
    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == "upscale":
                model = cached.model
                break

    if model is None:
        raise RuntimeError("Upscale model not loaded on this GPU.")

    device = gpu.device
    orig_w, orig_h = image.size

    # Pad odd dimensions (replicate edge pixel)
    pad_right = orig_w % 2
    pad_bottom = orig_h % 2

    if pad_right or pad_bottom:
        new_w = orig_w + pad_right
        new_h = orig_h + pad_bottom
        padded = Image.new("RGB", (new_w, new_h))
        padded.paste(image, (0, 0))
        if pad_right:
            edge = image.crop((orig_w - 1, 0, orig_w, orig_h))
            padded.paste(edge, (new_w - 1, 0))
        if pad_bottom:
            edge = image.crop((0, orig_h - 1, orig_w + pad_right, orig_h))
            padded.paste(edge, (0, new_h - 1))
        run_image = padded
    else:
        run_image = image

    # Convert to tensor [1, 3, H, W] in [0, 1]
    arr = np.array(run_image.convert("RGB"), dtype=np.float32) / 255.0
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = t.to(device=device, dtype=torch.float16)

    # Use tiling for large images to avoid OOM
    h, w = t.shape[2], t.shape[3]
    if TILE_SIZE > 0 and (h > TILE_SIZE or w > TILE_SIZE):
        output = _tile_forward(model, t, device, job=job)
    else:
        with torch.no_grad():
            output = model(t)

    # Convert back to PIL
    output = output.squeeze(0).clamp(0, 1)
    output = (output * 255).byte().permute(1, 2, 0).cpu().numpy()
    result = Image.fromarray(output, "RGB")

    # Crop to 2x original size
    target_w = orig_w * 2
    target_h = orig_h * 2
    if result.width != target_w or result.height != target_h:
        result = result.crop((0, 0, target_w, target_h))

    return result


def _tile_forward(model: nn.Module, img: torch.Tensor,
                  device: torch.device, *,
                  job: "InferenceJob | None" = None) -> torch.Tensor:
    """Process image in tiles to manage VRAM for large images."""
    scale = 2
    batch, channel, height, width = img.shape
    output_h = height * scale
    output_w = width * scale
    output = img.new_zeros(batch, channel, output_h, output_w)

    tiles_x = math.ceil(width / TILE_SIZE)
    tiles_y = math.ceil(height / TILE_SIZE)

    if job is not None:
        job.stage_total_steps = tiles_x * tiles_y
        job.stage_step = 0

    for y in range(tiles_y):
        for x in range(tiles_x):
            # Input tile with padding
            ofs_x = x * TILE_SIZE
            ofs_y = y * TILE_SIZE
            input_start_x = max(ofs_x - TILE_PAD, 0)
            input_end_x = min(ofs_x + TILE_SIZE + TILE_PAD, width)
            input_start_y = max(ofs_y - TILE_PAD, 0)
            input_end_y = min(ofs_y + TILE_SIZE + TILE_PAD, height)

            input_tile = img[:, :, input_start_y:input_end_y, input_start_x:input_end_x]

            with torch.no_grad():
                output_tile = model(input_tile)

            # Output tile coordinates
            output_start_x = input_start_x * scale
            output_end_x = input_end_x * scale
            output_start_y = input_start_y * scale
            output_end_y = input_end_y * scale

            # Remove padding from output tile
            output_start_x_tile = (ofs_x - input_start_x) * scale
            output_end_x_tile = output_tile.shape[3] - (input_end_x - min(ofs_x + TILE_SIZE, width)) * scale
            output_start_y_tile = (ofs_y - input_start_y) * scale
            output_end_y_tile = output_tile.shape[2] - (input_end_y - min(ofs_y + TILE_SIZE, height)) * scale

            output[:, :, ofs_y * scale:min((ofs_y + TILE_SIZE) * scale, output_h),
                   ofs_x * scale:min((ofs_x + TILE_SIZE) * scale, output_w)] = \
                output_tile[:, :, output_start_y_tile:output_end_y_tile,
                            output_start_x_tile:output_end_x_tile]

            if job is not None:
                job.stage_step += 1

    return output
