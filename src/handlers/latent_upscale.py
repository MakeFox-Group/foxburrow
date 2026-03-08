"""Latent-space upscaler: Lanczos-3 + Iterative Back-Projection (IBP).

Upscales cached denoised latents directly in latent space, skipping
RealESRGAN and VAE encode entirely. Pure tensor math — no model weights.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import log

if TYPE_CHECKING:
    from gpu.pool import GpuInstance
    from scheduling.job import InferenceJob

_IBP_ITERATIONS = 0
_IBP_ALPHA = 0.4


def _lanczos_resample_dim(tensor: torch.Tensor, dim: int, target_size: int, a: int = 3) -> torch.Tensor:
    """Resample a tensor along one spatial dimension using a Lanczos-a kernel.

    Args:
        tensor: [B, C, H, W] tensor
        dim: spatial dimension to resample (2=height, 3=width)
        target_size: desired output size along that dimension
        a: Lanczos kernel radius (3 for Lanczos-3)

    Returns:
        Resampled tensor with the specified dimension changed to target_size.
    """
    src_size = tensor.shape[dim]
    if src_size == target_size:
        return tensor

    device = tensor.device
    dtype = tensor.dtype

    # Work in float32 for precision
    tensor = tensor.float()

    ratio = src_size / target_size

    # For each output position, compute the fractional source coordinate
    # Center-aligned mapping: out_pos maps to (out_pos + 0.5) * ratio - 0.5
    out_pos = torch.arange(target_size, device=device, dtype=torch.float32)
    src_center = (out_pos + 0.5) * ratio - 0.5  # [target_size]

    # Symmetric kernel window: 2a+1 taps centered on floor(src_center)
    kernel_offsets = torch.arange(-a, a + 1, device=device, dtype=torch.float32)  # [2a+1]

    # Integer source positions: floor(src_center) + offsets
    src_idx_float = src_center.unsqueeze(1).floor() + kernel_offsets.unsqueeze(0)  # [target_size, 2a+1]

    # Fractional distances from src_center to each integer source position
    x = src_idx_float - src_center.unsqueeze(1)  # [target_size, 2a+1]

    # Lanczos kernel: sinc(x) * sinc(x/a) for |x| < a, else 0
    # Use clamped x to avoid NaN from sin(0)/0, then fix x≈0 with where()
    abs_x = x.abs()
    safe_x = abs_x.clamp(min=1e-7)
    pi_safe_x = torch.pi * safe_x
    pi_safe_x_over_a = pi_safe_x / a

    sinc_val = torch.sin(pi_safe_x) / pi_safe_x
    sinc_a_val = torch.sin(pi_safe_x_over_a) / pi_safe_x_over_a
    raw_weights = sinc_val * sinc_a_val

    # x=0 → weight=1, |x|>=a → weight=0
    weights = torch.where(abs_x < 1e-7, torch.ones_like(x), raw_weights)
    weights = torch.where(abs_x >= a, torch.zeros_like(x), weights)

    # Normalize weights per output position
    weights = weights / weights.sum(dim=1, keepdim=True)

    # Clamp source indices to valid range (src_idx_float is already integer-valued)
    src_idx = src_idx_float.long().clamp(0, src_size - 1)  # [target_size, 2a+1]

    # Gather and weighted sum
    if dim == 2:
        # Resample height: tensor is [B, C, H, W]
        gathered = tensor[:, :, src_idx, :]  # [B, C, target_size, 2a+1, W]
        w = weights.unsqueeze(0).unsqueeze(0).unsqueeze(-1)  # [1, 1, target_size, 2a+1, 1]
        result = (gathered * w).sum(dim=3)  # [B, C, target_size, W]
    else:
        # Resample width: tensor is [B, C, H, W]
        gathered = tensor[:, :, :, src_idx]  # [B, C, H, target_size, 2a+1]
        w = weights.unsqueeze(0).unsqueeze(0).unsqueeze(0)  # [1, 1, 1, target_size, 2a+1]
        result = (gathered * w).sum(dim=4)  # [B, C, H, target_size]

    return result.to(dtype)


def _lanczos_resize_2d(tensor: torch.Tensor, target_h: int, target_w: int, a: int = 3) -> torch.Tensor:
    """Separable Lanczos-a resize: horizontal pass then vertical pass.

    Args:
        tensor: [B, C, H, W] tensor
        target_h: desired output height
        target_w: desired output width
        a: Lanczos kernel radius

    Returns:
        Resized tensor [B, C, target_h, target_w].
    """
    result = _lanczos_resample_dim(tensor, dim=3, target_size=target_w, a=a)
    result = _lanczos_resample_dim(result, dim=2, target_size=target_h, a=a)
    return result


def lanczos_ibp_upscale(
    latents: torch.Tensor,
    target_h: int,
    target_w: int,
    a: int = 3,
    ibp_iterations: int = _IBP_ITERATIONS,
    ibp_alpha: float = _IBP_ALPHA,
) -> torch.Tensor:
    """Upscale latents using Lanczos-3 + Iterative Back-Projection.

    1. Initial Lanczos-3 upscale to target size
    2. IBP refinement: downscale estimate, compute residual vs original,
       upscale residual, add weighted correction. Repeat N times.

    Args:
        latents: [B, C, H, W] source latent tensor
        target_h: target latent height
        target_w: target latent width
        a: Lanczos kernel radius
        ibp_iterations: number of IBP refinement iterations
        ibp_alpha: weight for residual correction

    Returns:
        Upscaled latent tensor [B, C, target_h, target_w].
    """
    src_h, src_w = latents.shape[2], latents.shape[3]

    # Initial Lanczos upscale
    estimate = _lanczos_resize_2d(latents, target_h, target_w, a=a)

    # IBP refinement loop
    for _ in range(ibp_iterations):
        # Downscale estimate back to original size
        downscaled = _lanczos_resize_2d(estimate, src_h, src_w, a=a)
        # Residual: what's missing from the original
        residual = latents - downscaled
        # Upscale residual
        upscaled_residual = _lanczos_resize_2d(residual, target_h, target_w, a=a)
        # Correct estimate
        estimate = estimate + ibp_alpha * upscaled_residual

    return estimate


# Available upscale methods
_UPSCALE_METHOD = "bicubic"  # "bilinear", "bicubic", or "lanczos"


def execute(job: "InferenceJob", gpu: "GpuInstance") -> None:
    """Pipeline stage adapter: upscale latents in latent space.

    Reads job.latents and job.hires_input for target dimensions.
    Target latent dims are hires_width // 8, hires_height // 8.
    Sets job.is_hires_pass = True after upscaling.
    """
    if job.latents is None:
        raise RuntimeError("Latents are required for latent upscale stage.")
    if job.hires_input is None:
        raise RuntimeError("hires_input is required for latent upscale stage.")

    target_h = job.hires_input.hires_height // 8
    target_w = job.hires_input.hires_width // 8
    src_h, src_w = job.latents.shape[2], job.latents.shape[3]

    with torch.no_grad():
        if _UPSCALE_METHOD == "lanczos":
            log.debug(f"  Latent upscale: {src_h}x{src_w} → {target_h}x{target_w} "
                      f"(Lanczos-3 + {_IBP_ITERATIONS}-iter IBP α={_IBP_ALPHA})")
            job.latents = lanczos_ibp_upscale(job.latents, target_h, target_w)
        else:
            log.debug(f"  Latent upscale: {src_h}x{src_w} → {target_h}x{target_w} "
                      f"({_UPSCALE_METHOD})")
            job.latents = torch.nn.functional.interpolate(
                job.latents.float(),
                size=(target_h, target_w),
                mode=_UPSCALE_METHOD,
                align_corners=False if _UPSCALE_METHOD != "nearest" else None,
            ).to(job.latents.dtype)

    job.is_hires_pass = True
