"""Shared GPU worker utilities — VRAM estimation, slot calculation, working memory.

This module previously contained the GpuWorker class.  That class has been
split into worker_process.py (subprocess entry point) and worker_proxy.py
(main-process proxy).  What remains here are stateless utility functions
imported by both sides:

- Working-memory estimation (_get_min_free_vram, _get_stage_pixels, etc.)
- Slot estimation (estimate_gpu_slots)
- Runtime BPP tracking (_update_working_memory)
"""

from __future__ import annotations

import threading
from typing import TYPE_CHECKING

import log
from scheduling.job import (
    InferenceJob, StageType,
)

if TYPE_CHECKING:
    from scheduling.worker_proxy import GpuProxy, GpuWorkerProxy


# Per-session concurrency limits
_SESSION_MAX_CONCURRENCY: dict[str, int] = {
    "sdxl_unet": 1,
    "sdxl_vae": 1,
    "sdxl_te": 1,
    "upscale": 1,
    "bgremove": 1,
}

MAX_CONCURRENCY = 4

# ---- Working-memory VRAM thresholds (resolution-aware) ----
#
# Three-tier prediction system (checked in priority order):
#
# 1. Workspace profiler (gpu/workspace_profiler.py):
#    Pre-computed at first model load via instrumented forward passes at
#    a grid of resolutions.  Cached to data/profiling/{GPU_MODEL}/*.json.
#    Most accurate — measures actual peak VRAM at each resolution.
#
# 2. Runtime BPP (bytes-per-pixel):
#    Measured during real job execution.  Useful as a cross-check and for
#    stages not yet profiled.  Corrupted by concurrent executions.
#
# 3. Hardcoded fallbacks:
#    Conservative last resort.  Used only before profiling or measurement.

# Map StageType → workspace profiler component_type
_STAGE_TO_PROFILER_COMPONENT: dict[StageType, str] = {
    StageType.GPU_TEXT_ENCODE: "sdxl_te1",
    StageType.GPU_DENOISE: "sdxl_unet",
    StageType.GPU_VAE_DECODE: "sdxl_vae",
    StageType.GPU_VAE_ENCODE: "sdxl_vae_enc",
    StageType.GPU_UPSCALE: "upscale",
    StageType.GPU_BGREMOVE: "bgremove",
}

_VRAM_FALLBACK_BYTES: dict[StageType, int] = {
    # Absolute byte fallbacks — used before first measurement only.
    # Aggressively high on purpose: on 12GB cards with a 5.2GB UNet loaded,
    # these force ensure_free_vram() to evict non-essential cached models
    # (upscale, TEs from previous stages) BEFORE starting.  Once the first
    # successful execution produces a real measurement, these are never used
    # again.  It's better to evict+reload than to OOM and crash the CUDA context.
    StageType.GPU_TEXT_ENCODE: 512 * 1024**2,        # ~512 MB (not resolution-dependent)
    StageType.GPU_DENOISE: 5 * 1024**3,              # ~5 GB (UNet activations + attention)
    StageType.GPU_VAE_DECODE: 4 * 1024**3,           # ~4 GB (cuDNN conv workspace for upsampling)
    StageType.GPU_VAE_ENCODE: 4 * 1024**3,           # ~4 GB (cuDNN conv workspace for downsampling)
    StageType.GPU_HIRES_TRANSFORM: 5 * 1024**3,      # ~5 GB (VAE+upscale+VAE pipeline)
    StageType.GPU_UPSCALE: 2 * 1024**3,              # ~2 GB
    StageType.GPU_BGREMOVE: 1 * 1024**3,             # ~1 GB
}

# Max observed bytes-per-pixel ratio per stage type (populated at runtime).
_measured_bpp: dict[StageType, float] = {}
_bpp_lock = threading.Lock()

# Headroom multiplier applied on top of measured peaks.
_VRAM_HEADROOM = 1.20


def _get_job_resolution(stage_type: StageType, job: InferenceJob) -> tuple[int, int]:
    """Return (width, height) for the profiler lookup.

    For tiled stages (VAE encode/decode, UNet denoise), returns the effective
    tile size rather than the full image size — VRAM usage is bounded by the
    tile, not the total image.
    """
    from handlers.sdxl import (VAE_TILE_THRESHOLD, VAE_TILE_MAX,
                                UNET_TILE_THRESHOLD, UNET_TILE_MAX)

    w, h = 768, 768  # safe default
    if job.sdxl_input:
        w, h = job.sdxl_input.width, job.sdxl_input.height

    if stage_type == StageType.GPU_HIRES_TRANSFORM:
        if job.hires_input and job.hires_input.hires_width > 0:
            return job.hires_input.hires_width, job.hires_input.hires_height

    if job.input_image is not None and stage_type in (
            StageType.GPU_UPSCALE, StageType.GPU_BGREMOVE):
        return job.input_image.width, job.input_image.height

    # VAE encode/decode: tiled when any dimension >= threshold.
    # Working memory is per-tile, so cap to max tile size.
    if stage_type in (StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE):
        img_w, img_h = w, h
        if stage_type == StageType.GPU_VAE_ENCODE and job.input_image is not None:
            img_w, img_h = job.input_image.width, job.input_image.height
        if img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
            w = min(w, VAE_TILE_MAX)
            h = min(h, VAE_TILE_MAX)

    # UNet denoise: tiled (MultiDiffusion) when latent dim > threshold.
    if stage_type == StageType.GPU_DENOISE:
        lat_h, lat_w = h // 8, w // 8
        if lat_h > UNET_TILE_THRESHOLD or lat_w > UNET_TILE_THRESHOLD:
            w = min(w, UNET_TILE_MAX)
            h = min(h, UNET_TILE_MAX)

    return w, h


def _get_stage_pixels(stage_type: StageType, job: InferenceJob) -> int:
    """Return the pixel count that drives working-memory scaling for a stage.

    For tiled stages, returns the tile pixel count (not total image),
    since VRAM usage is bounded by the tile size.
    """
    if stage_type == StageType.GPU_TEXT_ENCODE:
        return 1  # absolute measurement, not per-pixel

    # Use _get_job_resolution which already handles tile capping
    w, h = _get_job_resolution(stage_type, job)

    if stage_type == StageType.GPU_DENOISE:
        return (w // 8) * (h // 8)  # latent dimensions

    if stage_type == StageType.GPU_HIRES_TRANSFORM:
        if job.hires_input and job.hires_input.hires_width > 0:
            return job.hires_input.hires_width * job.hires_input.hires_height
        return w * h

    # VAE decode/encode, upscale, bgremove: pixel area
    if job.input_image is not None and stage_type in (
            StageType.GPU_UPSCALE, StageType.GPU_BGREMOVE):
        return job.input_image.width * job.input_image.height
    return w * h


def _update_working_memory(stage_type: StageType, working_bytes: int,
                           pixels: int) -> None:
    """Record a working-memory measurement as a bytes-per-pixel ratio.

    Uses a "damped max" approach: new peaks are tracked immediately, but
    measurements that are significantly lower than the stored max cause the
    stored value to decay by half each update.  This prevents permanent
    inflation from outlier measurements (e.g. TRT engine deserialization
    captured by peak memory tracking) while still tracking genuine peaks.
    """
    if working_bytes <= 0 or pixels <= 0:
        return
    ratio = working_bytes / pixels
    with _bpp_lock:
        prev_ratio = _measured_bpp.get(stage_type, 0.0)
        if ratio >= prev_ratio:
            _measured_bpp[stage_type] = ratio
            if ratio > prev_ratio:
                log.debug(f"  Working memory: new peak for {stage_type.value}: "
                          f"{working_bytes // (1024**2)}MB @ {pixels} px "
                          f"({ratio:.1f} B/px, prev {prev_ratio:.1f} B/px)")
        elif prev_ratio > 0 and ratio < prev_ratio * 0.5:
            # Measurement drastically lower — decay stored max toward actual.
            new_ratio = max(ratio * 1.5, prev_ratio * 0.5)
            _measured_bpp[stage_type] = new_ratio
            log.debug(f"  Working memory: decaying {stage_type.value}: "
                      f"{working_bytes // (1024**2)}MB @ {pixels} px "
                      f"({new_ratio:.1f} B/px, was {prev_ratio:.1f} B/px)")


def _get_min_free_vram(stage_type: StageType, job: InferenceJob,
                       gpu_model: str | None = None) -> int:
    """Return the minimum free VRAM for a stage, scaled to the job's resolution.

    Priority chain:
    1. Workspace profiler cache (exact measurement at this resolution)
    2. Runtime BPP measurement (measured during execution, scaled by pixels)
    3. Hardcoded fallback (conservative last resort)
    """
    w, h = _get_job_resolution(stage_type, job)

    # --- Tier 1: workspace profiler ---
    if gpu_model is not None:
        from gpu.workspace_profiler import get_working_memory

        comp = _STAGE_TO_PROFILER_COMPONENT.get(stage_type)
        if comp is not None:
            profiled = get_working_memory(comp, gpu_model, w, h)
            # VAE encoder shares the same model file as VAE decoder, so the
            # model registry may store it under "sdxl_vae" instead of "sdxl_vae_enc".
            if profiled is None and comp == "sdxl_vae_enc":
                profiled = get_working_memory("sdxl_vae", gpu_model, w, h)
            # Text encode runs TE1 + TE2 sequentially — sum both
            if comp == "sdxl_te1":
                te2 = get_working_memory("sdxl_te2", gpu_model, w, h)
                if profiled is not None and te2 is not None:
                    profiled = profiled + te2
                elif te2 is not None:
                    profiled = te2
            if profiled is not None:
                return int(profiled * 1.10)

        # Hires transform: max of VAE decode + upscale + VAE encode + UNet
        if stage_type == StageType.GPU_HIRES_TRANSFORM:
            sub_vals = []
            for sub_comp in ("sdxl_unet", "sdxl_vae", "upscale"):
                v = get_working_memory(sub_comp, gpu_model, w, h)
                if v is not None:
                    sub_vals.append(v)
            if sub_vals:
                return int(max(sub_vals) * 1.10)

    # --- Tier 2: runtime BPP ---
    pixels = _get_stage_pixels(stage_type, job)
    with _bpp_lock:
        ratio = _measured_bpp.get(stage_type, 0.0)
    if ratio > 0.0:
        return int(ratio * pixels * _VRAM_HEADROOM)

    # --- Tier 3: hardcoded fallback ---
    return _VRAM_FALLBACK_BYTES.get(stage_type, 0)


# ── Slot estimation (main process, uses proxy data) ──────────────

# Reference resolutions for conservative VRAM estimation.
# Using 1536x1024 (common large-format SDXL size) rather than 1024x1024
# to avoid underestimating working memory for typical high-res jobs.
_REF_DENOISE_LATENT_PX = (1536 // 8) * (1024 // 8)  # 24576 latent pixels
_REF_IMAGE_PX = 1536 * 1024                           # 1572864 image pixels


def _vram_available(gpu: GpuProxy) -> int:
    """Return VRAM available for new allocations (NVML free + allocator slack + evictable).

    Works with GpuProxy's cached vram_stats from StatusSnapshot.
    """
    vram = gpu.get_vram_stats()
    if not vram:
        return 0
    nvml_free = vram.get("free", 0)
    pt_slack = max(0, vram.get("reserved", 0) - vram.get("allocated", 0))
    evictable = gpu.get_evictable_vram()
    return nvml_free + pt_slack + evictable


def _unloaded_model_cost(gpu: GpuProxy, categories: list[str]) -> int:
    """Sum estimated VRAM for model categories not currently loaded on the GPU."""
    from scheduling.model_registry import VramEstimates
    _model_vram = {
        "sdxl_unet": VramEstimates.SDXL_UNET,
        "sdxl_te1": VramEstimates.SDXL_TEXT_ENCODER_1,
        "sdxl_te2": VramEstimates.SDXL_TEXT_ENCODER_2,
        "sdxl_vae": VramEstimates.SDXL_VAE_DECODER,
        "upscale": VramEstimates.UPSCALE,
        "bgremove": VramEstimates.BGREMOVE,
    }
    loaded_cats = set(gpu.get_cached_categories())
    return sum(
        _model_vram.get(cat, 500 * 1024**2)
        for cat in categories
        if cat not in loaded_cats
    )


def _working_memory_cost(stage_type: StageType, ref_pixels: int,
                         gpu_model: str | None = None) -> int:
    """Estimate working memory for slot estimation.

    Uses workspace profiler -> BPP -> hardcoded fallback.
    The reference resolution (ref_pixels) is used for BPP scaling.
    For the workspace profiler, we use a reference resolution of 1536x1024.
    """
    # Tier 1: workspace profiler
    if gpu_model is not None:
        comp = _STAGE_TO_PROFILER_COMPONENT.get(stage_type)
        if comp is not None:
            from gpu.workspace_profiler import get_working_memory
            # Use reference resolution for slot estimation
            profiled = get_working_memory(comp, gpu_model, 1536, 1024)
            if profiled is not None:
                return int(profiled * 1.10)

    # Tier 2: BPP
    with _bpp_lock:
        bpp = _measured_bpp.get(stage_type, 0.0)
    if bpp > 0:
        return int(bpp * ref_pixels * _VRAM_HEADROOM)

    # Tier 3: hardcoded
    return _VRAM_FALLBACK_BYTES.get(stage_type, 1 * 1024**3)


def estimate_gpu_slots(worker: GpuWorkerProxy) -> dict[str, int]:
    """Estimate available job slots on a single GPU, per capability.

    Uses ``check_slot_availability`` for atomic concurrency + group checks,
    then VRAM budget for busy GPUs.  For SDXL, the bottleneck is UNet
    denoise (concurrency=1, ~5GB model + working memory), but total cost
    includes all pipeline components that need loading.

    Returns ``{capability: 0_or_1}`` per GPU.
    """
    gpu = worker.gpu
    if gpu.is_failed:
        return {}

    # GPU is drained for TRT build — no inference capacity
    if worker._draining or worker._building:
        return {}

    # Use cached GPU model name (avoids CUDA driver call on every scheduling cycle)
    gpu_model = worker._gpu_model_name

    slots: dict[str, int] = {}

    # -- SDXL slot --
    if gpu.supports_capability("sdxl"):
        can_accept, active = worker.check_slot_availability("sdxl_unet", "sdxl")
        if not can_accept:
            slots["sdxl"] = 0
        elif active == 0:
            slots["sdxl"] = 1  # idle GPU — _ensure_models_for_stage handles eviction
        else:
            # Busy — check VRAM for UNet + working memory.
            # Also include TE1/TE2/VAE loading cost if not cached, since
            # the full pipeline will need them across stages.
            available = _vram_available(gpu)
            model_cost = _unloaded_model_cost(
                gpu, ["sdxl_unet", "sdxl_te1", "sdxl_te2", "sdxl_vae"])
            working_cost = _working_memory_cost(
                StageType.GPU_DENOISE, _REF_DENOISE_LATENT_PX, gpu_model)
            slots["sdxl"] = 1 if available >= (model_cost + working_cost) else 0

    # -- Simple-task slots --
    for cap, session_key, group, stage_type in [
        ("upscale", "upscale", "upscale", StageType.GPU_UPSCALE),
        ("bgremove", "bgremove", "bgremove", StageType.GPU_BGREMOVE),
    ]:
        if not gpu.supports_capability(cap):
            continue
        can_accept, active = worker.check_slot_availability(session_key, group)
        if not can_accept:
            slots[cap] = 0
        elif active == 0:
            slots[cap] = 1
        else:
            available = _vram_available(gpu)
            model_cost = _unloaded_model_cost(gpu, [session_key])
            working_cost = _working_memory_cost(stage_type, _REF_IMAGE_PX, gpu_model)
            slots[cap] = 1 if available >= (model_cost + working_cost) else 0

    return slots
