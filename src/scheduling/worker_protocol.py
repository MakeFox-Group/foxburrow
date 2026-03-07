"""Cross-process message types for GPU worker IPC.

Both the main process and GPU worker subprocess import this module.
All types must be pickle-safe (no asyncio objects, no CUDA tensors).
Tensors crossing the boundary must be on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PIL import Image
import torch

from scheduling.job import (
    StageType,
    WorkStage,
    SdxlJobInput,
    SdxlHiresInput,
    SdxlTokenizeResult,
    ModelComponentId,
)


# ── Commands (main → worker) ─────────────────────────────────────

@dataclass
class ExecuteJobCmd:
    """Execute an entire job pipeline on the GPU (all stages, start to finish)."""
    job_id: str
    job_type_value: str                          # JobType value
    pipeline: list[WorkStage]                    # Full pipeline stages

    # Job input data (immutable)
    sdxl_input: SdxlJobInput | None
    hires_input: SdxlHiresInput | None
    input_image: Image.Image | None              # For enhance/img2img
    input_latents: torch.Tensor | None           # CPU tensor for decode/hires latent jobs

    # CPU tokenization results (already completed in main process)
    tokenize_result: SdxlTokenizeResult | None
    regional_tokenize_results: list[SdxlTokenizeResult] | None
    regional_base_tokenize: SdxlTokenizeResult | None
    regional_shared_neg_tokenize: SdxlTokenizeResult | None

    # Regional prompting info (serializable parts only)
    regional_info_data: dict | None

    # Job state
    priority: int
    oom_retries: int

    # Hires pass flag (set by routes for SdxlHiresLatents / SdxlGenerateHires)
    is_hires_pass: bool = False

    # Tiling overrides
    unet_tile_width: int = 0
    unet_tile_height: int = 0
    vae_tile_width: int = 0
    vae_tile_height: int = 0

    # Original dimensions for final resize
    orig_width: int | None = None
    orig_height: int | None = None


@dataclass
class DrainCmd:
    """Drain the GPU: stop accepting work, evict all models, prepare for TRT build."""
    pass


@dataclass
class ReleaseDrainCmd:
    """Release drain state, resume normal operation."""
    pass


@dataclass
class TrtBuildCmd:
    """Build TRT engines for a model on this GPU."""
    model_hash: str
    model_dir: str
    cache_dir: str
    arch_key: str
    max_workspace_gb: float = 0  # 0 = auto (total_vram - 1GB)
    dynamic_only: bool = True    # True = skip static engines


@dataclass
class OnloadCmd:
    """Pre-load models at startup."""
    types: set[str]
    onload_entries: set[str]
    unevictable_entries: set[str]
    models_dir: str
    sdxl_models: dict[str, str]
    lora_index: dict | None = None   # name -> LoraEntry (initial population)
    loras_dir: str | None = None


@dataclass
class UpdateLoraIndexCmd:
    """Replace the worker's LoRA index after a rescan in the main process."""
    lora_index: dict  # name -> LoraEntry


@dataclass
class UpdateSdxlModelsCmd:
    """Replace the worker's SDXL model map after a rescan in the main process."""
    sdxl_models: dict[str, str]  # name -> path


@dataclass
class TagImageCmd:
    """Tag an image using the tagger model."""
    image: Image.Image
    threshold: float = 0.2


@dataclass
class ShutdownCmd:
    """Gracefully shut down the worker process."""
    pass


@dataclass
class GetStatusCmd:
    """Request a fresh StatusSnapshot (worker always sends one after commands,
    but this forces an explicit one)."""
    pass


# ── Responses (worker → main) ────────────────────────────────────

@dataclass
class ClipCacheEntry:
    """Single CLIP encoder output for caching. Pickle-safe (raw bytes, no torch)."""
    encoder_type: str       # "clip_l", "clip_g", "clip_g_pooled"
    polarity: str           # "positive" or "negative"
    data: bytes             # raw tensor bytes (contiguous, CPU, fp16)
    dtype: str              # "fp16", "bf16", "fp32"
    dim0: int
    dim1: int | None


@dataclass
class JobComplete:
    """Result of executing an entire job pipeline."""
    job_id: str
    success: bool
    error: str | None = None
    oom: bool = False                   # True = re-route to different GPU
    fatal: bool = False                 # True = GPU context corrupted

    # Output data
    output_image: Image.Image | None = None
    output_latents: torch.Tensor | None = None  # CPU tensor

    # Timing
    gpu_time_s: float = 0.0
    model_load_time_s: float = 0.0
    stage_times: list[dict] = field(default_factory=list)  # Per-stage timing breakdown

    # CLIP embedding cache data (serialized tensors for DB storage)
    clip_cache: dict | None = None
    # {"prompt_hash": bytes(32), "model": str, "entries": list[ClipCacheEntry]}


@dataclass
class StatusSnapshot:
    """Lightweight GPU status for the proxy to cache."""
    cached_fingerprints: set[str]
    cached_categories: list[str]
    cached_models_info: list[dict]
    session_group: str | None
    vram_stats: dict
    loaded_models_vram: int
    evictable_vram: int
    trt_freeable_vram: int
    is_failed: bool
    fail_reason: str
    is_busy: bool
    trt_shared_memory_vram: int
    loaded_lora_count: int
    gpu_model_name: str
    arch_key: str

    # Per-fingerprint measured VRAM (actual bytes for each cached model).
    # Lets the main process use real model sizes instead of registry estimates.
    fingerprint_vram: dict[str, int] = field(default_factory=dict)

    # Runtime bytes-per-pixel measurements from the worker's stage executions.
    # Keys are StageType.value strings, values are max observed BPP ratios.
    # Propagated to the main process so _get_min_free_vram() uses real data.
    measured_bpp: dict[str, float] = field(default_factory=dict)


@dataclass
class ProgressUpdate:
    """Denoise progress callback from worker."""
    job_id: str
    denoise_step: int
    denoise_total_steps: int
    stage_step: int = 0
    stage_total_steps: int = 0
    stage_index: int = 0


@dataclass
class DrainComplete:
    """Worker has drained all active work and evicted models."""
    pass


@dataclass
class TrtBuildProgress:
    """Progress update during TRT engine building (sent before each engine)."""
    component: str   # e.g. "te1", "unet", "vae"
    engine: str      # e.g. "default", "static-640x768", "dynamic-standard"


@dataclass
class TrtBuildResult:
    """Result of TRT engine building."""
    success: bool
    results: dict[str, list[str]] = field(default_factory=dict)
    error: str | None = None


@dataclass
class TagResult:
    """Result of image tagging."""
    tags: dict[str, float] = field(default_factory=dict)
    error: str | None = None


@dataclass
class OnloadComplete:
    """Onload/unevictable processing finished."""
    pass


@dataclass
class ProcessError:
    """Unhandled error in worker process."""
    error: str
    fatal: bool = False


@dataclass
class LogMessage:
    """Log message from worker process, forwarded to main process for display."""
    message: str
    level: str  # LogLevel.value: "DEBUG", "INFO", "WARNING", "ERROR"


@dataclass
class WorkerReady:
    """Worker process has initialized and is ready for commands."""
    gpu_model_name: str
    arch_key: str
