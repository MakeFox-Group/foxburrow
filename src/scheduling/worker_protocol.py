"""Cross-process message types for GPU worker IPC.

Both the main process and GPU worker subprocess import this module.
All types must be pickle-safe (no asyncio objects, no CUDA tensors).
Tensors crossing the boundary must be on CPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from PIL import Image
import torch

from scheduling.job import (
    StageType,
    SdxlJobInput,
    SdxlHiresInput,
    SdxlTokenizeResult,
    ModelComponentId,
)


# ── Commands (main → worker) ─────────────────────────────────────

@dataclass
class ExecuteStageCmd:
    """Execute a single pipeline stage on the GPU."""
    job_id: str
    job_type_value: str
    stage_type: StageType
    required_components: list[ModelComponentId]
    required_capability: str | None

    # Job input data (immutable)
    sdxl_input: SdxlJobInput | None
    hires_input: SdxlHiresInput | None
    input_image: Image.Image | None

    # Intermediate state (CPU tensors only)
    tokenize_result: SdxlTokenizeResult | None
    encode_result_tensors: dict[str, torch.Tensor] | None
    regional_encode_tensors: dict[str, Any] | None
    regional_tokenize_results: list[SdxlTokenizeResult] | None
    regional_base_tokenize: SdxlTokenizeResult | None
    regional_shared_neg_tokenize: SdxlTokenizeResult | None
    latents: torch.Tensor | None  # CPU tensor

    # Regional prompting info (serializable parts only)
    regional_info_data: dict | None

    # Job progress / state
    is_hires_pass: bool
    oom_retries: int
    current_stage_index: int
    pipeline_length: int
    priority: int

    # Tiling overrides
    unet_tile_width: int = 0
    unet_tile_height: int = 0
    vae_tile_width: int = 0
    vae_tile_height: int = 0

    # Original dimensions for final resize
    orig_width: int | None = None
    orig_height: int | None = None

    # TE fingerprints from the pipeline (category -> fingerprint).
    # Populated by the proxy from the job's TE stage so non-TE stages
    # can protect TE TRT engines from eviction during model loading.
    te_fingerprints: dict[str, str] | None = None


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
    dynamic_only: bool = False   # True = skip static engines


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
class StageResult:
    """Result of executing a pipeline stage."""
    job_id: str
    success: bool
    error: str | None = None
    oom: bool = False
    fatal: bool = False

    # Output data
    output_image: Image.Image | None = None
    output_latents: torch.Tensor | None = None  # CPU tensor

    # Updated intermediate state (CPU tensors)
    encode_result_tensors: dict[str, torch.Tensor] | None = None
    regional_encode_tensors: dict[str, Any] | None = None
    latents: torch.Tensor | None = None  # CPU tensor
    tokenize_result: SdxlTokenizeResult | None = None
    regional_tokenize_results: list[SdxlTokenizeResult] | None = None
    regional_base_tokenize: SdxlTokenizeResult | None = None
    regional_shared_neg_tokenize: SdxlTokenizeResult | None = None
    regional_info_data: dict | None = None

    # Updated job state
    is_hires_pass: bool = False
    current_stage_index: int = 0

    # Progress info
    denoise_step: int = 0
    denoise_total_steps: int = 0

    # Timing
    model_load_time_s: float = 0.0
    gpu_time_s: float = 0.0
    gpu_stage_times: list[dict] = field(default_factory=list)


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
class WorkerReady:
    """Worker process has initialized and is ready for commands."""
    gpu_model_name: str
    arch_key: str
