"""Inference job definitions and types."""

from __future__ import annotations

import asyncio
import random
import string
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from utils.regional import RegionalPromptResult

import torch
from PIL import Image


class JobType(Enum):
    SDXL_GENERATE = "SdxlGenerate"
    SDXL_GENERATE_LATENTS = "SdxlGenerateLatents"
    SDXL_DECODE_LATENTS = "SdxlDecodeLatents"
    SDXL_ENCODE_LATENTS = "SdxlEncodeLatents"
    SDXL_GENERATE_HIRES = "SdxlGenerateHires"
    SDXL_HIRES_LATENTS = "SdxlHiresLatents"
    UPSCALE = "Upscale"
    BGREMOVE = "BGRemove"
    TAG = "Tag"
    ENHANCE = "Enhance"


class StageType(Enum):
    CPU_TOKENIZE = "CpuTokenize"
    CPU_PREPROCESS = "CpuPreprocess"
    GPU_TEXT_ENCODE = "GpuTextEncode"
    GPU_DENOISE = "GpuDenoise"
    GPU_VAE_DECODE = "GpuVaeDecode"
    GPU_VAE_ENCODE = "GpuVaeEncode"
    GPU_HIRES_TRANSFORM = "GpuHiresTransform"
    GPU_UPSCALE = "GpuUpscale"
    GPU_BGREMOVE = "GpuBGRemove"


@dataclass
class ModelComponentId:
    """Identifies a loadable model component by its content fingerprint."""
    fingerprint: str
    category: str
    estimated_vram_bytes: int

    def __hash__(self):
        return hash(self.fingerprint)

    def __eq__(self, other):
        if not isinstance(other, ModelComponentId):
            return False
        return self.fingerprint == other.fingerprint


@dataclass
class WorkStage:
    """A single pipeline stage: type, required components, and GPU capability."""
    type: StageType
    required_components: list[ModelComponentId] = field(default_factory=list)
    required_capability: str | None = None

    @property
    def is_cpu_only(self) -> bool:
        return self.required_capability is None

    def __str__(self):
        comps = "+".join(c.category for c in self.required_components)
        return f"{self.type.value}" + (f" [{comps}]" if comps else "")


@dataclass
class SdxlJobInput:
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = 0
    model_dir: str = ""
    loras: list = field(default_factory=list)  # list[LoraSpec] parsed from prompt
    regional_prompting: bool = False


@dataclass
class SdxlHiresInput:
    hires_width: int = 0
    hires_height: int = 0
    hires_steps: int = 15
    denoising_strength: float = 0.33


@dataclass
class SdxlTokenizeResult:
    prompt_tokens_1: list[int] = field(default_factory=list)
    prompt_weights_1: list[float] = field(default_factory=list)
    neg_tokens_1: list[int] = field(default_factory=list)
    neg_weights_1: list[float] = field(default_factory=list)
    prompt_tokens_2: list[int] = field(default_factory=list)
    prompt_weights_2: list[float] = field(default_factory=list)
    neg_tokens_2: list[int] = field(default_factory=list)
    neg_weights_2: list[float] = field(default_factory=list)
    prompt_mask_1: list[int] = field(default_factory=list)
    neg_mask_1: list[int] = field(default_factory=list)
    prompt_mask_2: list[int] = field(default_factory=list)
    neg_mask_2: list[int] = field(default_factory=list)


@dataclass
class SdxlEncodeResult:
    prompt_embeds: torch.Tensor | None = None       # [1, 77, 2048]
    neg_prompt_embeds: torch.Tensor | None = None    # [1, 77, 2048]
    pooled_prompt_embeds: torch.Tensor | None = None # [1, 1280]
    neg_pooled_prompt_embeds: torch.Tensor | None = None  # [1, 1280]


@dataclass
class SdxlRegionalEncodeResult:
    region_embeds: list[torch.Tensor] = field(default_factory=list)  # N × [1, 77, 2048]
    neg_prompt_embeds: torch.Tensor | None = None    # [1, 77, 2048] shared
    neg_region_embeds: list[torch.Tensor] | None = None  # N × [1, 77, 2048] per-region neg, or None if shared
    pooled_prompt_embeds: torch.Tensor | None = None  # [1, 1280] from base or first region
    neg_pooled_prompt_embeds: torch.Tensor | None = None  # [1, 1280]
    base_embeds: torch.Tensor | None = None          # [1, 77, 2048] if ADDBASE
    base_ratio: float = 0.2


@dataclass
class JobResult:
    success: bool
    output_image: Image.Image | None = None
    output_latents: torch.Tensor | None = None
    error: str | None = None


class InferenceJob:
    """A complete inference request that flows through a pipeline of stages."""

    def __init__(
        self,
        job_type: JobType,
        pipeline: list[WorkStage],
        sdxl_input: SdxlJobInput | None = None,
        hires_input: SdxlHiresInput | None = None,
        input_image: Image.Image | None = None,
        priority: int = 100,
    ):
        _JOB_CHARS = string.ascii_letters + string.digits  # a-zA-Z0-9
        self.job_id: str = ''.join(random.choices(_JOB_CHARS, k=10))
        self.created_at: datetime = datetime.utcnow()
        self.started_at: datetime | None = None
        self.completed_at: datetime | None = None

        self.type = job_type
        self.priority = priority
        self.pipeline = pipeline
        self.current_stage_index: int = 0

        # Completion future — HTTP handler awaits this
        # Use get_running_loop() since InferenceJob is always created from async context
        self._loop = asyncio.get_running_loop()
        self.completion: asyncio.Future[JobResult] = self._loop.create_future()

        # Input data (immutable after creation)
        self.sdxl_input = sdxl_input
        self.hires_input = hires_input
        self.input_image = input_image

        # Progress tracking
        self.denoise_step: int = 0
        self.denoise_total_steps: int = 0
        self.stage_step: int = 0          # generic stage progress (e.g. VAE tiles)
        self.stage_total_steps: int = 0   # total sub-steps for current stage
        self.stage_status: str = ""  # "loading" or "running" — for TUI display

        # GPU tracking — updated when worker starts/finishes a stage
        self.active_gpus: list[dict] = []  # [{"uuid": "GPU-...", "name": "RTX 4090", "stage": "GpuDenoise"}]

        # Actual GPU time tracking — accumulated per-stage
        self.gpu_time_s: float = 0.0  # total seconds of actual GPU execution
        self.gpu_stage_times: list[dict] = []  # [{"gpu": "GPU-...", "stage": "GpuDenoise", "model": "xavier_v10", "duration_s": 2.3}]

        # Intermediate state
        self.tokenize_result: SdxlTokenizeResult | None = None
        self.encode_result: SdxlEncodeResult | None = None
        self.latents: torch.Tensor | None = None
        self.is_hires_pass: bool = False

        # Regional prompting state
        self.regional_info: RegionalPromptResult | None = None
        self.regional_encode_result: SdxlRegionalEncodeResult | None = None
        self.regional_tokenize_results: list[SdxlTokenizeResult] | None = None
        self.regional_base_tokenize: SdxlTokenizeResult | None = None
        self.regional_shared_neg_tokenize: SdxlTokenizeResult | None = None  # global negative

        # Tiling overrides (0 = auto: divide into tiles ≤ 1024px)
        self.unet_tile_width: int = 0   # UNet / MultiDiffusion tile width (pixels)
        self.unet_tile_height: int = 0  # UNet / MultiDiffusion tile height (pixels)
        self.vae_tile_width: int = 0    # VAE encode/decode tile width (pixels)
        self.vae_tile_height: int = 0   # VAE encode/decode tile height (pixels)

    @property
    def current_stage(self) -> WorkStage | None:
        if self.current_stage_index < len(self.pipeline):
            return self.pipeline[self.current_stage_index]
        return None

    @property
    def is_complete(self) -> bool:
        return self.current_stage_index >= len(self.pipeline)

    def set_result(self, result: JobResult) -> None:
        """Thread-safe result setting via event loop."""
        if not self.completion.done():
            self._loop.call_soon_threadsafe(self.completion.set_result, result)

    def __str__(self):
        return f"Job[{self.job_id}] {self.type.value} stage={self.current_stage_index}/{len(self.pipeline)}"
