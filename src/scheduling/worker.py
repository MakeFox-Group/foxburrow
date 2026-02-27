"""Per-GPU async worker with model loading and stage execution."""

from __future__ import annotations

import asyncio
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import torch

import contextlib

import log
from gpu import nvml
from gpu.pool import GpuInstance
from gpu import torch_ext
from scheduling.job import (
    InferenceJob, JobResult, JobType, StageType, WorkStage,
)
from scheduling.queue import JobQueue
from scheduling.scheduler import (
    get_session_key, get_session_group, get_session_group_from_key,
    _COMPATIBLE_WITH_ALL,
)

if TYPE_CHECKING:
    pass

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
    # GPU_HIRES_TRANSFORM is no longer used (split into VaeDecode+Upscale+VaeEncode)
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
    """Return (width, height) for the profiler lookup."""
    w, h = 768, 768  # safe default
    if job.sdxl_input:
        w, h = job.sdxl_input.width, job.sdxl_input.height

    if stage_type == StageType.GPU_HIRES_TRANSFORM:
        if job.hires_input and job.hires_input.hires_width > 0:
            return job.hires_input.hires_width, job.hires_input.hires_height

    if job.input_image is not None and stage_type in (
            StageType.GPU_UPSCALE, StageType.GPU_BGREMOVE):
        return job.input_image.width, job.input_image.height

    return w, h


def _get_stage_pixels(stage_type: StageType, job: InferenceJob) -> int:
    """Return the pixel count that drives working-memory scaling for a stage.

    - Text encode: constant (77 tokens, resolution-independent) → returns 1
    - Denoise: scales with latent area = (W/8)×(H/8)
    - VAE decode/encode, bgremove: scales with output pixel area = W×H
    - Hires transform: scales with hires output area
    - Upscale: scales with input pixel area (model processes input)
    """
    if stage_type == StageType.GPU_TEXT_ENCODE:
        return 1  # absolute measurement, not per-pixel

    w, h = 768, 768  # safe default
    if job.sdxl_input:
        w, h = job.sdxl_input.width, job.sdxl_input.height

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
    """Record a working-memory measurement as a bytes-per-pixel ratio."""
    if working_bytes <= 0 or pixels <= 0:
        return
    ratio = working_bytes / pixels
    with _bpp_lock:
        prev_ratio = _measured_bpp.get(stage_type, 0.0)
        if ratio > prev_ratio:
            _measured_bpp[stage_type] = ratio
            log.debug(f"  Working memory: new peak for {stage_type.value}: "
                     f"{working_bytes // (1024**2)}MB @ {pixels} px "
                     f"({ratio:.1f} B/px, prev {prev_ratio:.1f} B/px)")


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


_CUDA_FATAL_PATTERNS = [
    "illegal memory access",           # cudaErrorIllegalAddress / Xid 31 MMU fault
    "unspecified launch failure",       # cudaErrorLaunchFailure
    "cuda error: an illegal instruction was encountered",  # cudaErrorIllegalInstruction
    "device-side assert",              # cudaErrorAssert (kernel assertion failure)
    "unable to find an engine",        # cuDNN workspace allocation failed → MMU fault
]


def _is_cuda_fatal(ex: Exception) -> bool:
    """Check if a CUDA error indicates permanent context corruption.

    These errors mean the CUDA context is irrecoverably broken — the GPU
    driver issued an MMU fault or similar fatal error. All GPUs sharing
    this process's CUDA context are affected and must be marked dead.

    OOM errors are explicitly excluded — they are recoverable.
    """
    if isinstance(ex, torch.cuda.OutOfMemoryError):
        return False
    msg = str(ex).lower()
    # OOM can surface as torch.AcceleratorError instead of OutOfMemoryError.
    # Check the message to catch these — OOM is always recoverable.
    if "out of memory" in msg or "cudaerrormemoryallocation" in msg:
        return False
    return any(p in msg for p in _CUDA_FATAL_PATTERNS)


class GpuWorker:
    """Per-GPU background worker. Receives work items, loads models, executes stages."""

    def __init__(self, gpu: GpuInstance, queue: JobQueue, scheduler_wake: asyncio.Event):
        self._gpu = gpu
        self._queue = queue
        self._scheduler_wake = scheduler_wake
        self._work_queue: asyncio.Queue[tuple[WorkStage, InferenceJob]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # set in start()

        # Cached GPU model name — stable for process lifetime, avoids
        # repeated CUDA driver calls on the scheduling hot path.
        from gpu.workspace_profiler import get_gpu_model_name
        self._gpu_model_name: str = get_gpu_model_name(gpu.device)

        # Concurrency tracking
        self._state_lock = threading.Lock()
        self._active_count = 0
        self._active_session_counts: dict[str, int] = defaultdict(int)
        self._active_stage_counts: dict[StageType, int] = defaultdict(int)
        self._active_jobs: dict[str, InferenceJob] = {}

        # Model loading serialization
        self._model_load_lock = asyncio.Lock()

        # Cross-GPU failure propagation — set by scheduler via set_workers()
        self._all_workers: list[GpuWorker] = []

    def has_session_capacity(self, session_key: str) -> bool:
        """Check if this worker can accept one more job of the given session type."""
        with self._state_lock:
            if self._active_count >= MAX_CONCURRENCY:
                return False
            max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
            return self._active_session_counts.get(session_key, 0) < max_for_session

    def check_slot_availability(self, session_key: str, group: str) -> tuple[bool, int]:
        """Atomically check capacity + group conflict and return active_count.

        Returns (can_accept, active_count) under a single lock acquisition.
        This prevents TOCTOU races between separate has_session_capacity()
        and active_count reads in the slot estimation path.
        """
        with self._state_lock:
            if self._active_count >= MAX_CONCURRENCY:
                return False, self._active_count
            max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
            if self._active_session_counts.get(session_key, 0) >= max_for_session:
                return False, self._active_count
            # Session group conflict — skip for lightweight compatible-with-all groups
            if self._active_count > 0:
                if group not in _COMPATIBLE_WITH_ALL:
                    for key, count in self._active_session_counts.items():
                        if count <= 0:
                            continue
                        active_group = get_session_group_from_key(key)
                        if active_group and active_group != group:
                            if active_group not in _COMPATIBLE_WITH_ALL:
                                return False, self._active_count
            return True, self._active_count

    @property
    def gpu(self) -> GpuInstance:
        return self._gpu

    @property
    def is_idle(self) -> bool:
        with self._state_lock:
            return self._active_count == 0

    @property
    def active_count(self) -> int:
        with self._state_lock:
            return self._active_count

    @property
    def active_jobs(self) -> list[InferenceJob]:
        with self._state_lock:
            return list(self._active_jobs.values())

    def start(self) -> None:
        self._loop = asyncio.get_running_loop()
        self._task = self._loop.create_task(self._run_loop())

    def get_loaded_categories(self) -> set[str]:
        """Return set of model categories currently loaded on this GPU."""
        cats = set()
        cached = self._gpu.get_cached_categories()
        has_te1 = "sdxl_te1" in cached
        has_te2 = "sdxl_te2" in cached
        for cat in cached:
            if cat == "sdxl_te1" or cat == "sdxl_te2":
                continue
            cats.add(cat)
        if has_te1 and has_te2:
            cats.add("sdxl_te")
        return cats

    def can_accept_work(self, stage: WorkStage) -> bool:
        """Whether this worker can accept a new work item for the given stage."""
        if self._gpu.is_failed:
            return False
        if stage.required_capability and not self._gpu.supports_capability(stage.required_capability):
            return False

        session_key = get_session_key(stage.type)
        if session_key is None:
            return False

        with self._state_lock:
            if self._active_count >= MAX_CONCURRENCY:
                return False

            # Per-stage-type limit: never run the same stage type twice on one GPU
            if self._active_stage_counts[stage.type] >= 1:
                return False

            max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
            if self._active_session_counts[session_key] >= max_for_session:
                return False

            # Session group must match all active items — unless either
            # side is a lightweight group that can coexist with anything.
            if self._active_count > 0:
                new_group = get_session_group(stage.type)
                if new_group not in _COMPATIBLE_WITH_ALL:
                    for key, count in self._active_session_counts.items():
                        if count <= 0:
                            continue
                        active_group = get_session_group_from_key(key)
                        if active_group and active_group != new_group:
                            if active_group not in _COMPATIBLE_WITH_ALL:
                                return False

            return True

    def check_vram_budget(self, stage: WorkStage, job: InferenceJob) -> bool:
        """Check if this GPU has enough VRAM for a new stage.

        When the GPU is idle, model loading handles eviction so we always
        return True (``_ensure_models_for_stage`` calls ``ensure_free_vram``
        before execution, which evicts LRU models as needed).
        When busy, we query NVML free VRAM, add back PyTorch's
        cached-allocator slack (released by ``empty_cache()``) and evictable
        model VRAM, then compare against the cost of loading new models plus
        the working-memory estimate for this stage.
        """
        with self._state_lock:
            active = self._active_count
        if active == 0:
            return True  # idle — _ensure_models_for_stage handles eviction

        gpu = self._gpu

        # ── Cost side ──────────────────────────────────────────────
        # Models not yet on this GPU
        model_cost = sum(
            c.estimated_vram_bytes
            for c in stage.required_components
            if not gpu.is_component_loaded(c.fingerprint)
        )
        # Working memory (activations, cuDNN workspace)
        working_cost = _get_min_free_vram(stage.type, job, self._gpu_model_name)
        total_needed = model_cost + working_cost

        # ── Available side ─────────────────────────────────────────
        _, _, nvml_free = nvml.get_memory_info(gpu.nvml_handle)

        # PyTorch allocator cache: reserved but not in use by live tensors.
        # empty_cache() in the model-load path frees these blocks.
        pt_reserved = torch.cuda.memory_reserved(gpu.device)
        pt_allocated = torch.cuda.memory_allocated(gpu.device)
        pt_slack = max(0, pt_reserved - pt_allocated)

        # Evictable models: cached but not protected by active jobs or this stage
        protect = gpu.get_active_fingerprints()
        protect.update(c.fingerprint for c in stage.required_components)
        evictable = gpu.get_evictable_vram(protect)

        available = nvml_free + pt_slack + evictable

        # When tag-based tracking is available, log exact model VRAM for
        # diagnostics (helps identify estimate drift vs actual usage).
        tag_info = ""
        if torch_ext.HAS_ALLOC_TAGS:
            actual_model_vram = torch.cuda.memory_allocated_by_tag(
                torch_ext.ALLOC_TAG_MODEL_WEIGHTS, gpu.device)
            tag_info = f", tag_models={actual_model_vram // (1024**2)}MB"

        fits = available >= total_needed

        log.debug(f"  VRAM budget [{gpu.uuid}]: "
                 f"need {total_needed // (1024**2)}MB "
                 f"(models={model_cost // (1024**2)}MB, "
                 f"working={working_cost // (1024**2)}MB) | "
                 f"avail {available // (1024**2)}MB "
                 f"(free={nvml_free // (1024**2)}MB, "
                 f"pt_slack={pt_slack // (1024**2)}MB, "
                 f"evictable={evictable // (1024**2)}MB{tag_info}) "
                 f"→ {'OK' if fits else 'REJECTED'}")

        return fits

    def log_vram_state(self, context: str) -> None:
        """Log detailed VRAM breakdown for this GPU."""
        gpu = self._gpu
        stats = gpu.get_vram_stats()
        models = gpu.get_cached_models_info()
        total_model_vram = sum(
            m['actual_vram'] if m['actual_vram'] > 0 else m['vram']
            for m in models
        )

        # Build tag-based VRAM suffix when available
        tag_info = ""
        if "model_vram" in stats:
            tag_info = (f" | Tags: model={stats['model_vram'] // (1024**2)}MB, "
                        f"activation={stats['activation_vram'] // (1024**2)}MB")

        log.debug(
            f"  VRAM [{gpu.uuid}] ({context}): "
            f"NVML {stats['used'] // (1024**2)}MB used / "
            f"{stats['free'] // (1024**2)}MB free / "
            f"{stats['total'] // (1024**2)}MB total | "
            f"PyTorch {stats['allocated'] // (1024**2)}MB alloc / "
            f"{stats['reserved'] // (1024**2)}MB reserved | "
            f"Models {total_model_vram // (1024**2)}MB in {len(models)} cached"
            f"{tag_info}"
        )
        for m in models:
            vram = m['actual_vram'] if m['actual_vram'] > 0 else m['vram']
            source = f" ({m['source']})" if m.get('source') else ""
            log.debug(f"    ├─ {m['category']}{source}: {vram // (1024**2)}MB")

        if torch_ext.HAS_HISTOGRAM:
            hist = torch.cuda.allocation_histogram(gpu.device)
            large_bins = {k: v for k, v in hist.items()
                          if v["count"] > 0 and v["total_bytes"] > 10 * 1024**2}
            if large_bins:
                log.debug(f"    Histogram [{gpu.uuid}]: {large_bins}")

    def dispatch(self, stage: WorkStage, job: InferenceJob) -> None:
        """Dispatch a work item to this worker."""
        if self._gpu.is_failed:
            raise RuntimeError(f"GPU [{self._gpu.uuid}] is permanently failed — cannot dispatch")
        with self._state_lock:
            if self._active_count == 0:
                if not self._gpu.try_acquire():
                    raise RuntimeError(f"GPU [{self._gpu.uuid}] semaphore unexpectedly held")
            self._active_count += 1
            self._active_stage_counts[stage.type] += 1
            key = get_session_key(stage.type)
            if key:
                self._active_session_counts[key] += 1

        self._work_queue.put_nowait((stage, job))

    async def _run_loop(self) -> None:
        log.debug(f"  GpuWorker[{self._gpu.uuid}]: Started")
        try:
            while True:
                stage, job = await self._work_queue.get()
                # Spawn concurrent — don't await
                self._loop.create_task(self._process_work_item(stage, job))
        except asyncio.CancelledError:
            log.debug(f"  GpuWorker[{self._gpu.uuid}]: Stopped")

    async def _process_work_item(self, stage: WorkStage, job: InferenceJob) -> None:
        try:
            log.debug(f"  GpuWorker[{self._gpu.uuid}]: Processing {job} at {stage} "
                     f"(active={self._active_count})")
            self.log_vram_state(f"pre-load {stage.type.value}")

            # Make job visible to TUI immediately (during model loading)
            if job.started_at is None:
                job.started_at = datetime.utcnow()
            job.denoise_step = 0
            job.denoise_total_steps = 0
            job.stage_step = 0
            job.stage_total_steps = 0
            job.stage_status = "loading"
            gpu_info = {"uuid": self._gpu.uuid, "name": self._gpu.name,
                        "stage": stage.type.value}
            job.active_gpus = [gpu_info]
            with self._state_lock:
                self._active_jobs[job.job_id] = job

            # Model loading (serialized) — included in GPU time
            import time as _time
            load_start = _time.monotonic()
            async with self._model_load_lock:
                # Release PyTorch's cached allocator memory before loading/checking VRAM.
                # Without this, NVML reports almost no free VRAM even when the cached
                # blocks hold no live tensors (e.g. after a previous stage's working
                # memory was freed but PyTorch kept the blocks reserved).
                # Target THIS worker's GPU — not the default device (cuda:0).
                with torch.cuda.device(self._gpu.device):
                    torch.cuda.empty_cache()

                await self._loop.run_in_executor(
                    None, self._ensure_models_for_stage, stage, job)

                min_free = _get_min_free_vram(stage.type, job, self._gpu_model_name)
                if min_free > 0:
                    protect = self._gpu.get_active_fingerprints()
                    self._gpu.ensure_free_vram(min_free, protect)
            load_duration = _time.monotonic() - load_start
            job.gpu_time_s += load_duration

            # Execute stage
            try:
                job.stage_status = "running"

                # Measure actual GPU execution time (model loading already measured above)
                import time as _time
                stage_start = _time.monotonic()

                # Snapshot active_count NOW (before execution) so we know
                # whether this was the only stage running during measurement.
                with self._state_lock:
                    active_at_start = self._active_count

                # Set up memory measurement: prefer PeakMemoryScope (no global
                # state reset, per-scope baseline) over reset_peak_memory_stats
                # which races with concurrent stages.
                peak_scope = None
                mem_baseline = 0
                if torch_ext.HAS_PEAK_SCOPE:
                    peak_scope = torch.cuda.PeakMemoryScope(device=self._gpu.device)
                    peak_scope.__enter__()
                else:
                    with torch.cuda.device(self._gpu.device):
                        torch.cuda.reset_peak_memory_stats()
                    mem_baseline = torch.cuda.memory_allocated(self._gpu.device)

                try:
                    output_image = await self._loop.run_in_executor(
                        None, self._execute_stage, job, stage)
                finally:
                    if peak_scope is not None:
                        peak_scope.__exit__(None, None, None)
                        working_mem = peak_scope.peak_bytes
                    else:
                        # Capture peak before empty_cache clears it
                        mem_peak = torch.cuda.max_memory_allocated(self._gpu.device)
                        working_mem = mem_peak - mem_baseline

                    # Release working memory (UNet activations, attention matrices, etc.)
                    # back to the OS immediately. This MUST run even on OOM — otherwise
                    # PyTorch's caching allocator keeps the failed allocation's blocks
                    # reserved, causing cascading OOM on all subsequent jobs.
                    # Target THIS worker's GPU — not the default device (cuda:0).
                    with torch.cuda.device(self._gpu.device):
                        torch.cuda.empty_cache()

                stage_duration = _time.monotonic() - stage_start
                job.gpu_time_s += stage_duration

                # Feed the measurement back so future VRAM checks are accurate.
                # IMPORTANT: Only update when running solo — when multiple
                # stages run concurrently, peak tracking (scoped or global)
                # captures ALL allocations on the device, producing inflated
                # per-component values.
                stage_pixels = _get_stage_pixels(stage.type, job)
                is_solo = active_at_start <= 1
                if is_solo:
                    _update_working_memory(stage.type, working_mem, stage_pixels)
                log.debug(f"  Stage {stage.type.value}: working memory "
                         f"{working_mem // (1024**2)}MB"
                         f"{' (scoped)' if peak_scope is not None else ''}"
                         f"{'' if is_solo else ' CONCURRENT — BPP update skipped'}")

                # Get model name for this stage
                model_name = None
                if job.sdxl_input and job.sdxl_input.model_dir:
                    model_name = os.path.basename(job.sdxl_input.model_dir)
                    for ext in (".safetensors", ".ckpt"):
                        if model_name.endswith(ext):
                            model_name = model_name[:-len(ext)]
                            break
                job.gpu_stage_times.append({
                    "gpu": self._gpu.uuid,
                    "gpu_name": self._gpu.name,
                    "stage": stage.type.value,
                    "model": model_name,
                    "duration_s": round(stage_duration, 3),
                })

                job.active_gpus = []

                # For intermediate stages that produce an image (e.g. upscale in
                # the enhance pipeline), store it back so the next stage can use it.
                if output_image is not None and not job.is_complete:
                    job.input_image = output_image
                    # Resize to hires target if needed (upscale may produce 2x
                    # which exceeds the target when capped at max resolution).
                    if job.hires_input:
                        tw = job.hires_input.hires_width
                        th = job.hires_input.hires_height
                        if job.input_image.width != tw or job.input_image.height != th:
                            from PIL import Image as _PILImage
                            job.input_image = job.input_image.resize(
                                (tw, th), _PILImage.LANCZOS)
                            log.debug(f"  GpuWorker[{self._gpu.uuid}]: Resized intermediate "
                                     f"output to {tw}x{th}")

                job.current_stage_index += 1

                if job.is_complete:
                    job.stage_status = ""
                    job.active_gpus = []
                    with self._state_lock:
                        self._active_jobs.pop(job.job_id, None)
                    result = JobResult(
                        success=True,
                        output_image=output_image,
                        output_latents=job.latents if job.type in (
                        JobType.SDXL_GENERATE_LATENTS,
                        JobType.SDXL_ENCODE_LATENTS,
                        JobType.SDXL_HIRES_LATENTS) else None,
                    )
                    self._store_job_result(job, result)
                    job.completed_at = datetime.utcnow()
                    job.set_result(result)
                    self._broadcast_complete(job, success=True)
                    self._gpu.record_success()
                    log.debug(f"  GpuWorker[{self._gpu.uuid}]: {job} completed")
                else:
                    job.stage_status = ""
                    job.active_gpus = []
                    with self._state_lock:
                        self._active_jobs.pop(job.job_id, None)
                    self._gpu.record_success()
                    self._queue.re_enqueue(job)

            except Exception as ex:
                job.stage_status = ""
                job.active_gpus = []
                with self._state_lock:
                    self._active_jobs.pop(job.job_id, None)
                if _is_cuda_fatal(ex):
                    log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: {job} failed at {stage}")
                    log.error(f"  FATAL: Unrecoverable CUDA error — all GPUs unusable. "
                              f"Process restart required.")
                    for w in self._all_workers:
                        w.gpu.mark_failed(f"CUDA context corrupted ({type(ex).__name__})")
                    job.completed_at = datetime.utcnow()
                    job.set_result(JobResult(success=False, error=str(ex)))
                    self._broadcast_complete(job, success=False, error=str(ex))
                elif isinstance(ex, torch.cuda.OutOfMemoryError):
                    # OOM during execution — always re-enqueue, never fail.
                    # Free cached memory first so the GPU can accept other work.
                    torch.cuda.empty_cache()
                    job.oom_retries += 1
                    job.oom_gpu_ids.add(self._gpu.uuid)
                    log.warning(f"  GpuWorker[{self._gpu.uuid}]: {job} OOM at {stage} "
                                f"— re-enqueuing (retry {job.oom_retries})")
                    self._queue.re_enqueue(job)
                else:
                    log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: {job} failed at {stage}")
                    self._gpu.record_failure()
                    job.completed_at = datetime.utcnow()
                    job.set_result(JobResult(success=False, error=str(ex)))
                    self._broadcast_complete(job, success=False, error=str(ex))

        except Exception as ex:
            # Outer handler catches model-loading failures and other setup errors.
            # Clear display state so TUI doesn't show stale GPU assignments.
            job.stage_status = ""
            job.active_gpus = []
            with self._state_lock:
                self._active_jobs.pop(job.job_id, None)
            if _is_cuda_fatal(ex):
                log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: Batch failed at {stage}")
                log.error(f"  FATAL: Unrecoverable CUDA error — all GPUs unusable. "
                          f"Process restart required.")
                for w in self._all_workers:
                    w.gpu.mark_failed(f"CUDA context corrupted ({type(ex).__name__})")
                job.completed_at = datetime.utcnow()
                job.set_result(JobResult(success=False, error=str(ex)))
                self._broadcast_complete(job, success=False, error=str(ex))
            elif isinstance(ex, torch.cuda.OutOfMemoryError):
                # OOM during model loading — always re-enqueue, never fail.
                torch.cuda.empty_cache()
                job.oom_retries += 1
                job.oom_gpu_ids.add(self._gpu.uuid)
                log.warning(f"  GpuWorker[{self._gpu.uuid}]: {job} OOM loading models for {stage} "
                            f"— re-enqueuing (retry {job.oom_retries})")
                self._queue.re_enqueue(job)
            else:
                log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: Batch failed at {stage}")
                job.completed_at = datetime.utcnow()
                job.set_result(JobResult(success=False, error=str(ex)))
                self._broadcast_complete(job, success=False, error=str(ex))

        finally:
            try:
                # Remove ref-counted active fingerprints so these models become evictable
                # when no other jobs are using them.
                active_fps = {c.fingerprint for c in stage.required_components}
                if active_fps:
                    self._gpu.remove_active_fingerprints(active_fps)

                became_idle = False
                with self._state_lock:
                    self._active_count -= 1
                    self._active_stage_counts[stage.type] = max(
                        0, self._active_stage_counts[stage.type] - 1)
                    key = get_session_key(stage.type)
                    if key:
                        self._active_session_counts[key] = max(
                            0, self._active_session_counts[key] - 1)
                    became_idle = self._active_count == 0

                if became_idle:
                    self._gpu.release()
            except Exception as ex:
                log.log_exception(ex,
                    f"GpuWorker[{self._gpu.uuid}]: FATAL error in cleanup for {stage}")
            finally:
                self._scheduler_wake.set()

    def _store_job_result(self, job: InferenceJob, result: JobResult) -> None:
        """Store result bytes in AppState for the queue-based API."""
        try:
            from state import app_state
            import io as _io

            if result.output_image is not None:
                image = result.output_image
                # Resize to originally-requested dimensions if the job was
                # generated at snapped-up sizes (orig_width/orig_height are
                # set by enqueue endpoints that use _snap_dims).
                orig_w = getattr(job, "orig_width", None)
                orig_h = getattr(job, "orig_height", None)
                if (orig_w is not None and orig_h is not None
                        and (image.width != orig_w or image.height != orig_h)):
                    from PIL import Image as _Image
                    image = image.resize((orig_w, orig_h), _Image.LANCZOS)
                buf = _io.BytesIO()
                mode = "RGBA" if image.mode == "RGBA" else "RGB"
                if mode == "RGBA":
                    image.save(buf, format="PNG")
                else:
                    image.convert("RGB").save(buf, format="PNG")
                app_state.job_results[job.job_id] = (buf.getvalue(), "image/png")
            elif result.output_latents is not None:
                import struct
                latents = result.output_latents
                shape = list(latents.shape)
                float_data = latents.cpu().float().numpy().tobytes()
                buf = _io.BytesIO()
                buf.write(b"FXLT")
                buf.write(struct.pack("<H", 1))
                buf.write(struct.pack("<H", len(shape)))
                for d in shape:
                    buf.write(struct.pack("<i", d))
                buf.write(struct.pack("<I", 0))  # no metadata
                buf.write(float_data)
                app_state.job_results[job.job_id] = (buf.getvalue(), "application/x-fox-latent")
        except Exception as ex:
            log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: Failed to store result for {job}")

    def _broadcast_complete(self, job: InferenceJob, success: bool,
                            error: str | None = None) -> None:
        """Broadcast job completion via WebSocket (thread-safe)."""
        try:
            from api.websocket import streamer
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    streamer.broadcast_complete(job, success, error), self._loop)
        except Exception:
            pass  # WebSocket broadcast is best-effort

    def _ensure_models_for_stage(self, stage: WorkStage, job: InferenceJob) -> None:
        """Load model components required for the given stage."""
        model_dir = job.sdxl_input.model_dir if job.sdxl_input else None

        if stage.type in (StageType.GPU_TEXT_ENCODE, StageType.GPU_DENOISE,
                          StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE):
            self._gpu.ensure_session_group("sdxl")
            self._load_sdxl_components(stage, model_dir)
        elif stage.type == StageType.GPU_UPSCALE:
            self._gpu.ensure_session_group("upscale")
            self._load_upscale_model()
        elif stage.type == StageType.GPU_BGREMOVE:
            self._gpu.ensure_session_group("bgremove")
            self._load_bgremove_model()

        # Add active fingerprints for VRAM eviction protection (ref-counted)
        active_fps = {c.fingerprint for c in stage.required_components}
        self._gpu.add_active_fingerprints(active_fps)

    def _load_sdxl_components(self, stage: WorkStage, model_dir: str | None) -> None:
        """Load SDXL sub-model components for a stage.
        Uses cached models when fingerprints match."""
        # Extract checkpoint name from model_dir for display
        source_name = ""
        if model_dir:
            source_name = os.path.basename(model_dir)
            # Strip .safetensors / .ckpt extension
            for ext in (".safetensors", ".ckpt"):
                if source_name.endswith(ext):
                    source_name = source_name[:-len(ext)]
                    break

        for component in stage.required_components:
            if self._gpu.is_component_loaded(component.fingerprint):
                continue
            # Ensure enough VRAM before loading (evicts non-current-group LRU first)
            active_fps = {c.fingerprint for c in stage.required_components}
            self._gpu.ensure_free_vram(component.estimated_vram_bytes, protect=active_fps)
            # Load the model component, measuring actual VRAM usage
            from handlers.sdxl import load_component
            before = torch.cuda.memory_allocated(self._gpu.device)
            tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
                       if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
            with tag_ctx:
                model = load_component(component.category, model_dir, self._gpu.device)
            after = torch.cuda.memory_allocated(self._gpu.device)
            actual_vram = after - before
            log.debug(f"  Loaded {component.category}: {actual_vram // (1024*1024)}MB actual "
                     f"(estimated {component.estimated_vram_bytes // (1024*1024)}MB)")
            # Feed actual VRAM measurement back to the registry so future
            # ensure_free_vram() calls use the real value instead of the estimate
            if actual_vram > 0:
                from state import app_state
                app_state.registry.update_actual_vram(component.fingerprint, actual_vram)
            # UNet eviction clears LoRA adapter tracking (adapters live in PEFT layers)
            evict_cb = None
            if component.category == "sdxl_unet":
                _gpu_ref = self._gpu
                def _on_unet_evict(_g=_gpu_ref):
                    _g._loaded_lora_adapters.clear()
                evict_cb = _on_unet_evict

            self._gpu.cache_model(
                component.fingerprint,
                component.category,
                model,
                component.estimated_vram_bytes,
                source=source_name,
                actual_vram=actual_vram,
                evict_callback=evict_cb,
            )

            # Profile workspace for this component type (one-time per GPU model).
            # Uses cached JSON after the first load — near-zero cost on subsequent loads.
            try:
                from gpu.workspace_profiler import ensure_profiled
                ensure_profiled(component.category, model,
                                self._gpu.device, self._gpu_model_name)
            except Exception as ex:
                log.warning(f"  WorkspaceProfiler: Failed to profile {component.category}: {ex}")

    def _load_upscale_model(self) -> None:
        """Load the upscale model if not already cached."""
        from handlers.upscale import load_model
        comp = None
        try:
            from state import app_state
            comp = app_state.registry.get_upscale_component()
        except Exception:
            pass
        if comp and self._gpu.is_component_loaded(comp.fingerprint):
            return
        if comp:
            self._gpu.ensure_free_vram(comp.estimated_vram_bytes,
                                       protect={comp.fingerprint})
        before = torch.cuda.memory_allocated(self._gpu.device)
        tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
                   if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
        with tag_ctx:
            model = load_model(self._gpu.device)
        after = torch.cuda.memory_allocated(self._gpu.device)
        actual_vram = after - before
        log.debug(f"  Loaded upscale: {actual_vram // (1024*1024)}MB actual"
                 + (f" (estimated {comp.estimated_vram_bytes // (1024*1024)}MB)" if comp else ""))
        if comp:
            if actual_vram > 0:
                from state import app_state
                app_state.registry.update_actual_vram(comp.fingerprint, actual_vram)
            self._gpu.cache_model(comp.fingerprint, "upscale", model, comp.estimated_vram_bytes,
                                  source="realesrgan")
            try:
                from gpu.workspace_profiler import ensure_profiled
                ensure_profiled("upscale", model,
                                self._gpu.device, self._gpu_model_name)
            except Exception as ex:
                log.warning(f"  WorkspaceProfiler: Failed to profile upscale: {ex}")

    def _load_bgremove_model(self) -> None:
        """Load the bgremove model if not already cached."""
        from handlers.bgremove import load_model
        comp = None
        try:
            from state import app_state
            comp = app_state.registry.get_bgremove_component()
        except Exception:
            pass
        if comp and self._gpu.is_component_loaded(comp.fingerprint):
            return
        if comp:
            self._gpu.ensure_free_vram(comp.estimated_vram_bytes,
                                       protect={comp.fingerprint})
        before = torch.cuda.memory_allocated(self._gpu.device)
        tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
                   if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
        with tag_ctx:
            model = load_model(self._gpu.device)
        after = torch.cuda.memory_allocated(self._gpu.device)
        actual_vram = after - before
        log.debug(f"  Loaded bgremove: {actual_vram // (1024*1024)}MB actual"
                 + (f" (estimated {comp.estimated_vram_bytes // (1024*1024)}MB)" if comp else ""))
        if comp:
            if actual_vram > 0:
                from state import app_state
                app_state.registry.update_actual_vram(comp.fingerprint, actual_vram)
            self._gpu.cache_model(comp.fingerprint, "bgremove", model, comp.estimated_vram_bytes,
                                  source="rmbg")
            try:
                from gpu.workspace_profiler import ensure_profiled
                ensure_profiled("bgremove", model,
                                self._gpu.device, self._gpu_model_name)
            except Exception as ex:
                log.warning(f"  WorkspaceProfiler: Failed to profile bgremove: {ex}")

    def _migrate_tensors_to_device(self, job: InferenceJob) -> None:
        """Move intermediate tensors to this worker's GPU device.

        When a job's stages run on different GPUs (e.g. text_encode on GPU 1,
        denoise on GPU 0), the intermediate tensors remain on the original
        device. This method moves them to the current worker's GPU before
        stage execution.
        """
        device = self._gpu.device

        if job.encode_result is not None:
            er = job.encode_result
            if er.prompt_embeds is not None and er.prompt_embeds.device != device:
                er.prompt_embeds = er.prompt_embeds.to(device)
            if er.neg_prompt_embeds is not None and er.neg_prompt_embeds.device != device:
                er.neg_prompt_embeds = er.neg_prompt_embeds.to(device)
            if er.pooled_prompt_embeds is not None and er.pooled_prompt_embeds.device != device:
                er.pooled_prompt_embeds = er.pooled_prompt_embeds.to(device)
            if er.neg_pooled_prompt_embeds is not None and er.neg_pooled_prompt_embeds.device != device:
                er.neg_pooled_prompt_embeds = er.neg_pooled_prompt_embeds.to(device)

        if job.latents is not None and job.latents.device != device:
            job.latents = job.latents.to(device)

        if job.regional_encode_result is not None:
            rer = job.regional_encode_result
            rer.region_embeds = [
                e.to(device) if e.device != device else e
                for e in rer.region_embeds
            ]
            if rer.neg_region_embeds is not None:
                rer.neg_region_embeds = [
                    e.to(device) if e.device != device else e
                    for e in rer.neg_region_embeds
                ]
            if rer.neg_prompt_embeds is not None and rer.neg_prompt_embeds.device != device:
                rer.neg_prompt_embeds = rer.neg_prompt_embeds.to(device)
            if rer.pooled_prompt_embeds is not None and rer.pooled_prompt_embeds.device != device:
                rer.pooled_prompt_embeds = rer.pooled_prompt_embeds.to(device)
            if rer.neg_pooled_prompt_embeds is not None and rer.neg_pooled_prompt_embeds.device != device:
                rer.neg_pooled_prompt_embeds = rer.neg_pooled_prompt_embeds.to(device)
            if rer.base_embeds is not None and rer.base_embeds.device != device:
                rer.base_embeds = rer.base_embeds.to(device)

    def _execute_stage(self, job: InferenceJob, stage: WorkStage):
        """Execute a single job's current stage. Returns output image for final stages."""
        # Tag all allocations during stage execution as activations (thread-local,
        # must be set here on the executor thread — not on the asyncio event loop).
        tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_ACTIVATIONS)
                   if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
        with tag_ctx:
            return self._dispatch_stage(job, stage)

    def _dispatch_stage(self, job: InferenceJob, stage: WorkStage):
        """Dispatch to the appropriate handler for a stage type."""
        # Store fingerprint mapping so handlers retrieve the correct model instance
        # when multiple models of the same category are cached (e.g. two sdxl_te1
        # from different checkpoints).  Without this, _get_cached_model returns the
        # LRU entry — which may be an unprotected model that gets evicted mid-inference.
        job._stage_model_fps = {c.category: c.fingerprint
                                for c in stage.required_components}

        # Move intermediate tensors to this GPU if they were produced on a different one
        self._migrate_tensors_to_device(job)

        if stage.type == StageType.GPU_TEXT_ENCODE:
            from handlers.sdxl import text_encode
            text_encode(job, self._gpu)
            return None

        elif stage.type == StageType.GPU_DENOISE:
            from handlers.sdxl import denoise
            denoise(job, self._gpu)
            return None

        elif stage.type == StageType.GPU_VAE_DECODE:
            from handlers.sdxl import vae_decode
            return vae_decode(job, self._gpu)

        elif stage.type == StageType.GPU_VAE_ENCODE:
            from handlers.sdxl import vae_encode
            vae_encode(job, self._gpu)
            # After encoding upscaled image back to latents, mark as hires pass
            # so the subsequent denoise uses hires parameters (strength, steps).
            if job.hires_input is not None:
                job.is_hires_pass = True
            return None

        elif stage.type == StageType.GPU_UPSCALE:
            from handlers.upscale import execute
            return execute(job, self._gpu)

        elif stage.type == StageType.GPU_BGREMOVE:
            from handlers.bgremove import execute
            return execute(job, self._gpu)

        else:
            raise RuntimeError(f"GPU worker cannot execute stage type {stage.type}")


# ── Slot estimation (for /api/status) ──────────────────────────────
#
# Reference resolutions for conservative VRAM estimation.
# Using 1536×1024 (common large-format SDXL size) rather than 1024×1024
# to avoid underestimating working memory for typical high-res jobs.
_REF_DENOISE_LATENT_PX = (1536 // 8) * (1024 // 8)  # 24576 latent pixels
_REF_IMAGE_PX = 1536 * 1024                           # 1572864 image pixels


def _vram_available(gpu: GpuInstance) -> int:
    """Return VRAM available for new allocations (NVML free + allocator slack + evictable)."""
    _, _, nvml_free = nvml.get_memory_info(gpu.nvml_handle)
    pt_slack = max(0,
                   torch.cuda.memory_reserved(gpu.device)
                   - torch.cuda.memory_allocated(gpu.device))
    # get_evictable_vram checks active fingerprints internally
    evictable = gpu.get_evictable_vram()
    return nvml_free + pt_slack + evictable


def _unloaded_model_cost(gpu: GpuInstance, categories: list[str]) -> int:
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
    loaded_cats = {m['category'] for m in gpu.get_cached_models_info()}
    return sum(
        _model_vram.get(cat, 500 * 1024**2)
        for cat in categories
        if cat not in loaded_cats
    )


def _working_memory_cost(stage_type: StageType, ref_pixels: int,
                         gpu_model: str | None = None) -> int:
    """Estimate working memory for slot estimation.

    Uses workspace profiler → BPP → hardcoded fallback.
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


def estimate_gpu_slots(worker: GpuWorker) -> dict[str, int]:
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

    # Use cached GPU model name (avoids CUDA driver call on every scheduling cycle)
    gpu_model = worker._gpu_model_name

    slots: dict[str, int] = {}

    # ── SDXL slot ──
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

    # ── Simple-task slots ──
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
