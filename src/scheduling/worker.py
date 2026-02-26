"""Per-GPU async worker with model loading and stage execution."""

from __future__ import annotations

import asyncio
import os
import threading
from collections import defaultdict
from datetime import datetime
from typing import TYPE_CHECKING

import torch

import log
from gpu.pool import GpuInstance
from scheduling.job import (
    InferenceJob, JobResult, JobType, StageType, WorkStage,
)
from scheduling.queue import JobQueue
from scheduling.scheduler import get_session_key, get_session_group, get_session_group_from_key

if TYPE_CHECKING:
    pass

# Per-session concurrency limits
_SESSION_MAX_CONCURRENCY: dict[str, int] = {
    "sdxl_unet": 1,
    "sdxl_vae": 2,
    "sdxl_te": 1,
    "sdxl_hires_xform": 1,
    "upscale": 1,
    "bgremove": 1,
}

MAX_CONCURRENCY = 4

# Minimum free VRAM before inference (bytes).
# These are the minimum NVML-reported free bytes required before starting a stage.
# Keep values modest — the model weights are already loaded/cached, so this is
# just working memory for intermediate tensors (activations, attention, etc.).
# PyTorch's caching allocator is cleared before each check (empty_cache), so
# these thresholds see real free VRAM, not inflated reserved values.
_MIN_FREE_VRAM: dict[StageType, int] = {
    StageType.GPU_TEXT_ENCODE: 256 * 1024**2,        # ~256 MB for text encoding
    StageType.GPU_DENOISE: 3 * 1024**3,              # ~3 GB working memory for UNet activations
    StageType.GPU_VAE_DECODE: 1 * 1024**3,           # ~1 GB for VAE decode
    StageType.GPU_VAE_ENCODE: 1 * 1024**3,           # ~1 GB for VAE encode (tiled)
    StageType.GPU_HIRES_TRANSFORM: 3 * 1024**3,      # ~3 GB for hires (VAE+upscale+VAE)
    StageType.GPU_UPSCALE: 512 * 1024**2,            # ~512 MB for RealESRGAN upscale
    StageType.GPU_BGREMOVE: 512 * 1024**2,           # ~512 MB for background removal
}


class GpuWorker:
    """Per-GPU background worker. Receives work items, loads models, executes stages."""

    def __init__(self, gpu: GpuInstance, queue: JobQueue, scheduler_wake: asyncio.Event):
        self._gpu = gpu
        self._queue = queue
        self._scheduler_wake = scheduler_wake
        self._work_queue: asyncio.Queue[tuple[WorkStage, InferenceJob]] = asyncio.Queue()
        self._task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None  # set in start()

        # Concurrency tracking
        self._state_lock = threading.Lock()
        self._active_count = 0
        self._active_session_counts: dict[str, int] = defaultdict(int)
        self._active_jobs: dict[str, InferenceJob] = {}

        # Model loading serialization
        self._model_load_lock = asyncio.Lock()

    @property
    def gpu(self) -> GpuInstance:
        return self._gpu

    @property
    def is_idle(self) -> bool:
        return self._active_count == 0

    @property
    def active_count(self) -> int:
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
        if stage.required_capability and not self._gpu.supports_capability(stage.required_capability):
            return False

        session_key = get_session_key(stage.type)
        if session_key is None:
            return False

        with self._state_lock:
            if self._active_count >= MAX_CONCURRENCY:
                return False

            max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
            if self._active_session_counts[session_key] >= max_for_session:
                return False

            # Session group must match all active items
            if self._active_count > 0:
                new_group = get_session_group(stage.type)
                for key, count in self._active_session_counts.items():
                    if count <= 0:
                        continue
                    active_group = get_session_group_from_key(key)
                    if active_group and active_group != new_group:
                        return False

            return True

    def dispatch(self, stage: WorkStage, job: InferenceJob) -> None:
        """Dispatch a work item to this worker."""
        with self._state_lock:
            if self._active_count == 0:
                if not self._gpu.try_acquire():
                    raise RuntimeError(f"GPU [{self._gpu.uuid}] semaphore unexpectedly held")
            self._active_count += 1
            key = get_session_key(stage.type)
            if key:
                self._active_session_counts[key] += 1

        self._work_queue.put_nowait((stage, job))

    async def _run_loop(self) -> None:
        log.info(f"  GpuWorker[{self._gpu.uuid}]: Started")
        try:
            while True:
                stage, job = await self._work_queue.get()
                # Spawn concurrent — don't await
                self._loop.create_task(self._process_work_item(stage, job))
        except asyncio.CancelledError:
            log.info(f"  GpuWorker[{self._gpu.uuid}]: Stopped")

    async def _process_work_item(self, stage: WorkStage, job: InferenceJob) -> None:
        try:
            log.info(f"  GpuWorker[{self._gpu.uuid}]: Processing {job} at {stage} "
                     f"(active={self._active_count})")

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

                min_free = _MIN_FREE_VRAM.get(stage.type, 0)
                if min_free > 0:
                    protect = self._gpu.get_active_fingerprints()
                    self._gpu.ensure_free_vram(min_free, protect)
            load_duration = _time.monotonic() - load_start
            job.gpu_time_s += load_duration

            # Execute stage
            try:
                if job.started_at is None:
                    job.started_at = datetime.utcnow()
                with self._state_lock:
                    self._active_jobs[job.job_id] = job

                # Track GPU for this stage
                gpu_info = {"uuid": self._gpu.uuid, "name": self._gpu.name,
                            "stage": stage.type.value}
                job.active_gpus = [gpu_info]

                # Measure actual GPU execution time (model loading already measured above)
                import time as _time
                stage_start = _time.monotonic()

                try:
                    output_image = await self._loop.run_in_executor(
                        None, self._execute_stage, job, stage)
                finally:
                    # Release working memory (UNet activations, attention matrices, etc.)
                    # back to the OS immediately. This MUST run even on OOM — otherwise
                    # PyTorch's caching allocator keeps the failed allocation's blocks
                    # reserved, causing cascading OOM on all subsequent jobs.
                    # Target THIS worker's GPU — not the default device (cuda:0).
                    with torch.cuda.device(self._gpu.device):
                        torch.cuda.empty_cache()

                stage_duration = _time.monotonic() - stage_start
                job.gpu_time_s += stage_duration

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
                            log.info(f"  GpuWorker[{self._gpu.uuid}]: Resized intermediate "
                                     f"output to {tw}x{th}")

                job.current_stage_index += 1

                if job.is_complete:
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
                    log.info(f"  GpuWorker[{self._gpu.uuid}]: {job} completed")
                else:
                    with self._state_lock:
                        self._active_jobs.pop(job.job_id, None)
                    self._queue.re_enqueue(job)

            except Exception as ex:
                with self._state_lock:
                    self._active_jobs.pop(job.job_id, None)
                job.active_gpus = []
                log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: {job} failed at {stage}")
                job.completed_at = datetime.utcnow()
                job.set_result(JobResult(success=False, error=str(ex)))
                self._broadcast_complete(job, success=False, error=str(ex))

        except Exception as ex:
            log.log_exception(ex, f"GpuWorker[{self._gpu.uuid}]: Batch failed at {stage}")
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
                buf = _io.BytesIO()
                mode = "RGBA" if result.output_image.mode == "RGBA" else "RGB"
                if mode == "RGBA":
                    result.output_image.save(buf, format="PNG")
                else:
                    result.output_image.convert("RGB").save(buf, format="PNG")
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
                          StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE,
                          StageType.GPU_HIRES_TRANSFORM):
            self._gpu.ensure_session_group("sdxl")
            self._load_sdxl_components(stage, model_dir)
            if stage.type == StageType.GPU_HIRES_TRANSFORM:
                self._load_upscale_model()
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
            model = load_component(component.category, model_dir, self._gpu.device)
            after = torch.cuda.memory_allocated(self._gpu.device)
            actual_vram = after - before
            log.info(f"  Loaded {component.category}: {actual_vram // (1024*1024)}MB actual "
                     f"(estimated {component.estimated_vram_bytes // (1024*1024)}MB)")
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
        model = load_model(self._gpu.device)
        after = torch.cuda.memory_allocated(self._gpu.device)
        actual_vram = after - before
        log.info(f"  Loaded upscale: {actual_vram // (1024*1024)}MB actual"
                 + (f" (estimated {comp.estimated_vram_bytes // (1024*1024)}MB)" if comp else ""))
        if comp:
            self._gpu.cache_model(comp.fingerprint, "upscale", model, comp.estimated_vram_bytes,
                                  source="realesrgan")

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
        model = load_model(self._gpu.device)
        after = torch.cuda.memory_allocated(self._gpu.device)
        actual_vram = after - before
        log.info(f"  Loaded bgremove: {actual_vram // (1024*1024)}MB actual"
                 + (f" (estimated {comp.estimated_vram_bytes // (1024*1024)}MB)" if comp else ""))
        if comp:
            self._gpu.cache_model(comp.fingerprint, "bgremove", model, comp.estimated_vram_bytes,
                                  source="rmbg")

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
        from PIL import Image

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
            return None

        elif stage.type == StageType.GPU_HIRES_TRANSFORM:
            from handlers.sdxl import hires_transform
            hires_transform(job, self._gpu)
            return None

        elif stage.type == StageType.GPU_UPSCALE:
            from handlers.upscale import execute
            return execute(job, self._gpu)

        elif stage.type == StageType.GPU_BGREMOVE:
            from handlers.bgremove import execute
            return execute(job, self._gpu)

        else:
            raise RuntimeError(f"GPU worker cannot execute stage type {stage.type}")
