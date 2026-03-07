"""Main-process proxy for a GPU worker subprocess.

Presents the same interface as the old GpuWorker so the scheduler,
TRT manager, and status APIs work without changes.  Backed by IPC
queues to the actual GPU worker process.

Each GPU runs one job at a time (entire pipeline start-to-finish).
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time as _time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

import log
from config import GpuConfig
from scheduling.job import (
    InferenceJob, JobResult, StageType,
)
from scheduling.worker_protocol import (
    DrainCmd,
    DrainComplete,
    ExecuteJobCmd,
    GetStatusCmd,
    JobComplete,
    LogMessage,
    OnloadCmd,
    OnloadComplete,
    ProcessError,
    ProgressUpdate,
    ReleaseDrainCmd,
    ShutdownCmd,
    StatusSnapshot,
    TagImageCmd,
    TagResult,
    TrtBuildCmd,
    TrtBuildProgress,
    TrtBuildResult,
    UpdateLoraIndexCmd,
    UpdateSdxlModelsCmd,
    WorkerReady,
)


# Global measured model VRAM — aggregated from all workers' StatusSnapshots.
# Once ANY worker loads a model and measures its VRAM, this value is available
# for budget calculations across all workers.  fingerprint -> actual bytes.
_global_measured_vram: dict[str, int] = {}

# Global category → max measured VRAM — tracks the largest observed VRAM
# per model category (e.g. "sdxl_unet", "sdxl_te1") across all workers.
_global_category_vram: dict[str, int] = {}


def get_measured_model_vram(fingerprint: str) -> int | None:
    """Look up the measured VRAM for a model component from any worker."""
    return _global_measured_vram.get(fingerprint)


def get_measured_category_vram(category: str) -> int | None:
    """Look up the max measured VRAM for any model of the given category."""
    return _global_category_vram.get(category)


class GpuProxy:
    """Lightweight read-only GPU info object for the scheduler.

    Replaces direct GpuInstance access in the main process.
    Updated from StatusSnapshot messages from the worker.
    """

    def __init__(self, uuid: str, name: str, capabilities: set[str],
                 total_memory: int, device_id: int):
        self.uuid = uuid
        self.name = name
        self.capabilities = set(capabilities)
        self.device_id = device_id
        self.total_memory = total_memory

        # Onload/unevictable config (for status reporting)
        self.onload: set[str] = set()
        self.unevictable: set[str] = set()

        # Status from worker (updated by proxy._reader_loop)
        self._status: StatusSnapshot | None = None

        # NVML handle for VRAM queries from main process
        self.nvml_handle = None

        # Failure state (can be set from main process too)
        self._failed = False
        self._fail_reason: str = ""
        self._consecutive_failures = 0

        # Busy tracking (maintained by proxy)
        self._busy = False

        # Reference to pool's GpuInstance for failure state sync.
        # Set by main.py after construction so AdmissionControl/queue see failures.
        self._pool_gpu = None

    def _update(self, status: StatusSnapshot) -> None:
        """Update from a fresh StatusSnapshot."""
        self._status = status
        # Failure is monotonic: once set by mark_failed() (proxy-side or
        # subprocess-side), never cleared by a stale snapshot from the worker
        # that hasn't yet observed its own failure.
        if status.is_failed:
            self._failed = True
            self._fail_reason = status.fail_reason
        elif not self._failed:
            self._fail_reason = status.fail_reason

    @property
    def is_failed(self) -> bool:
        return self._failed

    @property
    def is_busy(self) -> bool:
        return self._busy

    def mark_failed(self, reason: str) -> None:
        if not self._failed:
            self._failed = True
            self._fail_reason = reason
            log.error(f"  GPU [{self.uuid}]: PERMANENTLY FAILED — {reason}")
            # Sync to pool's GpuInstance so AdmissionControl sees the failure.
            # Set _failed directly to avoid a duplicate log.error from
            # GpuInstance.mark_failed().
            if self._pool_gpu is not None:
                self._pool_gpu._failed = True
                self._pool_gpu._fail_reason = reason

    def record_success(self) -> None:
        self._consecutive_failures = 0

    def record_failure(self) -> bool:
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self.mark_failed(f"{self._consecutive_failures} consecutive job failures")
            return True
        return False

    def supports_capability(self, cap: str) -> bool:
        return cap.lower() in self.capabilities

    def is_component_loaded(self, fingerprint: str) -> bool:
        if self._status is None:
            return False
        return fingerprint in self._status.cached_fingerprints

    def get_cached_categories(self) -> list[str]:
        if self._status is None:
            return []
        return list(self._status.cached_categories)

    def get_cached_models_info(self) -> list[dict]:
        if self._status is None:
            return []
        return list(self._status.cached_models_info)

    def get_vram_stats(self) -> dict:
        if self._status is None:
            return {}
        return dict(self._status.vram_stats)

    def get_loaded_models_vram(self) -> int:
        if self._status is None:
            return 0
        return self._status.loaded_models_vram

    def get_evictable_vram(self, protect: set[str] | None = None) -> int:
        if self._status is None:
            return 0
        return self._status.evictable_vram

    def get_trt_shared_memory_vram(self) -> int:
        if self._status is None:
            return 0
        return self._status.trt_shared_memory_vram

    def get_trt_freeable_vram(self) -> int:
        if self._status is None:
            return 0
        return self._status.trt_freeable_vram

    @property
    def loaded_lora_count(self) -> int:
        if self._status is None:
            return 0
        return self._status.loaded_lora_count

    def check_trt_affinity(self, model_fp: str, component: str,
                           width: int, height: int) -> int:
        """Check TRT engine affinity for a given model/resolution.

        Returns:
            +2 if the GPU has a TRT engine that exactly matches (static)
            +1 if the GPU has a dynamic TRT engine that covers the resolution
             0 if the GPU has no TRT engines for this model/component
            -1 if the GPU has TRT engine(s) that DON'T cover the resolution
               (loading will evict them — wasteful)
        """
        if self._status is None:
            return 0

        prefix = f"{model_fp}:{component}_trt:"
        cached_trt = [
            fp[len(prefix):] for fp in self._status.cached_fingerprints
            if fp.startswith(prefix)
        ]

        if not cached_trt:
            return 0

        from state import app_state
        dynamic_only = app_state.config.tensorrt.dynamic_only

        # Check for exact static match (skip if dynamic_only)
        if not dynamic_only:
            static_key = f"{width}x{height}"
            if static_key in cached_trt:
                return 2

        # Check dynamic engines
        from trt.builder import DYNAMIC_PROFILES
        has_relevant = False
        for label_suffix in cached_trt:
            # Skip stale static-resolution cache entries under dynamic_only
            if dynamic_only and "x" in label_suffix and label_suffix.replace("x", "").isdigit():
                continue
            has_relevant = True
            for profile in DYNAMIC_PROFILES:
                if profile["label"] != label_suffix:
                    continue
                min_w, min_h = profile["min"]
                max_w, max_h = profile["max"]
                if min_w <= width <= max_w and min_h <= height <= max_h:
                    return 1

        # Has TRT engines but none cover this resolution
        return -1 if has_relevant else 0

    def get_active_fingerprints(self) -> set[str]:
        # In the proxy, active fingerprints are tracked locally
        return set()

    @property
    def session_cache_count(self) -> int:
        if self._status is None:
            return 0
        return len(self._status.cached_fingerprints)


class GpuWorkerProxy:
    """Main-process proxy that replaces GpuWorker.

    Each GPU runs one job at a time (entire pipeline start-to-finish).
    No concurrent stages, no cross-GPU dispatch.
    """

    def __init__(
        self,
        process: mp.Process,
        cmd_queue: mp.Queue,
        result_queue: mp.Queue,
        gpu_config: GpuConfig,
        gpu_total_memory: int,
        gpu_index: int,
    ):
        self._process = process
        self._cmd_queue = cmd_queue
        self._result_queue = result_queue
        self._gpu_index = gpu_index

        # Create the lightweight GPU proxy object
        self._gpu_proxy = GpuProxy(
            uuid=gpu_config.uuid,
            name=gpu_config.name,
            capabilities=gpu_config.capabilities,
            total_memory=gpu_total_memory,
            device_id=gpu_index,
        )
        self._gpu_proxy.onload = gpu_config.onload
        self._gpu_proxy.unevictable = gpu_config.unevictable

        # Reader infrastructure
        self._reader_pool = ThreadPoolExecutor(max_workers=1,
                                                thread_name_prefix=f"gpu-reader-{gpu_index}")
        self._reader_task: asyncio.Task | None = None
        self._loop: asyncio.AbstractEventLoop | None = None

        # State tracking — one job at a time
        self._active_job: InferenceJob | None = None

        # TRT drain state
        self._draining = False
        self._building = False
        self._drain_event: asyncio.Event | None = None
        self._drain_future: asyncio.Future | None = None

        # TRT build state
        self._trt_build_future: asyncio.Future | None = None
        # TRT build info for TUI display (set by TrtBuildManager._gpu_build_loop)
        self.trt_build_model: str | None = None      # human-readable model name
        self.trt_build_component: str | None = None   # e.g. "te1", "unet", "vae"
        self.trt_build_engine: str | None = None      # e.g. "static-640x768"

        # Tag state
        self._tag_future: asyncio.Future | None = None

        # Onload state
        self._onload_future: asyncio.Future | None = None

        # Cached GPU model name and arch key (set on WorkerReady)
        self._gpu_model_name: str = ""
        self._arch_key: str = ""

        # Ready state
        self._ready = False
        self._ready_event = asyncio.Event()

        # Scheduler wake reference (set by scheduler)
        self._scheduler_wake: asyncio.Event | None = None

        # Watchdog
        self._last_activity = _time.monotonic()
        self._watchdog_timeout = 600.0  # 10 minutes

        # Dispatch tracking (for scheduler diversification)
        self._last_dispatch_time: float = 0.0  # monotonic timestamp

    @property
    def gpu(self) -> GpuProxy:
        return self._gpu_proxy

    @property
    def is_idle(self) -> bool:
        return self._active_job is None

    @property
    def active_count(self) -> int:
        return 0 if self._active_job is None else 1

    @property
    def active_jobs(self) -> list[InferenceJob]:
        if self._active_job is not None:
            return [self._active_job]
        return []

    def start(self) -> None:
        """Start the reader loop as an asyncio task."""
        self._loop = asyncio.get_running_loop()
        self._reader_task = self._loop.create_task(self._reader_loop())

    async def wait_ready(self, timeout: float = 60.0) -> bool:
        """Wait for the worker process to send WorkerReady."""
        try:
            await asyncio.wait_for(self._ready_event.wait(), timeout=timeout)
            return True
        except asyncio.TimeoutError:
            log.error(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Worker did not become ready "
                      f"within {timeout}s")
            return False

    def get_loaded_categories(self) -> set[str]:
        """Return set of model categories currently loaded on this GPU."""
        cats = set()
        cached = self._gpu_proxy.get_cached_categories()
        has_te1 = "sdxl_te1" in cached
        has_te2 = "sdxl_te2" in cached
        for cat in cached:
            if cat in ("sdxl_te1", "sdxl_te2"):
                continue
            cats.add(cat)
        if has_te1 and has_te2:
            cats.add("sdxl_te")
        return cats

    def can_accept_work(self) -> bool:
        """Whether this worker can accept a new job."""
        if self._gpu_proxy.is_failed:
            return False
        if self._draining or self._building:
            return False
        if not self._ready:
            return False
        if self._active_job is not None:
            return False
        return True

    def check_vram_budget(self, job: InferenceJob) -> bool:
        """Check if this GPU can run the job.

        One GPU = one safetensor model.  Any SDXL model fits on any GPU
        since the worker can evict cached models and the handlers use
        adaptive tiling for VAE.  This only rejects jobs whose required
        capabilities aren't supported.
        """
        return True

    def dispatch(self, job: InferenceJob) -> None:
        """Dispatch an entire job to the worker process."""
        if self._gpu_proxy.is_failed:
            raise RuntimeError(f"GPU [{self._gpu_proxy.uuid}] is permanently failed")

        self._last_dispatch_time = _time.monotonic()

        # Mark busy
        self._gpu_proxy._busy = True
        self._active_job = job

        # Track job state
        if job.started_at is None:
            job.started_at = datetime.utcnow()
        job.assigned_gpu_uuid = self._gpu_proxy.uuid
        job.stage_status = "loading"
        job.active_gpus = [{"uuid": self._gpu_proxy.uuid,
                            "name": self._gpu_proxy.name,
                            "stage": "pipeline"}]

        # Build and send command
        cmd = self._build_execute_cmd(job)
        self._cmd_queue.put(cmd)
        self._last_activity = _time.monotonic()

    def _build_execute_cmd(self, job: InferenceJob) -> ExecuteJobCmd:
        """Serialize job data into an ExecuteJobCmd for cross-process transfer."""
        # Move input latents to CPU for pickling
        input_latents = None
        if job.latents is not None:
            input_latents = job.latents.cpu() if job.latents.device.type != "cpu" else job.latents

        regional_info_data = None
        if job.regional_info is not None:
            regional_info_data = job.regional_info.to_dict()

        return ExecuteJobCmd(
            job_id=job.job_id,
            job_type_value=job.type.value,
            pipeline=job.pipeline,
            sdxl_input=job.sdxl_input,
            hires_input=job.hires_input,
            input_image=job.input_image,
            input_latents=input_latents,
            tokenize_result=job.tokenize_result,
            regional_tokenize_results=job.regional_tokenize_results,
            regional_base_tokenize=job.regional_base_tokenize,
            regional_shared_neg_tokenize=job.regional_shared_neg_tokenize,
            regional_info_data=regional_info_data,
            priority=job.priority,
            oom_retries=job.oom_retries,
            is_hires_pass=job.is_hires_pass,
            unet_tile_width=job.unet_tile_width,
            unet_tile_height=job.unet_tile_height,
            vae_tile_width=job.vae_tile_width,
            vae_tile_height=job.vae_tile_height,
            orig_width=getattr(job, "orig_width", None),
            orig_height=getattr(job, "orig_height", None),
        )

    # ── TRT drain control ─────────────────────────────────────────

    async def request_drain(self) -> None:
        """Request the worker to drain for TRT build.

        Waits for the active job to complete (if any), then sends DrainCmd
        to evict models in the worker process.
        """
        if self._draining or self._building:
            raise RuntimeError(f"GPU [{self._gpu_proxy.uuid}] is already draining/building")

        self._drain_event = asyncio.Event()
        self._draining = True

        active = 1 if self._active_job is not None else 0
        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drain requested — "
                 f"waiting for {active} active job(s)")

        # If already idle, drain is instant
        if self._active_job is None:
            self._drain_event.set()

        await self._drain_event.wait()

        # Now send drain command to worker to evict models
        self._drain_future = self._loop.create_future()
        self._cmd_queue.put(DrainCmd())
        await self._drain_future  # Wait for DrainComplete from worker

        self._building = True
        # Sync to pool's GpuInstance so AdmissionControl sees reduced capacity
        if self._gpu_proxy._pool_gpu is not None:
            self._gpu_proxy._pool_gpu._trt_building = True
        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drained — GPU ready for TRT build")

    async def release_drain(self) -> None:
        """Release drain state, resume normal operation.

        After releasing, re-sends onload commands to restore pre-configured
        models so the GPU is warm and ready for work immediately.
        Drain state (_draining) stays True until onload completes so the
        scheduler doesn't dispatch to a cold GPU.
        """
        self._cmd_queue.put(ReleaseDrainCmd())
        self._building = False
        # Sync to pool's GpuInstance so AdmissionControl restores capacity
        if self._gpu_proxy._pool_gpu is not None:
            self._gpu_proxy._pool_gpu._trt_building = False

        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drain released")

        # Re-load pre-configured models (onload) so the GPU isn't left cold.
        # Drain evicts everything; without this, the first job to hit this GPU
        # pays the full model-loading cost and the GPU sits at a scoring
        # disadvantage vs GPUs that kept their models warm.
        # Keep _draining=True during onload so the scheduler doesn't dispatch
        # to a GPU with no models loaded yet.
        try:
            from state import app_state
            if self._gpu_proxy.onload:
                models_dir = app_state.config.server.models_dir
                await self.send_onload(
                    types={"sdxl", "upscale", "bgremove"},
                    onload_entries=self._gpu_proxy.onload,
                    unevictable_entries=self._gpu_proxy.unevictable,
                    models_dir=models_dir,
                    sdxl_models=app_state.sdxl_models,
                    lora_index=dict(app_state.lora_index),
                    loras_dir=app_state.loras_dir,
                )
                log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                          f"Post-drain onload complete")
        except Exception as ex:
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                        f"Post-drain onload failed: {ex}")
        finally:
            # Only now mark as available for work
            self._draining = False
            self._drain_event = None

        if self._scheduler_wake:
            self._scheduler_wake.set()

    async def trt_build(self, model_hash: str, model_dir: str,
                        cache_dir: str, arch_key: str,
                        max_workspace_gb: float = 0,
                        dynamic_only: bool = False) -> TrtBuildResult:
        """Send a TRT build command and wait for the result."""
        self._trt_build_future = self._loop.create_future()
        self._cmd_queue.put(TrtBuildCmd(
            model_hash=model_hash,
            model_dir=model_dir,
            cache_dir=cache_dir,
            arch_key=arch_key,
            max_workspace_gb=max_workspace_gb,
            dynamic_only=dynamic_only,
        ))
        return await self._trt_build_future

    async def tag_image(self, image, threshold: float = 0.2) -> TagResult:
        """Send a tag command and wait for the result."""
        self._tag_future = self._loop.create_future()
        self._cmd_queue.put(TagImageCmd(image=image, threshold=threshold))
        return await self._tag_future

    async def send_onload(self, types: set[str], onload_entries: set[str],
                          unevictable_entries: set[str], models_dir: str,
                          sdxl_models: dict[str, str],
                          lora_index: dict | None = None,
                          loras_dir: str | None = None) -> None:
        """Send onload command and wait for completion."""
        self._onload_future = self._loop.create_future()
        self._cmd_queue.put(OnloadCmd(
            types=types,
            onload_entries=onload_entries,
            unevictable_entries=unevictable_entries,
            models_dir=models_dir,
            sdxl_models=sdxl_models,
            lora_index=lora_index,
            loras_dir=loras_dir,
        ))
        await self._onload_future

    @staticmethod
    def _merge_bpp(bpp_data: dict[str, float]) -> None:
        """Merge a worker's BPP measurements into the main process's tracking.

        Uses damped-max logic matching the worker's approach: new peaks are
        tracked immediately, but when a worker reports a significantly lower
        value than the stored max, the stored value decays toward it.  This
        prevents permanent inflation from outlier measurements.
        """
        from scheduling.worker import _measured_bpp, _bpp_lock
        with _bpp_lock:
            for stage_val, bpp in bpp_data.items():
                try:
                    st = StageType(stage_val)
                    prev = _measured_bpp.get(st, 0.0)
                    if bpp >= prev:
                        _measured_bpp[st] = bpp
                    elif prev > 0 and bpp < prev * 0.5:
                        # Worker has decayed its measurement — follow it down
                        _measured_bpp[st] = max(bpp * 1.5, prev * 0.5)
                except ValueError:
                    pass

    def update_lora_index(self, lora_index: dict) -> None:
        """Push an updated LoRA index to the worker process (fire-and-forget)."""
        self._cmd_queue.put(UpdateLoraIndexCmd(lora_index=lora_index))

    def update_sdxl_models(self, sdxl_models: dict[str, str]) -> None:
        """Push an updated SDXL model map to the worker process (fire-and-forget)."""
        self._cmd_queue.put(UpdateSdxlModelsCmd(sdxl_models=sdxl_models))

    def request_status(self) -> None:
        """Request a fresh status snapshot from the worker."""
        self._cmd_queue.put(GetStatusCmd())

    # ── Reader loop ───────────────────────────────────────────────

    async def _reader_loop(self) -> None:
        """Continuously read results from the worker process."""
        import queue as _queue_mod
        loop = asyncio.get_running_loop()
        log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Reader loop started")

        def _get_with_timeout():
            """Blocking get with timeout so we can check process liveness."""
            while True:
                try:
                    return self._result_queue.get(timeout=5.0)
                except _queue_mod.Empty:
                    if not self._process.is_alive():
                        raise EOFError("Worker process is no longer alive")
                    continue

        try:
            while True:
                try:
                    msg = await loop.run_in_executor(
                        self._reader_pool, _get_with_timeout)
                except (EOFError, OSError):
                    log.error(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                              f"Worker process connection lost")
                    self._handle_process_death()
                    break

                self._last_activity = _time.monotonic()

                if isinstance(msg, LogMessage):
                    try:
                        level = log.LogLevel(msg.level)
                    except ValueError:
                        level = log.LogLevel.INFO
                    log.write_line(msg.message, level)
                    continue

                elif isinstance(msg, WorkerReady):
                    self._gpu_model_name = msg.gpu_model_name
                    self._arch_key = msg.arch_key
                    self._ready = True
                    self._ready_event.set()
                    log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Worker ready "
                             f"(arch={msg.arch_key})")

                elif isinstance(msg, JobComplete):
                    self._handle_job_complete(msg)

                elif isinstance(msg, StatusSnapshot):
                    self._gpu_proxy._update(msg)
                    # Merge measured model VRAM into global registry
                    if msg.fingerprint_vram:
                        _global_measured_vram.update(msg.fingerprint_vram)
                    # Build category → max VRAM mapping from model info
                    for info in msg.cached_models_info:
                        cat = info.get("category", "")
                        vram = info.get("vram", 0)
                        if cat and vram > 0:
                            prev = _global_category_vram.get(cat, 0)
                            if vram > prev:
                                _global_category_vram[cat] = vram
                    # Propagate BPP measurements to the main process
                    if msg.measured_bpp:
                        self._merge_bpp(msg.measured_bpp)

                elif isinstance(msg, ProgressUpdate):
                    self._handle_progress(msg)

                elif isinstance(msg, DrainComplete):
                    if self._drain_future and not self._drain_future.done():
                        self._drain_future.set_result(True)

                elif isinstance(msg, TrtBuildProgress):
                    self.trt_build_component = msg.component
                    self.trt_build_engine = msg.engine

                elif isinstance(msg, TrtBuildResult):
                    if self._trt_build_future and not self._trt_build_future.done():
                        self._trt_build_future.set_result(msg)

                elif isinstance(msg, TagResult):
                    if self._tag_future and not self._tag_future.done():
                        self._tag_future.set_result(msg)

                elif isinstance(msg, OnloadComplete):
                    if self._onload_future and not self._onload_future.done():
                        self._onload_future.set_result(True)

                elif isinstance(msg, ProcessError):
                    log.error(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                              f"Worker error: {msg.error}")
                    if msg.fatal:
                        self._gpu_proxy.mark_failed(msg.error)
                    # Fail any pending futures
                    self._fail_pending_futures(msg.error)

        except asyncio.CancelledError:
            log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Reader loop cancelled")

    def _handle_job_complete(self, msg: JobComplete) -> None:
        """Process a job completion from the worker."""
        job = self._active_job
        if job is None or job.job_id != msg.job_id:
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                        f"Received result for unknown job {msg.job_id}")
            return

        # Update job timing
        job.model_load_time_s += msg.model_load_time_s
        job.gpu_time_s += msg.gpu_time_s
        job.gpu_stage_times.extend(msg.stage_times)

        # Mark GPU state BEFORE releasing (so scheduler sees correct state)
        if msg.fatal:
            self._gpu_proxy.mark_failed("CUDA context corrupted")
        elif not msg.oom and not msg.success:
            self._gpu_proxy.record_failure()
        elif msg.success:
            self._gpu_proxy.record_success()

        # Release the GPU (wakes scheduler)
        self._release_job(job)

        if msg.oom:
            # OOM — re-enqueue to a different GPU
            job.oom_retries += 1
            job.oom_gpu_ids.add(self._gpu_proxy.uuid)
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: {job} OOM "
                        f"— re-enqueuing (retry {job.oom_retries})")
            from state import app_state
            app_state.queue.re_enqueue(job)

        elif msg.fatal:
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=msg.error))
            self._broadcast_complete(job, success=False, error=msg.error)

        elif not msg.success:
            error_msg = msg.error or "Unknown worker error"
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=error_msg))
            self._broadcast_complete(job, success=False, error=error_msg)

        else:
            # Job completed successfully
            result = JobResult(
                success=True,
                output_image=msg.output_image,
                output_latents=msg.output_latents,
            )
            self._store_job_result(job, result)
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(result)
            self._broadcast_complete(job, success=True)
            if msg.clip_cache is not None:
                self._broadcast_clip_cache(msg.clip_cache)
            if msg.latent_cache is not None:
                self._broadcast_latent_cache(msg.latent_cache)
            log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: {job} completed")

    def _handle_progress(self, msg: ProgressUpdate) -> None:
        """Update job progress from worker and broadcast to WebSocket clients."""
        job = self._active_job
        if job is not None and job.job_id == msg.job_id:
            job.denoise_step = msg.denoise_step
            job.denoise_total_steps = msg.denoise_total_steps
            job.stage_step = msg.stage_step
            job.stage_total_steps = msg.stage_total_steps
            job.current_stage_index = msg.stage_index

            # Broadcast to WebSocket clients
            try:
                from api.websocket import streamer
                if self._loop is not None:
                    asyncio.run_coroutine_threadsafe(
                        streamer.broadcast_progress(job), self._loop)
            except Exception:
                pass

    def _release_job(self, job: InferenceJob) -> None:
        """Release tracking for a completed job."""
        job.stage_status = ""
        job.active_gpus = []
        job.assigned_gpu_uuid = None

        self._active_job = None
        self._gpu_proxy._busy = False

        # Signal drain completion if draining
        if self._draining and self._drain_event is not None:
            self._drain_event.set()

        # Wake scheduler
        if self._scheduler_wake:
            self._scheduler_wake.set()

        # Push fresh status
        try:
            from api.websocket import streamer
            from api.status_snapshot import compute_status_snapshot
            streamer.fire_event("status_update", compute_status_snapshot())
        except Exception:
            pass

    def _store_job_result(self, job: InferenceJob, result: JobResult) -> None:
        """Store result bytes in AppState for queue-based API."""
        try:
            from state import app_state
            import io as _io

            if result.output_image is not None:
                image = result.output_image
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
                buf.write(struct.pack("<I", 0))
                buf.write(float_data)
                app_state.job_results[job.job_id] = (buf.getvalue(), "application/x-fox-latent")
        except Exception as ex:
            log.log_exception(ex, f"GpuWorkerProxy: Failed to store result for {job}")

    def _broadcast_clip_cache(self, cache_data: dict) -> None:
        """Send CLIP embeddings to makefoxsrv for database caching."""
        try:
            import base64
            entries = []
            for e in cache_data["entries"]:
                entries.append({
                    "encoder_type": e.encoder_type,
                    "polarity": e.polarity,
                    "data": base64.b64encode(e.data).decode("ascii"),
                    "dtype": e.dtype,
                    "dim0": e.dim0,
                    "dim1": e.dim1,
                })
            from api.websocket import streamer
            streamer.fire_event("clip_embeddings", {
                "prompt_hash": base64.b64encode(cache_data["prompt_hash"]).decode("ascii"),
                "model": cache_data["model"],
                "entries": entries,
            })
        except Exception as ex:
            log.warning(f"  Failed to broadcast clip_embeddings: {ex}")

    def _broadcast_latent_cache(self, cache_data: dict) -> None:
        """Send denoised latents to makefoxsrv for database caching."""
        try:
            import base64
            from api.websocket import streamer
            streamer.fire_event("latent_cache", {
                "job_id": cache_data["job_id"],
                "data": base64.b64encode(cache_data["data"]).decode("ascii"),
                "dtype": cache_data["dtype"],
                "shape": cache_data["shape"],
            })
        except Exception as ex:
            log.warning(f"  Failed to broadcast latent_cache: {ex}")

    def _broadcast_complete(self, job: InferenceJob, success: bool,
                            error: str | None = None) -> None:
        """Broadcast job completion via WebSocket."""
        try:
            from api.websocket import streamer
            if self._loop is not None:
                asyncio.run_coroutine_threadsafe(
                    streamer.broadcast_complete(job, success, error), self._loop)
        except Exception:
            pass

    def _release_admission(self, job: InferenceJob) -> None:
        """Release the admission control slot."""
        from state import app_state
        if app_state.admission is not None:
            app_state.admission.release(job.type)

    def _fail_pending_futures(self, error: str) -> None:
        """Fail any pending futures on process error."""
        for future_attr in ("_drain_future", "_trt_build_future", "_tag_future", "_onload_future"):
            future = getattr(self, future_attr, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError(error))

        # Fail the active job
        job = self._active_job
        if job is not None:
            self._release_job(job)
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=error))
            self._broadcast_complete(job, success=False, error=error)

    def _handle_process_death(self) -> None:
        """Handle worker process crash."""
        self._gpu_proxy.mark_failed("Worker process died")
        # Reset drain/build flags so they don't stay stuck
        self._draining = False
        self._building = False
        self._drain_event = None
        if self._gpu_proxy._pool_gpu is not None:
            self._gpu_proxy._pool_gpu._trt_building = False
        self._fail_pending_futures("Worker process died")

    def shutdown(self) -> None:
        """Request graceful shutdown of the worker process."""
        try:
            self._cmd_queue.put(ShutdownCmd())
        except Exception:
            pass
        if self._reader_task:
            self._reader_task.cancel()
        self._reader_pool.shutdown(wait=False)

    def log_vram_state(self, context: str) -> None:
        """Log VRAM state from cached status."""
        status = self._gpu_proxy._status
        if status is None:
            return
        vram = status.vram_stats
        log.debug(
            f"  VRAM [{self._gpu_proxy.uuid}] ({context}): "
            f"models={status.loaded_models_vram // (1024**2)}MB, "
            f"evictable={status.evictable_vram // (1024**2)}MB"
        )
