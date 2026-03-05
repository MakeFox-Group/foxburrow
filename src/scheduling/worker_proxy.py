"""Main-process proxy for a GPU worker subprocess.

Presents the same interface as the old GpuWorker so the scheduler,
TRT manager, and status APIs work without changes.  Backed by IPC
queues to the actual GPU worker process.
"""

from __future__ import annotations

import asyncio
import multiprocessing as mp
import os
import time as _time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import TYPE_CHECKING

import log
from config import GpuConfig
from scheduling.job import (
    InferenceJob, JobResult, JobType, StageType, WorkStage,
    SdxlTokenizeResult, SdxlEncodeResult, SdxlRegionalEncodeResult,
)
from scheduling.queue import JobQueue
from scheduling.scheduler import (
    get_session_key, get_session_group, get_session_group_from_key,
    _COMPATIBLE_WITH_ALL,
)
from scheduling.worker_protocol import (
    DrainCmd,
    DrainComplete,
    ExecuteStageCmd,
    GetStatusCmd,
    OnloadCmd,
    OnloadComplete,
    ProcessError,
    ProgressUpdate,
    ReleaseDrainCmd,
    ShutdownCmd,
    StageResult,
    StatusSnapshot,
    TagImageCmd,
    TagResult,
    TrtBuildCmd,
    TrtBuildResult,
    WorkerReady,
)

if TYPE_CHECKING:
    pass

# Per-session concurrency limits (mirrors worker.py constants)
_SESSION_MAX_CONCURRENCY: dict[str, int] = {
    "sdxl_unet": 1,
    "sdxl_vae": 1,
    "sdxl_te": 1,
    "upscale": 1,
    "bgremove": 1,
}

MAX_CONCURRENCY = 4


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

    @property
    def current_group(self) -> str | None:
        if self._status is None:
            return None
        return self._status.session_group

    @property
    def loaded_lora_count(self) -> int:
        if self._status is None:
            return 0
        return self._status.loaded_lora_count

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

    Presents the same interface (gpu, is_idle, active_count, dispatch,
    can_accept_work, etc.) backed by IPC to the worker subprocess.
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

        # State tracking (main process side)
        self._is_busy = False
        self._pending: tuple[WorkStage, InferenceJob] | None = None
        self._active_count = 0
        self._active_session_counts: dict[str, int] = defaultdict(int)
        self._active_stage_counts: dict[StageType, int] = defaultdict(int)
        self._active_jobs: dict[str, InferenceJob] = {}

        # TRT drain state
        self._draining = False
        self._building = False
        self._drain_event: asyncio.Event | None = None
        self._drain_future: asyncio.Future | None = None

        # TRT build state
        self._trt_build_future: asyncio.Future | None = None

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

        # Cross-GPU failure propagation

        # Scheduler wake reference (set by scheduler)
        self._scheduler_wake: asyncio.Event | None = None

        # Watchdog
        self._last_activity = _time.monotonic()
        self._watchdog_timeout = 600.0  # 10 minutes

    @property
    def gpu(self) -> GpuProxy:
        return self._gpu_proxy

    @property
    def is_idle(self) -> bool:
        return self._active_count == 0

    @property
    def active_count(self) -> int:
        return self._active_count

    @property
    def active_jobs(self) -> list[InferenceJob]:
        return list(self._active_jobs.values())

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

    def can_accept_work(self, stage: WorkStage) -> bool:
        """Whether this worker can accept a new work item for the given stage."""
        if self._gpu_proxy.is_failed:
            return False
        if self._draining or self._building:
            return False
        if not self._ready:
            return False
        if stage.required_capability and not self._gpu_proxy.supports_capability(stage.required_capability):
            return False

        session_key = get_session_key(stage.type)
        if session_key is None:
            return False

        if self._active_count >= MAX_CONCURRENCY:
            return False

        # Per-stage-type limit
        if self._active_stage_counts[stage.type] >= 1:
            return False

        max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
        if self._active_session_counts[session_key] >= max_for_session:
            return False

        # Session group conflict
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

    def check_slot_availability(self, session_key: str, group: str) -> tuple[bool, int]:
        """Atomically check capacity + group conflict and return active_count."""
        if self._active_count >= MAX_CONCURRENCY:
            return False, self._active_count
        max_for_session = _SESSION_MAX_CONCURRENCY.get(session_key, 1)
        if self._active_session_counts.get(session_key, 0) >= max_for_session:
            return False, self._active_count
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

    def check_vram_budget(self, stage: WorkStage, job: InferenceJob) -> bool:
        """Check if this GPU has enough VRAM for a new stage.

        When idle, always returns True (model loading handles eviction).
        When busy, uses cached VRAM stats from the worker's StatusSnapshot.
        """
        if self._active_count == 0:
            return True

        status = self._gpu_proxy._status
        if status is None:
            return True  # No status yet, be optimistic

        # Cost: models not loaded + working memory
        model_cost = sum(
            c.estimated_vram_bytes
            for c in stage.required_components
            if not self._gpu_proxy.is_component_loaded(c.fingerprint)
        )

        from scheduling.worker import _get_min_free_vram
        working_cost = _get_min_free_vram(stage.type, job, self._gpu_model_name)
        total_needed = model_cost + working_cost

        # Available: NVML free + evictable
        vram = status.vram_stats
        nvml_free = vram.get("free", 0)
        pt_slack = max(0, vram.get("reserved", 0) - vram.get("allocated", 0))
        evictable = status.evictable_vram

        available = nvml_free + pt_slack + evictable
        return available >= total_needed

    def dispatch(self, stage: WorkStage, job: InferenceJob) -> None:
        """Dispatch a work item to the worker process."""
        if self._gpu_proxy.is_failed:
            raise RuntimeError(f"GPU [{self._gpu_proxy.uuid}] is permanently failed")

        # Update local concurrency tracking
        if self._active_count == 0:
            self._gpu_proxy._busy = True
        self._active_count += 1
        self._active_stage_counts[stage.type] += 1
        key = get_session_key(stage.type)
        if key:
            self._active_session_counts[key] += 1

        # Track active job
        if job.started_at is None:
            job.started_at = datetime.utcnow()
        job.stage_status = "loading"
        job.active_gpus = [{"uuid": self._gpu_proxy.uuid,
                            "name": self._gpu_proxy.name,
                            "stage": stage.type.value}]
        self._active_jobs[job.job_id] = job
        self._pending = (stage, job)

        # Serialize job data into command
        cmd = self._build_execute_cmd(stage, job)
        self._cmd_queue.put(cmd)
        self._last_activity = _time.monotonic()

    def _build_execute_cmd(self, stage: WorkStage, job: InferenceJob) -> ExecuteStageCmd:
        """Serialize job data into an ExecuteStageCmd for cross-process transfer."""
        # Move tensors to CPU for pickling
        encode_tensors = None
        if job.encode_result is not None:
            er = job.encode_result
            encode_tensors = {}
            for name in ("prompt_embeds", "neg_prompt_embeds",
                          "pooled_prompt_embeds", "neg_pooled_prompt_embeds"):
                t = getattr(er, name, None)
                if t is not None:
                    encode_tensors[name] = t.cpu() if t.device.type != "cpu" else t

        regional_tensors = None
        if job.regional_encode_result is not None:
            rer = job.regional_encode_result
            def _cpu(t):
                return t.cpu() if t is not None and t.device.type != "cpu" else t

            regional_tensors = {
                "region_embeds": [_cpu(e) for e in rer.region_embeds],
                "neg_prompt_embeds": _cpu(rer.neg_prompt_embeds),
                "neg_region_embeds": [_cpu(e) for e in rer.neg_region_embeds] if rer.neg_region_embeds else None,
                "pooled_prompt_embeds": _cpu(rer.pooled_prompt_embeds),
                "neg_pooled_prompt_embeds": _cpu(rer.neg_pooled_prompt_embeds),
                "base_embeds": _cpu(rer.base_embeds),
                "base_ratio": rer.base_ratio,
            }

        latents = None
        if job.latents is not None:
            latents = job.latents.cpu() if job.latents.device.type != "cpu" else job.latents

        regional_info_data = None
        if job.regional_info is not None:
            regional_info_data = job.regional_info.to_dict()

        return ExecuteStageCmd(
            job_id=job.job_id,
            job_type_value=job.type.value,
            stage_type=stage.type,
            required_components=list(stage.required_components),
            required_capability=stage.required_capability,
            sdxl_input=job.sdxl_input,
            hires_input=job.hires_input,
            input_image=job.input_image,
            tokenize_result=job.tokenize_result,
            encode_result_tensors=encode_tensors,
            regional_encode_tensors=regional_tensors,
            regional_tokenize_results=job.regional_tokenize_results,
            regional_base_tokenize=job.regional_base_tokenize,
            regional_shared_neg_tokenize=job.regional_shared_neg_tokenize,
            latents=latents,
            regional_info_data=regional_info_data,
            is_hires_pass=job.is_hires_pass,
            oom_retries=job.oom_retries,
            current_stage_index=job.current_stage_index,
            pipeline_length=len(job.pipeline),
            priority=job.priority,
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

        Waits for all active jobs to complete (tracked locally), then
        sends DrainCmd to evict models in the worker process.
        """
        if self._draining or self._building:
            raise RuntimeError(f"GPU [{self._gpu_proxy.uuid}] is already draining/building")

        self._drain_event = asyncio.Event()
        self._draining = True

        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drain requested — "
                 f"waiting for {self._active_count} active job(s)")

        # If already idle, drain is instant
        if self._active_count == 0:
            self._drain_event.set()

        await self._drain_event.wait()

        # Now send drain command to worker to evict models
        self._drain_future = self._loop.create_future()
        self._cmd_queue.put(DrainCmd())
        await self._drain_future  # Wait for DrainComplete from worker

        self._building = True
        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drained — GPU ready for TRT build")

    async def release_drain(self) -> None:
        """Release drain state, resume normal operation."""
        self._cmd_queue.put(ReleaseDrainCmd())
        self._building = False
        self._draining = False
        self._drain_event = None

        log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Drain released")
        if self._scheduler_wake:
            self._scheduler_wake.set()

    async def trt_build(self, model_hash: str, model_dir: str,
                        cache_dir: str, arch_key: str,
                        max_workspace_gb: float = 0) -> TrtBuildResult:
        """Send a TRT build command and wait for the result."""
        self._trt_build_future = self._loop.create_future()
        self._cmd_queue.put(TrtBuildCmd(
            model_hash=model_hash,
            model_dir=model_dir,
            cache_dir=cache_dir,
            arch_key=arch_key,
            max_workspace_gb=max_workspace_gb,
        ))
        return await self._trt_build_future

    async def tag_image(self, image, threshold: float = 0.2) -> TagResult:
        """Send a tag command and wait for the result."""
        self._tag_future = self._loop.create_future()
        self._cmd_queue.put(TagImageCmd(image=image, threshold=threshold))
        return await self._tag_future

    async def send_onload(self, types: set[str], onload_entries: set[str],
                          unevictable_entries: set[str], models_dir: str,
                          sdxl_models: dict[str, str]) -> None:
        """Send onload command and wait for completion."""
        self._onload_future = self._loop.create_future()
        self._cmd_queue.put(OnloadCmd(
            types=types,
            onload_entries=onload_entries,
            unevictable_entries=unevictable_entries,
            models_dir=models_dir,
            sdxl_models=sdxl_models,
        ))
        await self._onload_future

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

                if isinstance(msg, WorkerReady):
                    self._gpu_model_name = msg.gpu_model_name
                    self._arch_key = msg.arch_key
                    self._ready = True
                    self._ready_event.set()
                    log.info(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Worker ready "
                             f"(arch={msg.arch_key})")

                elif isinstance(msg, StageResult):
                    self._handle_stage_result(msg)

                elif isinstance(msg, StatusSnapshot):
                    self._gpu_proxy._update(msg)

                elif isinstance(msg, ProgressUpdate):
                    self._handle_progress(msg)

                elif isinstance(msg, DrainComplete):
                    if self._drain_future and not self._drain_future.done():
                        self._drain_future.set_result(True)

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
                        # No cross-GPU propagation: each worker has an isolated
                        # CUDA context (separate process + CUDA_VISIBLE_DEVICES),
                        # so one GPU's fatal error cannot corrupt another's.
                    # Fail any pending futures
                    self._fail_pending_futures(msg.error)

        except asyncio.CancelledError:
            log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: Reader loop cancelled")

    def _handle_stage_result(self, msg: StageResult) -> None:
        """Process a stage result from the worker."""
        job = self._active_jobs.get(msg.job_id)
        if job is None:
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                        f"Received result for unknown job {msg.job_id}")
            return

        stage = job.current_stage
        if stage is None:
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: "
                        f"Job {msg.job_id} has no current stage")
            return

        # Update job timing
        job.model_load_time_s += msg.model_load_time_s
        job.gpu_time_s += msg.gpu_time_s
        job.gpu_stage_times.extend(msg.gpu_stage_times)
        job.denoise_step = msg.denoise_step
        job.denoise_total_steps = msg.denoise_total_steps
        job.is_hires_pass = msg.is_hires_pass

        # Restore tensors to job
        if msg.encode_result_tensors:
            job.encode_result = SdxlEncodeResult(
                prompt_embeds=msg.encode_result_tensors.get("prompt_embeds"),
                neg_prompt_embeds=msg.encode_result_tensors.get("neg_prompt_embeds"),
                pooled_prompt_embeds=msg.encode_result_tensors.get("pooled_prompt_embeds"),
                neg_pooled_prompt_embeds=msg.encode_result_tensors.get("neg_pooled_prompt_embeds"),
            )
        if msg.regional_encode_tensors:
            job.regional_encode_result = SdxlRegionalEncodeResult(
                region_embeds=msg.regional_encode_tensors.get("region_embeds", []),
                neg_prompt_embeds=msg.regional_encode_tensors.get("neg_prompt_embeds"),
                neg_region_embeds=msg.regional_encode_tensors.get("neg_region_embeds"),
                pooled_prompt_embeds=msg.regional_encode_tensors.get("pooled_prompt_embeds"),
                neg_pooled_prompt_embeds=msg.regional_encode_tensors.get("neg_pooled_prompt_embeds"),
                base_embeds=msg.regional_encode_tensors.get("base_embeds"),
                base_ratio=msg.regional_encode_tensors.get("base_ratio", 0.2),
            )
        if msg.latents is not None:
            job.latents = msg.latents
        if msg.tokenize_result is not None:
            job.tokenize_result = msg.tokenize_result
        if msg.regional_tokenize_results is not None:
            job.regional_tokenize_results = msg.regional_tokenize_results
        if msg.regional_base_tokenize is not None:
            job.regional_base_tokenize = msg.regional_base_tokenize
        if msg.regional_shared_neg_tokenize is not None:
            job.regional_shared_neg_tokenize = msg.regional_shared_neg_tokenize
        if msg.regional_info_data is not None:
            from utils.regional import RegionalPromptResult
            job.regional_info = RegionalPromptResult.from_dict(msg.regional_info_data)

        # Update intermediate image
        if msg.output_image is not None and msg.success:
            if msg.current_stage_index < len(job.pipeline):
                job.input_image = msg.output_image

        # Release concurrency tracking
        self._release_job(job, stage)

        if msg.oom:
            # OOM — re-enqueue
            job.oom_retries += 1
            job.oom_gpu_ids.add(self._gpu_proxy.uuid)
            log.warning(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: {job} OOM "
                        f"— re-enqueuing (retry {job.oom_retries})")
            from state import app_state
            app_state.queue.re_enqueue(job)

        elif msg.fatal:
            # Fatal CUDA error — only this GPU is affected (isolated process)
            self._gpu_proxy.mark_failed(f"CUDA context corrupted")
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=msg.error))
            self._broadcast_complete(job, success=False, error=msg.error)

        elif not msg.success:
            # Non-fatal error
            self._gpu_proxy.record_failure()
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=msg.error))
            self._broadcast_complete(job, success=False, error=msg.error)

        elif msg.current_stage_index >= len(job.pipeline):
            # Job complete
            self._gpu_proxy.record_success()
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
            log.debug(f"  GpuWorkerProxy[{self._gpu_proxy.uuid}]: {job} completed")

        else:
            # More stages — advance and re-enqueue
            self._gpu_proxy.record_success()
            job.current_stage_index = msg.current_stage_index
            from state import app_state
            app_state.queue.re_enqueue(job)

    def _handle_progress(self, msg: ProgressUpdate) -> None:
        """Update job progress from worker."""
        job = self._active_jobs.get(msg.job_id)
        if job is not None:
            job.denoise_step = msg.denoise_step
            job.denoise_total_steps = msg.denoise_total_steps
            job.stage_step = msg.stage_step
            job.stage_total_steps = msg.stage_total_steps

    def _release_job(self, job: InferenceJob, stage: WorkStage) -> None:
        """Release concurrency tracking for a completed stage."""
        job.stage_status = ""
        job.active_gpus = []
        self._active_jobs.pop(job.job_id, None)

        self._active_count -= 1
        self._active_stage_counts[stage.type] = max(
            0, self._active_stage_counts[stage.type] - 1)
        key = get_session_key(stage.type)
        if key:
            self._active_session_counts[key] = max(
                0, self._active_session_counts[key] - 1)

        if self._active_count == 0:
            self._gpu_proxy._busy = False
            # Signal drain completion if draining
            if self._draining and self._drain_event is not None:
                self._drain_event.set()

        self._pending = None

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
        for future_attr in ("_trt_build_future", "_tag_future", "_onload_future"):
            future = getattr(self, future_attr, None)
            if future is not None and not future.done():
                future.set_exception(RuntimeError(error))

        # Fail all active jobs
        for job_id, job in list(self._active_jobs.items()):
            stage = job.current_stage
            if stage:
                self._release_job(job, stage)
            job.completed_at = datetime.utcnow()
            self._release_admission(job)
            job.set_result(JobResult(success=False, error=error))
            self._broadcast_complete(job, success=False, error=error)

    def _handle_process_death(self) -> None:
        """Handle worker process crash."""
        self._gpu_proxy.mark_failed("Worker process died")
        self._fail_pending_futures("Worker process died")
        if self._drain_event is not None and not self._drain_event.is_set():
            self._drain_event.set()

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
