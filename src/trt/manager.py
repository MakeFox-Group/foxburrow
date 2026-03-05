"""Background TRT build manager — pipelined architecture.

Scans registered SDXL models, exports missing ONNX files, and builds
TRT engines on drained GPUs.  Runs as a set of asyncio tasks alongside
the scheduler, coordinating GPU access through the drain mechanism.

Three-layer pipeline:
  Layer 1 — ONNX export coordinator (CPU-bound, parallelized via thread pool)
  Layer 2 — Per-architecture coordinators (drain GPUs on-demand as ONNX completes)
  Layer 3 — Per-GPU build loops (pull from arch queue, build engines, linger, release)

Key improvement over the old two-phase batch approach:
  - ONNX exports feed directly into per-architecture build queues
  - GPUs drain on-demand only when work arrives for their architecture
  - Remaining GPUs stay available for inference throughout the build cycle
"""

from __future__ import annotations

import asyncio
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import torch

import log
from trt.builder import (
    all_engines_exist,
    get_arch_key,
    get_onnx_path,
)
from trt.exporter import export_te1_onnx, export_te2_onnx, export_unet_onnx, export_vae_onnx

if TYPE_CHECKING:
    from scheduling.worker_proxy import GpuWorkerProxy as GpuWorker

# Seconds a GPU lingers after finishing a build, waiting for more work
# before releasing the drain.  Avoids repeated drain/release cycles when
# ONNX exports complete in quick succession.
_BUILD_LINGER_TIMEOUT = 5.0

# Seconds between arch coordinator drain-escalation checks.
_ARCH_CHECK_INTERVAL = 3.0


class TrtBuildManager:
    """Manages background ONNX export and TRT engine compilation.

    On startup, scans all registered SDXL models and queues builds for
    any missing engines.  When new models are registered at runtime (via
    filesystem watcher), they can be queued via ``queue_model()``.

    Pipeline flow:
    1. ``queue_model()`` / ``scan_and_queue()`` → ``_export_queue``
    2. ``_export_coordinator`` pulls from queue, exports ONNX in parallel
    3. Completed ONNX → per-architecture build queues
    4. ``_arch_coordinator(arch)`` drains GPUs on-demand, spawns build loops
    5. ``_gpu_build_loop(worker)`` builds engines, lingers, releases GPU
    """

    def __init__(
        self,
        cache_dir: str,
        workers: list[GpuWorker],
        export_threads: int = 0,
        one_at_a_time: bool = False,
        trt_config: "TensorrtConfig | None" = None,
    ):
        from config import TensorrtConfig
        self._cache_dir = os.path.abspath(cache_dir)
        self._workers = workers
        self._one_at_a_time = one_at_a_time
        self._trt_config = trt_config or TensorrtConfig()
        self._ready_models: set[str] = set()
        self._queued_models: set[str] = set()
        self._lock = threading.Lock()

        # Thread pool for parallel ONNX export (CPU-bound).
        if export_threads <= 0:
            from config import _auto_threads
            export_threads = _auto_threads(0, 8)
        self._export_pool = ThreadPoolExecutor(
            max_workers=export_threads, thread_name_prefix="trt-export")
        self._export_threads = export_threads

        # Layer 1: input queue for models needing export+build
        self._export_queue: asyncio.Queue[_BuildRequest] = asyncio.Queue()

        # Layer 2: per-architecture build queues and wake events
        self._arch_queues: dict[str, asyncio.Queue[_BuildRequest]] = {}
        self._arch_work_available: dict[str, asyncio.Event] = {}
        self._arch_workers: dict[str, list[GpuWorker]] = {}

        # Precompute arch → workers mapping (stable for process lifetime).
        # With multiprocessing workers, arch_key comes from the worker's
        # cached status (set after WorkerReady).
        for w in workers:
            if w.gpu.is_failed:
                continue
            # Use the proxy's cached arch_key from the worker process
            ak = w._arch_key
            if not ak:
                # Worker may not be ready yet — use device_id as fallback
                ak = get_arch_key(w.gpu.device_id)
            if ak not in self._arch_workers:
                self._arch_workers[ak] = []
                self._arch_queues[ak] = asyncio.Queue()
                self._arch_work_available[ak] = asyncio.Event()
            self._arch_workers[ak].append(w)

        # Task handles for cleanup
        self._export_task: asyncio.Task | None = None
        self._arch_tasks: dict[str, asyncio.Task] = {}
        self._active_export_tasks: set[asyncio.Task] = set()

    def start(self) -> None:
        """Start the pipeline as asyncio tasks."""
        loop = asyncio.get_running_loop()

        self._export_task = loop.create_task(self._export_coordinator())

        self._arch_tasks = {}
        for arch_key in self._arch_workers:
            self._arch_tasks[arch_key] = loop.create_task(
                self._arch_coordinator(arch_key))

        archs = list(self._arch_workers.keys())
        gpu_counts = {ak: len(ws) for ak, ws in self._arch_workers.items()}
        mode = "one-at-a-time" if self._one_at_a_time else "pipelined"
        ws_default = self._trt_config.workspace_gb
        ws_overrides = self._trt_config.workspace_gb_per_arch
        ws_info = f"workspace={ws_default:.0f}GB"
        if ws_overrides:
            ws_info += " (" + ", ".join(f"{k}={v:.0f}GB" for k, v in ws_overrides.items()) + ")"
        log.info(f"  TRT: Build manager started (cache: {self._cache_dir}, "
                 f"export_threads: {self._export_threads}, mode: {mode}, "
                 f"architectures: {gpu_counts}, {ws_info})")

    def _get_workspace_gb(self, arch_key: str) -> float:
        """Resolve workspace GB for an architecture from config.

        Priority: per-arch override > default workspace_gb.
        A value of 0 passes through to the builder, which will
        auto-detect from available GPU VRAM in that case.
        """
        # Check per-arch override first (strip _cuXXX suffix for matching)
        # e.g. arch_key "sm_89_cu130" should match config key "sm_89"
        arch_base = arch_key.rsplit("_cu", 1)[0] if "_cu" in arch_key else arch_key
        ws = self._trt_config.workspace_gb_per_arch.get(arch_base)
        if ws is not None:
            return ws
        return self._trt_config.workspace_gb

    # ── Public API ────────────────────────────────────────────────────

    def queue_model(
        self,
        model_hash: str,
        model_dir: str,
        checkpoint_path: str | None = None,
    ) -> None:
        """Queue a model for TRT engine building.

        Args:
            model_hash: Content fingerprint (UNet component fingerprint).
            model_dir: Path to the model directory (diffusers) or checkpoint file.
            checkpoint_path: For single-file checkpoints, the .safetensors path.
        """
        with self._lock:
            if model_hash in self._ready_models:
                return
            if model_hash in self._queued_models:
                return
            self._queued_models.add(model_hash)

        self._export_queue.put_nowait(_BuildRequest(
            model_hash=model_hash,
            model_dir=model_dir,
            checkpoint_path=checkpoint_path,
        ))
        log.debug(f"  TRT: Queued build for model {model_hash[:16]}")

    def is_ready(self, model_hash: str) -> bool:
        """Check if all TRT engines are built for a model."""
        with self._lock:
            return model_hash in self._ready_models

    def scan_and_queue(self, sdxl_models: dict[str, str], registry) -> None:
        """Scan all registered SDXL models and queue builds for missing engines.

        Called during startup after model scanning completes.
        """
        if not self._workers:
            return

        arch_keys = set(self._arch_workers.keys())

        queued = 0
        for model_name, model_dir in sdxl_models.items():
            try:
                unet_comp = registry.get_sdxl_unet_component(model_dir)
                vae_comp = registry.get_sdxl_vae_component(model_dir)
            except (KeyError, RuntimeError, IndexError):
                continue

            model_hash = unet_comp.fingerprint

            all_exist = True
            for arch_key in arch_keys:
                for component_type in ("te1", "te2", "unet", "vae"):
                    if not all_engines_exist(self._cache_dir, model_hash,
                                             component_type, arch_key):
                        all_exist = False
                        break
                if not all_exist:
                    break

            if all_exist:
                with self._lock:
                    self._ready_models.add(model_hash)
                continue

            checkpoint_path = model_dir if os.path.isfile(model_dir) else None
            self.queue_model(model_hash, model_dir, checkpoint_path)
            queued += 1

        if queued:
            log.info(f"  TRT: Queued {queued} model(s) for engine building")
        else:
            log.info(f"  TRT: All engines up to date")

    # ── Layer 1: Export Coordinator ───────────────────────────────────

    async def _export_coordinator(self) -> None:
        """Continuously pulls models from _export_queue, runs ONNX export,
        and feeds completed models into per-architecture build queues."""
        try:
            while True:
                request = await self._export_queue.get()
                # Fire off the export+dispatch as a concurrent task so
                # multiple exports can run in parallel (bounded by thread pool)
                task = asyncio.get_running_loop().create_task(
                    self._handle_export(request))
                self._active_export_tasks.add(task)
                task.add_done_callback(self._active_export_tasks.discard)
                task.add_done_callback(self._on_export_done)
        except asyncio.CancelledError:
            log.debug("  TRT: Export coordinator stopped")

    async def _handle_export(self, request: _BuildRequest) -> None:
        """Export a single model to ONNX (if needed), then dispatch to arch queues."""
        short_hash = request.model_hash[:16]
        loop = asyncio.get_running_loop()

        # Check if ONNX files already exist (skip export if so)
        missing_onnx = False
        for ct in ("te1", "te2", "unet", "vae"):
            if not os.path.isfile(get_onnx_path(self._cache_dir, request.model_hash, ct)):
                missing_onnx = True
                break

        if missing_onnx:
            try:
                await loop.run_in_executor(
                    self._export_pool, self._do_export, request)
            except Exception as ex:
                log.log_exception(ex, f"TRT: ONNX export failed for {short_hash}")
                with self._lock:
                    self._queued_models.discard(request.model_hash)
                return

        # Verify UNet ONNX exists (required minimum for building)
        unet_onnx = get_onnx_path(self._cache_dir, request.model_hash, "unet")
        if not os.path.isfile(unet_onnx):
            log.warning(f"  TRT: No UNet ONNX for {short_hash} after export — skipping")
            with self._lock:
                self._queued_models.discard(request.model_hash)
            return

        # Dispatch to per-architecture build queues
        dispatched = False
        for arch_key in self._arch_workers:
            needs_build = False
            for component_type in ("te1", "te2", "unet", "vae"):
                onnx = get_onnx_path(self._cache_dir, request.model_hash, component_type)
                if not os.path.isfile(onnx):
                    continue
                if not all_engines_exist(self._cache_dir, request.model_hash,
                                         component_type, arch_key):
                    needs_build = True
                    break

            if needs_build:
                self._arch_queues[arch_key].put_nowait(request)
                self._arch_work_available[arch_key].set()
                dispatched = True
                log.debug(f"  TRT: {short_hash} → {arch_key} build queue")

        if not dispatched:
            # All engines already exist for all architectures
            self._check_model_ready(request.model_hash)
            log.debug(f"  TRT: {short_hash} — all engines already exist, no build needed")

    @staticmethod
    def _on_export_done(task: asyncio.Task) -> None:
        """Callback for fire-and-forget export tasks — log unhandled exceptions."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            log.log_exception(exc, "TRT: Unhandled error in export task")

    # ── Layer 2: Per-Architecture Coordinator ─────────────────────────

    async def _arch_coordinator(self, arch_key: str) -> None:
        """Sleeps until work arrives for this architecture, then manages
        GPU drain escalation and build loop lifecycle."""
        queue = self._arch_queues[arch_key]
        wake = self._arch_work_available[arch_key]
        all_workers = self._arch_workers[arch_key]

        try:
            while True:
                await wake.wait()
                wake.clear()

                if queue.empty():
                    continue

                # Build a fresh idle list (exclude failed GPUs)
                idle_workers = [w for w in all_workers if not w.gpu.is_failed]

                if not idle_workers:
                    log.warning(f"  TRT: [{arch_key}] No healthy GPUs available for build")
                    continue

                active_loops: dict[asyncio.Task, GpuWorker] = {}

                # Start first build loop
                w = idle_workers.pop(0)
                task = asyncio.get_running_loop().create_task(
                    self._gpu_build_loop(w, arch_key))
                active_loops[task] = w
                log.info(f"  TRT: [{arch_key}] GPU [{w.gpu.uuid}] draining for build "
                         f"({queue.qsize()} item(s) queued)")

                # Monitor loop: escalate drain or wind down
                while active_loops:
                    done, _ = await asyncio.wait(
                        active_loops.keys(),
                        timeout=_ARCH_CHECK_INTERVAL,
                        return_when=asyncio.FIRST_COMPLETED,
                    )

                    # Return completed workers to idle pool
                    for t in done:
                        w = active_loops.pop(t)
                        idle_workers.append(w)
                        if t.cancelled():
                            continue
                        exc = t.exception()
                        if exc is not None:
                            log.log_exception(
                                exc, f"TRT: [{arch_key}] Build loop failed on "
                                     f"GPU [{w.gpu.uuid}]")

                    # Escalate: drain another GPU if work is accumulating
                    if not queue.empty() and idle_workers:
                        can_escalate = (not self._one_at_a_time
                                        or len(active_loops) == 0)
                        if can_escalate:
                            w = idle_workers.pop(0)
                            task = asyncio.get_running_loop().create_task(
                                self._gpu_build_loop(w, arch_key))
                            active_loops[task] = w
                            log.info(f"  TRT: [{arch_key}] Escalating — "
                                     f"GPU [{w.gpu.uuid}] draining for build "
                                     f"({queue.qsize()} item(s) remaining)")

                log.info(f"  TRT: [{arch_key}] All GPUs back to inference")

        except asyncio.CancelledError:
            log.debug(f"  TRT: [{arch_key}] Arch coordinator stopped")

    # ── Layer 3: Per-GPU Build Loop ───────────────────────────────────

    async def _gpu_build_loop(self, worker: GpuWorker, arch_key: str) -> None:
        """Drain a GPU, build engines from the arch queue, linger for more
        work, then release the GPU back to inference.

        With multiprocessing workers, TRT builds run inside the worker
        subprocess via the trt_build() proxy method — true parallel execution
        across GPUs with no GIL contention.
        """
        queue = self._arch_queues[arch_key]
        device_id = worker.gpu.device_id
        drained = False

        try:
            await worker.request_drain()
            drained = True

            while True:
                # Wait for work with linger timeout
                try:
                    request = await asyncio.wait_for(
                        queue.get(), timeout=_BUILD_LINGER_TIMEOUT)
                except asyncio.TimeoutError:
                    # No more work arrived within linger period — release GPU
                    log.debug(f"  TRT: [{arch_key}] GPU [{worker.gpu.uuid}] "
                              f"idle for {_BUILD_LINGER_TIMEOUT}s — releasing")
                    break

                short_hash = request.model_hash[:16]

                # Skip if another GPU already built these engines
                needs_build = False
                for component_type in ("te1", "te2", "unet", "vae"):
                    onnx = get_onnx_path(self._cache_dir, request.model_hash,
                                         component_type)
                    if not os.path.isfile(onnx):
                        continue
                    if not all_engines_exist(self._cache_dir, request.model_hash,
                                             component_type, arch_key):
                        needs_build = True
                        break

                if not needs_build:
                    log.debug(f"  TRT: [{arch_key}] {short_hash} — "
                              f"engines already built, skipping")
                    self._check_model_ready(request.model_hash)
                    continue

                # Build engines via worker subprocess (true parallel per GPU)
                log.info(f"  TRT: [{arch_key}] GPU {device_id} building "
                         f"{short_hash}...")
                try:
                    trt_result = await worker.trt_build(
                        model_hash=request.model_hash,
                        model_dir=request.model_dir,
                        cache_dir=self._cache_dir,
                        arch_key=arch_key,
                        max_workspace_gb=self._get_workspace_gb(arch_key),
                    )
                    if trt_result.success:
                        results = trt_result.results
                        built_te1 = results.get("te1", [])
                        built_te2 = results.get("te2", [])
                        built_unet = results.get("unet", [])
                        built_vae = results.get("vae", [])
                        log.info(f"  TRT: [{arch_key}] {short_hash}: "
                                 f"TE1 [{', '.join(built_te1)}] + "
                                 f"TE2 [{', '.join(built_te2)}] + "
                                 f"UNet [{', '.join(built_unet)}] + "
                                 f"VAE [{', '.join(built_vae)}]")
                        self._check_model_ready(request.model_hash)
                    else:
                        log.error(f"  TRT: [{arch_key}] Engine build failed for "
                                  f"{short_hash}: {trt_result.error}")
                except Exception as ex:
                    log.log_exception(
                        ex, f"TRT: [{arch_key}] Engine build failed for "
                            f"{short_hash} on GPU {device_id}")
        finally:
            if drained:
                await worker.release_drain()

    # ── Helpers ───────────────────────────────────────────────────────

    def _check_model_ready(self, model_hash: str) -> None:
        """Check if all engines are ready across all architectures.
        If so, mark the model as TRT-ready."""
        short_hash = model_hash[:16]

        if not self._arch_workers:
            return

        for arch_key in self._arch_workers:
            for component_type in ("te1", "te2", "unet", "vae"):
                onnx = get_onnx_path(self._cache_dir, model_hash, component_type)
                if not os.path.isfile(onnx):
                    # If no ONNX exists, engines can't exist either — not ready
                    return
                if not all_engines_exist(self._cache_dir, model_hash,
                                         component_type, arch_key):
                    return

        with self._lock:
            if model_hash not in self._ready_models:
                self._ready_models.add(model_hash)
                self._queued_models.discard(model_hash)
                log.info(f"  TRT: Model {short_hash} — all engines ready "
                         f"across all architectures")

    def _do_export(self, request: _BuildRequest) -> None:
        """Export a single model to ONNX (runs in thread pool)."""
        from handlers.sdxl import load_component

        short_hash = request.model_hash[:16]
        model_dir = request.model_dir

        te1_onnx = get_onnx_path(self._cache_dir, request.model_hash, "te1")
        te2_onnx = get_onnx_path(self._cache_dir, request.model_hash, "te2")
        unet_onnx = get_onnx_path(self._cache_dir, request.model_hash, "unet")
        vae_onnx = get_onnx_path(self._cache_dir, request.model_hash, "vae")

        log.info(f"  TRT: Exporting ONNX for {short_hash}...")

        # Export TE1 (CLIP-L)
        if not os.path.isfile(te1_onnx):
            try:
                te1 = load_component("sdxl_te1", model_dir, torch.device("cpu"))
                export_te1_onnx(te1, te1_onnx)
                del te1
            except Exception as ex:
                log.log_exception(ex, f"TRT: TE1 ONNX export failed for {short_hash}")

        # Export TE2 (CLIP-bigG)
        if not os.path.isfile(te2_onnx):
            try:
                te2 = load_component("sdxl_te2", model_dir, torch.device("cpu"))
                export_te2_onnx(te2, te2_onnx)
                del te2
            except Exception as ex:
                log.log_exception(ex, f"TRT: TE2 ONNX export failed for {short_hash}")

        # Export UNet
        if not os.path.isfile(unet_onnx):
            try:
                unet = load_component("sdxl_unet", model_dir, torch.device("cpu"))
                ca_dim = getattr(getattr(unet, "config", None),
                                 "cross_attention_dim", None)
                if ca_dim is not None and ca_dim != 2048:
                    log.warning(f"  TRT: Skipping {short_hash} — UNet "
                                f"cross_attention_dim is {ca_dim}, expected "
                                f"2048 (not SDXL?)")
                    del unet
                    return
                export_unet_onnx(unet, unet_onnx)
                del unet
            except Exception as ex:
                log.log_exception(ex, f"TRT: UNet ONNX export failed for {short_hash}")

        # Export VAE
        if not os.path.isfile(vae_onnx):
            try:
                vae = load_component("sdxl_vae", model_dir, torch.device("cpu"))
                export_vae_onnx(vae, vae_onnx)
                del vae
            except Exception as ex:
                log.log_exception(ex, f"TRT: VAE ONNX export failed for {short_hash}")

        log.info(f"  TRT: ONNX export complete for {short_hash}")


class _BuildRequest:
    """Internal build request."""
    __slots__ = ("model_hash", "model_dir", "checkpoint_path")

    def __init__(self, model_hash: str, model_dir: str,
                 checkpoint_path: str | None):
        self.model_hash = model_hash
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
