"""Background TRT build manager.

Scans registered SDXL models, exports missing ONNX files, and builds
TRT engines on drained GPUs.  Runs as an asyncio task alongside the
scheduler, coordinating GPU access through the drain mechanism.

Two-phase pipeline:
  Phase 1 — ONNX export (CPU-bound, parallelized via thread pool)
  Phase 2 — Engine build (GPU-bound, drain once per batch, sequential)
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
    build_all_engines,
    get_arch_key,
    get_onnx_path,
)
from trt.exporter import export_unet_onnx, export_vae_onnx

if TYPE_CHECKING:
    from scheduling.worker import GpuWorker


class TrtBuildManager:
    """Manages background ONNX export and TRT engine compilation.

    On startup, scans all registered SDXL models and queues builds for
    any missing engines.  When new models are registered at runtime (via
    filesystem watcher), they can be queued via ``queue_model()``.

    Build flow (batched):
    1. Collect all pending models from the queue
    2. Export to ONNX in parallel (CPU work, thread pool)
    3. For each unique GPU architecture in the pool:
       a. Request a GPU of that arch to drain (once per batch)
       b. Build engines for ALL models that have ONNX ready
       c. Release the GPU back to the scheduler
    4. Mark successful models as "TRT ready"
    """

    def __init__(
        self,
        cache_dir: str,
        workers: list[GpuWorker],
        export_threads: int = 0,
    ):
        self._cache_dir = os.path.abspath(cache_dir)
        self._workers = workers
        self._task: asyncio.Task | None = None
        self._build_queue: asyncio.Queue[_BuildRequest] = asyncio.Queue()
        self._ready_models: set[str] = set()  # model_hashes with all engines built
        self._lock = threading.Lock()

        # Thread pool for parallel ONNX export (CPU-bound).
        # 0 = auto: use fingerprint threads config (resolved by caller).
        if export_threads <= 0:
            from config import _auto_threads
            export_threads = _auto_threads(0, 8)
        self._export_pool = ThreadPoolExecutor(
            max_workers=export_threads, thread_name_prefix="trt-export")
        self._export_threads = export_threads

    def start(self) -> None:
        """Start the background build loop as an asyncio task."""
        self._task = asyncio.get_running_loop().create_task(self._run_loop())
        log.info(f"  TRT: Build manager started (cache: {self._cache_dir}, "
                 f"export_threads: {self._export_threads})")

    async def _run_loop(self) -> None:
        """Main build loop — batches requests, exports in parallel, builds on GPU."""
        try:
            while True:
                # Block until at least one request arrives
                first = await self._build_queue.get()
                batch = [first]

                # Drain any additional queued requests into the batch.
                # Short sleep lets scan_and_queue() finish queueing all models
                # before we start processing (avoids partial first batch).
                await asyncio.sleep(0.1)
                while not self._build_queue.empty():
                    try:
                        batch.append(self._build_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break

                try:
                    await self._process_batch(batch)
                except Exception as ex:
                    log.log_exception(ex, "TRT batch processing failed")
                finally:
                    for _ in batch:
                        self._build_queue.task_done()
        except asyncio.CancelledError:
            log.debug("  TRT: Build manager stopped")

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

        self._build_queue.put_nowait(_BuildRequest(
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

        # Determine unique architectures across all workers
        arch_keys = set()
        for w in self._workers:
            arch_keys.add(get_arch_key(w.gpu.device_id))

        queued = 0
        for model_name, model_dir in sdxl_models.items():
            try:
                unet_comp = registry.get_sdxl_unet_component(model_dir)
                vae_comp = registry.get_sdxl_vae_component(model_dir)
            except (KeyError, RuntimeError, IndexError):
                continue  # model not yet registered/scanned

            model_hash = unet_comp.fingerprint

            # Check if all engines (static + dynamic) exist for all architectures
            all_exist = True
            for arch_key in arch_keys:
                for component_type in ("unet", "vae"):
                    if not all_engines_exist(self._cache_dir, model_hash, component_type, arch_key):
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

    async def _process_batch(self, batch: list[_BuildRequest]) -> None:
        """Process a batch of models: parallel ONNX export, then batched engine build."""
        log.info(f"  TRT: Processing batch of {len(batch)} model(s)...")

        # ── Phase 1: Parallel ONNX export (CPU-bound) ────────────────────
        needs_export = []
        for req in batch:
            unet_onnx = get_onnx_path(self._cache_dir, req.model_hash, "unet")
            vae_onnx = get_onnx_path(self._cache_dir, req.model_hash, "vae")
            if not os.path.isfile(unet_onnx) or not os.path.isfile(vae_onnx):
                needs_export.append(req)

        if needs_export:
            log.info(f"  TRT: Exporting ONNX for {len(needs_export)} model(s) "
                     f"({self._export_threads} threads)...")
            loop = asyncio.get_running_loop()
            export_tasks = []
            for req in needs_export:
                export_tasks.append(
                    loop.run_in_executor(self._export_pool, self._do_export, req))
            results = await asyncio.gather(*export_tasks, return_exceptions=True)
            for req, result in zip(needs_export, results):
                if isinstance(result, Exception):
                    log.log_exception(
                        result, f"TRT: ONNX export failed for {req.model_hash[:16]}")

        # ── Phase 2: Batched engine build (GPU-bound) ────────────────────
        # Filter to models that have at least a UNet ONNX (required)
        buildable = []
        for req in batch:
            unet_onnx = get_onnx_path(self._cache_dir, req.model_hash, "unet")
            if os.path.isfile(unet_onnx):
                buildable.append(req)
            else:
                log.warning(f"  TRT: No UNet ONNX for {req.model_hash[:16]} — skipping build")

        if not buildable:
            log.warning("  TRT: No models ready for engine building")
            return

        # Determine unique GPU architectures
        arch_workers: dict[str, GpuWorker] = {}
        for w in self._workers:
            if w.gpu.is_failed:
                continue
            ak = get_arch_key(w.gpu.device_id)
            if ak not in arch_workers:
                arch_workers[ak] = w

        for arch_key, worker in arch_workers.items():
            # Find models that need engines for this arch
            needs_build = []
            for req in buildable:
                missing = False
                for component_type in ("unet", "vae"):
                    onnx = get_onnx_path(self._cache_dir, req.model_hash, component_type)
                    if not os.path.isfile(onnx):
                        continue
                    if not all_engines_exist(self._cache_dir, req.model_hash,
                                             component_type, arch_key):
                        missing = True
                        break
                if missing:
                    needs_build.append(req)

            if not needs_build:
                log.debug(f"  TRT: All {arch_key} engines exist — no build needed")
                continue

            # Drain the GPU once for the entire batch
            log.info(f"  TRT: Requesting drain on GPU [{worker.gpu.uuid}] "
                     f"for {arch_key} engine build ({len(needs_build)} models)...")

            try:
                await worker.request_drain()
                log.info(f"  TRT: GPU [{worker.gpu.uuid}] drained — building "
                         f"{len(needs_build)} model(s)")

                loop = asyncio.get_running_loop()
                for i, req in enumerate(needs_build):
                    short_hash = req.model_hash[:16]
                    log.info(f"  TRT: Building engines for {short_hash} "
                             f"[{i + 1}/{len(needs_build)}]...")
                    try:
                        results = await loop.run_in_executor(
                            None,
                            build_all_engines,
                            req.model_hash,
                            self._cache_dir,
                            arch_key,
                            worker.gpu.device_id,
                        )
                        built_unet = results.get("unet", [])
                        built_vae = results.get("vae", [])
                        log.info(f"  TRT: {short_hash}: UNet [{', '.join(built_unet)}] + "
                                 f"VAE [{', '.join(built_vae)}]")
                    except Exception as ex:
                        log.log_exception(
                            ex, f"TRT: Engine build failed for {short_hash}")

            finally:
                await worker.release_drain()
                log.info(f"  TRT: GPU [{worker.gpu.uuid}] released from drain")

        # ── Mark ready models ─────────────────────────────────────────────
        for req in buildable:
            model_hash = req.model_hash
            all_built = True
            for arch_key in arch_workers:
                for component_type in ("unet", "vae"):
                    onnx = get_onnx_path(self._cache_dir, model_hash, component_type)
                    if not os.path.isfile(onnx):
                        all_built = False
                        break
                    if not all_engines_exist(self._cache_dir, model_hash,
                                             component_type, arch_key):
                        all_built = False
                        break
                if not all_built:
                    break

            short_hash = model_hash[:16]
            if all_built:
                with self._lock:
                    self._ready_models.add(model_hash)
                log.info(f"  TRT: Model {short_hash} — all engines ready")
            else:
                log.warning(f"  TRT: Model {short_hash} — some engines failed, "
                            f"will retry on next startup")

    def _do_export(self, request: _BuildRequest) -> None:
        """Export a single model to ONNX (runs in thread pool)."""
        from handlers.sdxl import load_component

        short_hash = request.model_hash[:16]
        model_dir = request.model_dir

        unet_onnx = get_onnx_path(self._cache_dir, request.model_hash, "unet")
        vae_onnx = get_onnx_path(self._cache_dir, request.model_hash, "vae")

        log.info(f"  TRT: Exporting ONNX for {short_hash}...")

        # Export UNet
        if not os.path.isfile(unet_onnx):
            try:
                unet = load_component("sdxl_unet", model_dir, torch.device("cpu"))
                # Validate this is actually an SDXL UNet (cross_attention_dim=2048).
                # SD 1.5 models (768) can be mis-classified as SDXL and will fail
                # with a shape mismatch during tracing.
                ca_dim = getattr(getattr(unet, "config", None), "cross_attention_dim", None)
                if ca_dim is not None and ca_dim != 2048:
                    log.warning(f"  TRT: Skipping {short_hash} — UNet cross_attention_dim "
                                f"is {ca_dim}, expected 2048 (not SDXL?)")
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

    def __init__(self, model_hash: str, model_dir: str, checkpoint_path: str | None):
        self.model_hash = model_hash
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
