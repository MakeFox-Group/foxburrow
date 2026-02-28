"""Background TRT build manager.

Scans registered SDXL models, exports missing ONNX files, and builds
TRT engines on drained GPUs.  Runs as an asyncio task alongside the
scheduler, coordinating GPU access through the drain mechanism.
"""

from __future__ import annotations

import asyncio
import os
import threading
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

    Build flow per model:
    1. Export to ONNX if not already done (CPU work, no GPU needed)
    2. For each unique GPU architecture in the pool:
       a. Request a GPU of that arch to drain (stop accepting work)
       b. Wait for drain to complete (GPU idle, all models evicted)
       c. Build engines for all target resolutions on that GPU
       d. Release the GPU back to the scheduler
    3. Mark model as "TRT ready"
    """

    def __init__(
        self,
        cache_dir: str,
        workers: list[GpuWorker],
    ):
        self._cache_dir = os.path.abspath(cache_dir)
        self._workers = workers
        self._task: asyncio.Task | None = None
        self._build_queue: asyncio.Queue[_BuildRequest] = asyncio.Queue()
        self._ready_models: set[str] = set()  # model_hashes with all engines built
        self._lock = threading.Lock()

    def start(self) -> None:
        """Start the background build loop as an asyncio task."""
        self._task = asyncio.get_running_loop().create_task(self._run_loop())
        log.info(f"  TRT: Build manager started (cache: {self._cache_dir})")

    async def _run_loop(self) -> None:
        """Main build loop — processes queued build requests sequentially."""
        try:
            while True:
                request = await self._build_queue.get()
                try:
                    await self._process_request(request)
                except Exception as ex:
                    log.log_exception(ex, f"TRT build failed for model {request.model_hash[:16]}")
                finally:
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

    async def _process_request(self, request: _BuildRequest) -> None:
        """Process a single build request: export ONNX then build engines."""
        model_hash = request.model_hash
        short_hash = model_hash[:16]

        log.info(f"  TRT: Processing build for model {short_hash}...")

        # Step 1: Export to ONNX if needed (CPU work)
        unet_onnx = get_onnx_path(self._cache_dir, model_hash, "unet")
        vae_onnx = get_onnx_path(self._cache_dir, model_hash, "vae")

        if not os.path.isfile(unet_onnx) or not os.path.isfile(vae_onnx):
            await self._export_onnx(request, unet_onnx, vae_onnx)

        if not os.path.isfile(unet_onnx):
            log.error(f"  TRT: UNet ONNX export failed for {short_hash} — skipping engine build")
            return

        # Step 2: Build engines for each unique GPU architecture
        arch_workers: dict[str, GpuWorker] = {}
        for w in self._workers:
            if w.gpu.is_failed:
                continue
            ak = get_arch_key(w.gpu.device_id)
            if ak not in arch_workers:
                arch_workers[ak] = w

        for arch_key, worker in arch_workers.items():
            # Check which engines are missing for this arch
            missing = False
            for component_type in ("unet", "vae"):
                onnx = unet_onnx if component_type == "unet" else vae_onnx
                if not os.path.isfile(onnx):
                    continue
                if not all_engines_exist(self._cache_dir, model_hash, component_type, arch_key):
                    missing = True
                    break

            if not missing:
                log.debug(f"  TRT: All {arch_key} engines exist for {short_hash}")
                continue

            # Drain the GPU, build, release
            log.info(f"  TRT: Requesting drain on GPU [{worker.gpu.uuid}] "
                     f"for {arch_key} engine build...")

            try:
                await worker.request_drain()
                log.info(f"  TRT: GPU [{worker.gpu.uuid}] drained — starting build")

                # Run the build in a thread to avoid blocking the event loop
                results = await asyncio.get_running_loop().run_in_executor(
                    None,
                    build_all_engines,
                    model_hash,
                    self._cache_dir,
                    arch_key,
                    worker.gpu.device_id,
                )

                built_unet = results.get("unet", [])
                built_vae = results.get("vae", [])
                log.info(f"  TRT: Built UNet [{', '.join(built_unet)}] + "
                         f"VAE [{', '.join(built_vae)}] on {arch_key}")

            finally:
                await worker.release_drain()
                log.info(f"  TRT: GPU [{worker.gpu.uuid}] released from drain")

        # Verify all engines were actually built before marking ready.
        # If ONNX export or engine build failed for any component, the model
        # stays un-ready so it can be retried on next startup.
        all_built = True
        for arch_key in arch_workers:
            for component_type in ("unet", "vae"):
                onnx = unet_onnx if component_type == "unet" else vae_onnx
                if not os.path.isfile(onnx):
                    all_built = False
                    break
                if not all_engines_exist(self._cache_dir, model_hash, component_type, arch_key):
                    all_built = False
                    break
            if not all_built:
                break

        if all_built:
            with self._lock:
                self._ready_models.add(model_hash)
            log.info(f"  TRT: Model {short_hash} — all engines ready")
        else:
            log.warning(f"  TRT: Model {short_hash} — some engines failed to build, "
                        f"will retry on next startup")

    async def _export_onnx(
        self,
        request: _BuildRequest,
        unet_onnx: str,
        vae_onnx: str,
    ) -> None:
        """Export model to ONNX format (CPU work, run in executor)."""
        short_hash = request.model_hash[:16]
        model_dir = request.model_dir

        log.info(f"  TRT: Exporting ONNX for model {short_hash}...")

        def _do_export():
            from handlers.sdxl import load_component

            # Export UNet
            if not os.path.isfile(unet_onnx):
                try:
                    unet = load_component("sdxl_unet", model_dir, torch.device("cpu"))
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

            torch.cuda.empty_cache()

        await asyncio.get_running_loop().run_in_executor(None, _do_export)


class _BuildRequest:
    """Internal build request."""
    __slots__ = ("model_hash", "model_dir", "checkpoint_path")

    def __init__(self, model_hash: str, model_dir: str, checkpoint_path: str | None):
        self.model_hash = model_hash
        self.model_dir = model_dir
        self.checkpoint_path = checkpoint_path
