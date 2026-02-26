"""Background model scanning with progressive availability."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import log
from api.websocket import streamer

if TYPE_CHECKING:
    from scheduling.model_registry import ModelRegistry
    from state import AppState


class ModelScanner:
    """Background fingerprinting + progressive registration of SDXL models.

    Single-file .safetensors checkpoints register instantly (fast fingerprint
    via per-directory .fpcache). Diffusers-format models are queued for
    background SHA256 hashing in a thread pool, becoming available to the API
    as each one finishes.

    Fires WebSocket events as models become available:
      - ``model_discovered``: per-model with model_type/name/path/format
      - ``model_scan_progress``: running completed/total counts
    """

    def __init__(
        self,
        registry: ModelRegistry,
        app_state: AppState,
        max_workers: int = 4,
    ):
        self._registry = registry
        self._app_state = app_state
        self._max_workers = max_workers
        self._thread: threading.Thread | None = None
        self._scanning = False
        self._completed = 0
        self._total = 0
        self._lock = threading.Lock()

    def start(self, discovered_models: dict[str, str]) -> threading.Thread:
        """Register models with progressive availability.

        Single-file models are registered immediately in the calling thread.
        Diffusers models are queued for background fingerprinting.
        Fires ``model_available`` and ``model_scan_progress`` WebSocket events
        for every model (both types).

        Returns the background daemon thread (already started).
        """
        single_file: dict[str, str] = {}
        diffusers: dict[str, str] = {}

        for name, path in discovered_models.items():
            if os.path.isfile(path) and path.endswith(".safetensors"):
                single_file[name] = path
            else:
                diffusers[name] = path

        # Total spans both types so progress is meaningful end-to-end.
        grand_total = len(single_file) + len(diffusers)
        with self._lock:
            self._total = grand_total
            self._completed = 0
            self._scanning = True

        # Register single-file models immediately — fast pseudo-fingerprint,
        # no SHA256 needed (uses .fpcache stat-based lookup or fast hash).
        for name, path in single_file.items():
            success = False
            try:
                self._registry.register_sdxl_checkpoint(path)
                self._app_state.sdxl_models[name] = path
                success = True
            except Exception as ex:
                log.log_exception(ex, f"Failed to register single-file model: {name}")

            with self._lock:
                self._completed += 1
                done = self._completed

            if success:
                streamer.fire_event("model_discovered", {
                    "model_type": "sdxl", "name": name, "path": path,
                    "format": "single_file"})
            streamer.fire_event("model_scan_progress", {
                "completed": done, "total": grand_total,
                "scanning": done < grand_total})

        if single_file:
            log.info(f"  ModelScanner: {len(single_file)} single-file model(s) available immediately")

        if not diffusers:
            with self._lock:
                self._scanning = False
            streamer.fire_event("model_scan_progress", {
                "completed": grand_total, "total": grand_total, "scanning": False})
            log.info("  ModelScanner: No diffusers models to background-hash")
            return _make_noop_thread()

        def _background():
            log.info(f"  ModelScanner: Background hashing {len(diffusers)} diffusers model(s) "
                     f"({self._max_workers} workers)")
            start_time = time.monotonic()

            def _register_one(item: tuple[str, str]) -> tuple[str, str, bool]:
                name, path = item
                try:
                    self._registry.register_sdxl_checkpoint(path)
                    self._app_state.sdxl_models[name] = path
                    return name, path, True
                except Exception as ex:
                    log.log_exception(ex, f"Failed to register diffusers model: {name}")
                    return name, path, False

            workers = min(self._max_workers, len(diffusers))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                for name, path, success in pool.map(_register_one, diffusers.items()):
                    with self._lock:
                        self._completed += 1
                        done = self._completed
                        total = self._total

                    if success:
                        log.info(f"  ModelScanner: Model available: {name} ({done}/{total})")
                        streamer.fire_event("model_discovered", {
                            "model_type": "sdxl", "name": name, "path": path,
                            "format": "diffusers"})
                    else:
                        log.warning(f"  ModelScanner: Failed: {name} ({done}/{total})")

                    streamer.fire_event("model_scan_progress", {
                        "completed": done, "total": total,
                        "scanning": done < total})

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._scanning = False
            log.info(f"  ModelScanner: Background scan complete "
                     f"({self._completed}/{self._total} in {elapsed:.1f}s)")

        t = threading.Thread(target=_background, name="model-scanner", daemon=True)
        t.start()
        self._thread = t
        return t

    @property
    def is_scanning(self) -> bool:
        with self._lock:
            return self._scanning

    @property
    def progress(self) -> tuple[int, int]:
        """Return (completed, total) counts."""
        with self._lock:
            return self._completed, self._total


def _make_noop_thread() -> threading.Thread:
    """Return an already-finished daemon thread (for consistent API)."""
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    t.join()
    return t
