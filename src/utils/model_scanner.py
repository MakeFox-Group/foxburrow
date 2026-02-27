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

    All models (single-file and diffusers) are fingerprinted in a thread pool
    and become available to the API as each one finishes.

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

    def start(self, discovered_models: dict[str, str], pool: ThreadPoolExecutor | None = None) -> threading.Thread:
        """Register all models using a background thread pool.

        All models (single-file and diffusers) are fingerprinted in parallel.
        On first run (no .fpcache), even single-file models need a full
        SHA256 hash, so parallelism matters.

        If *pool* is provided, uses that executor (caller manages lifetime).
        Otherwise creates an internal pool with ``max_workers`` threads.

        Fires ``model_discovered`` and ``model_scan_progress`` WebSocket
        events for every model as it completes.

        Returns the background daemon thread (already started).
        """
        if not discovered_models:
            log.debug("  ModelScanner: No models to register")
            return _make_noop_thread()

        # Classify for logging / event payloads
        model_formats: dict[str, str] = {}
        for name, path in discovered_models.items():
            if os.path.isfile(path) and path.endswith(".safetensors"):
                model_formats[name] = "single_file"
            else:
                model_formats[name] = "diffusers"

        grand_total = len(discovered_models)
        with self._lock:
            self._total = grand_total
            self._completed = 0
            self._scanning = True

        def _background():
            sf_count = sum(1 for f in model_formats.values() if f == "single_file")
            df_count = grand_total - sf_count
            log.debug(f"  ModelScanner: Registering {grand_total} model(s) "
                     f"({sf_count} single-file, {df_count} diffusers, "
                     f"{self._max_workers} workers)")
            start_time = time.monotonic()

            def _register_one(item: tuple[str, str]) -> tuple[str, str, str, bool]:
                name, path = item
                fmt = model_formats[name]
                try:
                    self._registry.register_sdxl_checkpoint(path)
                    self._app_state.sdxl_models[name] = path
                    return name, path, fmt, True
                except Exception as ex:
                    log.log_exception(ex, f"Failed to register model: {name}")
                    return name, path, fmt, False

            # Use caller's pool if provided; otherwise create a local one.
            own_pool: ThreadPoolExecutor | None = None
            use_pool = pool
            if use_pool is None:
                workers = min(self._max_workers, grand_total)
                use_pool = ThreadPoolExecutor(max_workers=workers)
                own_pool = use_pool

            try:
                for name, path, fmt, success in use_pool.map(_register_one, discovered_models.items()):
                    with self._lock:
                        self._completed += 1
                        done = self._completed
                        total = self._total

                    if success:
                        log.debug(f"  ModelScanner: Model available: {name} ({done}/{total})")
                        streamer.fire_event("model_discovered", {
                            "model_type": "sdxl", "name": name, "path": path,
                            "format": fmt})
                    else:
                        log.warning(f"  ModelScanner: Failed: {name} ({done}/{total})")

                    streamer.fire_event("model_scan_progress", {
                        "completed": done, "total": total,
                        "scanning": done < total})
            finally:
                if own_pool is not None:
                    own_pool.shutdown(wait=True)

            elapsed = time.monotonic() - start_time
            with self._lock:
                self._scanning = False
            log.debug(f"  ModelScanner: Background scan complete "
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
