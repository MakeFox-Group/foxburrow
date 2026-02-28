"""GPU profiling tracer — appends structured JSONL events to per-architecture trace files.

Delivery guarantee: at-most-once for in-flight events. A reader opening
the same file may see a partial trailing line if a write is in progress;
the query engine silently skips malformed lines.
"""

from __future__ import annotations

import json
import os
import threading
from datetime import datetime, timezone
from typing import Any

import log

# Base directory for trace files
_TRACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiling", "traces",
)

# Per-file locks to prevent interleaving from concurrent workers on the same arch
_file_locks: dict[str, threading.Lock] = {}
_file_locks_guard = threading.Lock()


def _get_file_lock(path: str) -> threading.Lock:
    with _file_locks_guard:
        if path not in _file_locks:
            _file_locks[path] = threading.Lock()
        return _file_locks[path]


class GpuTracer:
    """Per-GPU tracer. Created once per GpuWorker at init."""

    def __init__(self, gpu_uuid: str, gpu_arch: str, gpu_name: str):
        self.gpu_uuid = gpu_uuid
        self.gpu_arch = gpu_arch
        self.gpu_name = gpu_name

        os.makedirs(_TRACES_DIR, exist_ok=True)
        self._path = os.path.join(_TRACES_DIR, f"{gpu_arch}.jsonl")
        self._lock = _get_file_lock(self._path)
        self._file = open(self._path, "a", encoding="utf-8")

    def close(self) -> None:
        """Flush and close the underlying file handle."""
        try:
            with self._lock:
                self._file.flush()
                self._file.close()
        except Exception:
            pass

    def record(self, event_type: str, job_id: str, model: str | None,
               duration_s: float, **extra: Any) -> None:
        """Append a JSONL line with common fields + extras."""
        event: dict[str, Any] = {
            "type": event_type,
            "ts": datetime.now(timezone.utc).isoformat(),
            "job_id": job_id,
            "gpu_uuid": self.gpu_uuid,
            "gpu_arch": self.gpu_arch,
            "model": model,
            "duration_s": round(duration_s, 6),
        }
        event.update(extra)

        line = json.dumps(event, separators=(",", ":")) + "\n"
        with self._lock:
            self._file.write(line)
            self._file.flush()

    # ── Convenience methods ──────────────────────────────────────────

    def model_load(self, job_id: str, model: str | None, component: str,
                   duration_s: float, vram_delta: int) -> None:
        self.record("model_load", job_id, model, duration_s,
                    component=component, vram_delta=vram_delta)

    def text_encode(self, job_id: str, model: str | None, encoder: str,
                    chunks: int, direction: str, duration_s: float) -> None:
        self.record("text_encode", job_id, model, duration_s,
                    encoder=encoder, chunks=chunks, direction=direction)

    def denoise_step(self, job_id: str, model: str | None, step: int,
                     total_steps: int, timestep: int, duration_s: float,
                     tiled: bool) -> None:
        self.record("denoise_step", job_id, model, duration_s,
                    step=step, total_steps=total_steps,
                    timestep=timestep, tiled=tiled)

    def denoise_setup(self, job_id: str, model: str | None, duration_s: float,
                      has_lora: bool, lora_name: str | None) -> None:
        self.record("denoise_setup", job_id, model, duration_s,
                    has_lora=has_lora, lora_name=lora_name)

    def vae_tile(self, job_id: str, model: str | None, tile: int,
                 total_tiles: int, tile_w: int, tile_h: int,
                 op: str, duration_s: float) -> None:
        self.record("vae_tile", job_id, model, duration_s,
                    tile=tile, total_tiles=total_tiles,
                    tile_w=tile_w, tile_h=tile_h, op=op)

    def stage_complete(self, job_id: str, model: str | None, stage: str,
                       width: int, height: int, steps: int | None,
                       total_duration_s: float) -> None:
        self.record("stage_complete", job_id, model, total_duration_s,
                    stage=stage, width=width, height=height, steps=steps)


# ── Module-level registry ────────────────────────────────────────────

_tracers: dict[str, GpuTracer] = {}  # gpu_uuid → GpuTracer
_tracers_lock = threading.Lock()


def get_tracer(gpu_uuid: str) -> GpuTracer | None:
    """Look up tracer by GPU UUID."""
    return _tracers.get(gpu_uuid)


def register_tracer(gpu_uuid: str, gpu_arch: str, gpu_name: str) -> GpuTracer:
    """Create and register a tracer for a GPU. Closes any existing tracer for the same UUID."""
    with _tracers_lock:
        old = _tracers.get(gpu_uuid)
        if old is not None:
            old.close()
        tracer = GpuTracer(gpu_uuid, gpu_arch, gpu_name)
        _tracers[gpu_uuid] = tracer
    log.info(f"Profiling tracer registered for {gpu_name} ({gpu_arch}, {gpu_uuid})")
    return tracer


# ── Thread-local current tracer ──────────────────────────────────────

_thread_local = threading.local()


def set_current_tracer(tracer: GpuTracer | None) -> None:
    """Set the tracer for the current thread (called by worker before stage execution)."""
    _thread_local.tracer = tracer


def get_current_tracer() -> GpuTracer | None:
    """Get the tracer for the current thread (called by handlers)."""
    return getattr(_thread_local, "tracer", None)
