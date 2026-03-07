"""Lightweight status snapshot shared by /api/status and WebSocket push.

Excludes per-job active details (already streamed via WS progress events),
full queue job list, and model_scan progress to keep periodic pushes small.
"""

from __future__ import annotations

import os
from typing import Any

import log


def compute_status_snapshot() -> dict[str, Any]:
    """Build a status snapshot suitable for WebSocket push and /api/status base data.

    Returns dict with: gpus, readiness, available_slots, max_concurrent,
    admission, queue_depth.
    """
    from state import app_state

    pool = app_state.gpu_pool
    scheduler = app_state.scheduler

    workers = list(scheduler.workers) if scheduler and scheduler.workers else []

    # Per-GPU info (lightweight: no active_jobs detail)
    gpus = []
    for w in workers:
        gpu_info: dict[str, Any] = {
            "uuid": w.gpu.uuid,
            "busy": w.gpu.is_busy,
            "active_count": w.active_count,
            "loaded_models": w.gpu.get_cached_models_info(),
            "vram": w.gpu.get_vram_stats(),
        }

        from scheduling.worker import estimate_gpu_slots
        try:
            gpu_info["slots"] = estimate_gpu_slots(w)
        except Exception as ex:
            log.log_exception(ex, f"status_snapshot: slots for GPU [{w.gpu.uuid}]")
            gpu_info["slots"] = {}

        gpus.append(gpu_info)

    # Readiness scores
    readiness = {}
    sdxl_models_snapshot = dict(app_state.sdxl_models)
    if workers:
        from scheduling.readiness import compute_readiness
        try:
            readiness = compute_readiness(
                pool, workers, app_state.registry, sdxl_models_snapshot)
        except Exception as ex:
            log.log_exception(ex, "status_snapshot: readiness")

    # Available slots (VRAM-aware)
    available_slots: dict[str, int] = {}
    if scheduler:
        try:
            available_slots = scheduler.estimate_available_slots()
        except Exception as ex:
            log.log_exception(ex, "status_snapshot: available_slots")

    # Tag slots — check via proxy cached categories
    tag_count = sum(1 for w in workers
                    if w.gpu.supports_capability("tag")
                    and "tagger" in w.gpu.get_cached_categories())
    available_slots["tag"] = tag_count

    # Max concurrent + admission
    admission_snapshot = None
    if app_state.admission is not None:
        admission_snapshot = app_state.admission.snapshot()
        max_concurrent = app_state.admission.max_concurrent
    else:
        num_idle_gpus = sum(1 for w in workers
                            if not w.gpu.is_busy and not w.gpu.is_failed
                            and not w._draining and not w._building)
        max_concurrent = num_idle_gpus + 2

    # Model affinity: which SDXL checkpoint sources are loaded on which GPUs
    # makefoxsrv uses this for intelligent routing — send same-model jobs to
    # the foxburrow instance that already has the model loaded.
    model_affinity: dict[str, dict[str, int]] = {}
    for w in workers:
        if w.gpu.is_failed:
            continue
        for m_info in w.gpu.get_cached_models_info():
            if m_info.get("category") == "sdxl_unet":
                source = m_info.get("source", "")
                if not source:
                    continue
                if source not in model_affinity:
                    model_affinity[source] = {"idle": 0, "busy": 0}
                if w.gpu.is_busy:
                    model_affinity[source]["busy"] += 1
                else:
                    model_affinity[source]["idle"] += 1

    return {
        "gpus": gpus,
        "readiness": readiness,
        "available_slots": available_slots,
        "max_concurrent": max_concurrent,
        "admission": admission_snapshot,
        "queue_depth": app_state.queue.count if app_state.queue else 0,
        "model_affinity": model_affinity,
    }
