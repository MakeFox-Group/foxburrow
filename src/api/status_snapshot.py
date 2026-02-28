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

    workers_by_uuid: dict[str, Any] = {}
    if scheduler and scheduler.workers:
        for w in scheduler.workers:
            workers_by_uuid[w.gpu.uuid] = w

    # Per-GPU info (lightweight: no active_jobs detail)
    gpus = []
    for g in pool.gpus:
        gpu_info: dict[str, Any] = {
            "uuid": g.uuid,
            "busy": g.is_busy,
            "active_count": 0,
            "loaded_models": g.get_cached_models_info(),
            "vram": g.get_vram_stats(),
        }

        worker = workers_by_uuid.get(g.uuid)
        if worker:
            gpu_info["active_count"] = worker.active_count
            from scheduling.worker import estimate_gpu_slots
            try:
                gpu_info["slots"] = estimate_gpu_slots(worker)
            except Exception as ex:
                log.log_exception(ex, f"status_snapshot: slots for GPU [{g.uuid}]")
                gpu_info["slots"] = {}

        gpus.append(gpu_info)

    # Readiness scores
    readiness = {}
    sdxl_models_snapshot = dict(app_state.sdxl_models)
    if scheduler and scheduler.workers:
        from scheduling.readiness import compute_readiness
        try:
            readiness = compute_readiness(
                pool, scheduler.workers, app_state.registry, sdxl_models_snapshot)
        except Exception as ex:
            log.log_exception(ex, "status_snapshot: readiness")

    # Available slots (VRAM-aware)
    available_slots: dict[str, int] = {}
    if scheduler:
        try:
            available_slots = scheduler.estimate_available_slots()
        except Exception as ex:
            log.log_exception(ex, "status_snapshot: available_slots")

    # Tag slots
    from handlers.tagger import is_loaded_on
    tag_count = sum(1 for g in pool.gpus
                    if g.supports_capability("tag") and is_loaded_on(g.device))
    available_slots["tag"] = tag_count

    # Max concurrent + admission
    admission_snapshot = None
    if app_state.admission is not None:
        admission_snapshot = app_state.admission.snapshot()
        max_concurrent = app_state.admission.max_concurrent
    else:
        num_active_gpus = sum(1 for g in pool.gpus if not g.is_failed)
        max_concurrent = num_active_gpus + (num_active_gpus // 2)

    return {
        "gpus": gpus,
        "readiness": readiness,
        "available_slots": available_slots,
        "max_concurrent": max_concurrent,
        "admission": admission_snapshot,
        "queue_depth": app_state.queue.count if app_state.queue else 0,
    }
