"""Readiness scoring for C# dispatch — computes per-task-type scores for each GPU.

Score scale (approximate ranges):
  Simple tasks: -700 to +700  (model loaded+idle = ~700, cold+busy = deeply negative)
  SDXL tasks:   -9999 to +1280 (full pipeline+idle = ~1280, nothing loaded+busy = negative)

Penalty/bonus magnitudes are calibrated so that:
  - Having the model loaded always beats not having it (unless GPU is extremely busy)
  - VRAM pressure is meaningful but doesn't dominate cache affinity
  - Busy penalties taper — a GPU finishing in 2s is still attractive if it has models loaded
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu.pool import GpuInstance, GpuPool
    from scheduling.model_registry import ModelRegistry
    from scheduling.worker import GpuWorker


# Duplicated from gpu/pool.py to avoid circular imports at runtime.
# Must stay in sync with gpu.pool._get_group_for_category.
def _get_group_for_category(category: str) -> str:
    """Map model category to session group."""
    if category.startswith("sdxl") or category == "sdxl_lora":
        return "sdxl"
    if category == "upscale":
        return "upscale"
    if category == "bgremove":
        return "bgremove"
    if category == "tagger":
        return "tagger"
    return "unknown"


def _estimate_remaining_s(worker: GpuWorker) -> float:
    """Estimate seconds until a GPU worker becomes idle.

    For denoise stages with progress, extrapolates from step timing.
    Uses gpu_stage_times for per-stage timing to avoid counting text-encode
    time in the denoise rate estimate.  Falls back to job.started_at if no
    stage timing is available.
    """
    from datetime import datetime
    import time as _time

    max_remaining = 0.0
    for job in worker.active_jobs:
        stage = job.current_stage
        if stage is None:
            continue

        is_denoise = stage.type.value == "GpuDenoise"
        if is_denoise and job.denoise_step > 0 and job.denoise_total_steps > 0:
            # Try to use per-stage timing from gpu_stage_times to get
            # an accurate per-step rate that excludes text-encode time.
            denoise_elapsed = None
            for st in job.gpu_stage_times:
                if st.get("stage") == "GpuDenoise":
                    denoise_elapsed = (denoise_elapsed or 0) + st.get("duration_s", 0)

            if denoise_elapsed is not None and denoise_elapsed > 0:
                time_per_step = denoise_elapsed / job.denoise_step
            elif job.started_at:
                # Fallback: use total elapsed but clamp per-step to a
                # reasonable max to avoid the text-encode inflation bug.
                elapsed = (datetime.utcnow() - job.started_at).total_seconds()
                time_per_step = min(elapsed / job.denoise_step, 5.0)
            else:
                time_per_step = 0.5  # reasonable default for SDXL

            remaining = (job.denoise_total_steps - job.denoise_step) * time_per_step
        else:
            remaining = 3.0  # flat estimate for non-denoise stages

        max_remaining = max(max_remaining, remaining)

    return max_remaining


def _would_evict_for_vram(gpu: GpuInstance, target_group: str, needed_vram: int) -> list[str]:
    """Return categories that would actually need evicting to free needed_vram.

    Only counts models from non-target groups that LRU eviction would remove,
    stopping once enough VRAM would be freed.  Returns empty list if enough
    free VRAM exists without eviction.
    """
    vram = gpu.get_vram_stats()
    deficit = needed_vram - vram["free"]
    if deficit <= 0:
        return []

    evicted = []
    freed = 0
    with gpu._cache_lock:
        active_fps = gpu.get_active_fingerprints()
        # Walk LRU order (oldest first), matching evict_lru's two-pass strategy:
        # Pass 1: non-current-group models
        for fp in list(gpu._cache.keys()):
            if freed >= deficit:
                break
            if fp in active_fps or fp in gpu._unevictable_fingerprints:
                continue
            m = gpu._cache[fp]
            if _get_group_for_category(m.category) != target_group:
                evicted.append(m.category)
                freed += m.actual_vram if m.actual_vram > 0 else m.estimated_vram
        # Pass 2: any evictable model (if still short)
        if freed < deficit:
            for fp in list(gpu._cache.keys()):
                if freed >= deficit:
                    break
                if fp in active_fps or fp in gpu._unevictable_fingerprints:
                    continue
                m = gpu._cache[fp]
                cat = m.category
                if cat not in evicted:
                    evicted.append(cat)
                    freed += m.actual_vram if m.actual_vram > 0 else m.estimated_vram
    return evicted


def _score_simple_task(
    gpu: GpuInstance,
    worker: GpuWorker,
    target_category: str,
    target_group: str,
) -> dict:
    """Score a simple task (upscale/bgremove/tag) for a specific GPU+worker.

    Returns dict with score, estimated_wait_s, model_loaded, would_evict.
    """
    score = 0
    model_loaded = False

    # Check if the model is already loaded (single lock acquisition for both
    # model_loaded check and eviction analysis)
    with gpu._cache_lock:
        for m in gpu._cache.values():
            if m.category == target_category:
                model_loaded = True
                break

    if model_loaded:
        score += 500
    else:
        score -= 200

    # GPU busy state — penalty coefficient of 15/sec means a 30s wait = -450,
    # which still loses to a model-loaded bonus of +500 but beats a cold GPU at -200.
    wait_s = _estimate_remaining_s(worker)
    if worker.is_idle:
        score += 200
    else:
        score -= int(15 * wait_s)

    # Eviction analysis — only penalize models that would actually be evicted
    # due to VRAM pressure, not all cross-group cached models
    from scheduling.model_registry import VramEstimates
    needed = {
        "upscale": VramEstimates.UPSCALE,
        "bgremove": VramEstimates.BGREMOVE,
        "tagger": VramEstimates.TAGGER,
    }.get(target_category, 300 * 1024 * 1024)

    if not model_loaded:
        would_evict = _would_evict_for_vram(gpu, target_group, needed)
    else:
        would_evict = []

    for cat in would_evict:
        cat_group = _get_group_for_category(cat)
        if cat_group == "sdxl":
            score -= 300
        else:
            score -= 100

    return {
        "score": score,
        "best_gpu": gpu.uuid,
        "estimated_wait_s": round(wait_s, 1),
        "model_loaded": model_loaded,
        "would_evict": would_evict,
    }


def _score_sdxl_checkpoint(
    gpu: GpuInstance,
    worker: GpuWorker,
    model_dir: str,
    registry: ModelRegistry,
) -> dict:
    """Score an SDXL checkpoint for a specific GPU+worker.

    Returns dict with score, estimated_wait_s, components_loaded/needed, missing_vram_bytes.
    """
    try:
        components = registry.get_sdxl_components(model_dir)
    except KeyError:
        return {
            "score": -9999,
            "best_gpu": gpu.uuid,
            "estimated_wait_s": 0.0,
            "components_loaded": 0,
            "components_needed": 0,
            "missing_vram_bytes": 0,
        }

    # Components: [te1, te2, unet, vae_dec, vae_enc]
    # For scoring we care about the first 4 (te1, te2, unet, vae_dec)
    scoring_components = components[:4]
    component_weights = {
        "sdxl_unet": 500,
        "sdxl_te2": 150,
        "sdxl_te1": 50,
        "sdxl_vae": 30,
    }

    score = 0
    loaded_count = 0
    missing_vram = 0

    for comp in scoring_components:
        if gpu.is_component_loaded(comp.fingerprint):
            loaded_count += 1
            score += component_weights.get(comp.category, 30)
        else:
            missing_vram += comp.estimated_vram_bytes

    # Full pipeline bonus
    if loaded_count == len(scoring_components):
        score += 300

    # GPU busy state — 15/sec so a 30s wait = -450, still loses to full pipeline (~1030)
    wait_s = _estimate_remaining_s(worker)
    if worker.is_idle:
        score += 200
    else:
        score -= int(15 * wait_s)

    # Session group affinity
    if gpu._current_group == "sdxl":
        score += 50
    elif gpu._current_group is not None:
        score -= 50

    # VRAM eviction penalty — 30 points per 100MB of deficit,
    # so a full 3.5GB deficit = ~1050 penalty (competitive with component bonuses)
    if missing_vram > 0:
        vram = gpu.get_vram_stats()
        deficit = max(0, missing_vram - vram["free"])
        if deficit > 0:
            score -= (deficit // (100 * 1024 * 1024)) * 30

    return {
        "score": score,
        "best_gpu": gpu.uuid,
        "estimated_wait_s": round(wait_s, 1),
        "components_loaded": loaded_count,
        "components_needed": len(scoring_components),
        "missing_vram_bytes": missing_vram,
        "cached_lora_count": len(gpu._loaded_lora_adapters),
    }


def compute_readiness(
    pool: GpuPool,
    workers: list[GpuWorker],
    registry: ModelRegistry,
    sdxl_models: dict[str, str],
) -> dict:
    """Compute readiness scores for all task types and SDXL checkpoints.

    Args:
        pool: GPU pool with all GPU instances
        workers: list of GpuWorker (one per GPU)
        registry: model registry with SDXL checkpoint info
        sdxl_models: dict of model_name -> model_dir

    Returns:
        Dict matching the readiness response structure for /api/status.
    """
    # Snapshot worker list to avoid iteration-during-mutation
    workers = list(workers)

    # Build GPU UUID -> worker lookup
    worker_by_uuid: dict[str, GpuWorker] = {}
    for w in workers:
        worker_by_uuid[w.gpu.uuid] = w

    result: dict = {}

    # Simple tasks: upscale, bgremove, tag
    simple_tasks = {
        "upscale": ("upscale", "upscale"),
        "bgremove": ("bgremove", "bgremove"),
        "tag": ("tagger", "tagger"),
    }

    for task_key, (category, group) in simple_tasks.items():
        best: dict | None = None

        for gpu in pool.gpus:
            if not gpu.supports_capability(task_key):
                continue

            worker = worker_by_uuid.get(gpu.uuid)
            if worker is None:
                continue

            entry = _score_simple_task(gpu, worker, category, group)
            if best is None or entry["score"] > best["score"]:
                best = entry

        if best is not None:
            result[task_key] = best

    # SDXL tasks: per checkpoint
    sdxl_result: dict = {}

    for model_name, model_dir in sdxl_models.items():
        best: dict | None = None

        for gpu in pool.gpus:
            if not gpu.supports_capability("sdxl"):
                continue

            worker = worker_by_uuid.get(gpu.uuid)
            if worker is None:
                continue

            entry = _score_sdxl_checkpoint(gpu, worker, model_dir, registry)
            if best is None or entry["score"] > best["score"]:
                best = entry

        if best is not None:
            sdxl_result[model_name] = best

    if sdxl_result:
        result["sdxl"] = sdxl_result

    return result
