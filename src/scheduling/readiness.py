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

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from gpu.pool import GpuPool
    from scheduling.model_registry import ModelRegistry
    from scheduling.worker_proxy import GpuProxy, GpuWorkerProxy as GpuWorker


# Default model load rate (bytes/sec) for estimated_ready_s calculations.
# 500 MB/s = NVMe safetensors sequential read speed.
LOAD_RATE = 500 * 1024 * 1024


def _estimate_remaining_s(worker: GpuWorker) -> float:
    """Estimate seconds until a GPU worker becomes idle.

    With one-job-per-GPU, this estimates the remaining time for the
    single active job (if any).
    """
    from datetime import datetime

    jobs = worker.active_jobs
    if not jobs:
        return 0.0

    job = jobs[0]

    # If the job is in denoise and we have step progress, estimate from that
    if job.denoise_step > 0 and job.denoise_total_steps > 0:
        # Try to use per-stage timing from gpu_stage_times to get
        # an accurate per-step rate that excludes text-encode time.
        denoise_elapsed = None
        for st in job.gpu_stage_times:
            if st.get("stage") == "GpuDenoise":
                denoise_elapsed = (denoise_elapsed or 0) + st.get("duration_s", 0)

        if denoise_elapsed is not None and denoise_elapsed > 0:
            time_per_step = denoise_elapsed / job.denoise_step
        elif job.started_at:
            elapsed = (datetime.utcnow() - job.started_at).total_seconds()
            time_per_step = min(elapsed / job.denoise_step, 5.0)
        else:
            time_per_step = 0.5

        return (job.denoise_total_steps - job.denoise_step) * time_per_step

    # Non-denoise or no progress yet
    return 3.0


def _score_simple_task(
    gpu: GpuProxy,
    worker: GpuWorker,
    target_category: str,
) -> dict:
    """Score a simple task (upscale/bgremove/tag) for a specific GPU+worker.

    Returns dict with score, estimated_wait_s, model_loaded.
    """
    score = 0
    model_loaded = target_category in gpu.get_cached_categories()

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

    # VRAM deficit penalty — if model isn't loaded and VRAM is tight,
    # loading it may require evicting something.
    from scheduling.model_registry import VramEstimates
    needed = {
        "upscale": VramEstimates.UPSCALE,
        "bgremove": VramEstimates.BGREMOVE,
        "tagger": VramEstimates.TAGGER,
    }.get(target_category, 300 * 1024 * 1024)

    if not model_loaded:
        vram = gpu.get_vram_stats()
        free = vram.get("free", 0) if vram else 0
        effective_free = free + gpu.get_evictable_vram()
        deficit = max(0, needed - effective_free)
        if deficit > 0:
            score -= (deficit // (100 * 1024 * 1024)) * 30

    load_time = (needed / LOAD_RATE) if not model_loaded else 0.0
    return {
        "score": score,
        "best_gpu": gpu.uuid,
        "estimated_wait_s": round(wait_s, 1),
        "estimated_ready_s": round(wait_s + load_time, 1),
        "model_loaded": model_loaded,
    }


def _score_sdxl_checkpoint(
    gpu: GpuProxy,
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

    # VRAM eviction penalty — 30 points per 100MB of deficit,
    # so a full 3.5GB deficit = ~1050 penalty (competitive with component bonuses)
    if missing_vram > 0:
        vram = gpu.get_vram_stats()
        free = vram.get("free", 0) if vram else 0
        effective_free = free + gpu.get_evictable_vram()
        deficit = max(0, missing_vram - effective_free)
        if deficit > 0:
            score -= (deficit // (100 * 1024 * 1024)) * 30

    load_time = (missing_vram / LOAD_RATE) if missing_vram > 0 else 0.0
    return {
        "score": score,
        "best_gpu": gpu.uuid,
        "estimated_wait_s": round(wait_s, 1),
        "estimated_ready_s": round(wait_s + load_time, 1),
        "components_loaded": loaded_count,
        "components_needed": len(scoring_components),
        "missing_vram_bytes": missing_vram,
        "cached_lora_count": gpu.loaded_lora_count,
    }


def compute_readiness(
    pool: GpuPool,
    workers: list[GpuWorker],
    registry: ModelRegistry,
    sdxl_models: dict[str, str],
) -> dict:
    """Compute readiness scores for all task types and SDXL checkpoints.

    Args:
        pool: GPU pool (unused, kept for API compatibility)
        workers: list of GpuWorkerProxy (one per GPU)
        registry: model registry with SDXL checkpoint info
        sdxl_models: dict of model_name -> model_dir

    Returns:
        Dict matching the readiness response structure for /api/status.
    """
    # Snapshot worker list to avoid iteration-during-mutation
    workers = list(workers)

    result: dict = {}

    # Simple tasks: upscale, bgremove, tag
    simple_tasks = {
        "upscale": "upscale",
        "bgremove": "bgremove",
        "tag": "tagger",
    }

    for task_key, category in simple_tasks.items():
        best: dict | None = None

        for worker in workers:
            if not worker.gpu.supports_capability(task_key):
                continue

            entry = _score_simple_task(worker.gpu, worker, category)
            if best is None or entry["score"] > best["score"]:
                best = entry

        if best is not None:
            result[task_key] = best

    # SDXL tasks: per checkpoint
    sdxl_result: dict = {}

    for model_name, model_dir in sdxl_models.items():
        best: dict | None = None

        for worker in workers:
            if not worker.gpu.supports_capability("sdxl"):
                continue

            entry = _score_sdxl_checkpoint(worker.gpu, worker, model_dir, registry)
            if best is None or entry["score"] > best["score"]:
                best = entry

        if best is not None:
            sdxl_result[model_name] = best

    if sdxl_result:
        result["sdxl"] = sdxl_result

    return result
