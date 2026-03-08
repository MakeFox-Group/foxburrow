"""REST API endpoints — served under /api/ prefix."""

from __future__ import annotations

import asyncio
import io
import os
import random
import struct
import json
from datetime import datetime
from typing import Any

import numpy as np
import torch
from fastapi import APIRouter, Query, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field, field_validator

import log

router = APIRouter()


# ====================================================================
# Request Models
# ====================================================================

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1024
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = 0
    subseed: int = 0
    subseed_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    model: str | None = None
    vae_tile_width: int = 0    # VAE decode tile width in pixels (0 = auto ≤ 1024)
    vae_tile_height: int = 0   # VAE decode tile height in pixels (0 = auto ≤ 1024)
    loras: list[dict] | None = None  # [{"name": "xxx", "weight": 1.0}, ...]
    regional_prompting: bool = False
    sampler: str = "Euler A"
    scheduler: str | None = None
    priority: int = Field(default=100, ge=1, le=100)
    clip_embeddings: list[dict] | None = None  # Pre-computed CLIP cache entries (6 items)


class GenerateHiresRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1536
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = 0
    subseed: int = 0
    subseed_strength: float = Field(default=0.0, ge=0.0, le=1.0)
    hires_steps: int = 15
    hires_denoising_strength: float = 0.33
    base_width: int | None = None
    base_height: int | None = None
    model: str | None = None
    unet_tile_width: int = 0   # Reserved (unused)
    unet_tile_height: int = 0  # Reserved (unused)
    loras: list[dict] | None = None  # [{"name": "xxx", "weight": 1.0}, ...]
    regional_prompting: bool = False
    sampler: str = "Euler A"
    scheduler: str | None = None
    vae_tile_width: int = 0    # VAE encode/decode tile width in pixels (0 = auto ≤ 1024)
    vae_tile_height: int = 0   # VAE encode/decode tile height in pixels (0 = auto ≤ 1024)
    priority: int = Field(default=100, ge=1, le=100)
    clip_embeddings: list[dict] | None = None  # Pre-computed CLIP cache entries (6 items)


# ====================================================================
# Helpers
# ====================================================================

def _get_state():
    """Get the global app state."""
    from state import app_state
    return app_state


def _error(status: int, message: str) -> JSONResponse:
    return JSONResponse(status_code=status, content={"error": message})


def _resolve_model(model_name: str | None) -> tuple[str | None, str | None]:
    """Resolve model name to model dir. Returns (model_dir, error_message)."""
    state = _get_state()
    if not model_name:
        return None, "Model name is required."
    models = state.sdxl_models
    if model_name not in models:
        return None, f'Unknown model "{model_name}".'
    return models[model_name], None


def _check_admission(job_type) -> JSONResponse | None:
    """Check admission control. Returns a 503 error response if rejected, None if admitted."""
    state = _get_state()
    if state.admission is None:
        return None
    err = state.admission.try_admit(job_type)
    if err:
        return _error(503, err)
    return None


def _release_admission(job_type) -> None:
    """Release an admission slot (used when enqueue fails after admission)."""
    state = _get_state()
    if state.admission is not None:
        state.admission.release(job_type)


def _find_proxy_with_tagger(state):
    """Find a worker proxy with the tagger loaded.

    Taggers can always run regardless of other GPU work (tiny VRAM footprint,
    fast inference), so we don't filter by busyness — just pick the first
    loaded tagger, preferring idle GPUs for best latency.
    """
    if state.scheduler is None:
        return None
    best = None
    for proxy in state.scheduler.workers:
        if proxy.gpu.supports_capability("tag") and "tagger" in proxy.gpu.get_cached_categories():
            if proxy.is_idle:
                return proxy  # prefer idle for best latency
            if best is None:
                best = proxy
    return best


def _parse_request_loras(prompt: str, negative_prompt: str,
                          explicit_loras: list[dict] | None) -> tuple[str, str, list]:
    """Parse LoRA tags from prompts and merge with explicit request loras.

    Returns (cleaned_prompt, cleaned_neg, all_lora_specs).
    """
    from utils.lora_parser import parse_lora_tags, LoraSpec

    cleaned_prompt, prompt_loras = parse_lora_tags(prompt)
    cleaned_neg, neg_loras = parse_lora_tags(negative_prompt)
    all_loras = prompt_loras + neg_loras

    # Also accept explicit loras from request body
    if explicit_loras:
        for l in explicit_loras:
            spec = LoraSpec(name=l["name"], weight=l.get("weight", 1.0))
            if not any(s.name == spec.name for s in all_loras):
                all_loras.append(spec)

    return cleaned_prompt, cleaned_neg, all_loras


def _parse_clip_cache_entries(entries: list[dict]):
    """Parse clip_embeddings from request into ClipCacheEntry list."""
    import base64
    from scheduling.worker_protocol import ClipCacheEntry
    result = []
    for e in entries:
        result.append(ClipCacheEntry(
            encoder_type=e["encoder_type"],
            polarity=e["polarity"],
            data=base64.b64decode(e["data"]),
            dtype=e["dtype"],
            dim0=e["dim0"],
            dim1=e.get("dim1"),
        ))
    return result


def _apply_clip_cache(req, job) -> None:
    """If the request has valid clip_embeddings, strip tokenize/text_encode
    from the job pipeline and store the parsed entries on the job."""
    from scheduling.job import StageType
    if not req.clip_embeddings or getattr(req, 'regional_prompting', False):
        return
    if len(req.clip_embeddings) != 6:
        return
    try:
        cached_clip = _parse_clip_cache_entries(req.clip_embeddings)
        # Preserve TE components before stripping so TRT protection still works
        te_stage = next((s for s in job.pipeline
                         if s.type == StageType.GPU_TEXT_ENCODE), None)
        if te_stage is not None:
            job.stripped_te_components = te_stage.required_components
        job.pipeline = [s for s in job.pipeline
                        if s.type not in (StageType.CPU_TOKENIZE,
                                          StageType.GPU_TEXT_ENCODE)]
        job.cached_clip_entries = cached_clip
    except Exception as ex:
        log.warning(f"Invalid clip_embeddings, ignoring cache: {ex}")


def _validate_dimensions(width: int, height: int, max_dim: int = 2048,
                          multiple: int = 64) -> str | None:
    if (width < multiple or width > max_dim
            or height < multiple or height > max_dim):
        return f"Width and height must be between {multiple}-{max_dim}."
    return None


def _snap_up(value: int, multiple: int = 64) -> int:
    """Round *value* up to the next multiple (e.g. 64)."""
    return ((value + multiple - 1) // multiple) * multiple


def _snap_dims(width: int, height: int, multiple: int = 64) -> tuple[int, int]:
    """Snap both dimensions up to the nearest *multiple*."""
    return _snap_up(width, multiple), _snap_up(height, multiple)


def _resize_if_needed(image: Image.Image, target_w: int, target_h: int) -> Image.Image:
    """Resize *image* down to target dimensions if it was generated at a
    snapped-up size.  Uses LANCZOS for quality."""
    if image.width != target_w or image.height != target_h:
        image = image.resize((target_w, target_h), Image.LANCZOS)
    return image


def _image_response(image: Image.Image, mode: str = "RGB") -> Response:
    """Convert PIL image to PNG response."""
    buf = io.BytesIO()
    if mode == "RGBA":
        image.save(buf, format="PNG")
    else:
        image.convert("RGB").save(buf, format="PNG")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="image/png")


# ====================================================================
# Endpoints
# ====================================================================

@router.get("/status")
async def status():
    from api.status_snapshot import compute_status_snapshot

    state = _get_state()
    pool = state.gpu_pool
    queue = state.queue
    scheduler = state.scheduler

    # Start from the shared snapshot (same data WebSocket push uses)
    result = compute_status_snapshot()

    # Augment GPU entries with full details (active jobs, name, device_id, etc.)
    workers_by_uuid: dict[str, Any] = {}
    if scheduler and scheduler.workers:
        for w in scheduler.workers:
            workers_by_uuid[w.gpu.uuid] = w

    enriched_gpus = []
    snapshot_gpus_by_uuid = {g["uuid"]: g for g in result["gpus"]}
    for g in pool.gpus:
        gpu_info = snapshot_gpus_by_uuid.get(g.uuid, {})
        # Add fields not included in the lightweight snapshot
        gpu_info["name"] = g.name
        gpu_info["device_id"] = g.device_id
        gpu_info["capabilities"] = sorted(g.capabilities)
        worker = workers_by_uuid.get(g.uuid)
        gpu_info["session_cache_count"] = worker.gpu.session_cache_count if worker else 0

        if worker:
            active_jobs = []
            for job in worker.active_jobs:
                stage = job.current_stage
                stage_type = stage.type.value if stage else None
                is_denoise = stage and stage.type.value == "GpuDenoise"
                denoise_step = job.denoise_step
                denoise_total = job.denoise_total_steps
                progress = (denoise_step / denoise_total) if is_denoise and denoise_total > 0 else None
                elapsed_s = None
                if job.started_at:
                    elapsed_s = round((datetime.utcnow() - job.started_at).total_seconds(), 1)

                model_name = None
                if job.sdxl_input and job.sdxl_input.model_dir:
                    model_name = os.path.splitext(os.path.basename(job.sdxl_input.model_dir))[0]

                active_jobs.append({
                    "job_id": job.job_id,
                    "type": job.type.value,
                    "model": model_name,
                    "stage": stage_type,
                    "stage_index": job.current_stage_index,
                    "stage_count": len(job.pipeline),
                    "denoise_step": denoise_step if is_denoise else None,
                    "denoise_total": denoise_total if is_denoise else None,
                    "progress": round(progress, 4) if progress is not None else None,
                    "elapsed_s": elapsed_s,
                })
            gpu_info["active_jobs"] = active_jobs

        enriched_gpus.append(gpu_info)

    all_caps = pool.get_all_capabilities()
    available = {cap: pool.available_count(cap) for cap in sorted(all_caps)}

    # Build queued jobs summary
    queued_jobs = []
    if queue:
        with queue._lock:
            for job in queue._jobs:
                if job.completion.done():
                    continue
                stage = job.current_stage
                model_name = None
                if job.sdxl_input and job.sdxl_input.model_dir:
                    model_name = os.path.splitext(os.path.basename(job.sdxl_input.model_dir))[0]
                queued_jobs.append({
                    "job_id": job.job_id,
                    "type": job.type.value,
                    "model": model_name,
                    "stage": stage.type.value if stage else None,
                })

    # Model scan progress
    model_scan = None
    scanner = state.model_scanner
    if scanner is not None:
        completed, total = scanner.progress
        model_scan = {
            "scanning": scanner.is_scanning,
            "completed": completed,
            "total": total,
        }

    result["gpus"] = enriched_gpus
    result["available"] = available
    result["total"] = len(pool.gpus)
    result["queue"] = {"depth": queue.count if queue else 0, "jobs": queued_jobs}
    result["model_scan"] = model_scan

    return result


@router.get("/lora-list")
async def lora_list():
    state = _get_state()
    return [{
        "name": entry.name,
        "path": entry.path,
        "fingerprint": entry.fingerprint,
        "size_bytes": entry.size_bytes,
    } for entry in sorted(state.lora_index.values(), key=lambda e: e.name)]


@router.post("/rescan-loras")
async def rescan_loras():
    state = _get_state()
    loras_dir = state.loras_dir
    if not loras_dir:
        return JSONResponse({"error": "No LoRA directory configured"}, status_code=500)

    from utils.lora_index import rescan_loras as do_rescan

    # Run the blocking rescan in a thread to avoid blocking the event loop
    summary = await asyncio.get_running_loop().run_in_executor(
        None, do_rescan, loras_dir, state.lora_index)

    # Propagate updated index to GPU worker processes
    from state import propagate_lora_index
    propagate_lora_index()

    return summary


@router.get("/model-list")
async def model_list():
    state = _get_state()
    models = dict(state.sdxl_models)
    return [{
        "name": name,
        "path": path,
        "type": "single_file" if path.endswith(".safetensors") else "diffusers",
    } for name, path in sorted(models.items())]


@router.post("/rescan-models")
async def rescan_models():
    """Rediscover SDXL models, register new ones, remove deleted ones."""
    from main import discover_sdxl_models
    from api.websocket import streamer
    from utils.model_scanner import ModelScanner
    from config import _auto_threads

    state = _get_state()
    models_dir = os.path.normpath(os.path.abspath(state.config.server.models_dir))
    sdxl_dir = os.path.join(models_dir, "sdxl")
    if not os.path.isdir(sdxl_dir):
        return JSONResponse({"error": "SDXL models directory not found"}, status_code=500)

    # Discover fresh model list (fast stat-based walk, no hashing)
    fresh = await asyncio.get_running_loop().run_in_executor(
        None, discover_sdxl_models, models_dir)

    old_names = set(state.sdxl_models.keys())
    new_names = set(fresh.keys())

    added = new_names - old_names
    removed = old_names - new_names

    # Remove deleted models
    for name in removed:
        path = state.sdxl_models.pop(name)
        streamer.fire_event("model_removed", {
            "model_type": "sdxl", "name": name, "path": path})

    # Propagate removals to GPU worker processes immediately
    if removed:
        from state import propagate_sdxl_models
        propagate_sdxl_models()

    # Register new models via ModelScanner (handles fingerprinting)
    if added:
        new_models = {name: fresh[name] for name in added}
        fp_threads = _auto_threads(state.config.threads.fingerprint, 8)
        scanner = ModelScanner(state.registry, state, max_workers=fp_threads)
        scanner.start(new_models)
        state.model_scanner = scanner

    summary = {
        "added": len(added),
        "removed": len(removed),
        "unchanged": len(old_names & new_names),
        "total": len(state.sdxl_models) + len(added),
    }

    log.debug(f"  Model rescan: +{summary['added']} -{summary['removed']} "
             f"={summary['unchanged']} (total: {summary['total']})")

    return summary


@router.delete("/gpu/{uuid}")
async def remove_gpu(uuid: str):
    """Remove a GPU from the pool at runtime."""
    state = _get_state()
    pool = state.gpu_pool
    scheduler = state.scheduler

    # Find the GPU
    gpu = None
    for g in pool.gpus:
        if g.uuid.lower() == uuid.lower():
            gpu = g
            break

    if gpu is None:
        return JSONResponse({"error": f"GPU {uuid} not found in pool"}, status_code=404)

    # Refuse if GPU has active jobs
    if scheduler and scheduler.workers:
        for w in scheduler.workers:
            if w.gpu is gpu and not w.is_idle:
                return JSONResponse(
                    {"error": f"GPU {uuid} has active jobs — wait for completion"},
                    status_code=409)

    # Cancel the worker task and remove from scheduler
    if scheduler and scheduler.workers:
        for i, w in enumerate(scheduler.workers):
            if w.gpu is gpu:
                if w._task is not None:
                    w._task.cancel()
                scheduler.workers.pop(i)
                break

    # Remove from pool (fires gpu_removed event)
    pool.remove_gpu(uuid)

    return {
        "removed": True,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "remaining_gpus": len(pool.gpus),
    }


@router.put("/gpu/{uuid}")
async def add_gpu(uuid: str):
    """Re-add a GPU to the pool at runtime from the original config."""
    state = _get_state()
    pool = state.gpu_pool
    scheduler = state.scheduler

    # Check if already active
    for g in pool.gpus:
        if g.uuid.lower() == uuid.lower():
            return JSONResponse(
                {"error": f"GPU {uuid} is already in the pool"}, status_code=409)

    # Find the GPU config
    config = state.config
    gpu_cfg = None
    for cfg in config.gpus:
        if cfg.uuid.lower() == uuid.lower():
            gpu_cfg = cfg
            break

    if gpu_cfg is None:
        return JSONResponse(
            {"error": f"GPU {uuid} not found in foxburrow.ini config"},
            status_code=404)

    # Re-initialize from NVML
    from gpu import nvml
    from gpu.pool import GpuInstance, _cuda_index_from_pci_bus_id

    try:
        nvml.init()
        devices = nvml.get_devices()
    except Exception as ex:
        return JSONResponse(
            {"error": f"NVML initialization failed: {ex}"}, status_code=500)

    # Match UUID to NVML device
    nvml_dev = None
    config_uuid = uuid.lower()
    for dev in devices:
        if dev.uuid.lower() == config_uuid:
            nvml_dev = dev
            break

    if nvml_dev is None:
        return JSONResponse(
            {"error": f"GPU {uuid} not found via NVML — hardware unavailable"},
            status_code=404)

    # Resolve CUDA index
    cuda_idx = _cuda_index_from_pci_bus_id(nvml_dev.pci_bus_id)
    if cuda_idx is None:
        cuda_idx = nvml_dev.index

    # Create GPU instance and add to pool
    gpu = GpuInstance(gpu_cfg, nvml_dev, cuda_idx)
    pool.gpus.append(gpu)

    # Fire event (set_loop was called during on_startup)
    from api.websocket import streamer
    await streamer.broadcast_event("gpu_added", {
        "uuid": gpu.uuid,
        "name": gpu.name,
        "device_id": cuda_idx,
        "total_memory": nvml_dev.total_memory,
        "capabilities": sorted(gpu.capabilities),
    })

    # TODO: hot-add needs rework for multiprocessing architecture.
    # Previously created a GpuWorker (thread-based); now needs to spawn a
    # worker subprocess + GpuWorkerProxy.  For now, GPU is added to pool
    # but has no worker — it won't receive dispatched work.
    log.warning(f"  GPU [{gpu.uuid}] added to pool but no worker spawned "
                f"(hot-add not yet supported with multiprocessing workers)")

    return {
        "added": True,
        "uuid": gpu.uuid,
        "name": gpu.name,
        "device_id": cuda_idx,
        "total_memory": nvml_dev.total_memory,
        "capabilities": sorted(gpu.capabilities),
        "total_gpus": len(pool.gpus),
    }


@router.post("/generate")
async def generate(req: GenerateRequest):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    err = _validate_dimensions(req.width, req.height)
    if err:
        return _error(400, err)
    if req.steps < 1 or req.steps > 150:
        return _error(400, "Steps must be between 1 and 150.")
    if req.cfg_scale < 1.0 or req.cfg_scale > 30.0:
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    if not req.prompt.strip():
        return _error(400, '"prompt" is required.')

    model_dir, err = _resolve_model(req.model)
    if err:
        return _error(400, err)

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)
    orig_w, orig_h = req.width, req.height
    gen_w, gen_h = _snap_dims(orig_w, orig_h)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"Generate request: {orig_w}x{orig_h} steps={req.steps} "
             f"cfg={req.cfg_scale} seed={seed} model={model_short}"
             + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.SDXL_GENERATE,
        pipeline=factory.create_sdxl_pipeline(model_dir),
        sdxl_input=SdxlJobInput(
            prompt=cleaned_prompt,
            negative_prompt=cleaned_neg,
            width=gen_w,
            height=gen_h,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            subseed=req.subseed,
            subseed_strength=req.subseed_strength,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
            sampler=req.sampler,
            scheduler=req.scheduler,
        ),
        priority=req.priority,
    )
    job.vae_tile_width  = req.vae_tile_width
    job.vae_tile_height = req.vae_tile_height

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    output = _resize_if_needed(result.output_image, orig_w, orig_h)
    log.debug(f"Generate complete: {result.output_image.width}x{result.output_image.height}"
             + (f" → {orig_w}x{orig_h}" if (gen_w, gen_h) != (orig_w, orig_h) else ""))
    return _image_response(output)


@router.post("/generate-hires")
async def generate_hires(req: GenerateHiresRequest):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlHiresInput
    from handlers.sdxl import calculate_base_resolution

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    err = _validate_dimensions(req.width, req.height, max_dim=4096)
    if err:
        return _error(400, err)
    if req.steps < 1 or req.steps > 150:
        return _error(400, "Steps must be between 1 and 150.")
    if req.hires_steps < 1 or req.hires_steps > 150:
        return _error(400, "hires_steps must be between 1 and 150.")
    if req.cfg_scale < 1.0 or req.cfg_scale > 30.0:
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    if req.hires_denoising_strength < 0.0 or req.hires_denoising_strength > 1.0:
        return _error(400, "hires_denoising_strength must be between 0.0 and 1.0.")
    if not req.prompt.strip():
        return _error(400, '"prompt" is required.')

    model_dir, err = _resolve_model(req.model)
    if err:
        return _error(400, err)

    orig_w, orig_h = req.width, req.height
    gen_w, gen_h = _snap_dims(orig_w, orig_h)

    # Calculate base resolution
    if req.base_width is not None and req.base_height is not None:
        base_w, base_h = req.base_width, req.base_height
        err = _validate_dimensions(base_w, base_h)
        if err:
            return _error(400, f"base_width/base_height: {err}")
        base_w, base_h = _snap_dims(base_w, base_h)
    else:
        base_w, base_h = calculate_base_resolution(req.width, req.height)

    if base_w > gen_w or base_h > gen_h:
        return _error(400, "Base dimensions must not exceed hires dimensions.")
    if base_w == gen_w and base_h == gen_h:
        return _error(400, "Base and hires dimensions are identical — use /generate instead.")

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"GenerateHires request: base={base_w}x{base_h} hires={orig_w}x{orig_h} "
             f"steps={req.steps} hires_steps={req.hires_steps} "
             f"strength={req.hires_denoising_strength:.2f} cfg={req.cfg_scale} "
             f"seed={seed} model={model_short}"
             + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.SDXL_GENERATE_HIRES,
        pipeline=factory.create_sdxl_hires_pipeline(model_dir),
        sdxl_input=SdxlJobInput(
            prompt=cleaned_prompt,
            negative_prompt=cleaned_neg,
            width=base_w,
            height=base_h,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            subseed=req.subseed,
            subseed_strength=req.subseed_strength,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
            sampler=req.sampler,
            scheduler=req.scheduler,
        ),
        hires_input=SdxlHiresInput(
            hires_width=gen_w,
            hires_height=gen_h,
            hires_steps=req.hires_steps,
            denoising_strength=req.hires_denoising_strength,
        ),
        priority=req.priority,
    )
    job.unet_tile_width  = req.unet_tile_width
    job.unet_tile_height = req.unet_tile_height
    job.vae_tile_width   = req.vae_tile_width
    job.vae_tile_height  = req.vae_tile_height

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    output = _resize_if_needed(result.output_image, orig_w, orig_h)
    log.debug(f"GenerateHires complete: {result.output_image.width}x{result.output_image.height}"
             + (f" → {orig_w}x{orig_h}" if (gen_w, gen_h) != (orig_w, orig_h) else ""))
    return _image_response(output)


@router.post("/generate-latents")
async def generate_latents(req: GenerateRequest):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    err = _validate_dimensions(req.width, req.height)
    if err:
        return _error(400, err)
    if req.steps < 1 or req.steps > 150:
        return _error(400, "Steps must be between 1 and 150.")
    if req.cfg_scale < 1.0 or req.cfg_scale > 30.0:
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    if not req.prompt.strip():
        return _error(400, '"prompt" is required.')

    model_dir, err = _resolve_model(req.model)
    if err:
        return _error(400, err)

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)
    gen_w, gen_h = _snap_dims(req.width, req.height)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"GenerateLatents request: {req.width}x{req.height} steps={req.steps} "
             f"cfg={req.cfg_scale} seed={seed} model={model_short}"
             + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.SDXL_GENERATE_LATENTS,
        pipeline=factory.create_sdxl_latents_pipeline(model_dir),
        sdxl_input=SdxlJobInput(
            prompt=cleaned_prompt,
            negative_prompt=cleaned_neg,
            width=gen_w,
            height=gen_h,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            subseed=req.subseed,
            subseed_strength=req.subseed_strength,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
            sampler=req.sampler,
            scheduler=req.scheduler,
        ),
        priority=req.priority,
    )

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    latents = result.output_latents
    shape = list(latents.shape)
    float_data = latents.cpu().float().numpy().tobytes()

    metadata = json.dumps({
        "width": req.width,
        "height": req.height,
        "seed": seed,
        "model": model_short,
        "steps": req.steps,
        "cfg_scale": req.cfg_scale,
    }).encode("utf-8")

    # Build FXLT binary
    buf = io.BytesIO()
    buf.write(b"FXLT")
    buf.write(struct.pack("<H", 1))          # version
    buf.write(struct.pack("<H", len(shape)))  # ndims
    for d in shape:
        buf.write(struct.pack("<i", d))       # shape[i]
    buf.write(struct.pack("<I", len(metadata)))
    buf.write(metadata)
    buf.write(float_data)

    buf.seek(0)
    log.debug(f"GenerateLatents complete: shape={shape} "
             f"({len(float_data)} bytes latent, {len(metadata)} bytes meta)")

    return Response(content=buf.getvalue(), media_type="application/x-fox-latent")


@router.post("/decode-latents")
async def decode_latents(request: Request):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    body = await request.body()
    buf = io.BytesIO(body)

    # Parse FXLT header
    magic = buf.read(4)
    if magic != b"FXLT":
        return _error(400, f'Invalid FXLT magic: "{magic.decode(errors="replace")}". Expected "FXLT".')

    version = struct.unpack("<H", buf.read(2))[0]
    if version != 1:
        return _error(400, f"Unsupported FXLT version: {version}. Expected 1.")

    ndims = struct.unpack("<H", buf.read(2))[0]
    if ndims > 8:
        return _error(400, f"FXLT ndims {ndims} exceeds maximum (8).")
    shape = [struct.unpack("<i", buf.read(4))[0] for _ in range(ndims)]
    for d in shape:
        if d <= 0 or d > 32768:
            return _error(400, f"Invalid FXLT shape dimension: {d}.")

    meta_len = struct.unpack("<I", buf.read(4))[0]
    if meta_len > 65536:
        return _error(400, f"FXLT metadata too large: {meta_len} bytes (max 65536).")
    model_name = None
    if meta_len > 0:
        try:
            meta = json.loads(buf.read(meta_len))
            model_name = meta.get("model")
        except Exception:
            pass  # cursor already advanced by buf.read above

    # Allow model override via query param
    query_model = request.query_params.get("model")
    if query_model:
        model_name = query_model

    total_floats = 1
    for d in shape:
        total_floats *= d
    expected_bytes = total_floats * 4
    latent_bytes = buf.read(expected_bytes)
    if len(latent_bytes) != expected_bytes:
        return _error(400,
            f"Latent data size mismatch: got {len(latent_bytes)} bytes, "
            f"expected {expected_bytes} for shape {shape}.")

    float_data = np.frombuffer(latent_bytes, dtype=np.float32).reshape(shape)
    latent_tensor = torch.from_numpy(float_data.copy())

    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"DecodeLatents request: shape={shape} model={model_short}")

    qp = request.query_params
    try:
        tile_w = int(qp.get("vae_tile_width",  qp.get("tile_w", "0")))
        tile_h = int(qp.get("vae_tile_height", qp.get("tile_h", "0")))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid tile parameter: {ex}")

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.SDXL_DECODE_LATENTS,
        pipeline=factory.create_sdxl_decode_latents_pipeline(model_dir),
        sdxl_input=SdxlJobInput(
            prompt="",
            negative_prompt="",
            model_dir=model_dir,
        ),
    )
    job.latents = latent_tensor
    job.vae_tile_width = tile_w
    job.vae_tile_height = tile_h

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    log.debug(f"DecodeLatents complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image)


@router.post("/encode-latents")
async def encode_latents(request: Request):
    """VAE-encode a PNG image to an FXLT latent file.

    Body: PNG (or any image format Pillow can open)
    Query params:
      model  (required unless a default is configured)
    """
    from scheduling.job import InferenceJob, JobType, SdxlJobInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    if input_image.width % 8 != 0 or input_image.height % 8 != 0:
        return _error(400, f"Image dimensions must be multiples of 8 "
                          f"(got {input_image.width}x{input_image.height}).")

    qp = request.query_params
    model_name = qp.get("model")
    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    vae_tile_w = int(qp.get("vae_tile_width",  "0"))
    vae_tile_h = int(qp.get("vae_tile_height", "0"))

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"EncodeLatents request: {input_image.width}x{input_image.height} model={model_short}"
             + (f" vae_tile={vae_tile_w}x{vae_tile_h}" if vae_tile_w or vae_tile_h else ""))

    factory = state.pipeline_factory
    queue   = state.queue

    job = InferenceJob(
        job_type=JobType.SDXL_ENCODE_LATENTS,
        pipeline=factory.create_sdxl_encode_latents_pipeline(model_dir),
        sdxl_input=SdxlJobInput(prompt="", negative_prompt="", model_dir=model_dir),
        input_image=input_image,
    )
    job.vae_tile_width  = vae_tile_w
    job.vae_tile_height = vae_tile_h

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    latents = result.output_latents
    shape = list(latents.shape)
    float_data = latents.cpu().float().numpy().tobytes()

    metadata = json.dumps({
        "width": input_image.width,
        "height": input_image.height,
        "model": model_short,
    }).encode("utf-8")

    buf = io.BytesIO()
    buf.write(b"FXLT")
    buf.write(struct.pack("<H", 1))
    buf.write(struct.pack("<H", len(shape)))
    for d in shape:
        buf.write(struct.pack("<i", d))
    buf.write(struct.pack("<I", len(metadata)))
    buf.write(metadata)
    buf.write(float_data)

    log.debug(f"EncodeLatents complete: shape={shape}")
    buf.seek(0)
    return Response(content=buf.getvalue(), media_type="application/x-fox-latent")


@router.post("/hires-latents")
async def hires_latents(request: Request):
    """Hires fix an input FXLT latent file, returning an FXLT latent file.

    Body: raw FXLT binary
    Query params:
      prompt          (required)
      negative_prompt (default: "")
      model           (default: from FXLT metadata)
      hires_width     (required)
      hires_height    (required)
      hires_steps     (default: 15)
      denoising_strength (default: 0.33)
      cfg_scale       (default: 4.0)
      seed            (default: random)
    """
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlHiresInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    body = await request.body()
    buf = io.BytesIO(body)

    # Parse FXLT
    magic = buf.read(4)
    if magic != b"FXLT":
        return _error(400, f'Invalid FXLT magic. Expected "FXLT".')
    version = struct.unpack("<H", buf.read(2))[0]
    if version != 1:
        return _error(400, f"Unsupported FXLT version: {version}.")
    ndims = struct.unpack("<H", buf.read(2))[0]
    if ndims > 8:
        return _error(400, f"FXLT ndims {ndims} exceeds maximum (8).")
    shape = [struct.unpack("<i", buf.read(4))[0] for _ in range(ndims)]
    for d in shape:
        if d <= 0 or d > 32768:
            return _error(400, f"Invalid FXLT shape dimension: {d}.")
    meta_len = struct.unpack("<I", buf.read(4))[0]
    if meta_len > 65536:
        return _error(400, f"FXLT metadata too large: {meta_len} bytes (max 65536).")
    fxlt_meta = {}
    if meta_len > 0:
        try:
            fxlt_meta = json.loads(buf.read(meta_len))
        except Exception:
            pass  # cursor already advanced by buf.read above
    total_floats = 1
    for d in shape:
        total_floats *= d
    latent_bytes = buf.read(total_floats * 4)
    if len(latent_bytes) != total_floats * 4:
        return _error(400, f"FXLT data truncated: got {len(latent_bytes)}, expected {total_floats * 4}.")
    latent_tensor = torch.from_numpy(
        np.frombuffer(latent_bytes, dtype=np.float32).reshape(shape).copy()
    )

    # Query params
    qp = request.query_params
    prompt = qp.get("prompt", "").strip()
    if not prompt:
        return _error(400, '"prompt" query param is required.')
    negative_prompt = qp.get("negative_prompt", "")

    try:
        hires_width  = int(qp["hires_width"])
        hires_height = int(qp["hires_height"])
    except (KeyError, ValueError):
        return _error(400, '"hires_width" and "hires_height" query params are required integers.')

    err = _validate_dimensions(hires_width, hires_height, max_dim=4096)
    if err:
        return _error(400, err)

    gen_hires_w, gen_hires_h = _snap_dims(hires_width, hires_height)

    try:
        hires_steps        = int(qp.get("hires_steps", "15"))
        denoising_strength = float(qp.get("denoising_strength", "0.33"))
        cfg_scale          = float(qp.get("cfg_scale", "4.0"))
        seed_param         = int(qp.get("seed", "0"))
        subseed_param      = int(qp.get("subseed", "0"))
        subseed_str        = max(0.0, min(1.0, float(qp.get("subseed_strength", "0.0"))))
        unet_tile_w        = int(qp.get("unet_tile_width",  "0"))
        unet_tile_h        = int(qp.get("unet_tile_height", "0"))
        vae_tile_w         = int(qp.get("vae_tile_width",   "0"))
        vae_tile_h         = int(qp.get("vae_tile_height",  "0"))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid query parameter: {ex}")
    sampler_param   = qp.get("sampler", "Euler A")
    scheduler_param = qp.get("scheduler")
    if not (1 <= hires_steps <= 150):
        return _error(400, "hires_steps must be between 1 and 150.")
    if not (0.0 <= denoising_strength <= 1.0):
        return _error(400, "denoising_strength must be between 0.0 and 1.0.")
    if not (1.0 <= cfg_scale <= 30.0):
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    seed               = seed_param if seed_param != 0 else random.randint(1, 2**31 - 1)

    # Model: query param overrides FXLT metadata
    model_name = qp.get("model") or fxlt_meta.get("model")
    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    if len(shape) != 4:
        return _error(400, f"Expected 4D latent tensor, got shape {shape}.")

    # skip_transform=1: latents are already at hires resolution (pre-upscaled externally).
    # Skips VAE decode → RealESRGAN → VAE encode; goes straight to hires denoise.
    skip_transform = qp.get("skip_transform", "0") in ("1", "true", "yes")

    # Base dims: from latent shape if transforming, or from explicit params / hires-halved if skipping.
    if skip_transform:
        base_w = int(qp.get("base_width",  hires_width  // 2))
        base_h = int(qp.get("base_height", hires_height // 2))
    else:
        base_h = shape[2] * 8
        base_w = shape[3] * 8

    # Parse LoRA tags from prompt
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        prompt, negative_prompt, None)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.debug(f"HiresLatents request: base={base_w}x{base_h} hires={hires_width}x{hires_height} "
             f"hires_steps={hires_steps} strength={denoising_strength:.2f} "
             f"cfg={cfg_scale} seed={seed} model={model_short}"
             + (" [skip_transform]" if skip_transform else "")
             + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

    factory = state.pipeline_factory
    queue   = state.queue

    if skip_transform:
        pipeline = factory.create_sdxl_hires_denoise_pipeline(model_dir)
    else:
        pipeline = factory.create_sdxl_hires_latents_pipeline(model_dir)

    job = InferenceJob(
        job_type=JobType.SDXL_HIRES_LATENTS,
        pipeline=pipeline,
        sdxl_input=SdxlJobInput(
            prompt=cleaned_prompt,
            negative_prompt=cleaned_neg,
            width=base_w,
            height=base_h,
            steps=0,
            cfg_scale=cfg_scale,
            seed=seed,
            subseed=subseed_param,
            subseed_strength=subseed_str,
            model_dir=model_dir,
            loras=all_loras,
            sampler=sampler_param,
            scheduler=scheduler_param,
        ),
        hires_input=SdxlHiresInput(
            hires_width=gen_hires_w,
            hires_height=gen_hires_h,
            hires_steps=hires_steps,
            denoising_strength=denoising_strength,
        ),
    )
    job.latents = latent_tensor
    job.unet_tile_width  = unet_tile_w
    job.unet_tile_height = unet_tile_h
    job.vae_tile_width   = vae_tile_w
    job.vae_tile_height  = vae_tile_h
    if skip_transform:
        job.is_hires_pass = True  # tell denoise() to use hires path immediately

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    latents = result.output_latents
    shape_out = list(latents.shape)
    float_data = latents.cpu().float().numpy().tobytes()

    metadata = json.dumps({
        "width": hires_width,
        "height": hires_height,
        "seed": seed,
        "model": model_short,
        "hires_steps": hires_steps,
        "denoising_strength": denoising_strength,
        "cfg_scale": cfg_scale,
    }).encode("utf-8")

    out = io.BytesIO()
    out.write(b"FXLT")
    out.write(struct.pack("<H", 1))
    out.write(struct.pack("<H", len(shape_out)))
    for d in shape_out:
        out.write(struct.pack("<i", d))
    out.write(struct.pack("<I", len(metadata)))
    out.write(metadata)
    out.write(float_data)

    log.debug(f"HiresLatents complete: shape={shape_out}")
    out.seek(0)
    return Response(content=out.getvalue(), media_type="application/x-fox-latent")


@router.post("/upscale")
async def upscale(request: Request):
    from scheduling.job import InferenceJob, JobType

    state = _get_state()

    if not state.gpu_pool.has_capability("upscale"):
        return _error(404, "No GPUs configured for upscale on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.debug(f"Upscale request: {input_image.width}x{input_image.height}")

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.UPSCALE,
        pipeline=factory.create_upscale_pipeline(),
        input_image=input_image,
    )

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    log.debug(f"Upscale complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image, mode="RGBA")


@router.post("/bgremove")
async def bgremove(request: Request):
    from scheduling.job import InferenceJob, JobType

    state = _get_state()

    if not state.gpu_pool.has_capability("bgremove"):
        return _error(404, "No GPUs configured for bgremove on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.debug(f"BGRemove request: {input_image.width}x{input_image.height}")

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.BGREMOVE,
        pipeline=factory.create_bgremove_pipeline(),
        input_image=input_image,
    )

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    log.debug(f"BGRemove complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image, mode="RGBA")


@router.post("/tag")
async def tag(request: Request):
    state = _get_state()
    proxy = _find_proxy_with_tagger(state)
    if proxy is None:
        return _error(404, "No GPUs with tagger loaded on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.debug(f"Tag request: {input_image.width}x{input_image.height} on GPU [{proxy.gpu.uuid}]")

    try:
        result = await proxy.tag_image(input_image)
        if result.error:
            return _error(500, result.error)
        log.debug(f"Tag complete: {len(result.tags)} tags")
        return {"tags": result.tags}
    except Exception as ex:
        log.log_exception(ex, "Tag failed")
        return _error(500, str(ex))


# ====================================================================
# Enqueue Endpoints (non-blocking, queue-based API)
# ====================================================================

def _register_job(job):
    """Register a job in the global job registry."""
    from state import app_state
    app_state.jobs[job.job_id] = job


@router.post("/enqueue/generate")
async def enqueue_generate(req: GenerateRequest):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    err = _validate_dimensions(req.width, req.height)
    if err:
        return _error(400, err)
    if req.steps < 1 or req.steps > 150:
        return _error(400, "Steps must be between 1 and 150.")
    if req.cfg_scale < 1.0 or req.cfg_scale > 30.0:
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    if not req.prompt.strip():
        return _error(400, '"prompt" is required.')

    model_dir, err = _resolve_model(req.model)
    if err:
        return _error(400, err)

    admitted = False
    try:
        rejected = _check_admission(JobType.SDXL_GENERATE)
        if rejected:
            return rejected
        admitted = True

        seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)
        orig_w, orig_h = req.width, req.height
        gen_w, gen_h = _snap_dims(orig_w, orig_h)

        # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
        cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
            req.prompt, req.negative_prompt, req.loras)

        model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
        log.debug(f"Enqueue generate: {orig_w}x{orig_h} steps={req.steps} "
                 f"cfg={req.cfg_scale} seed={seed} model={model_short}"
                 + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

        factory = state.pipeline_factory
        queue = state.queue

        job = InferenceJob(
            job_type=JobType.SDXL_GENERATE,
            pipeline=factory.create_sdxl_pipeline(model_dir),
            sdxl_input=SdxlJobInput(
                prompt=cleaned_prompt,
                negative_prompt=cleaned_neg,
                width=gen_w,
                height=gen_h,
                steps=req.steps,
                cfg_scale=req.cfg_scale,
                seed=seed,
                subseed=req.subseed,
                subseed_strength=req.subseed_strength,
                model_dir=model_dir,
                loras=all_loras,
                regional_prompting=req.regional_prompting,
                sampler=req.sampler,
                scheduler=req.scheduler,
            ),
            priority=req.priority,
        )
        job.vae_tile_width = req.vae_tile_width
        job.vae_tile_height = req.vae_tile_height
        job.orig_width = orig_w
        job.orig_height = orig_h

        _apply_clip_cache(req, job)

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.SDXL_GENERATE)
        raise


@router.post("/enqueue/generate-hires")
async def enqueue_generate_hires(req: GenerateHiresRequest):
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlHiresInput
    from handlers.sdxl import calculate_base_resolution

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    err = _validate_dimensions(req.width, req.height, max_dim=4096)
    if err:
        return _error(400, err)
    if req.steps < 1 or req.steps > 150:
        return _error(400, "Steps must be between 1 and 150.")
    if req.hires_steps < 1 or req.hires_steps > 150:
        return _error(400, "hires_steps must be between 1 and 150.")
    if req.cfg_scale < 1.0 or req.cfg_scale > 30.0:
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")
    if req.hires_denoising_strength < 0.0 or req.hires_denoising_strength > 1.0:
        return _error(400, "hires_denoising_strength must be between 0.0 and 1.0.")
    if not req.prompt.strip():
        return _error(400, '"prompt" is required.')

    model_dir, err = _resolve_model(req.model)
    if err:
        return _error(400, err)

    orig_w, orig_h = req.width, req.height
    gen_w, gen_h = _snap_dims(orig_w, orig_h)

    if req.base_width is not None and req.base_height is not None:
        base_w, base_h = req.base_width, req.base_height
        err = _validate_dimensions(base_w, base_h)
        if err:
            return _error(400, f"base_width/base_height: {err}")
        base_w, base_h = _snap_dims(base_w, base_h)
    else:
        base_w, base_h = calculate_base_resolution(req.width, req.height)

    if base_w > gen_w or base_h > gen_h:
        return _error(400, "Base dimensions must not exceed hires dimensions.")
    if base_w == gen_w and base_h == gen_h:
        return _error(400, "Base and hires dimensions are identical — use /enqueue/generate instead.")

    admitted = False
    try:
        rejected = _check_admission(JobType.SDXL_GENERATE_HIRES)
        if rejected:
            return rejected
        admitted = True

        seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)

        # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
        cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
            req.prompt, req.negative_prompt, req.loras)

        model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
        log.debug(f"Enqueue generate-hires: base={base_w}x{base_h} hires={orig_w}x{orig_h} "
                 f"steps={req.steps} hires_steps={req.hires_steps} "
                 f"strength={req.hires_denoising_strength:.2f} cfg={req.cfg_scale} "
                 f"seed={seed} model={model_short}"
                 + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

        factory = state.pipeline_factory
        queue = state.queue

        job = InferenceJob(
            job_type=JobType.SDXL_GENERATE_HIRES,
            pipeline=factory.create_sdxl_hires_pipeline(model_dir),
            sdxl_input=SdxlJobInput(
                prompt=cleaned_prompt,
                negative_prompt=cleaned_neg,
                width=base_w,
                height=base_h,
                steps=req.steps,
                cfg_scale=req.cfg_scale,
                seed=seed,
                subseed=req.subseed,
                subseed_strength=req.subseed_strength,
                model_dir=model_dir,
                loras=all_loras,
                regional_prompting=req.regional_prompting,
                sampler=req.sampler,
                scheduler=req.scheduler,
            ),
            hires_input=SdxlHiresInput(
                hires_width=gen_w,
                hires_height=gen_h,
                hires_steps=req.hires_steps,
                denoising_strength=req.hires_denoising_strength,
            ),
            priority=req.priority,
        )
        job.unet_tile_width = req.unet_tile_width
        job.unet_tile_height = req.unet_tile_height
        job.vae_tile_width = req.vae_tile_width
        job.vae_tile_height = req.vae_tile_height
        job.orig_width = orig_w
        job.orig_height = orig_h

        _apply_clip_cache(req, job)

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.SDXL_GENERATE_HIRES)
        raise


@router.post("/enqueue/upscale")
async def enqueue_upscale(request: Request):
    from scheduling.job import InferenceJob, JobType

    state = _get_state()

    if not state.gpu_pool.has_capability("upscale"):
        return _error(404, "No GPUs configured for upscale on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    admitted = False
    try:
        rejected = _check_admission(JobType.UPSCALE)
        if rejected:
            return rejected
        admitted = True

        log.debug(f"Enqueue upscale: {input_image.width}x{input_image.height}")

        factory = state.pipeline_factory
        queue = state.queue

        job = InferenceJob(
            job_type=JobType.UPSCALE,
            pipeline=factory.create_upscale_pipeline(),
            input_image=input_image,
        )

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.UPSCALE)
        raise


@router.post("/enqueue/bgremove")
async def enqueue_bgremove(request: Request):
    from scheduling.job import InferenceJob, JobType

    state = _get_state()

    if not state.gpu_pool.has_capability("bgremove"):
        return _error(404, "No GPUs configured for bgremove on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    admitted = False
    try:
        rejected = _check_admission(JobType.BGREMOVE)
        if rejected:
            return rejected
        admitted = True

        log.debug(f"Enqueue bgremove: {input_image.width}x{input_image.height}")

        factory = state.pipeline_factory
        queue = state.queue

        job = InferenceJob(
            job_type=JobType.BGREMOVE,
            pipeline=factory.create_bgremove_pipeline(),
            input_image=input_image,
        )

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.BGREMOVE)
        raise


@router.post("/enqueue/tag")
async def enqueue_tag(request: Request):
    state = _get_state()
    proxy = _find_proxy_with_tagger(state)
    if proxy is None:
        return _error(404, "No GPUs with tagger loaded on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.debug(f"Enqueue tag: {input_image.width}x{input_image.height}")

    from scheduling.job import InferenceJob, JobType, JobResult
    job = InferenceJob(
        job_type=JobType.TAG,
        pipeline=[],
        input_image=input_image,
    )

    _register_job(job)

    try:
        result = await proxy.tag_image(input_image)
        if result.error:
            job.completed_at = datetime.utcnow()
            job.set_result(JobResult(success=False, error=result.error))
        else:
            tag_bytes = json.dumps({"tags": result.tags}).encode("utf-8")
            state.job_results[job.job_id] = (tag_bytes, "application/json")
            job.completed_at = datetime.utcnow()
            job.set_result(JobResult(success=True))
            log.debug(f"Enqueue tag complete: {len(result.tags)} tags, job_id={job.job_id}")
    except Exception as ex:
        job.completed_at = datetime.utcnow()
        job.set_result(JobResult(success=False, error=str(ex)))
        log.log_exception(ex, "Enqueue tag failed")

    return {"job_id": job.job_id}


@router.post("/enqueue/enhance")
async def enqueue_enhance(request: Request):
    """Enhance (img2img hires): upscale input image, VAE encode, hires denoise, VAE decode.

    Multipart form data:
      image  (required unless cached_latents provided) — PNG/JPEG image bytes
      params (required) — JSON string with fields:
        prompt          (required)
        negative_prompt (default: "")
        model           (required or default configured)
        hires_width     (required) — target output width
        hires_height    (required) — target output height
        hires_steps     (default: 15)
        denoising_strength (default: 0.33)
        cfg_scale       (default: 7.0)
        seed            (default: random)
        unet_tile_width, unet_tile_height (default: 0 = auto)
        vae_tile_width, vae_tile_height   (default: 0 = auto)
        clip_embeddings  (optional) — pre-computed CLIP cache entries
        cached_latents   (optional) — {data: base64, dtype: str, shape: [4 ints]}
    """
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlHiresInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    # Parse multipart form data (raise limit from default 1MB to 32MB for large images)
    form = await request.form(max_part_size=32 * 1024 * 1024)
    try:
        image_file = form.get("image")
        params_str = form.get("params")
        latent_file = form.get("latent_data")

        if params_str is None:
            return _error(400, 'Multipart form must include "params" part.')

        image_bytes = (await image_file.read()) if image_file is not None else None
        latent_bytes = (await latent_file.read()) if latent_file is not None else None
    finally:
        await form.close()

    input_image = None
    if image_bytes is not None:
        try:
            input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        except Exception as ex:
            return _error(400, f"Could not decode image: {ex}")

    try:
        p = json.loads(params_str)
    except (json.JSONDecodeError, TypeError) as ex:
        return _error(400, f"Invalid params JSON: {ex}")

    prompt = p.get("prompt", "").strip()
    if not prompt:
        return _error(400, '"prompt" is required.')
    negative_prompt = p.get("negative_prompt", "")

    try:
        hires_width  = int(p["hires_width"])
        hires_height = int(p["hires_height"])
    except (KeyError, ValueError, TypeError):
        return _error(400, '"hires_width" and "hires_height" are required integers.')

    # Enhance only needs multiples of 8 (VAE latent alignment), not 64
    err = _validate_dimensions(hires_width, hires_height, max_dim=4096, multiple=8)
    if err:
        return _error(400, err)

    orig_hires_w, orig_hires_h = hires_width, hires_height
    gen_hires_w, gen_hires_h = _snap_dims(hires_width, hires_height, multiple=8)

    try:
        hires_steps        = int(p.get("hires_steps", 15))
        denoising_strength = float(p.get("denoising_strength", 0.33))
        cfg_scale          = float(p.get("cfg_scale", 7.0))
        seed_param         = int(p.get("seed", 0))
        subseed_param      = int(p.get("subseed", 0))
        subseed_strength   = max(0.0, min(1.0, float(p.get("subseed_strength", 0.0))))
        unet_tile_w        = int(p.get("unet_tile_width",  0))
        unet_tile_h        = int(p.get("unet_tile_height", 0))
        vae_tile_w         = int(p.get("vae_tile_width",   0))
        vae_tile_h         = int(p.get("vae_tile_height",  0))
        job_priority       = max(1, min(100, int(p.get("priority", 100))))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid parameter: {ex}")
    sampler_param   = p.get("sampler", "Euler A")
    scheduler_param = p.get("scheduler")

    if not (1 <= hires_steps <= 150):
        return _error(400, "hires_steps must be between 1 and 150.")
    if not (0.0 <= denoising_strength <= 1.0):
        return _error(400, "denoising_strength must be between 0.0 and 1.0.")
    if not (1.0 <= cfg_scale <= 30.0):
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")

    seed = seed_param if seed_param != 0 else random.randint(1, 2**31 - 1)

    model_name = p.get("model")
    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    regional_prompting = bool(p.get("regional_prompting", False))
    clip_embeddings = p.get("clip_embeddings")
    cached_latents_meta = p.get("cached_latents")

    # Try to parse cached latents for latent-space upscale path
    # Latent data comes as a separate binary multipart part (latent_data),
    # metadata (dtype, shape) comes in the JSON params (cached_latents).
    use_latent_pipeline = False
    latent_tensor = None
    if cached_latents_meta and latent_bytes and not regional_prompting:
        try:
            _DTYPE_MAP = {"float16": np.float16, "fp16": np.float16,
                          "float32": np.float32, "fp32": np.float32}

            lat_shape = cached_latents_meta["shape"]
            if (not isinstance(lat_shape, list) or len(lat_shape) != 4
                    or lat_shape[0] != 1 or lat_shape[1] != 4
                    or any(not isinstance(d, int) or d <= 0 for d in lat_shape)):
                raise ValueError(f"shape must be [1, 4, h, w], got {lat_shape!r}")

            lat_dtype_str = cached_latents_meta.get("dtype", "float16")
            np_dtype = _DTYPE_MAP.get(lat_dtype_str)
            if np_dtype is None:
                raise ValueError(f"Unsupported latent dtype: {lat_dtype_str!r}")

            expected_bytes = int(np.prod(lat_shape)) * np.dtype(np_dtype).itemsize
            if expected_bytes > 64 * 1024 * 1024:  # 64MB cap
                raise ValueError(f"cached_latents too large: {expected_bytes} bytes")

            if len(latent_bytes) != expected_bytes:
                raise ValueError(f"latent_data length {len(latent_bytes)} != expected {expected_bytes}")

            lat_arr = np.frombuffer(latent_bytes, dtype=np_dtype).reshape(lat_shape)
            latent_tensor = torch.from_numpy(lat_arr.copy())  # CPU tensor
            use_latent_pipeline = True
        except Exception as ex:
            log.warning(f"Invalid cached_latents, falling back to image pipeline: {ex}")

    # Image pipeline requires upscale capability; latent pipeline does not
    if not use_latent_pipeline:
        if not state.gpu_pool.has_capability("upscale"):
            return _error(404, "No GPUs configured for upscale on this worker.")
        if input_image is None:
            return _error(400, 'Image is required when cached_latents are not provided.')

    admitted = False
    try:
        rejected = _check_admission(JobType.ENHANCE)
        if rejected:
            return rejected
        admitted = True

        # Parse LoRA tags from prompt
        cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
            prompt, negative_prompt, None)

        model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
        factory = state.pipeline_factory
        queue = state.queue

        if use_latent_pipeline:
            # Latent-space upscale path: skip RealESRGAN and VAE encode
            lat_h, lat_w = latent_tensor.shape[2], latent_tensor.shape[3]
            base_w = lat_w * 8
            base_h = lat_h * 8

            log.debug(f"Enqueue enhance (latent path): latents={lat_h}x{lat_w} "
                     f"target={orig_hires_w}x{orig_hires_h} "
                     f"hires_steps={hires_steps} strength={denoising_strength:.2f} "
                     f"cfg={cfg_scale} seed={seed} model={model_short}"
                     + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

            job = InferenceJob(
                job_type=JobType.ENHANCE,
                pipeline=factory.create_enhance_latent_pipeline(model_dir),
                sdxl_input=SdxlJobInput(
                    prompt=cleaned_prompt,
                    negative_prompt=cleaned_neg,
                    width=base_w,
                    height=base_h,
                    steps=0,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    subseed=subseed_param,
                    subseed_strength=subseed_strength,
                    model_dir=model_dir,
                    loras=all_loras,
                    regional_prompting=regional_prompting,
                    sampler=sampler_param,
                    scheduler=scheduler_param,
                ),
                hires_input=SdxlHiresInput(
                    hires_width=gen_hires_w,
                    hires_height=gen_hires_h,
                    hires_steps=hires_steps,
                    denoising_strength=denoising_strength,
                ),
                input_image=input_image,  # may be None, not used by latent path
                priority=job_priority,
            )
            job.latents = latent_tensor
        else:
            # Image-based upscale path (existing behavior)
            base_w = input_image.width
            base_h = input_image.height

            needs_upscale = base_w < gen_hires_w or base_h < gen_hires_h

            if not needs_upscale:
                if base_w != gen_hires_w or base_h != gen_hires_h:
                    input_image = input_image.resize((gen_hires_w, gen_hires_h), Image.LANCZOS)
                    log.debug(f"Enhance: input already large enough, resized {base_w}x{base_h} → "
                             f"{gen_hires_w}x{gen_hires_h} (no upscale needed)")

            log.debug(f"Enqueue enhance (image path): input={base_w}x{base_h} "
                     f"target={orig_hires_w}x{orig_hires_h} "
                     f"upscale={'yes' if needs_upscale else 'no'} "
                     f"hires_steps={hires_steps} strength={denoising_strength:.2f} "
                     f"cfg={cfg_scale} seed={seed} model={model_short}"
                     + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

            job = InferenceJob(
                job_type=JobType.ENHANCE,
                pipeline=factory.create_enhance_pipeline(model_dir, needs_upscale=needs_upscale),
                sdxl_input=SdxlJobInput(
                    prompt=cleaned_prompt,
                    negative_prompt=cleaned_neg,
                    width=base_w,
                    height=base_h,
                    steps=0,
                    cfg_scale=cfg_scale,
                    seed=seed,
                    subseed=subseed_param,
                    subseed_strength=subseed_strength,
                    model_dir=model_dir,
                    loras=all_loras,
                    regional_prompting=regional_prompting,
                    sampler=sampler_param,
                    scheduler=scheduler_param,
                ),
                hires_input=SdxlHiresInput(
                    hires_width=gen_hires_w,
                    hires_height=gen_hires_h,
                    hires_steps=hires_steps,
                    denoising_strength=denoising_strength,
                ),
                input_image=input_image,
                priority=job_priority,
            )

        job.is_hires_pass = True  # denoise uses hires path from the start
        job.unet_tile_width  = unet_tile_w
        job.unet_tile_height = unet_tile_h
        job.vae_tile_width   = vae_tile_w
        job.vae_tile_height  = vae_tile_h
        job.orig_width = orig_hires_w
        job.orig_height = orig_hires_h

        # Apply CLIP cache if provided
        if clip_embeddings and not regional_prompting and len(clip_embeddings) == 6:
            try:
                from scheduling.job import StageType
                cached_clip = _parse_clip_cache_entries(clip_embeddings)
                te_stage = next((s for s in job.pipeline
                                 if s.type == StageType.GPU_TEXT_ENCODE), None)
                if te_stage is not None:
                    job.stripped_te_components = te_stage.required_components
                job.pipeline = [s for s in job.pipeline
                                if s.type not in (StageType.CPU_TOKENIZE,
                                                  StageType.GPU_TEXT_ENCODE)]
                job.cached_clip_entries = cached_clip
            except Exception as ex:
                log.warning(f"Invalid clip_embeddings in enhance, ignoring cache: {ex}")

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.ENHANCE)
        raise


def _create_outpaint_canvas(input_image: Image.Image, target_w: int, target_h: int):
    """Create canvas + mask for outpainting when aspect ratios differ.

    The canvas is filled by reflecting the input image's edges outward,
    giving the UNet contextual content at the borders instead of flat gray.

    Returns (canvas_image, mask_array) where mask is [H, W] float32
    with 1.0 = generate (borders), 0.0 = keep (original image area).
    """
    iw, ih = input_image.size
    target_aspect = target_w / target_h
    input_aspect = iw / ih

    if input_aspect > target_aspect:
        # Input is wider — extend vertically
        canvas_w = iw
        canvas_h = round(iw / target_aspect)
    else:
        # Input is taller — extend horizontally
        canvas_h = ih
        canvas_w = round(ih * target_aspect)

    # Snap to multiples of 8 (VAE latent alignment)
    canvas_w = ((canvas_w + 7) // 8) * 8
    canvas_h = ((canvas_h + 7) // 8) * 8

    offset_x = (canvas_w - iw) // 2
    offset_y = (canvas_h - ih) // 2

    # Build canvas: edge-replicate the original's border pixels outward.
    # This creates a smooth color transition that the VAE encodes without
    # artifacts (unlike gray fill which creates a sharp boundary the VAE
    # turns into visible latent seams).
    img_arr = np.array(input_image)  # [ih, iw, 3]
    pad_top = offset_y
    pad_bottom = canvas_h - ih - offset_y
    pad_left = offset_x
    pad_right = canvas_w - iw - offset_x
    canvas_arr = np.pad(img_arr,
                        ((pad_top, pad_bottom), (pad_left, pad_right), (0, 0)),
                        mode='edge')
    canvas = Image.fromarray(canvas_arr)

    # Mask: 1.0 = generate (borders), 0.0 = keep (original)
    mask = np.ones((canvas_h, canvas_w), dtype=np.float32)
    mask[offset_y:offset_y + ih, offset_x:offset_x + iw] = 0.0

    return canvas, mask


@router.post("/enqueue/img2img")
async def enqueue_img2img(request: Request):
    """IMG2IMG: VAE encode input image, partial denoise, VAE decode.

    Multipart form data:
      image  (required) — PNG/JPEG image bytes
      params (required) — JSON string with fields:
        prompt          (required)
        negative_prompt (default: "")
        model           (required)
        width           (default: 0 = use input size)
        height          (default: 0 = use input size)
        steps           (default: 25)
        denoising_strength (default: 0.75)
        cfg_scale       (default: 7.0)
        seed            (default: random)
        sampler         (default: "Euler A")
        scheduler       (optional)
        regional_prompting (default: false)
        priority        (default: 100)
        clip_embeddings (optional) — pre-computed CLIP cache entries
    """
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlImg2ImgInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")

    # Parse multipart form data (raise limit from default 1MB to 32MB for large images)
    form = await request.form(max_part_size=32 * 1024 * 1024)
    try:
        image_file = form.get("image")
        params_str = form.get("params")

        if params_str is None:
            return _error(400, 'Multipart form must include "params" part.')
        if image_file is None:
            return _error(400, 'Multipart form must include "image" part.')

        image_bytes = await image_file.read()
    finally:
        await form.close()

    try:
        input_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    try:
        p = json.loads(params_str)
    except (json.JSONDecodeError, TypeError) as ex:
        return _error(400, f"Invalid params JSON: {ex}")

    prompt = p.get("prompt", "").strip()
    if not prompt:
        return _error(400, '"prompt" is required.')
    negative_prompt = p.get("negative_prompt", "")

    try:
        target_w        = int(p.get("width", 0))
        target_h        = int(p.get("height", 0))
        steps           = int(p.get("steps", 25))
        denoising_str   = float(p.get("denoising_strength", 0.75))
        cfg_scale       = float(p.get("cfg_scale", 7.0))
        seed_param      = int(p.get("seed", 0))
        subseed_param   = int(p.get("subseed", 0))
        subseed_strength = max(0.0, min(1.0, float(p.get("subseed_strength", 0.0))))
        vae_tile_w      = int(p.get("vae_tile_width", 0))
        vae_tile_h      = int(p.get("vae_tile_height", 0))
        job_priority    = max(1, min(100, int(p.get("priority", 100))))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid parameter: {ex}")
    sampler_param   = p.get("sampler", "Euler A")
    scheduler_param = p.get("scheduler")

    if not (1 <= steps <= 150):
        return _error(400, "steps must be between 1 and 150.")
    if not (0.0 <= denoising_str <= 1.0):
        return _error(400, "denoising_strength must be between 0.0 and 1.0.")
    if not (1.0 <= cfg_scale <= 30.0):
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")

    seed = seed_param if seed_param != 0 else random.randint(1, 2**31 - 1)

    model_name = p.get("model")
    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    regional_prompting = bool(p.get("regional_prompting", False))
    clip_embeddings = p.get("clip_embeddings")

    iw, ih = input_image.size

    # If target dimensions not specified, use input image size
    if target_w <= 0:
        target_w = iw
    if target_h <= 0:
        target_h = ih

    # Validate target dimensions
    err = _validate_dimensions(target_w, target_h, max_dim=4096, multiple=8)
    if err:
        return _error(400, err)

    orig_target_w, orig_target_h = target_w, target_h
    target_w, target_h = _snap_dims(target_w, target_h, multiple=8)

    # Determine canvas/mask strategy based on aspect ratio comparison
    img2img_mask = None
    target_aspect = target_w / target_h
    input_aspect = iw / ih
    aspect_diff = abs(target_aspect - input_aspect) / max(target_aspect, input_aspect)

    if aspect_diff > 0.01:
        # Aspect ratios differ — create outpaint canvas + mask
        input_image, img2img_mask = _create_outpaint_canvas(input_image, target_w, target_h)
        # Update dimensions to canvas size
        canvas_w, canvas_h = input_image.size
        gen_w, gen_h = canvas_w, canvas_h
    elif iw != target_w or ih != target_h:
        # Same aspect, different size — resize input to target
        input_image = input_image.resize((target_w, target_h), Image.LANCZOS)
        gen_w, gen_h = target_w, target_h
    else:
        # Same dimensions — snap to multiples of 8 for VAE alignment
        snapped_w = ((iw + 7) // 8) * 8
        snapped_h = ((ih + 7) // 8) * 8
        if snapped_w != iw or snapped_h != ih:
            input_image = input_image.resize((snapped_w, snapped_h), Image.LANCZOS)
        gen_w, gen_h = snapped_w, snapped_h

    admitted = False
    try:
        rejected = _check_admission(JobType.SDXL_IMG2IMG)
        if rejected:
            return rejected
        admitted = True

        # Parse LoRA tags from prompt
        cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
            prompt, negative_prompt, None)

        model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
        factory = state.pipeline_factory
        queue = state.queue

        log.debug(f"Enqueue img2img: input={iw}x{ih} target={orig_target_w}x{orig_target_h} "
                 f"gen={gen_w}x{gen_h} mask={'yes' if img2img_mask is not None else 'no'} "
                 f"steps={steps} strength={denoising_str:.2f} "
                 f"cfg={cfg_scale} seed={seed} model={model_short}"
                 + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

        job = InferenceJob(
            job_type=JobType.SDXL_IMG2IMG,
            pipeline=factory.create_sdxl_img2img_pipeline(model_dir),
            sdxl_input=SdxlJobInput(
                prompt=cleaned_prompt,
                negative_prompt=cleaned_neg,
                width=gen_w,
                height=gen_h,
                steps=steps,
                cfg_scale=cfg_scale,
                seed=seed,
                subseed=subseed_param,
                subseed_strength=subseed_strength,
                model_dir=model_dir,
                loras=all_loras,
                regional_prompting=regional_prompting,
                sampler=sampler_param,
                scheduler=scheduler_param,
            ),
            img2img_input=SdxlImg2ImgInput(
                denoising_strength=denoising_str,
                width=orig_target_w,
                height=orig_target_h,
            ),
            input_image=input_image,
            priority=job_priority,
        )
        job.img2img_mask = img2img_mask
        job.vae_tile_width = vae_tile_w
        job.vae_tile_height = vae_tile_h
        job.orig_width = orig_target_w
        job.orig_height = orig_target_h

        # Apply CLIP cache if provided
        if clip_embeddings and not regional_prompting and len(clip_embeddings) == 6:
            try:
                from scheduling.job import StageType
                cached_clip = _parse_clip_cache_entries(clip_embeddings)
                te_stage = next((s for s in job.pipeline
                                 if s.type == StageType.GPU_TEXT_ENCODE), None)
                if te_stage is not None:
                    job.stripped_te_components = te_stage.required_components
                job.pipeline = [s for s in job.pipeline
                                if s.type not in (StageType.CPU_TOKENIZE,
                                                  StageType.GPU_TEXT_ENCODE)]
                job.cached_clip_entries = cached_clip
            except Exception as ex:
                log.warning(f"Invalid clip_embeddings in img2img, ignoring cache: {ex}")

        _register_job(job)
        queue.enqueue(job)
        return {"job_id": job.job_id}
    except Exception:
        if admitted:
            _release_admission(JobType.SDXL_IMG2IMG)
        raise


# ====================================================================
# Job Status & Result Endpoints
# ====================================================================

@router.get("/job/{job_id}")
async def job_status(job_id: str):
    state = _get_state()
    job = state.jobs.get(job_id)
    if job is None:
        return _error(404, f"Job {job_id} not found.")

    # Determine status — use completion future as authoritative source
    if job.completion.done():
        result = job.completion.result()
        status = "failed" if (result and not result.success) else "completed"
    elif job.started_at is not None:
        status = "running"
    else:
        status = "queued"

    stage = job.current_stage
    stage_type = stage.type.value if stage else None
    is_denoise = stage and stage.type.value == "GpuDenoise"
    denoise_step = job.denoise_step if is_denoise else None
    denoise_total = job.denoise_total_steps if is_denoise else None

    progress = None
    if is_denoise and denoise_total and denoise_total > 0:
        progress = round(denoise_step / denoise_total, 4)

    elapsed_s = None
    if job.started_at:
        elapsed_s = round((datetime.utcnow() - job.started_at).total_seconds(), 1)

    error = None
    if status == "failed" and job.completion.done():
        result = job.completion.result()
        error = result.error if result else None

    pool = state.gpu_pool

    return {
        "job_id": job.job_id,
        "status": status,
        "type": job.type.value,
        "stage": stage_type,
        "stage_index": job.current_stage_index,
        "stage_count": len(job.pipeline),
        "denoise_step": denoise_step,
        "denoise_total": denoise_total,
        "progress": progress,
        "gpu_count": len(pool.gpus),
        "gpus": job.active_gpus,
        "elapsed_s": elapsed_s,
        "gpu_time_s": round(job.gpu_time_s, 3),
        "gpu_stages": job.gpu_stage_times,
        "created_at": job.created_at.isoformat() if job.created_at else None,
        "error": error,
    }


@router.post("/job/{job_id}/cancel")
async def job_cancel(job_id: str):
    from scheduling.job import JobResult
    from datetime import datetime

    state = _get_state()
    job = state.jobs.get(job_id)
    if job is None:
        return _error(404, f"Job {job_id} not found.")

    if job.completion.done():
        return {"job_id": job_id, "status": "already_completed"}

    # If still queued (not yet dispatched to a GPU), resolve directly
    if job.assigned_gpu_uuid is None:
        state.queue.remove(job)
        job.completed_at = datetime.utcnow()
        if not job._admission_released:
            job._admission_released = True
            if state.admission is not None:
                state.admission.release(job.type)
        job.set_result(JobResult(success=False, error="Cancelled"))
        from api.websocket import streamer
        await streamer.broadcast_complete(job, success=False, error="Cancelled")
        log.info(f"Cancelled queued job {job_id}")
        return {"job_id": job_id, "status": "cancelled"}

    # Job has a GPU assigned — find the worker and signal cancel_event.
    # Use assigned_gpu_uuid (not _active_job) so we catch the dispatch-to-dequeue window.
    for worker in state.scheduler.workers:
        if worker.gpu.uuid == job.assigned_gpu_uuid:
            worker.cancel_active_job()
            log.info(f"Cancelled running job {job_id} on GPU [{worker.gpu.uuid}]")
            return {"job_id": job_id, "status": "cancelling"}

    # Worker not found for this GPU UUID (shouldn't happen)
    log.warning(f"Cancel requested for job {job_id} but worker for GPU "
                f"{job.assigned_gpu_uuid} not found, signalling cancel")
    return {"job_id": job_id, "status": "cancelling"}


@router.get("/job/{job_id}/result")
async def job_result(job_id: str):
    state = _get_state()
    job = state.jobs.get(job_id)
    if job is None:
        return _error(404, f"Job {job_id} not found.")

    # Still running — check the future, not completed_at, to avoid race
    if not job.completion.done():
        return JSONResponse(status_code=202, content={"status": "running", "job_id": job_id})

    # Check for failure
    result = job.completion.result()
    if result and not result.success:
        return _error(500, result.error or "Job failed")

    # Check for stored result bytes (from result storage hook)
    stored = state.job_results.get(job_id)
    if stored is not None:
        result_bytes, media_type = stored
        return Response(content=result_bytes, media_type=media_type)

    # Fallback: try to serialize from the job's completion result
    if result and result.output_image:
        image = result.output_image
        orig_w = getattr(job, "orig_width", None)
        orig_h = getattr(job, "orig_height", None)
        if (orig_w is not None and orig_h is not None
                and (image.width != orig_w or image.height != orig_h)):
            image = image.resize((orig_w, orig_h), Image.LANCZOS)
        buf = io.BytesIO()
        if image.mode == "RGBA":
            image.save(buf, format="PNG")
        else:
            image.convert("RGB").save(buf, format="PNG")
        return Response(content=buf.getvalue(), media_type="image/png")

    if result and result.output_latents is not None:
        # Build FXLT binary
        latents = result.output_latents
        shape = list(latents.shape)
        float_data = latents.cpu().float().numpy().tobytes()
        buf = io.BytesIO()
        buf.write(b"FXLT")
        buf.write(struct.pack("<H", 1))
        buf.write(struct.pack("<H", len(shape)))
        for d in shape:
            buf.write(struct.pack("<i", d))
        buf.write(struct.pack("<I", 0))  # no metadata
        buf.write(float_data)
        return Response(content=buf.getvalue(), media_type="application/x-fox-latent")

    return _error(500, "Result not available")


# ====================================================================
# Log Viewer
# ====================================================================

@router.get("/logs")
async def get_logs(
    lines: int = 100,
    offset: int = 0,
    level: str | None = None,
    search: str | None = None,
    before: str | None = None,
    after: str | None = None,
    output_format: str = Query("text", alias="format"),
):
    """Return recent log lines with optional filtering.

    Query params:
        lines:   Number of lines to return (default 100, max 10000)
        offset:  Skip this many lines from the tail before returning
        level:   Filter by log level: DEBUG, INFO, WARNING, ERROR
        search:  Case-insensitive substring search across log lines
        before:  Only lines before this timestamp (ISO 8601 or HH:MM:SS)
        after:   Only lines after this timestamp (ISO 8601 or HH:MM:SS)
        format:  Response format: "text" (default, human-readable) or "jsonl" (raw JSONL)
    """
    log_path = log.get_log_path()
    if log_path is None or not os.path.isfile(log_path):
        return _error(404, "Log file not available")

    lines = max(1, min(lines, 10000))

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            raw_lines = f.readlines()
    except OSError as ex:
        return _error(500, f"Failed to read log file: {ex}")

    # Parse each line as JSON, with fallback for malformed lines
    records: list[dict] = []
    for raw in raw_lines:
        raw = raw.rstrip("\n")
        if not raw:
            continue
        try:
            rec = json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            # Fallback for malformed or legacy plain-text lines
            rec = {"ts": "", "level": "INFO", "msg": raw}
        records.append(rec)

    # Apply filters on parsed records
    if level:
        level_upper = level.upper()
        records = [r for r in records if r.get("level", "").upper() == level_upper]

    if search:
        search_lower = search.lower()
        records = [r for r in records if search_lower in r.get("msg", "").lower()]

    if after or before:
        records = _filter_by_time(records, after=after, before=before)

    # Apply offset + limit from the tail
    total = len(records)
    if offset > 0:
        records = records[:max(0, total - offset)]
    records = records[-lines:]

    headers = {"X-Total-Lines": str(total), "X-Returned-Lines": str(len(records))}

    if output_format == "jsonl":
        content = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        return Response(content=content, media_type="application/x-ndjson", headers=headers)

    # Default: human-readable text
    text_lines = []
    for r in records:
        ts = r.get("ts", "")
        # Convert ISO-T format to space-separated for display
        ts_display = ts.replace("T", " ") if ts else "?"
        lvl = r.get("level", "?")
        msg = r.get("msg", "")
        text_lines.append(f"[{ts_display}] [{lvl}] {msg}\n")

    return Response(
        content="".join(text_lines),
        media_type="text/plain; charset=utf-8",
        headers=headers,
    )


def _filter_by_time(records: list[dict], after: str | None, before: str | None) -> list[dict]:
    """Filter parsed log records by timestamp range."""
    from datetime import datetime as _dt

    def _parse_ts(s: str) -> _dt | None:
        for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S",
                     "%Y-%m-%dT%H:%M:%S.%f", "%Y-%m-%dT%H:%M:%S",
                     "%H:%M:%S.%f", "%H:%M:%S"):
            try:
                t = _dt.strptime(s, fmt)
                # Time-only formats: use today's date
                if t.year == 1900:
                    now = _dt.now()
                    t = t.replace(year=now.year, month=now.month, day=now.day)
                return t
            except ValueError:
                continue
        return None

    after_dt = _parse_ts(after) if after else None
    before_dt = _parse_ts(before) if before else None
    if after_dt is None and before_dt is None:
        return records

    result = []
    for rec in records:
        ts_str = rec.get("ts", "")
        if ts_str:
            ts = _parse_ts(ts_str)
            if ts is not None:
                if after_dt and ts < after_dt:
                    continue
                if before_dt and ts > before_dt:
                    continue
        result.append(rec)
    return result


# ====================================================================
# Profiling Endpoints
# ====================================================================

@router.get("/profiling/traces")
async def profiling_traces():
    """List available profiling trace files with metadata."""
    from profiling.query import list_trace_files
    loop = asyncio.get_running_loop()
    traces = await loop.run_in_executor(None, list_trace_files)
    return {"traces": traces}


@router.get("/profiling/search")
async def profiling_search(
    job_id: str | None = None,
    type: str | None = None,
    model: str | None = None,
    gpu_uuid: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = Query(default=500, ge=1, le=10000),
    offset: int = Query(default=0, ge=0),
):
    """Search profiling trace events with filters."""
    from profiling.query import search_events
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(
        None, lambda: search_events(
            gpu_uuid=gpu_uuid, job_id=job_id, event_type=type, model=model,
            after=after, before=before,
            limit=limit, offset=offset,
        ))
    return {"events": events, "count": len(events)}


@router.get("/profiling/stats")
async def profiling_stats(
    group_by: str = Query(default="model"),
    type: str | None = None,
    model: str | None = None,
    gpu_uuid: str | None = None,
    after: str | None = None,
    before: str | None = None,
):
    """Aggregate profiling stats (count, sum, avg, min, max, p50, p95) grouped by field."""
    from profiling.query import aggregate_stats
    loop = asyncio.get_running_loop()
    try:
        groups = await loop.run_in_executor(
            None, lambda: aggregate_stats(
                group_by=group_by, event_type=type, model=model,
                gpu_uuid=gpu_uuid, after=after, before=before,
            ))
    except ValueError as e:
        return JSONResponse(status_code=422, content={"error": str(e)})
    return {"groups": groups}


@router.get("/profiling/job/{job_id}")
async def profiling_job_timeline(job_id: str):
    """Get complete event timeline for a specific job across all GPUs."""
    from profiling.query import job_timeline
    loop = asyncio.get_running_loop()
    events = await loop.run_in_executor(
        None, lambda: job_timeline(job_id))
    return {"job_id": job_id, "events": events, "count": len(events)}
