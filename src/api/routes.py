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
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image
from pydantic import BaseModel, Field

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
    model: str | None = None
    vae_tile_width: int = 0    # VAE decode tile width in pixels (0 = auto ≤ 1024)
    vae_tile_height: int = 0   # VAE decode tile height in pixels (0 = auto ≤ 1024)
    loras: list[dict] | None = None  # [{"name": "xxx", "weight": 1.0}, ...]
    regional_prompting: bool = False


class GenerateHiresRequest(BaseModel):
    prompt: str
    negative_prompt: str = ""
    width: int = 1536
    height: int = 1024
    steps: int = 25
    cfg_scale: float = 7.0
    seed: int = 0
    hires_steps: int = 15
    hires_denoising_strength: float = 0.33
    base_width: int | None = None
    base_height: int | None = None
    model: str | None = None
    unet_tile_width: int = 0   # UNet / MultiDiffusion tile width in pixels (0 = auto ≤ 1024)
    unet_tile_height: int = 0  # UNet / MultiDiffusion tile height in pixels (0 = auto ≤ 1024)
    loras: list[dict] | None = None  # [{"name": "xxx", "weight": 1.0}, ...]
    regional_prompting: bool = False
    vae_tile_width: int = 0    # VAE encode/decode tile width in pixels (0 = auto ≤ 1024)
    vae_tile_height: int = 0   # VAE encode/decode tile height in pixels (0 = auto ≤ 1024)


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


def _find_gpu_with_tagger(state) -> "GpuInstance | None":
    """Find a GPU with the tagger loaded, preferring non-busy GPUs."""
    from handlers.tagger import is_loaded_on
    best = None
    for g in state.gpu_pool.gpus:
        if g.supports_capability("tag") and is_loaded_on(g.device):
            if not g.is_busy:
                return g  # prefer idle GPU
            if best is None:
                best = g
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


def _validate_dimensions(width: int, height: int, max_dim: int = 2048,
                          multiple: int = 64) -> str | None:
    if (width % multiple != 0 or height % multiple != 0
            or width < multiple or width > max_dim
            or height < multiple or height > max_dim):
        return (f"Width and height must be divisible by {multiple} "
                f"and between {multiple}-{max_dim}.")
    return None


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
    state = _get_state()
    pool = state.gpu_pool
    queue = state.queue
    scheduler = state.scheduler

    workers_by_uuid: dict[str, Any] = {}
    if scheduler and scheduler.workers:
        for w in scheduler.workers:
            workers_by_uuid[w.gpu.uuid] = w

    gpus = []
    for g in pool.gpus:
        gpu_info: dict[str, Any] = {
            "uuid": g.uuid,
            "name": g.name,
            "device_id": g.device_id,
            "capabilities": sorted(g.capabilities),
            "busy": g.is_busy,
            "session_cache_count": g.session_cache_count,
            "vram": g.get_vram_stats(),
        }

        worker = workers_by_uuid.get(g.uuid)
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
                    from datetime import datetime
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

        gpu_info["loaded_models"] = g.get_cached_models_info()
        gpus.append(gpu_info)

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

    # Compute readiness scores for C# dispatch
    sdxl_models_snapshot = dict(state.sdxl_models)
    readiness = {}
    if scheduler and scheduler.workers:
        from scheduling.readiness import compute_readiness
        try:
            readiness = compute_readiness(
                pool, scheduler.workers, state.registry, sdxl_models_snapshot)
        except Exception as ex:
            log.log_exception(ex, "Failed to compute readiness scores")

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

    return {
        "gpus": gpus,
        "available": available,
        "total": len(pool.gpus),
        "queue": {"depth": queue.count if queue else 0, "jobs": queued_jobs},
        "model_scan": model_scan,
        "readiness": readiness,
    }


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

    log.info(f"  Model rescan: +{summary['added']} -{summary['removed']} "
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

    # Create and start a worker for this GPU
    if scheduler:
        from scheduling.worker import GpuWorker
        worker = GpuWorker(gpu, state.queue, scheduler._wake)
        scheduler.workers.append(worker)
        worker.start()
        log.info(f"  GPU [{gpu.uuid}] re-added with worker")

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

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"Generate request: {req.width}x{req.height} steps={req.steps} "
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
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
        ),
    )
    job.vae_tile_width  = req.vae_tile_width
    job.vae_tile_height = req.vae_tile_height

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    log.info(f"Generate complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image)


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

    # Calculate base resolution
    if req.base_width is not None and req.base_height is not None:
        base_w, base_h = req.base_width, req.base_height
        err = _validate_dimensions(base_w, base_h)
        if err:
            return _error(400, f"base_width/base_height: {err}")
    else:
        base_w, base_h = calculate_base_resolution(req.width, req.height)

    if base_w > req.width or base_h > req.height:
        return _error(400, "Base dimensions must not exceed hires dimensions.")
    if base_w == req.width and base_h == req.height:
        return _error(400, "Base and hires dimensions are identical — use /generate instead.")

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"GenerateHires request: base={base_w}x{base_h} hires={req.width}x{req.height} "
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
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
        ),
        hires_input=SdxlHiresInput(
            hires_width=req.width,
            hires_height=req.height,
            hires_steps=req.hires_steps,
            denoising_strength=req.hires_denoising_strength,
        ),
    )
    job.unet_tile_width  = req.unet_tile_width
    job.unet_tile_height = req.unet_tile_height
    job.vae_tile_width   = req.vae_tile_width
    job.vae_tile_height  = req.vae_tile_height

    queue.enqueue(job)
    result = await job.completion

    if not result.success:
        return _error(500, result.error or "Unknown error")

    log.info(f"GenerateHires complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image)


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

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"GenerateLatents request: {req.width}x{req.height} steps={req.steps} "
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
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
        ),
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
    log.info(f"GenerateLatents complete: shape={shape} "
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
    log.info(f"DecodeLatents request: shape={shape} model={model_short}")

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

    log.info(f"DecodeLatents complete: {result.output_image.width}x{result.output_image.height}")
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
    log.info(f"EncodeLatents request: {input_image.width}x{input_image.height} model={model_short}"
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

    log.info(f"EncodeLatents complete: shape={shape}")
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

    try:
        hires_steps        = int(qp.get("hires_steps", "15"))
        denoising_strength = float(qp.get("denoising_strength", "0.33"))
        cfg_scale          = float(qp.get("cfg_scale", "4.0"))
        seed_param         = int(qp.get("seed", "0"))
        unet_tile_w        = int(qp.get("unet_tile_width",  "0"))
        unet_tile_h        = int(qp.get("unet_tile_height", "0"))
        vae_tile_w         = int(qp.get("vae_tile_width",   "0"))
        vae_tile_h         = int(qp.get("vae_tile_height",  "0"))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid query parameter: {ex}")
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
    log.info(f"HiresLatents request: base={base_w}x{base_h} hires={hires_width}x{hires_height} "
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
            model_dir=model_dir,
            loras=all_loras,
        ),
        hires_input=SdxlHiresInput(
            hires_width=hires_width,
            hires_height=hires_height,
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

    log.info(f"HiresLatents complete: shape={shape_out}")
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

    log.info(f"Upscale request: {input_image.width}x{input_image.height}")

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

    log.info(f"Upscale complete: {result.output_image.width}x{result.output_image.height}")
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

    log.info(f"BGRemove request: {input_image.width}x{input_image.height}")

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

    log.info(f"BGRemove complete: {result.output_image.width}x{result.output_image.height}")
    return _image_response(result.output_image, mode="RGBA")


@router.post("/tag")
async def tag(request: Request):
    from handlers.tagger import process_image, is_loaded_on

    state = _get_state()
    gpu = _find_gpu_with_tagger(state)
    if gpu is None:
        return _error(404, "No GPUs with tagger loaded on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.info(f"Tag request: {input_image.width}x{input_image.height} on GPU [{gpu.uuid}]")

    try:
        tags = process_image(input_image, gpu)
        log.info(f"Tag complete: {len(tags)} tags")
        return {"tags": tags}
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

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"Enqueue generate: {req.width}x{req.height} steps={req.steps} "
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
            width=req.width,
            height=req.height,
            steps=req.steps,
            cfg_scale=req.cfg_scale,
            seed=seed,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
        ),
    )
    job.vae_tile_width = req.vae_tile_width
    job.vae_tile_height = req.vae_tile_height

    _register_job(job)
    queue.enqueue(job)
    return {"job_id": job.job_id}


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

    if req.base_width is not None and req.base_height is not None:
        base_w, base_h = req.base_width, req.base_height
        err = _validate_dimensions(base_w, base_h)
        if err:
            return _error(400, f"base_width/base_height: {err}")
    else:
        base_w, base_h = calculate_base_resolution(req.width, req.height)

    if base_w > req.width or base_h > req.height:
        return _error(400, "Base dimensions must not exceed hires dimensions.")
    if base_w == req.width and base_h == req.height:
        return _error(400, "Base and hires dimensions are identical — use /enqueue/generate instead.")

    seed = req.seed if req.seed != 0 else random.randint(1, 2**31 - 1)

    # Parse LoRA tags from prompt (A1111 syntax) + explicit request loras
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        req.prompt, req.negative_prompt, req.loras)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"Enqueue generate-hires: base={base_w}x{base_h} hires={req.width}x{req.height} "
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
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=req.regional_prompting,
        ),
        hires_input=SdxlHiresInput(
            hires_width=req.width,
            hires_height=req.height,
            hires_steps=req.hires_steps,
            denoising_strength=req.hires_denoising_strength,
        ),
    )
    job.unet_tile_width = req.unet_tile_width
    job.unet_tile_height = req.unet_tile_height
    job.vae_tile_width = req.vae_tile_width
    job.vae_tile_height = req.vae_tile_height

    _register_job(job)
    queue.enqueue(job)
    return {"job_id": job.job_id}


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

    log.info(f"Enqueue upscale: {input_image.width}x{input_image.height}")

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

    log.info(f"Enqueue bgremove: {input_image.width}x{input_image.height}")

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


@router.post("/enqueue/tag")
async def enqueue_tag(request: Request):
    from handlers.tagger import process_image as _tag_process

    state = _get_state()
    gpu = _find_gpu_with_tagger(state)
    if gpu is None:
        return _error(404, "No GPUs with tagger loaded on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGBA")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

    log.info(f"Enqueue tag: {input_image.width}x{input_image.height}")

    from scheduling.job import InferenceJob, JobType, JobResult
    job = InferenceJob(
        job_type=JobType.TAG,
        pipeline=[],
        input_image=input_image,
    )

    _register_job(job)

    # Run in executor to avoid blocking the event loop
    try:
        loop = asyncio.get_running_loop()
        tags = await loop.run_in_executor(None, _tag_process, input_image, gpu)
        tag_bytes = json.dumps({"tags": tags}).encode("utf-8")
        state.job_results[job.job_id] = (tag_bytes, "application/json")
        job.completed_at = datetime.utcnow()
        job.set_result(JobResult(success=True))
        log.info(f"Enqueue tag complete: {len(tags)} tags, job_id={job.job_id}")
    except Exception as ex:
        job.completed_at = datetime.utcnow()
        job.set_result(JobResult(success=False, error=str(ex)))
        log.log_exception(ex, "Enqueue tag failed")

    return {"job_id": job.job_id}


@router.post("/enqueue/enhance")
async def enqueue_enhance(request: Request):
    """Enhance (img2img hires): upscale input image, VAE encode, hires denoise, VAE decode.

    Body: PNG/JPEG image bytes
    Query params:
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
    """
    from scheduling.job import InferenceJob, JobType, SdxlJobInput, SdxlHiresInput

    state = _get_state()

    if not state.gpu_pool.has_capability("sdxl"):
        return _error(404, "No GPUs configured for sdxl on this worker.")
    if not state.gpu_pool.has_capability("upscale"):
        return _error(404, "No GPUs configured for upscale on this worker.")

    body = await request.body()
    try:
        input_image = Image.open(io.BytesIO(body)).convert("RGB")
    except Exception as ex:
        return _error(400, f"Could not decode image: {ex}")

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

    # Enhance only needs multiples of 8 (VAE latent alignment), not 64
    err = _validate_dimensions(hires_width, hires_height, max_dim=4096, multiple=8)
    if err:
        return _error(400, err)

    try:
        hires_steps        = int(qp.get("hires_steps", "15"))
        denoising_strength = float(qp.get("denoising_strength", "0.33"))
        cfg_scale          = float(qp.get("cfg_scale", "7.0"))
        seed_param         = int(qp.get("seed", "0"))
        unet_tile_w        = int(qp.get("unet_tile_width",  "0"))
        unet_tile_h        = int(qp.get("unet_tile_height", "0"))
        vae_tile_w         = int(qp.get("vae_tile_width",   "0"))
        vae_tile_h         = int(qp.get("vae_tile_height",  "0"))
    except (ValueError, TypeError) as ex:
        return _error(400, f"Invalid query parameter: {ex}")

    if not (1 <= hires_steps <= 150):
        return _error(400, "hires_steps must be between 1 and 150.")
    if not (0.0 <= denoising_strength <= 1.0):
        return _error(400, "denoising_strength must be between 0.0 and 1.0.")
    if not (1.0 <= cfg_scale <= 30.0):
        return _error(400, "cfg_scale must be between 1.0 and 30.0.")

    seed = seed_param if seed_param != 0 else random.randint(1, 2**31 - 1)

    model_name = qp.get("model")
    model_dir, err = _resolve_model(model_name)
    if err:
        return _error(400, err)

    # Base dimensions = original input image size
    base_w = input_image.width
    base_h = input_image.height

    # Only upscale if the input image is smaller than the target in either dimension
    needs_upscale = base_w < hires_width or base_h < hires_height

    if not needs_upscale:
        # Image is already large enough — just resize to target dimensions
        if base_w != hires_width or base_h != hires_height:
            input_image = input_image.resize((hires_width, hires_height), Image.LANCZOS)
            log.info(f"Enhance: input already large enough, resized {base_w}x{base_h} → "
                     f"{hires_width}x{hires_height} (no upscale needed)")

    # Parse LoRA tags from prompt
    cleaned_prompt, cleaned_neg, all_loras = _parse_request_loras(
        prompt, negative_prompt, None)

    model_short = os.path.splitext(os.path.basename(model_dir))[0] if model_dir else "default"
    log.info(f"Enqueue enhance: input={base_w}x{base_h} target={hires_width}x{hires_height} "
             f"upscale={'yes' if needs_upscale else 'no'} "
             f"hires_steps={hires_steps} strength={denoising_strength:.2f} "
             f"cfg={cfg_scale} seed={seed} model={model_short}"
             + (f" loras={[s.name for s in all_loras]}" if all_loras else ""))

    factory = state.pipeline_factory
    queue = state.queue

    job = InferenceJob(
        job_type=JobType.ENHANCE,
        pipeline=factory.create_enhance_pipeline(model_dir, needs_upscale=needs_upscale),
        sdxl_input=SdxlJobInput(
            prompt=cleaned_prompt,
            negative_prompt=cleaned_neg,
            width=base_w,
            height=base_h,
            steps=0,  # not used for enhance — hires_steps drives denoise
            cfg_scale=cfg_scale,
            seed=seed,
            model_dir=model_dir,
            loras=all_loras,
            regional_prompting=qp.get("regional_prompting", "false").lower() == "true",
        ),
        hires_input=SdxlHiresInput(
            hires_width=hires_width,
            hires_height=hires_height,
            hires_steps=hires_steps,
            denoising_strength=denoising_strength,
        ),
        input_image=input_image,
    )
    job.is_hires_pass = True  # denoise uses hires path from the start
    job.unet_tile_width  = unet_tile_w
    job.unet_tile_height = unet_tile_h
    job.vae_tile_width   = vae_tile_w
    job.vae_tile_height  = vae_tile_h

    _register_job(job)
    queue.enqueue(job)
    return {"job_id": job.job_id}


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
        buf = io.BytesIO()
        if result.output_image.mode == "RGBA":
            result.output_image.save(buf, format="PNG")
        else:
            result.output_image.convert("RGB").save(buf, format="PNG")
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
):
    """Return recent log lines with optional filtering.

    Query params:
        lines:   Number of lines to return (default 100, max 10000)
        offset:  Skip this many lines from the tail before returning
        level:   Filter by log level: DEBUG, INFO, WARNING, ERROR
        search:  Case-insensitive substring search across log lines
        before:  Only lines before this timestamp (ISO 8601 or HH:MM:SS)
        after:   Only lines after this timestamp (ISO 8601 or HH:MM:SS)
    """
    log_path = log.get_log_path()
    if log_path is None or not os.path.isfile(log_path):
        return _error(404, "Log file not available")

    lines = max(1, min(lines, 10000))

    try:
        with open(log_path, "r", encoding="utf-8", errors="replace") as f:
            all_lines = f.readlines()
    except OSError as ex:
        return _error(500, f"Failed to read log file: {ex}")

    # Apply filters
    filtered = all_lines

    if level:
        level_tag = f"[{level.upper()}]"
        filtered = [ln for ln in filtered if level_tag in ln]

    if search:
        search_lower = search.lower()
        filtered = [ln for ln in filtered if search_lower in ln.lower()]

    if after or before:
        filtered = _filter_by_time(filtered, after=after, before=before)

    # Apply offset + limit from the tail
    total = len(filtered)
    if offset > 0:
        filtered = filtered[:max(0, total - offset)]
    filtered = filtered[-lines:]

    return Response(
        content="".join(filtered),
        media_type="text/plain; charset=utf-8",
        headers={"X-Total-Lines": str(total), "X-Returned-Lines": str(len(filtered))},
    )


def _filter_by_time(lines: list[str], after: str | None, before: str | None) -> list[str]:
    """Filter log lines by timestamp range."""
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
        return lines

    result = []
    for ln in lines:
        # Extract timestamp: [2026-02-26 10:14:24.918]
        if len(ln) > 25 and ln[0] == "[":
            ts_str = ln[1:24]
            ts = _parse_ts(ts_str)
            if ts is None:
                result.append(ln)
                continue
            if after_dt and ts < after_dt:
                continue
            if before_dt and ts > before_dt:
                continue
        result.append(ln)
    return result
