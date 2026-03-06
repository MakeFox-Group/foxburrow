"""GPU worker subprocess — owns all CUDA state for a single GPU.

Launched by the main process with CUDA_VISIBLE_DEVICES set so this
process sees exactly one GPU as cuda:0.  Runs a sequential command
loop: no concurrency, no locks, no GIL contention with other GPUs.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import sys
import time as _time
import traceback
from collections import OrderedDict
from datetime import datetime

# NOTE: torch is imported AFTER CUDA_VISIBLE_DEVICES is set in gpu_worker_main().
# Do not import torch at module level.

from scheduling.worker_protocol import (
    DrainCmd,
    DrainComplete,
    ExecuteStageCmd,
    GetStatusCmd,
    OnloadCmd,
    OnloadComplete,
    ProcessError,
    ProgressUpdate,
    ReleaseDrainCmd,
    ShutdownCmd,
    StageResult,
    StatusSnapshot,
    TagImageCmd,
    TagResult,
    TrtBuildCmd,
    TrtBuildProgress,
    TrtBuildResult,
    UpdateLoraIndexCmd,
    UpdateSdxlModelsCmd,
    WorkerReady,
)


def gpu_worker_main(
    cuda_visible_devices: str,
    gpu_index: int,
    gpu_uuid: str,
    gpu_name: str,
    gpu_capabilities: set[str],
    gpu_onload: set[str],
    gpu_unevictable: set[str],
    gpu_total_memory: int,
    server_config_dict: dict,
    cmd_queue: mp.Queue,
    result_queue: mp.Queue,
) -> None:
    """Entry point for each GPU worker process.

    Args:
        cuda_visible_devices: Value for CUDA_VISIBLE_DEVICES (e.g. "GPU-abc123...")
        gpu_index: Original GPU index in the main process (for logging)
        gpu_uuid: GPU UUID string
        gpu_name: Human-readable GPU name
        gpu_capabilities: Set of capability strings
        gpu_onload: Set of onload config entries
        gpu_unevictable: Set of unevictable config entries
        gpu_total_memory: Total GPU memory in bytes (from NVML)
        server_config_dict: Serialized server config
        cmd_queue: Commands from main process
        result_queue: Results back to main process
    """
    # Ask the kernel to send SIGTERM when our parent process dies.
    # This prevents orphaned GPU workers if the main process is OOM-killed.
    try:
        import ctypes
        import signal
        _PR_SET_PDEATHSIG = 1
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass  # Non-Linux or libc unavailable — fall back to queue timeout

    # Track parent PID for periodic liveness checks
    _parent_pid = os.getppid()

    # Set CUDA visibility BEFORE any torch import
    os.environ["CUDA_VISIBLE_DEVICES"] = cuda_visible_devices
    os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

    # NOTE: PYTORCH_ALLOC_CONF (e.g. backend:cudaMallocAsync) is NOT set here.
    # Recent PyTorch nightlies fix the allocator backend at compile time.
    # Setting backend:cudaMallocAsync on a build compiled with "native"
    # corrupts the allocator config during import and hangs on first alloc.
    # If cudaMallocAsync is desired, set PYTORCH_ALLOC_CONF in the environment
    # (e.g. foxburrow.sh) ONLY when the PyTorch build supports it.

    import torch
    import torch.nn as nn

    # Enable SDP attention backends
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch._inductor.config.conv_1x1_as_mm = True

    import log
    log.init_file(os.path.abspath(f"data/worker_{gpu_index}.jsonl"))
    log.info(f"GPU worker process started: GPU[{gpu_index}] {gpu_name} ({gpu_uuid})")

    device = torch.device("cuda:0")

    # Cap VRAM at 98%
    try:
        torch.cuda.set_per_process_memory_fraction(0.98, 0)
    except Exception as e:
        log.warning(f"  Could not set VRAM cap: {e}")

    # Patch accelerate for thread safety (even though we're single-threaded,
    # libraries like diffusers use it internally)
    try:
        from gpu.pool import patch_accelerate_thread_safety
        patch_accelerate_thread_safety()
    except Exception as e:
        log.warning(f"  Could not patch accelerate: {e}")

    # Check runtime support for allocation tags etc.
    from gpu import torch_ext
    torch_ext.check_runtime_support(device)

    # Initialize NVML in this process for VRAM queries
    from gpu import nvml
    nvml.init()
    nvml_devices = nvml.get_devices()
    # With CUDA_VISIBLE_DEVICES set to a single GPU UUID, NVML still sees all
    # GPUs. Find our GPU by UUID.
    nvml_handle = None
    for dev in nvml_devices:
        if dev.uuid.lower() == gpu_uuid.lower():
            nvml_handle = dev.handle
            break
    if nvml_handle is None:
        log.error(f"  Could not find GPU {gpu_uuid} in NVML devices")
        result_queue.put(ProcessError("NVML device not found", fatal=True))
        return

    # Warm CUDA context
    torch.zeros(1, device=device)

    # Get GPU model name for workspace profiler
    from gpu.workspace_profiler import get_gpu_model_name
    gpu_model_name = get_gpu_model_name(device)

    # Get architecture key for TRT
    arch_key = ""
    try:
        from trt.builder import get_arch_key
        arch_key = get_arch_key(0)
    except Exception:
        pass

    # Create GpuInstance for this process
    from gpu.pool import GpuInstance
    from config import GpuConfig

    gpu_config = GpuConfig(
        uuid=gpu_uuid,
        name=gpu_name,
        capabilities=gpu_capabilities,
        onload=gpu_onload,
        unevictable=gpu_unevictable,
    )

    # We need a minimal NvmlDeviceInfo-like object for GpuInstance
    class _NvmlInfo:
        def __init__(self, handle, total_memory):
            self.handle = handle
            self.total_memory = total_memory

    nvml_info = _NvmlInfo(nvml_handle, gpu_total_memory)
    gpu = GpuInstance(gpu_config, nvml_info, torch_device_id=0)

    # Initialize app_state in this process — handlers access it for config
    # values (tensorrt_cache, models_dir) via app_state.config.server.
    from state import app_state
    from config import ServerConfig, FoxBurrowConfig
    _server_cfg = ServerConfig()
    _server_cfg.tensorrt_cache = server_config_dict.get(
        "tensorrt_cache", _server_cfg.tensorrt_cache)
    _server_cfg.models_dir = server_config_dict.get(
        "models_dir", _server_cfg.models_dir)
    from config import TensorrtConfig
    _trt_cfg = TensorrtConfig()
    _trt_cfg.enabled = server_config_dict.get("trt_enabled", True)
    app_state.config = FoxBurrowConfig(
        server=_server_cfg, gpus=[gpu_config], tensorrt=_trt_cfg)

    # Initialize profiling tracer
    from profiling.tracer import register_tracer
    tracer = register_tracer(gpu_uuid, gpu_model_name, gpu_name)

    # Send ready signal
    result_queue.put(WorkerReady(gpu_model_name=gpu_model_name, arch_key=arch_key))
    result_queue.put(_build_status_snapshot(gpu, gpu_model_name, arch_key))

    log.info(f"  GPU worker ready: cuda:0 = {gpu_name} (arch={arch_key})")

    # Create a single event loop for the worker process lifetime.
    # Handlers use job._loop for WebSocket progress broadcasts and
    # set_result callbacks. Creating one per _execute_stage_cmd leaks loops.
    import asyncio
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    # ── Command loop ──────────────────────────────────────────────
    import queue as _queue_mod
    while True:
        try:
            cmd = cmd_queue.get(timeout=5.0)
        except _queue_mod.Empty:
            # Periodic liveness check: exit if parent died
            if os.getppid() != _parent_pid:
                log.info("  GPU worker: parent process died, exiting")
                break
            continue
        except (EOFError, OSError):
            log.info("  GPU worker: command queue closed, exiting")
            break

        try:
            if isinstance(cmd, ExecuteStageCmd):
                result = _execute_stage_cmd(gpu, cmd, gpu_model_name, tracer)
                result_queue.put(result)

            elif isinstance(cmd, DrainCmd):
                _drain(gpu)
                result_queue.put(DrainComplete())

            elif isinstance(cmd, ReleaseDrainCmd):
                log.info(f"  GPU worker [{gpu_uuid}]: Drain released")

            elif isinstance(cmd, TrtBuildCmd):
                trt_result = _handle_trt_build(cmd, result_queue)
                result_queue.put(trt_result)

            elif isinstance(cmd, TagImageCmd):
                tag_result = _handle_tag(gpu, cmd)
                result_queue.put(tag_result)

            elif isinstance(cmd, OnloadCmd):
                _handle_onload(gpu, cmd, server_config_dict)
                result_queue.put(OnloadComplete())

            elif isinstance(cmd, UpdateLoraIndexCmd):
                from state import app_state
                app_state.lora_index = cmd.lora_index
                log.debug(f"  GPU worker [{gpu_uuid}]: LoRA index updated "
                          f"({len(cmd.lora_index)} entries)")

            elif isinstance(cmd, UpdateSdxlModelsCmd):
                from state import app_state
                app_state.sdxl_models = dict(cmd.sdxl_models)
                # Re-register any new checkpoints in the worker's registry
                for model_name, model_dir in cmd.sdxl_models.items():
                    try:
                        app_state.registry.register_sdxl_checkpoint(model_dir)
                    except Exception:
                        pass
                log.debug(f"  GPU worker [{gpu_uuid}]: SDXL models updated "
                          f"({len(cmd.sdxl_models)} entries)")

            elif isinstance(cmd, GetStatusCmd):
                pass  # Status is sent below after every command

            elif isinstance(cmd, ShutdownCmd):
                log.info(f"  GPU worker [{gpu_uuid}]: Shutdown requested")
                break

            else:
                log.warning(f"  GPU worker: Unknown command type: {type(cmd).__name__}")

        except Exception as ex:
            fatal = _is_cuda_fatal_local(ex)
            log.log_exception(ex, f"GPU worker [{gpu_uuid}]: Command failed")
            result_queue.put(ProcessError(
                error=f"{type(ex).__name__}: {ex}",
                fatal=fatal,
            ))

        # Always send fresh status after every command
        try:
            result_queue.put(_build_status_snapshot(gpu, gpu_model_name, arch_key))
        except Exception:
            pass

    log.info(f"  GPU worker [{gpu_uuid}]: Exiting")


# ── Status snapshot builder ───────────────────────────────────────

def _build_status_snapshot(gpu, gpu_model_name: str, arch_key: str) -> StatusSnapshot:
    """Build a StatusSnapshot from the current GPU state."""
    import torch
    from gpu import nvml

    cached_fps = set()
    cached_cats = []
    fp_vram: dict[str, int] = {}
    with gpu._cache_lock:
        for fp, m in gpu._cache.items():
            cached_fps.add(fp)
            cached_cats.append(m.category)
            fp_vram[fp] = m.actual_vram if m.actual_vram > 0 else m.estimated_vram

    # Collect runtime BPP measurements from this worker process
    from scheduling.worker import _measured_bpp, _bpp_lock
    with _bpp_lock:
        bpp_copy = {st.value: v for st, v in _measured_bpp.items()}

    try:
        vram_stats = gpu.get_vram_stats()
    except Exception:
        vram_stats = {}

    return StatusSnapshot(
        cached_fingerprints=cached_fps,
        cached_categories=cached_cats,
        cached_models_info=gpu.get_cached_models_info(),
        session_group=gpu._current_group,
        vram_stats=vram_stats,
        loaded_models_vram=gpu.get_loaded_models_vram(),
        evictable_vram=gpu.get_evictable_vram(),
        is_failed=gpu.is_failed,
        fail_reason=gpu._fail_reason,
        is_busy=gpu.is_busy,
        trt_shared_memory_vram=gpu.get_trt_shared_memory_vram(),
        loaded_lora_count=len(getattr(gpu, "_loaded_lora_adapters", {})),
        gpu_model_name=gpu_model_name,
        arch_key=arch_key,
        fingerprint_vram=fp_vram,
        measured_bpp=bpp_copy,
    )


# ── Stage execution ───────────────────────────────────────────────

def _execute_stage_cmd(gpu, cmd: ExecuteStageCmd, gpu_model_name: str, tracer) -> StageResult:
    """Execute a single pipeline stage. Returns StageResult with all outputs on CPU."""
    import torch
    from gpu import torch_ext
    from scheduling.job import (
        InferenceJob, JobType, SdxlEncodeResult, SdxlRegionalEncodeResult,
        WorkStage, StageType, ModelComponentId,
    )
    from profiling.tracer import set_current_tracer

    device = torch.device("cuda:0")

    # Reconstruct a WorkStage
    stage = WorkStage(
        type=cmd.stage_type,
        required_components=list(cmd.required_components),
        required_capability=cmd.required_capability,
    )

    # Reconstruct a minimal InferenceJob for handler compatibility
    job = InferenceJob.__new__(InferenceJob)
    job.job_id = cmd.job_id
    job.type = JobType(cmd.job_type_value)
    job.sdxl_input = cmd.sdxl_input
    job.hires_input = cmd.hires_input
    job.input_image = cmd.input_image
    job.is_hires_pass = cmd.is_hires_pass
    job.oom_retries = cmd.oom_retries
    job.current_stage_index = cmd.current_stage_index
    job.pipeline = [stage]  # Minimal pipeline for current_stage property
    job.tokenize_result = cmd.tokenize_result
    job.latents = cmd.latents
    job.denoise_step = 0
    job.denoise_total_steps = 0
    job.stage_step = 0
    job.stage_total_steps = 0
    job.stage_status = ""
    job.active_gpus = []
    job.gpu_time_s = 0.0
    job.model_load_time_s = 0.0
    job.gpu_stage_times = []
    job.started_at = datetime.utcnow()
    job.completed_at = None
    job.priority = cmd.priority
    job.unet_tile_width = cmd.unet_tile_width
    job.unet_tile_height = cmd.unet_tile_height
    job.vae_tile_width = cmd.vae_tile_width
    job.vae_tile_height = cmd.vae_tile_height

    # Set original dimensions for final resize
    if cmd.orig_width is not None:
        job.orig_width = cmd.orig_width
    if cmd.orig_height is not None:
        job.orig_height = cmd.orig_height

    # Reconstruct encode results from CPU tensors
    job.encode_result = None
    if cmd.encode_result_tensors:
        from scheduling.job import SdxlEncodeResult
        job.encode_result = SdxlEncodeResult(
            prompt_embeds=cmd.encode_result_tensors.get("prompt_embeds"),
            neg_prompt_embeds=cmd.encode_result_tensors.get("neg_prompt_embeds"),
            pooled_prompt_embeds=cmd.encode_result_tensors.get("pooled_prompt_embeds"),
            neg_pooled_prompt_embeds=cmd.encode_result_tensors.get("neg_pooled_prompt_embeds"),
        )

    # Reconstruct regional encode results
    job.regional_encode_result = None
    if cmd.regional_encode_tensors:
        from scheduling.job import SdxlRegionalEncodeResult
        job.regional_encode_result = SdxlRegionalEncodeResult(
            region_embeds=cmd.regional_encode_tensors.get("region_embeds", []),
            neg_prompt_embeds=cmd.regional_encode_tensors.get("neg_prompt_embeds"),
            neg_region_embeds=cmd.regional_encode_tensors.get("neg_region_embeds"),
            pooled_prompt_embeds=cmd.regional_encode_tensors.get("pooled_prompt_embeds"),
            neg_pooled_prompt_embeds=cmd.regional_encode_tensors.get("neg_pooled_prompt_embeds"),
            base_embeds=cmd.regional_encode_tensors.get("base_embeds"),
            base_ratio=cmd.regional_encode_tensors.get("base_ratio", 0.2),
        )

    job.regional_tokenize_results = cmd.regional_tokenize_results
    job.regional_base_tokenize = cmd.regional_base_tokenize
    job.regional_shared_neg_tokenize = cmd.regional_shared_neg_tokenize

    # Reconstruct regional info
    job.regional_info = None
    if cmd.regional_info_data:
        from utils.regional import RegionalPromptResult
        job.regional_info = RegionalPromptResult.from_dict(cmd.regional_info_data)

    # Use the worker-level event loop (created once in gpu_worker_main).
    # Handlers use job._loop for WebSocket progress broadcasts.
    import asyncio
    loop = asyncio.get_event_loop()
    job._loop = loop
    job.completion = loop.create_future()

    # Pass TE fingerprints from proxy so non-TE stages can protect
    # TE TRT engines from eviction during model loading.
    job._te_fingerprints = cmd.te_fingerprints

    # ── Model loading ─────────────────────────────────────────────
    load_start = _time.monotonic()

    # Release PyTorch cached allocator memory before loading
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

    vram_before = torch.cuda.memory_allocated(device)
    _ensure_models_for_stage(gpu, stage, job, gpu_model_name)

    # Ensure free VRAM for working memory
    from scheduling.worker import _get_min_free_vram
    min_free = _get_min_free_vram(stage.type, job, gpu_model_name)
    if min_free > 0:
        gpu.ensure_free_vram(min_free)

    load_duration = _time.monotonic() - load_start
    vram_delta = torch.cuda.memory_allocated(device) - vram_before

    # Record model load event
    if load_duration > 0.01:
        model_name = _get_model_name_from_job(job)
        tracer.model_load(
            job.job_id, model_name, stage.type.value,
            load_duration, vram_delta)

    # ── Execute stage ─────────────────────────────────────────────
    set_current_tracer(tracer)

    stage_start = _time.monotonic()

    # Memory measurement
    peak_scope = None
    mem_baseline = 0
    if torch_ext.HAS_PEAK_SCOPE:
        peak_scope = torch.cuda.PeakMemoryScope(device=device)
        peak_scope.__enter__()
    else:
        with torch.cuda.device(device):
            torch.cuda.reset_peak_memory_stats()
        mem_baseline = torch.cuda.memory_allocated(device)

    # Tag activations
    tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_ACTIVATIONS)
               if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())

    output_image = None
    oom = False
    fatal = False
    error = None

    migrate_to_gpu_s = 0.0
    try:
        # Move tensors to GPU (CPU→VRAM transfer)
        _migrate_start = _time.monotonic()
        _migrate_tensors_to_device(job, device)
        migrate_to_gpu_s = _time.monotonic() - _migrate_start

        with tag_ctx:
            output_image = _dispatch_stage(job, stage, gpu)

        gpu.record_success()

    except torch.cuda.OutOfMemoryError as ex:
        oom = True
        error = str(ex)
        torch.cuda.empty_cache()

    except Exception as ex:
        if _is_cuda_fatal_local(ex):
            fatal = True
            gpu.mark_failed(f"CUDA context corrupted ({type(ex).__name__})")
        error = str(ex)

    finally:
        if peak_scope is not None:
            peak_scope.__exit__(None, None, None)
            working_mem = peak_scope.peak_bytes
        else:
            mem_peak = torch.cuda.max_memory_allocated(device)
            working_mem = mem_peak - mem_baseline

        with torch.cuda.device(device):
            torch.cuda.empty_cache()

        set_current_tracer(None)

    stage_duration = _time.monotonic() - stage_start

    # Update BPP measurement (always solo in worker process)
    if error is None:
        from scheduling.worker import _get_stage_pixels, _update_working_memory
        stage_pixels = _get_stage_pixels(stage.type, job)
        _update_working_memory(stage.type, working_mem, stage_pixels)

    # Record per-stage GPU timing for the API (migrate_to_cpu_s added after serialization)
    model_name = _get_model_name_from_job(job)
    vram_alloc = torch.cuda.memory_allocated(device)
    vram_reserved = torch.cuda.memory_reserved(device)
    _stage_timing = {
        "gpu": gpu.uuid,
        "gpu_name": gpu.name,
        "gpu_arch": gpu_model_name,
        "stage": stage.type.value,
        "model": model_name or "",
        "duration_s": round(stage_duration, 3),
        "load_s": round(load_duration, 3),
        "migrate_to_gpu_s": round(migrate_to_gpu_s, 4),
        "working_mem_bytes": working_mem,
        "vram_allocated": vram_alloc,
        "vram_reserved": vram_reserved,
    }

    # Record stage completion for profiling
    if error is None:
        _stage_w = job.sdxl_input.width if job.sdxl_input else 0
        _stage_h = job.sdxl_input.height if job.sdxl_input else 0
        _stage_steps = job.denoise_step if job.denoise_step > 0 else None
        tracer.stage_complete(
            job.job_id, model_name, stage.type.value,
            _stage_w, _stage_h, _stage_steps, stage_duration)

    # Release active fingerprints added by _ensure_models_for_stage (line 676-678)
    # and _load_sdxl_components (te_protect_fps only).
    release_fps: set[str] = set()
    release_fps.update(c.fingerprint for c in stage.required_components)
    release_fps.update(getattr(stage, '_trt_active_fps', set()))
    if release_fps:
        gpu.remove_active_fingerprints(release_fps)
    te_protect = getattr(stage, '_te_protect_fps', set())
    if te_protect:
        gpu.remove_active_fingerprints(te_protect)

    gpu.release()

    # Handle intermediate image for multi-stage pipelines
    if output_image is not None and not oom and error is None:
        if cmd.current_stage_index + 1 < cmd.pipeline_length:
            # Store as input_image for next stage
            job.input_image = output_image
            if job.hires_input:
                tw = job.hires_input.hires_width
                th = job.hires_input.hires_height
                if job.input_image.width != tw or job.input_image.height != th:
                    from PIL import Image as _PILImage
                    job.input_image = job.input_image.resize((tw, th), _PILImage.LANCZOS)

    # ── Build result ──────────────────────────────────────────────
    # Move all tensors to CPU for pickling (VRAM→SYSRAM transfer)
    _serialize_start = _time.monotonic()
    encode_tensors = None
    if job.encode_result is not None:
        er = job.encode_result
        encode_tensors = {}
        for name in ("prompt_embeds", "neg_prompt_embeds",
                      "pooled_prompt_embeds", "neg_pooled_prompt_embeds"):
            t = getattr(er, name, None)
            if t is not None:
                encode_tensors[name] = t.cpu()

    regional_tensors = None
    if job.regional_encode_result is not None:
        rer = job.regional_encode_result
        regional_tensors = {
            "region_embeds": [e.cpu() for e in rer.region_embeds],
            "neg_prompt_embeds": rer.neg_prompt_embeds.cpu() if rer.neg_prompt_embeds is not None else None,
            "neg_region_embeds": [e.cpu() for e in rer.neg_region_embeds] if rer.neg_region_embeds is not None else None,
            "pooled_prompt_embeds": rer.pooled_prompt_embeds.cpu() if rer.pooled_prompt_embeds is not None else None,
            "neg_pooled_prompt_embeds": rer.neg_pooled_prompt_embeds.cpu() if rer.neg_pooled_prompt_embeds is not None else None,
            "base_embeds": rer.base_embeds.cpu() if rer.base_embeds is not None else None,
            "base_ratio": rer.base_ratio,
        }

    latents_cpu = None
    if job.latents is not None:
        latents_cpu = job.latents.cpu()

    regional_info_data = None
    if job.regional_info is not None:
        regional_info_data = job.regional_info.to_dict()

    migrate_to_cpu_s = _time.monotonic() - _serialize_start
    _stage_timing["migrate_to_cpu_s"] = round(migrate_to_cpu_s, 4)
    job.gpu_stage_times.append(_stage_timing)

    return StageResult(
        job_id=cmd.job_id,
        success=error is None,
        error=error,
        oom=oom,
        fatal=fatal,
        output_image=output_image if error is None else None,
        output_latents=latents_cpu if error is None and job.type.value in (
            "SdxlGenerateLatents", "SdxlEncodeLatents", "SdxlHiresLatents") else None,
        encode_result_tensors=encode_tensors,
        regional_encode_tensors=regional_tensors,
        latents=latents_cpu,
        tokenize_result=job.tokenize_result,
        regional_tokenize_results=job.regional_tokenize_results,
        regional_base_tokenize=job.regional_base_tokenize,
        regional_shared_neg_tokenize=job.regional_shared_neg_tokenize,
        regional_info_data=regional_info_data,
        is_hires_pass=job.is_hires_pass,
        current_stage_index=cmd.current_stage_index + 1,
        denoise_step=job.denoise_step,
        denoise_total_steps=job.denoise_total_steps,
        model_load_time_s=load_duration,
        gpu_time_s=stage_duration,
        gpu_stage_times=job.gpu_stage_times,
    )


# ── Model loading (extracted from GpuWorker._ensure_models_for_stage) ──

def _ensure_models_for_stage(gpu, stage, job, gpu_model_name: str) -> None:
    """Load model components required for the given stage."""
    import torch
    from gpu import torch_ext
    from scheduling.job import StageType

    model_dir = job.sdxl_input.model_dir if job.sdxl_input else None

    if stage.type in (StageType.GPU_TEXT_ENCODE, StageType.GPU_DENOISE,
                      StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE):
        gpu.ensure_session_group("sdxl")
        _load_sdxl_components(gpu, stage, model_dir, job, gpu_model_name)
    elif stage.type == StageType.GPU_UPSCALE:
        gpu.ensure_session_group("upscale")
        _load_upscale_model(gpu)
    elif stage.type == StageType.GPU_BGREMOVE:
        gpu.ensure_session_group("bgremove")
        _load_bgremove_model(gpu)

    # Add active fingerprints for eviction protection
    active_fps = {c.fingerprint for c in stage.required_components}
    active_fps.update(getattr(stage, '_trt_active_fps', set()))
    gpu.add_active_fingerprints(active_fps)


def _load_sdxl_components(gpu, stage, model_dir, job, gpu_model_name: str) -> None:
    """Load SDXL sub-model components for a stage."""
    import torch
    import contextlib
    from gpu import torch_ext
    from scheduling.job import StageType
    import log

    source_name = ""
    if model_dir:
        source_name = os.path.basename(model_dir)
        for ext in (".safetensors", ".ckpt"):
            if source_name.endswith(ext):
                source_name = source_name[:-len(ext)]
                break

    trt_fps: set[str] = set()

    # Build a protect set for ensure_free_vram() calls during loading.
    # This prevents eviction of models needed by THIS stage while loading
    # other models. The protect= parameter does NOT increment ref-counts —
    # ref-counts are managed solely by _ensure_models_for_stage() after
    # loading completes (lines 676-678).
    all_required_fps = {c.fingerprint for c in stage.required_components}

    # Also protect TE TRT engines from eviction during non-TE stages.
    # Uses _te_fingerprints passed from the proxy (extracted from the
    # job's pipeline TE stage) since each stage gets a fresh job object.
    te_protect_fps: set[str] = set()
    if stage.type != StageType.GPU_TEXT_ENCODE:
        te_fps = getattr(job, '_te_fingerprints', None) or {}
        for te_key, comp_name in (("sdxl_te1", "te1"), ("sdxl_te2", "te2")):
            base = te_fps.get(te_key)
            if base:
                te_fp = f"{base}:{comp_name}_trt:default"
                if gpu.is_component_loaded(te_fp):
                    te_protect_fps.add(te_fp)

    # TE TRT engines need ref-count protection because _ensure_models_for_stage
    # doesn't know about them (they're not in stage.required_components).
    if te_protect_fps:
        gpu.add_active_fingerprints(te_protect_fps)
    stage._te_protect_fps = te_protect_fps

    # protect_fps is passed to ensure_free_vram(protect=...) during loading.
    # This does NOT touch ref-counts — it only prevents eviction of these
    # specific fingerprints during the ensure_free_vram call.
    protect_fps = all_required_fps | te_protect_fps

    device = torch.device("cuda:0")

    for component in stage.required_components:
        if gpu.is_component_loaded(component.fingerprint):
            continue

        # Check TRT coverage
        if _trt_covers_component(gpu, stage, job, component):
            log.debug(f"  Skipping PyTorch {component.category} — TRT engine available")
            new_trt_fps = _preload_trt_engine(gpu, stage, job, component)
            trt_fps.update(new_trt_fps)
            # Add to protect set so later ensure_free_vram calls don't evict
            # newly loaded TRT engines. No add_active_fingerprints here —
            # ref-counts are managed by _ensure_models_for_stage via _trt_active_fps.
            protect_fps.update(new_trt_fps)
            continue

        # Ensure VRAM before loading — protect set prevents eviction of
        # required components and TRT engines loaded earlier in this stage
        gpu.ensure_free_vram(component.estimated_vram_bytes, protect=protect_fps)

        # Load the model component
        from handlers.sdxl import load_component
        before = torch.cuda.memory_allocated(device)
        tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
                   if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
        with tag_ctx:
            model = load_component(component.category, model_dir, device)
        after = torch.cuda.memory_allocated(device)
        actual_vram = after - before

        log.debug(f"  Loaded {component.category}: {actual_vram // (1024*1024)}MB actual "
                  f"(estimated {component.estimated_vram_bytes // (1024*1024)}MB)")

        # Feed actual VRAM back to registry
        if actual_vram > 0:
            from state import app_state
            if app_state.registry is not None:
                app_state.registry.update_actual_vram(component.fingerprint, actual_vram)

        # UNet eviction clears LoRA adapter tracking
        evict_cb = None
        if component.category == "sdxl_unet":
            _gpu_ref = gpu
            def _on_unet_evict(_g=_gpu_ref):
                _g._loaded_lora_adapters.clear()
            evict_cb = _on_unet_evict

        gpu.cache_model(
            component.fingerprint, component.category, model,
            component.estimated_vram_bytes, source=source_name,
            actual_vram=actual_vram, evict_callback=evict_cb,
        )

        # Profile workspace for this component type
        try:
            from gpu.workspace_profiler import ensure_profiled
            ensure_profiled(component.category, model, device, gpu_model_name)
        except Exception as ex:
            log.warning(f"  WorkspaceProfiler: Failed to profile {component.category}: {ex}")

    stage._trt_active_fps = trt_fps


def _trt_covers_component(gpu, stage, job, component) -> bool:
    """Check if a TRT engine can handle this component for the job's resolution."""
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return False

    inp = job.sdxl_input
    if inp is None:
        return False

    from scheduling.job import StageType

    if component.category == "sdxl_unet":
        if inp.loras:
            return False
        trt_component = "unet"
    elif component.category in ("sdxl_vae", "sdxl_vae_enc"):
        trt_component = "vae"
    elif component.category == "sdxl_te1":
        trt_component = "te1"
    elif component.category == "sdxl_te2":
        trt_component = "te2"
    else:
        return False

    if trt_component in ("te1", "te2"):
        try:
            from trt.builder import has_trt_coverage, get_arch_key
            from state import app_state
            cache_dir = app_state.config.server.tensorrt_cache
            arch_key = get_arch_key(0)
            return has_trt_coverage(
                cache_dir, component.fingerprint, trt_component, arch_key, 0, 0)
        except Exception:
            return False

    if stage.type == StageType.GPU_DENOISE:
        if job.is_hires_pass and job.hires_input:
            width, height = job.hires_input.hires_width, job.hires_input.hires_height
        else:
            width, height = inp.width, inp.height
    elif stage.type == StageType.GPU_VAE_DECODE:
        if job.is_hires_pass and job.hires_input:
            width, height = job.hires_input.hires_width, job.hires_input.hires_height
        else:
            width, height = inp.width, inp.height
    elif stage.type == StageType.GPU_VAE_ENCODE:
        width, height = inp.width, inp.height
    else:
        return False

    try:
        from trt.builder import has_trt_coverage, get_arch_key
        from state import app_state
        cache_dir = app_state.config.server.tensorrt_cache
        arch_key = get_arch_key(0)
        return has_trt_coverage(
            cache_dir, component.fingerprint, trt_component, arch_key, width, height)
    except Exception:
        return False


def _preload_trt_engine(gpu, stage, job, component) -> set[str]:
    """Pre-load a TRT engine during model-loading phase."""
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return set()

    from handlers.sdxl import (
        _get_trt_unet_runner, _get_trt_vae_runner,
        _get_trt_te1_runner, _get_trt_te2_runner,
    )

    inp = job.sdxl_input
    if inp is None:
        return set()

    from scheduling.job import StageType
    import log

    width, height = inp.width, inp.height
    if stage.type in (StageType.GPU_DENOISE, StageType.GPU_VAE_DECODE):
        if job.is_hires_pass and job.hires_input:
            width, height = job.hires_input.hires_width, job.hires_input.hires_height

    try:
        if component.category == "sdxl_unet":
            _get_trt_unet_runner(gpu, job, width, height)
        elif component.category in ("sdxl_vae", "sdxl_vae_enc"):
            _get_trt_vae_runner(gpu, job, width, height)
        elif component.category == "sdxl_te1":
            _get_trt_te1_runner(gpu, job)
        elif component.category == "sdxl_te2":
            _get_trt_te2_runner(gpu, job)
    except Exception as ex:
        log.debug(f"  TRT pre-load for {component.category} failed: {ex}")
        return set()

    _fp_keys = {"sdxl_unet": "sdxl_unet", "sdxl_vae": "sdxl_vae",
                "sdxl_vae_enc": "sdxl_vae", "sdxl_te1": "sdxl_te1",
                "sdxl_te2": "sdxl_te2"}
    _trt_comps = {"sdxl_unet": "unet", "sdxl_vae": "vae",
                  "sdxl_vae_enc": "vae", "sdxl_te1": "te1",
                  "sdxl_te2": "te2"}
    fp_key = _fp_keys.get(component.category)
    trt_comp = _trt_comps.get(component.category)
    if not fp_key or not trt_comp:
        return set()

    base_fp = getattr(job, '_stage_model_fps', {}).get(fp_key)
    if not base_fp:
        return set()

    candidates = []
    if trt_comp in ("te1", "te2"):
        candidates.append(f"{base_fp}:{trt_comp}_trt:default")
    else:
        candidates.append(f"{base_fp}:{trt_comp}_trt:{width}x{height}")
        from trt.builder import DYNAMIC_PROFILES
        for profile in DYNAMIC_PROFILES:
            candidates.append(f"{base_fp}:{trt_comp}_trt:{profile['label']}")

    loaded = set()
    for c in candidates:
        if gpu.is_component_loaded(c):
            loaded.add(c)
    return loaded


def _load_upscale_model(gpu) -> None:
    """Load the upscale model if not already cached."""
    import torch
    import contextlib
    from gpu import torch_ext
    import log

    device = torch.device("cuda:0")
    from handlers.upscale import load_model
    comp = None
    try:
        from state import app_state
        comp = app_state.registry.get_upscale_component()
    except Exception:
        pass
    if comp and gpu.is_component_loaded(comp.fingerprint):
        return
    if comp:
        gpu.ensure_free_vram(comp.estimated_vram_bytes, protect={comp.fingerprint})
    before = torch.cuda.memory_allocated(device)
    tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
               if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
    with tag_ctx:
        model = load_model(device)
    after = torch.cuda.memory_allocated(device)
    actual_vram = after - before
    if comp:
        if actual_vram > 0:
            from state import app_state
            app_state.registry.update_actual_vram(comp.fingerprint, actual_vram)
        gpu.cache_model(comp.fingerprint, "upscale", model, comp.estimated_vram_bytes,
                        source="realesrgan", actual_vram=actual_vram)
        try:
            from gpu.workspace_profiler import ensure_profiled
            ensure_profiled("upscale", model, device, "")
        except Exception:
            pass


def _load_bgremove_model(gpu) -> None:
    """Load the bgremove model if not already cached."""
    import torch
    import contextlib
    from gpu import torch_ext
    import log

    device = torch.device("cuda:0")
    from handlers.bgremove import load_model
    comp = None
    try:
        from state import app_state
        comp = app_state.registry.get_bgremove_component()
    except Exception:
        pass
    if comp and gpu.is_component_loaded(comp.fingerprint):
        return
    if comp:
        gpu.ensure_free_vram(comp.estimated_vram_bytes, protect={comp.fingerprint})
    before = torch.cuda.memory_allocated(device)
    tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
               if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())
    with tag_ctx:
        model = load_model(device)
    after = torch.cuda.memory_allocated(device)
    actual_vram = after - before
    if comp:
        if actual_vram > 0:
            from state import app_state
            app_state.registry.update_actual_vram(comp.fingerprint, actual_vram)
        gpu.cache_model(comp.fingerprint, "bgremove", model, comp.estimated_vram_bytes,
                        source="rmbg", actual_vram=actual_vram)
        try:
            from gpu.workspace_profiler import ensure_profiled
            ensure_profiled("bgremove", model, device, "")
        except Exception:
            pass


# ── Stage dispatch ────────────────────────────────────────────────

def _dispatch_stage(job, stage, gpu):
    """Dispatch to the appropriate handler for a stage type."""
    from scheduling.job import StageType

    # Store fingerprint mapping for handlers
    job._stage_model_fps = {c.category: c.fingerprint
                            for c in stage.required_components}

    # Move intermediate tensors to cuda:0
    _migrate_tensors_to_device(job, gpu.device)

    if stage.type == StageType.GPU_TEXT_ENCODE:
        from handlers.sdxl import text_encode
        text_encode(job, gpu)
        return None

    elif stage.type == StageType.GPU_DENOISE:
        from handlers.sdxl import denoise
        denoise(job, gpu)
        return None

    elif stage.type == StageType.GPU_VAE_DECODE:
        from handlers.sdxl import vae_decode
        return vae_decode(job, gpu)

    elif stage.type == StageType.GPU_VAE_ENCODE:
        from handlers.sdxl import vae_encode
        vae_encode(job, gpu)
        if job.hires_input is not None:
            job.is_hires_pass = True
        return None

    elif stage.type == StageType.GPU_UPSCALE:
        from handlers.upscale import execute
        return execute(job, gpu)

    elif stage.type == StageType.GPU_BGREMOVE:
        from handlers.bgremove import execute
        return execute(job, gpu)

    else:
        raise RuntimeError(f"GPU worker cannot execute stage type {stage.type}")


# ── Tensor migration ─────────────────────────────────────────────

def _migrate_tensors_to_device(job, device) -> None:
    """Move intermediate tensors to the worker's GPU device (cuda:0)."""
    import torch

    def _move(tensor, name):
        if tensor is None:
            return None
        if tensor.device == device:
            return tensor
        if tensor.device.type == device.type:
            src_idx = tensor.device.index or 0
            dst_idx = device.index or 0
            if src_idx == dst_idx:
                return tensor
        return tensor.to(device)

    if job.encode_result is not None:
        er = job.encode_result
        er.prompt_embeds = _move(er.prompt_embeds, "prompt_embeds")
        er.neg_prompt_embeds = _move(er.neg_prompt_embeds, "neg_prompt_embeds")
        er.pooled_prompt_embeds = _move(er.pooled_prompt_embeds, "pooled_prompt_embeds")
        er.neg_pooled_prompt_embeds = _move(er.neg_pooled_prompt_embeds, "neg_pooled_prompt_embeds")

    if job.latents is not None:
        job.latents = _move(job.latents, "latents")

    if job.regional_encode_result is not None:
        rer = job.regional_encode_result
        rer.region_embeds = [_move(e, f"region_embeds[{i}]")
                             for i, e in enumerate(rer.region_embeds)]
        if rer.neg_region_embeds is not None:
            rer.neg_region_embeds = [_move(e, f"neg_region_embeds[{i}]")
                                     for i, e in enumerate(rer.neg_region_embeds)]
        rer.neg_prompt_embeds = _move(rer.neg_prompt_embeds, "regional_neg")
        rer.pooled_prompt_embeds = _move(rer.pooled_prompt_embeds, "regional_pooled")
        rer.neg_pooled_prompt_embeds = _move(rer.neg_pooled_prompt_embeds, "regional_neg_pooled")
        rer.base_embeds = _move(rer.base_embeds, "regional_base")


# ── Drain handling ────────────────────────────────────────────────

def _drain(gpu) -> None:
    """Evict all cached models to free VRAM for TRT build."""
    import torch
    import log

    log.info(f"  GPU worker [{gpu.uuid}]: Draining — evicting all models")
    while gpu.evict_lru() is not None:
        pass
    with torch.cuda.device(gpu.device):
        torch.cuda.empty_cache()
    log.info(f"  GPU worker [{gpu.uuid}]: Drain complete")


# ── TRT build handling ────────────────────────────────────────────

def _handle_trt_build(cmd: TrtBuildCmd, result_queue) -> TrtBuildResult:
    """Build TRT engines for a model."""
    import log

    def _progress(component: str, engine: str) -> None:
        result_queue.put(TrtBuildProgress(component=component, engine=engine))

    try:
        from trt.builder import build_all_engines
        results = build_all_engines(
            cmd.model_hash, cmd.cache_dir, cmd.arch_key, device_id=0,
            max_workspace_gb=cmd.max_workspace_gb,
            progress_cb=_progress)
        return TrtBuildResult(success=True, results=results)
    except Exception as ex:
        log.log_exception(ex, f"TRT build failed for {cmd.model_hash[:16]}")
        return TrtBuildResult(success=False, error=str(ex))


# ── Tag handling ──────────────────────────────────────────────────

def _handle_tag(gpu, cmd: TagImageCmd) -> TagResult:
    """Tag an image using the tagger model."""
    import log

    try:
        from handlers.tagger import process_image
        tags = process_image(cmd.image, gpu, cmd.threshold)
        return TagResult(tags=tags)
    except Exception as ex:
        log.log_exception(ex, "Tag failed")
        return TagResult(error=str(ex))


# ── Onload handling ───────────────────────────────────────────────

def _handle_onload(gpu, cmd: OnloadCmd, server_config_dict: dict) -> None:
    """Process onload and unevictable entries."""
    import log

    from state import app_state

    # Populate app_state.sdxl_models from the OnloadCmd — the main process
    # passes the authoritative model map, but the worker's app_state starts empty.
    if cmd.sdxl_models:
        app_state.sdxl_models.update(cmd.sdxl_models)

    # Populate LoRA index and directory — needed for _ensure_loras() in handlers
    if cmd.lora_index is not None:
        app_state.lora_index = cmd.lora_index
    if cmd.loras_dir is not None:
        app_state.loras_dir = cmd.loras_dir

    # Register SDXL checkpoints in the worker's registry so that
    # get_sdxl_components() works for onload and unevictable resolution.
    if cmd.sdxl_models:
        for model_name, model_dir in cmd.sdxl_models.items():
            try:
                app_state.registry.register_sdxl_checkpoint(model_dir)
            except Exception:
                pass  # Already registered or missing — will fail at onload time

    for entry in cmd.onload_entries:
        entry_lower = entry.strip().lower()
        if cmd.types is not None:
            if entry_lower in ("tag", "upscale", "bgremove"):
                if entry_lower not in cmd.types:
                    continue
            else:
                if "sdxl" not in cmd.types:
                    continue

        try:
            if entry_lower == "tag":
                if not gpu.supports_capability("tag"):
                    continue
                import torch
                from handlers.tagger import init_tagger, unload_tagger
                device = gpu.device
                before = torch.cuda.memory_allocated(device)
                init_tagger(device)
                after = torch.cuda.memory_allocated(device)
                actual_vram = after - before
                _dev = device
                def _tagger_evict_cb(_d=_dev):
                    unload_tagger(_d)
                fp = f"tagger:{gpu.uuid}"
                gpu.cache_model(
                    fp, "tagger", None,
                    estimated_vram=150 * 1024 * 1024,
                    source="JTP_PILOT2_SigLIP",
                    actual_vram=actual_vram,
                    evict_callback=_tagger_evict_cb,
                )
                gpu.mark_onload(fp)
                log.debug(f"  GPU worker [{gpu.uuid}]: Pre-loaded tagger ({actual_vram // (1024*1024)}MB)")

            elif entry_lower == "upscale":
                from handlers.upscale import load_model as load_upscale
                model = load_upscale(gpu.device)
                comp = app_state.registry.get_upscale_component()
                gpu.cache_model(comp.fingerprint, "upscale", model,
                                comp.estimated_vram_bytes, source="realesrgan")
                gpu.mark_onload(comp.fingerprint)
                log.debug(f"  GPU worker [{gpu.uuid}]: Pre-loaded upscale model")

            elif entry_lower == "bgremove":
                from handlers.bgremove import load_model as load_bgremove
                model = load_bgremove(gpu.device)
                comp = app_state.registry.get_bgremove_component()
                gpu.cache_model(comp.fingerprint, "bgremove", model,
                                comp.estimated_vram_bytes, source="rmbg")
                gpu.mark_onload(comp.fingerprint)
                log.debug(f"  GPU worker [{gpu.uuid}]: Pre-loaded bgremove model")

            else:
                # SDXL model onload
                _handle_sdxl_onload(gpu, entry, app_state)

        except Exception as ex:
            log.log_exception(ex, f"GPU worker [{gpu.uuid}]: onload={entry} failed")

    # Process unevictable entries
    for entry in cmd.unevictable_entries:
        try:
            _handle_unevictable(gpu, entry, app_state)
        except Exception as ex:
            log.log_exception(ex, f"GPU worker [{gpu.uuid}]: unevictable={entry} failed")


def _handle_sdxl_onload(gpu, entry: str, app_state) -> None:
    """Handle SDXL model onload in worker process."""
    import torch
    import log

    # Parse entry (model_name or model_name:component)
    entry_lower = entry.strip().lower()
    from main import _resolve_onload_entry, _CATEGORY_TO_REGISTRY_INDEX
    actions = _resolve_onload_entry(entry_lower, app_state)
    if isinstance(actions, str):
        log.warning(f"  GPU worker [{gpu.uuid}]: onload={entry}: {actions}")
        return

    for action in actions:
        if action["type"] != "sdxl":
            continue
        model_name = action["model"]
        model_dir = app_state.sdxl_models[model_name]
        registry_comps = app_state.registry.get_sdxl_components(model_dir)
        from handlers.sdxl import load_component

        for category in action["categories"]:
            idx = _CATEGORY_TO_REGISTRY_INDEX.get(category)
            if idx is None:
                continue
            comp = registry_comps[idx]
            if gpu.is_component_loaded(comp.fingerprint):
                gpu.mark_onload(comp.fingerprint)
                continue
            model = load_component(category, model_dir, gpu.device)
            source = os.path.basename(model_dir)
            for ext in (".safetensors", ".ckpt"):
                if source.endswith(ext):
                    source = source[:-len(ext)]
                    break
            gpu.cache_model(comp.fingerprint, category, model,
                            comp.estimated_vram_bytes, source=source)
            gpu.mark_onload(comp.fingerprint)
            log.debug(f"  GPU worker [{gpu.uuid}]: Pre-loaded {category} from {source}")


def _handle_unevictable(gpu, entry: str, app_state) -> None:
    """Handle unevictable marking in worker process."""
    import log

    entry_lower = entry.strip().lower()
    from main import _resolve_onload_entry
    actions = _resolve_onload_entry(entry_lower, app_state)
    if isinstance(actions, str):
        log.warning(f"  GPU worker [{gpu.uuid}]: unevictable={entry}: {actions}")
        return

    for action in actions:
        if action["type"] == "tag":
            fp = f"tagger:{gpu.uuid}"
            gpu.mark_unevictable(fp)
        elif action["type"] == "upscale":
            comp = app_state.registry.get_upscale_component()
            gpu.mark_unevictable(comp.fingerprint)
        elif action["type"] == "bgremove":
            comp = app_state.registry.get_bgremove_component()
            gpu.mark_unevictable(comp.fingerprint)
        elif action["type"] == "sdxl":
            model_name = action["model"]
            model_dir = app_state.sdxl_models[model_name]
            registry_comps = app_state.registry.get_sdxl_components(model_dir)
            from main import _CATEGORY_TO_REGISTRY_INDEX
            for category in action["categories"]:
                idx = _CATEGORY_TO_REGISTRY_INDEX.get(category)
                if idx is None:
                    continue
                comp = registry_comps[idx]
                gpu.mark_unevictable(comp.fingerprint)
                log.debug(f"  GPU worker [{gpu.uuid}]: Marked {category} as unevictable")


# ── Utilities ─────────────────────────────────────────────────────

def _get_model_name_from_job(job) -> str | None:
    """Extract clean model name from job's sdxl_input.model_dir."""
    inp = job.sdxl_input
    if not inp or not inp.model_dir:
        return None
    name = os.path.basename(inp.model_dir)
    for ext in (".safetensors", ".ckpt"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


def _is_cuda_fatal_local(ex: Exception) -> bool:
    """Check if a CUDA error indicates permanent context corruption."""
    import torch
    if isinstance(ex, torch.cuda.OutOfMemoryError):
        return False
    msg = str(ex).lower()
    if "out of memory" in msg or "cudaerrormemoryallocation" in msg:
        return False
    patterns = [
        "illegal memory access",
        "unspecified launch failure",
        "cuda error: an illegal instruction was encountered",
        "device-side assert",
        "unable to find an engine",
    ]
    return any(p in msg for p in patterns)
