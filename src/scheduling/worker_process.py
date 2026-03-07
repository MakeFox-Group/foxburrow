"""GPU worker subprocess — owns all CUDA state for a single GPU.

Launched by the main process with CUDA_VISIBLE_DEVICES set so this
process sees exactly one GPU as cuda:0.  Each job runs its entire
pipeline (text_encode → denoise → vae_decode, etc.) sequentially
on this GPU.  No concurrent stage execution — one job at a time.

A background ModelPreloader thread loads the next stage's model
weights to CPU RAM while the current stage executes on GPU.
"""

from __future__ import annotations

import contextlib
import multiprocessing as mp
import os
import threading
import time as _time
from datetime import datetime

# NOTE: torch is imported AFTER CUDA_VISIBLE_DEVICES is set in gpu_worker_main().
# Do not import torch at module level.

from scheduling.worker_protocol import (
    DrainCmd,
    DrainComplete,
    ExecuteJobCmd,
    GetStatusCmd,
    JobComplete,
    ClipCacheEntry,
    LogMessage,
    OnloadCmd,
    OnloadComplete,
    ProcessError,
    ProgressUpdate,
    ReleaseDrainCmd,
    ShutdownCmd,
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


# ── Model preloader ──────────────────────────────────────────────

class ModelPreloader:
    """Loads model weights to CPU RAM in a background thread.

    GIL safety: safetensors file I/O releases GIL.  Python model
    construction holds GIL briefly but progresses in gaps between
    SDPA attention calls on the main thread.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._request_event = threading.Event()
        self._shutdown_event = threading.Event()
        self._request: tuple[str, str, str] | None = None  # (category, model_dir, fingerprint)
        self._result: tuple[str, object] | None = None  # (fingerprint, model)
        self._thread = threading.Thread(target=self._run, daemon=True, name="model-preloader")
        self._thread.start()

    def _run(self) -> None:
        import log
        while not self._shutdown_event.is_set():
            self._request_event.wait(timeout=5.0)
            self._request_event.clear()

            if self._shutdown_event.is_set():
                break

            with self._lock:
                req = self._request
                self._request = None

            if req is None:
                continue

            category, model_dir, fingerprint = req
            try:
                import torch
                from handlers.sdxl import load_component
                model = load_component(category, model_dir, torch.device("cpu"))
                with self._lock:
                    # Only store if no new request superseded this one
                    if self._request is None:
                        self._result = (fingerprint, model)
                log.debug(f"  Preloader: Preloaded {category} to CPU RAM")
            except Exception as ex:
                log.debug(f"  Preloader: Failed to preload {category}: {ex}")

    def request(self, category: str, model_dir: str, fingerprint: str) -> None:
        """Request preloading. Non-blocking. Supersedes previous request."""
        with self._lock:
            self._request = (category, model_dir, fingerprint)
            self._result = None
        self._request_event.set()

    def get(self, fingerprint: str) -> object | None:
        """Get preloaded model if ready and matches fingerprint."""
        with self._lock:
            if self._result is not None and self._result[0] == fingerprint:
                model = self._result[1]
                self._result = None
                return model
            return None

    def clear(self) -> None:
        """Discard any preloaded model (e.g., job cancelled)."""
        with self._lock:
            self._result = None
            self._request = None

    def shutdown(self) -> None:
        """Stop background thread."""
        self._shutdown_event.set()
        self._request_event.set()
        if self._thread:
            self._thread.join(timeout=5)


# ── Worker main ──────────────────────────────────────────────────

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

    import torch
    import torch.nn as nn

    # Enable SDP attention backends
    torch.backends.cuda.enable_math_sdp(True)
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch._inductor.config.conv_1x1_as_mm = True

    import log

    def _log_to_queue(message: str, level_value: str) -> None:
        try:
            result_queue.put(LogMessage(message=message, level=level_value))
        except Exception:
            pass  # Queue broken — nothing we can do

    log.set_remote_callback(_log_to_queue)
    log.info(f"GPU worker process started: GPU[{gpu_index}] {gpu_name} ({gpu_uuid})")

    device = torch.device("cuda:0")

    # Cap VRAM at 98%
    try:
        torch.cuda.set_per_process_memory_fraction(0.98, 0)
    except Exception as e:
        log.warning(f"  Could not set VRAM cap: {e}")

    # Patch accelerate for thread safety
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

    class _NvmlInfo:
        def __init__(self, handle, total_memory):
            self.handle = handle
            self.total_memory = total_memory

    nvml_info = _NvmlInfo(nvml_handle, gpu_total_memory)
    cpu_cache_gb = server_config_dict.get("cpu_cache_gb", 64.0)
    cpu_cache_bytes = int(cpu_cache_gb * 1024 * 1024 * 1024)
    gpu = GpuInstance(gpu_config, nvml_info, torch_device_id=0,
                      cpu_cache_bytes=cpu_cache_bytes)

    # Initialize app_state in this process
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
    _trt_cfg.dynamic_only = server_config_dict.get("trt_dynamic_only", True)
    app_state.config = FoxBurrowConfig(
        server=_server_cfg, gpus=[gpu_config], tensorrt=_trt_cfg)

    # Set handler model paths
    _up_path = server_config_dict.get("upscale_model_path")
    if _up_path:
        from handlers.upscale import set_model_path as _set_upscale_path
        _set_upscale_path(_up_path)
        app_state.registry.register_upscale_model(_up_path)

    _bgr_path = server_config_dict.get("bgremove_model_path")
    if _bgr_path:
        from handlers.bgremove import set_model_path as _set_bgremove_path
        _set_bgremove_path(_bgr_path)
        _bgr_file = (os.path.join(_bgr_path, "model.safetensors")
                     if os.path.isdir(_bgr_path) else _bgr_path)
        if os.path.isfile(_bgr_file):
            app_state.registry.register_bgremove_model(_bgr_file)

    _tag_path = server_config_dict.get("tagger_model_path")
    if _tag_path:
        from handlers.tagger import set_model_path as _set_tagger_path
        _set_tagger_path(_tag_path)

    # Initialize profiling tracer
    from profiling.tracer import register_tracer
    tracer = register_tracer(gpu_uuid, gpu_model_name, gpu_name)

    # Send ready signal
    result_queue.put(WorkerReady(gpu_model_name=gpu_model_name, arch_key=arch_key))
    result_queue.put(_build_status_snapshot(gpu, gpu_model_name, arch_key))

    log.info(f"  GPU worker ready: cuda:0 = {gpu_name} (arch={arch_key})")

    # Create event loop for WebSocket progress broadcasts
    import asyncio
    _worker_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_worker_loop)

    # Set up progress callback for handler → IPC progress forwarding
    from handlers.sdxl import set_progress_callback as _set_progress_cb

    def _send_progress(job):
        try:
            result_queue.put(ProgressUpdate(
                job_id=job.job_id,
                denoise_step=job.denoise_step,
                denoise_total_steps=job.denoise_total_steps,
                stage_step=getattr(job, "stage_step", 0),
                stage_total_steps=getattr(job, "stage_total_steps", 0),
                stage_index=getattr(job, "current_stage_index", 0),
            ))
        except Exception:
            pass

    _set_progress_cb(_send_progress)

    # Create model preloader (background CPU loading thread)
    preloader = ModelPreloader()

    # ── Command loop (single-threaded, one job at a time) ─────────
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
            if isinstance(cmd, ExecuteJobCmd):
                # Execute entire job pipeline on main thread (single-threaded)
                try:
                    job_result = _execute_job(
                        gpu, cmd, gpu_model_name, tracer, _worker_loop, preloader)
                    result_queue.put(job_result)
                except Exception as ex:
                    fatal = _is_cuda_fatal_local(ex)
                    log.log_exception(ex, f"GPU worker [{gpu_uuid}]: Job failed (unhandled)")
                    if fatal:
                        result_queue.put(ProcessError(
                            error=f"{type(ex).__name__}: {ex}",
                            fatal=True,
                        ))
                    else:
                        result_queue.put(JobComplete(
                            job_id=cmd.job_id,
                            success=False,
                            error=f"{type(ex).__name__}: {ex}",
                        ))

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

    # Shutdown
    preloader.shutdown()
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
        session_group=None,
        vram_stats=vram_stats,
        loaded_models_vram=gpu.get_loaded_models_vram(),
        evictable_vram=gpu.get_evictable_vram(),
        trt_freeable_vram=gpu._get_freeable_trt_shared_memory_vram(),
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


# ── Job execution (entire pipeline on one GPU) ───────────────────

def _execute_job(
    gpu, cmd: ExecuteJobCmd, gpu_model_name: str, tracer, loop, preloader: ModelPreloader,
) -> JobComplete:
    """Execute an entire job pipeline sequentially on one GPU."""
    import log
    import torch
    from gpu import torch_ext
    from scheduling.job import (
        InferenceJob, JobType, StageType, WorkStage,
    )
    from profiling.tracer import set_current_tracer

    device = torch.device("cuda:0")

    # Reconstruct a minimal InferenceJob for handler compatibility
    job = _reconstruct_job(cmd, loop)

    pipeline = cmd.pipeline
    total_load_time = 0.0
    total_gpu_time = 0.0
    stage_times: list[dict] = []
    output_image = None
    oom = False
    fatal = False
    error = None

    # Mark GPU busy
    gpu.acquire()
    preloader.clear()

    # One GPU = one checkpoint.  Evict any models from a different
    # checkpoint so the current model's components all fit.
    if cmd.sdxl_input and cmd.sdxl_input.model_dir:
        source_name = os.path.basename(cmd.sdxl_input.model_dir)
        for ext in (".safetensors", ".ckpt"):
            if source_name.endswith(ext):
                source_name = source_name[:-len(ext)]
                break
        gpu.evict_other_sources(source_name)

    try:
        for stage_idx, stage in enumerate(pipeline):
            if stage.is_cpu_only:
                # CPU work already done in main process — skip
                continue

            # Update stage index so progress reports reflect current stage
            job.current_stage_index = stage_idx

            # ── Model loading (serialized) ────────────────────────
            load_start = _time.monotonic()

            with gpu.model_load_lock:
                with torch.cuda.device(device):
                    torch.cuda.empty_cache()

                vram_before = torch.cuda.memory_allocated(device)
                _ensure_models_for_stage(gpu, stage, job, gpu_model_name, preloader)

                # Ensure free VRAM for working memory.
                # VAE decode/encode stages are excluded — they use VRAM-aware
                # adaptive tiling that picks the largest tile size that fits in
                # available VRAM.  Pre-evicting models for VAE working memory
                # forces the UNet out on 12GB GPUs (untiled 1024x1024 needs ~4GB),
                # causing it to be reloaded for every subsequent job.
                from scheduling.worker import _get_min_free_vram
                _vae_stages = (StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE,
                               StageType.GPU_HIRES_TRANSFORM)
                if stage.type not in _vae_stages:
                    active_fps = {c.fingerprint for c in stage.required_components}
                    active_fps.update(getattr(stage, '_trt_active_fps', set()))
                    min_free = _get_min_free_vram(stage.type, job, gpu_model_name)
                    if min_free > 0:
                        if not gpu.ensure_free_vram(min_free, protect=active_fps):
                            raise torch.cuda.OutOfMemoryError(
                                f"Cannot free {min_free // (1024*1024)}MB working memory "
                                f"for {stage.type.value} — predicted OOM, re-routing")

                vram_delta = torch.cuda.memory_allocated(device) - vram_before

            load_duration = _time.monotonic() - load_start
            total_load_time += load_duration

            # Record model load event
            if load_duration > 0.01:
                model_name = _get_model_name_from_job(job)
                tracer.model_load(
                    job.job_id, model_name, stage.type.value,
                    load_duration, vram_delta)

            # ── Preload next stage's model(s) to CPU ──────────────
            next_gpu_stage = _find_next_gpu_stage(pipeline, stage_idx)
            if next_gpu_stage is not None:
                _request_preload(preloader, next_gpu_stage, job, gpu)

            # ── Execute stage ─────────────────────────────────────
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

            working_mem = 0
            tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_ACTIVATIONS)
                       if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())

            try:
                with tag_ctx:
                    stage_output = _dispatch_stage(job, stage, gpu)

                # Handle stage output
                if stage_output is not None:
                    # Stages that produce images (vae_decode, upscale, bgremove)
                    is_last_gpu = _find_next_gpu_stage(pipeline, stage_idx) is None
                    if is_last_gpu:
                        output_image = stage_output
                    else:
                        # Intermediate image — store for next stage
                        job.input_image = stage_output
                        # Hires resize if needed
                        if job.hires_input:
                            tw = job.hires_input.hires_width
                            th = job.hires_input.hires_height
                            if (tw > 0 and th > 0
                                    and (job.input_image.width != tw
                                         or job.input_image.height != th)):
                                from PIL import Image as _PILImage
                                log.debug(f"  Resizing intermediate image "
                                          f"{job.input_image.width}x{job.input_image.height}"
                                          f" → {tw}x{th} (hires target)")
                                job.input_image = job.input_image.resize(
                                    (tw, th), _PILImage.LANCZOS)

                gpu.record_success()

            except torch.cuda.OutOfMemoryError as ex:
                oom = True
                error = str(ex)
                torch.cuda.empty_cache()
                break

            except Exception as ex:
                if _is_cuda_fatal_local(ex):
                    fatal = True
                    gpu.mark_failed(f"CUDA context corrupted ({type(ex).__name__})")
                error = str(ex)
                import traceback as _tb
                log.error(f"  GPU worker: Stage {stage.type.value} failed: {_tb.format_exc()}")
                break

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

                # Release active fingerprints for this stage
                release_fps: set[str] = set()
                release_fps.update(c.fingerprint for c in stage.required_components)
                release_fps.update(getattr(stage, '_trt_active_fps', set()))
                if release_fps:
                    gpu.remove_active_fingerprints(release_fps)
                te_protect = getattr(stage, '_te_protect_fps', set())
                if te_protect:
                    gpu.remove_active_fingerprints(te_protect)

            stage_duration = _time.monotonic() - stage_start
            total_gpu_time += stage_duration

            # Update BPP measurement
            if error is None:
                from scheduling.worker import _get_stage_pixels, _update_working_memory
                stage_pixels = _get_stage_pixels(stage.type, job)
                _update_working_memory(stage.type, working_mem, stage_pixels)

            # Record per-stage timing
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
                "working_mem_bytes": working_mem,
                "vram_allocated": vram_alloc,
                "vram_reserved": vram_reserved,
            }
            stage_times.append(_stage_timing)

            # Serialize CLIP tensors for caching after text_encode stage
            if (stage.type == StageType.GPU_TEXT_ENCODE
                    and hasattr(job, '_clip_cache_tensors')
                    and job._clip_cache_tensors is not None):
                try:
                    p_h1, n_h1, p_h2, n_h2, p_pooled, n_pooled = job._clip_cache_tensors

                    def _ser(tensor, enc_type, polarity):
                        t = tensor.squeeze(0).contiguous().half().cpu()
                        return ClipCacheEntry(
                            encoder_type=enc_type, polarity=polarity,
                            data=t.numpy().tobytes(), dtype="fp16",
                            dim0=t.shape[0],
                            dim1=t.shape[1] if t.dim() >= 2 else None,
                        )

                    job._clip_cache_result = {
                        "job_id": cmd.job_id,
                        "entries": [
                            _ser(p_h1, "clip_l", "positive"),
                            _ser(n_h1, "clip_l", "negative"),
                            _ser(p_h2, "clip_g", "positive"),
                            _ser(n_h2, "clip_g", "negative"),
                            _ser(p_pooled, "clip_g_pooled", "positive"),
                            _ser(n_pooled, "clip_g_pooled", "negative"),
                        ],
                    }
                except Exception as ex:
                    log.warning(f"  CLIP cache serialization failed: {ex}")
                    job._clip_cache_result = None
                finally:
                    job._clip_cache_tensors = None  # Release GPU tensor references

            # Serialize latent tensor for caching after denoise stage
            if (stage.type == StageType.GPU_DENOISE
                    and job.latents is not None
                    and error is None):
                try:
                    t = job.latents.contiguous().half().cpu()
                    job._latent_cache_result = {
                        "job_id": cmd.job_id,
                        "data": t.numpy().tobytes(),
                        "dtype": "fp16",
                        "shape": list(t.shape),
                    }
                except Exception as ex:
                    log.warning(f"  Latent cache serialization failed: {ex}")
                    job._latent_cache_result = None

            # Record stage completion for profiling
            if error is None:
                _stage_w = job.sdxl_input.width if job.sdxl_input else 0
                _stage_h = job.sdxl_input.height if job.sdxl_input else 0
                _stage_steps = job.denoise_step if job.denoise_step > 0 else None
                tracer.stage_complete(
                    job.job_id, model_name, stage.type.value,
                    _stage_w, _stage_h, _stage_steps, stage_duration)

    except torch.cuda.OutOfMemoryError as ex:
        # OOM during model loading (before stage execution)
        oom = True
        error = str(ex)
        torch.cuda.empty_cache()

    except Exception as ex:
        if _is_cuda_fatal_local(ex):
            fatal = True
            gpu.mark_failed(f"CUDA context corrupted ({type(ex).__name__})")
        error = str(ex)
        import traceback as _tb
        log.error(f"  GPU worker: Job {cmd.job_id} failed: {_tb.format_exc()}")

    finally:
        # Clear all active fingerprints and release GPU
        gpu.clear_active_fingerprints()
        gpu.release()
        preloader.clear()
        # Release GPU tensor references from CLIP capture if serialization was
        # skipped due to break-on-error (OOM, CUDA fatal, etc.)
        if hasattr(job, '_clip_cache_tensors'):
            job._clip_cache_tensors = None

    # ── Build result ──────────────────────────────────────────────
    output_latents = None
    if job.latents is not None and error is None and job.type.value in (
            "SdxlGenerateLatents", "SdxlEncodeLatents", "SdxlHiresLatents"):
        output_latents = job.latents.cpu()

    return JobComplete(
        job_id=cmd.job_id,
        success=error is None,
        error=error,
        oom=oom,
        fatal=fatal,
        output_image=output_image if error is None else None,
        output_latents=output_latents,
        gpu_time_s=total_gpu_time,
        model_load_time_s=total_load_time,
        stage_times=stage_times,
        clip_cache=getattr(job, '_clip_cache_result', None),
        latent_cache=getattr(job, '_latent_cache_result', None),
    )


def _reconstruct_job(cmd: ExecuteJobCmd, loop) -> object:
    """Reconstruct a minimal InferenceJob from an ExecuteJobCmd."""
    import asyncio
    from scheduling.job import InferenceJob, JobType

    job = InferenceJob.__new__(InferenceJob)
    job.job_id = cmd.job_id
    job.type = JobType(cmd.job_type_value)
    job.sdxl_input = cmd.sdxl_input
    job.hires_input = cmd.hires_input
    job.input_image = cmd.input_image
    job.is_hires_pass = cmd.is_hires_pass
    job.oom_retries = cmd.oom_retries
    job.current_stage_index = 0
    job.pipeline = cmd.pipeline
    job.tokenize_result = cmd.tokenize_result
    job.latents = cmd.input_latents
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
    job.encode_result = None
    job.regional_encode_result = None
    job.regional_info = None
    job.regional_tokenize_results = cmd.regional_tokenize_results
    job.regional_base_tokenize = cmd.regional_base_tokenize
    job.regional_shared_neg_tokenize = cmd.regional_shared_neg_tokenize

    # Set original dimensions for final resize
    if cmd.orig_width is not None:
        job.orig_width = cmd.orig_width
    if cmd.orig_height is not None:
        job.orig_height = cmd.orig_height

    # Reconstruct regional info
    if cmd.regional_info_data:
        from utils.regional import RegionalPromptResult
        job.regional_info = RegionalPromptResult.from_dict(cmd.regional_info_data)

    # Event loop for WebSocket progress broadcasts
    job._loop = loop
    job.completion = loop.create_future()

    return job


def _find_next_gpu_stage(pipeline: list, current_idx: int):
    """Find the next GPU stage in the pipeline after current_idx."""
    for i in range(current_idx + 1, len(pipeline)):
        if not pipeline[i].is_cpu_only:
            return pipeline[i]
    return None


def _request_preload(preloader: ModelPreloader, next_stage, job, gpu) -> None:
    """Request preloading the first missing component of the next stage."""
    import log

    model_dir = job.sdxl_input.model_dir if job.sdxl_input else None
    if model_dir is None:
        return

    # Only preload SDXL components (TE, UNet, VAE)
    sdxl_categories = {"sdxl_te1", "sdxl_te2", "sdxl_unet", "sdxl_vae", "sdxl_vae_enc"}

    for component in next_stage.required_components:
        if component.category not in sdxl_categories:
            continue
        if gpu.is_component_loaded(component.fingerprint):
            continue
        # Request preload of the first (largest) missing component
        preloader.request(component.category, model_dir, component.fingerprint)
        log.debug(f"  Preloader: Requested preload of {component.category}")
        break


# ── Model loading ────────────────────────────────────────────────

def _ensure_models_for_stage(gpu, stage, job, gpu_model_name: str,
                             preloader: ModelPreloader | None = None) -> None:
    """Load model components required for the given stage."""
    import torch
    from gpu import torch_ext
    from scheduling.job import StageType

    model_dir = job.sdxl_input.model_dir if job.sdxl_input else None

    if stage.type in (StageType.GPU_TEXT_ENCODE, StageType.GPU_DENOISE,
                      StageType.GPU_VAE_DECODE, StageType.GPU_VAE_ENCODE):
        _load_sdxl_components(gpu, stage, model_dir, job, gpu_model_name, preloader)
    elif stage.type == StageType.GPU_UPSCALE:
        _load_upscale_model(gpu)
    elif stage.type == StageType.GPU_BGREMOVE:
        _load_bgremove_model(gpu)

    # Add active fingerprints for eviction protection
    active_fps = {c.fingerprint for c in stage.required_components}
    active_fps.update(getattr(stage, '_trt_active_fps', set()))
    gpu.add_active_fingerprints(active_fps)


def _load_sdxl_components(gpu, stage, model_dir, job, gpu_model_name: str,
                          preloader: ModelPreloader | None = None) -> None:
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

    # Build protect set for ensure_free_vram
    all_required_fps = {c.fingerprint for c in stage.required_components}

    # Protect TE TRT engines from eviction during non-TE stages
    te_protect_fps: set[str] = set()
    if stage.type != StageType.GPU_TEXT_ENCODE:
        # Look for TE fingerprints from the pipeline
        te_fps: dict[str, str] = {}
        for ps in job.pipeline:
            if ps.type == StageType.GPU_TEXT_ENCODE:
                te_fps = {c.category: c.fingerprint for c in ps.required_components}
                break
        for te_key, comp_name in (("sdxl_te1", "te1"), ("sdxl_te2", "te2")):
            base = te_fps.get(te_key)
            if base:
                te_fp = f"{base}:{comp_name}_trt:default"
                if gpu.is_component_loaded(te_fp):
                    te_protect_fps.add(te_fp)

    if te_protect_fps:
        gpu.add_active_fingerprints(te_protect_fps)
    stage._te_protect_fps = te_protect_fps

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
            protect_fps.update(new_trt_fps)
            continue

        # Check CPU cache (evicted models kept on CPU RAM)
        cpu_cached = gpu.get_cpu_cached_model(component.fingerprint)

        # Check preloader for CPU-cached model (background thread)
        preloaded_model = None
        if cpu_cached is None and preloader is not None:
            preloaded_model = preloader.get(component.fingerprint)

        # Ensure VRAM before loading
        if not gpu.ensure_free_vram(component.estimated_vram_bytes, protect=protect_fps):
            raise torch.cuda.OutOfMemoryError(
                f"Cannot free {component.estimated_vram_bytes // (1024*1024)}MB "
                f"for {component.category} — predicted OOM, re-routing")

        tag_ctx = (torch.cuda.tag_allocations(torch_ext.ALLOC_TAG_MODEL_WEIGHTS)
                   if torch_ext.HAS_ALLOC_TAGS else contextlib.nullcontext())

        if cpu_cached is not None:
            # Fastest path: CPU cache → GPU transfer
            before = torch.cuda.memory_allocated(device)
            with tag_ctx:
                cpu_cached.model.to(device)
            after = torch.cuda.memory_allocated(device)
            actual_vram = after - before
            model = cpu_cached.model

            log.debug(f"  Loaded {component.category} from CPU cache: "
                      f"{actual_vram // (1024*1024)}MB actual")
        elif preloaded_model is not None:
            # Fast path: preloader CPU → GPU transfer
            before = torch.cuda.memory_allocated(device)
            with tag_ctx:
                preloaded_model.to(device)
            after = torch.cuda.memory_allocated(device)
            actual_vram = after - before
            model = preloaded_model

            log.debug(f"  Loaded {component.category} from preloader: "
                      f"{actual_vram // (1024*1024)}MB actual")
        else:
            # Slow path: load from disk to GPU
            from handlers.sdxl import load_component
            before = torch.cuda.memory_allocated(device)
            with tag_ctx:
                model = load_component(component.category, model_dir, device)
            after = torch.cuda.memory_allocated(device)
            actual_vram = after - before

            log.debug(f"  Loaded {component.category} from disk: {actual_vram // (1024*1024)}MB actual "
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
    elif component.category == "sdxl_vae":
        trt_component = "vae"
    elif component.category == "sdxl_vae_enc":
        trt_component = "vae_enc"
    elif component.category == "sdxl_te1":
        trt_component = "te1"
    elif component.category == "sdxl_te2":
        trt_component = "te2"
    else:
        return False

    from state import app_state
    dynamic_only = app_state.config.tensorrt.dynamic_only

    if trt_component in ("te1", "te2"):
        try:
            from trt.builder import has_trt_coverage, get_arch_key
            cache_dir = app_state.config.server.tensorrt_cache
            arch_key = get_arch_key(0)
            return has_trt_coverage(
                cache_dir, component.fingerprint, trt_component, arch_key, 0, 0,
                dynamic_only=dynamic_only)
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
        cache_dir = app_state.config.server.tensorrt_cache
        arch_key = get_arch_key(0)
        return has_trt_coverage(
            cache_dir, component.fingerprint, trt_component, arch_key, width, height,
            dynamic_only=dynamic_only)
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
        elif component.category == "sdxl_vae":
            _get_trt_vae_runner(gpu, job, width, height)
        elif component.category == "sdxl_vae_enc":
            from handlers.sdxl import _get_trt_vae_enc_runner
            _get_trt_vae_enc_runner(gpu, job, width, height)
        elif component.category == "sdxl_te1":
            _get_trt_te1_runner(gpu, job)
        elif component.category == "sdxl_te2":
            _get_trt_te2_runner(gpu, job)
    except Exception as ex:
        log.debug(f"  TRT pre-load for {component.category} failed: {ex}")
        return set()

    _fp_keys = {"sdxl_unet": "sdxl_unet", "sdxl_vae": "sdxl_vae",
                "sdxl_vae_enc": "sdxl_vae_enc", "sdxl_te1": "sdxl_te1",
                "sdxl_te2": "sdxl_te2"}
    _trt_comps = {"sdxl_unet": "unet", "sdxl_vae": "vae",
                  "sdxl_vae_enc": "vae_enc", "sdxl_te1": "te1",
                  "sdxl_te2": "te2"}
    fp_key = _fp_keys.get(component.category)
    trt_comp = _trt_comps.get(component.category)
    if not fp_key or not trt_comp:
        return set()

    # Get the base fingerprint for this component from the stage
    base_fp = None
    for c in stage.required_components:
        if c.category == fp_key:
            base_fp = c.fingerprint
            break
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
        if not gpu.ensure_free_vram(comp.estimated_vram_bytes, protect={comp.fingerprint}):
            raise torch.cuda.OutOfMemoryError(
                f"Cannot free {comp.estimated_vram_bytes // (1024*1024)}MB for upscale model")
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
        if not gpu.ensure_free_vram(comp.estimated_vram_bytes, protect={comp.fingerprint}):
            raise torch.cuda.OutOfMemoryError(
                f"Cannot free {comp.estimated_vram_bytes // (1024*1024)}MB for bgremove model")
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

    # Move intermediate tensors to cuda:0 (no-op if already on GPU)
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
            progress_cb=_progress,
            dynamic_only=cmd.dynamic_only)
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

    if cmd.sdxl_models:
        app_state.sdxl_models.update(cmd.sdxl_models)

    if cmd.lora_index is not None:
        app_state.lora_index = cmd.lora_index
    if cmd.loras_dir is not None:
        app_state.loras_dir = cmd.loras_dir

    if cmd.sdxl_models:
        for model_name, model_dir in cmd.sdxl_models.items():
            try:
                app_state.registry.register_sdxl_checkpoint(model_dir)
            except Exception:
                pass

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
                _handle_sdxl_onload(gpu, entry, app_state)

        except Exception as ex:
            log.log_exception(ex, f"GPU worker [{gpu.uuid}]: onload={entry} failed")

    for entry in cmd.unevictable_entries:
        try:
            _handle_unevictable(gpu, entry, app_state)
        except Exception as ex:
            log.log_exception(ex, f"GPU worker [{gpu.uuid}]: unevictable={entry} failed")


def _handle_sdxl_onload(gpu, entry: str, app_state) -> None:
    """Handle SDXL model onload in worker process."""
    import torch
    import log

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
