"""GPU workspace profiler: measure VRAM working memory at multiple resolutions.

Runs instrumented forward passes at first model load to capture actual peak
working memory (activations + cuDNN workspace) for a grid of common
resolutions.  Additionally queries nvidia-cudnn-frontend for per-conv
workspace sizes as a diagnostic breakdown.

Results are cached per GPU model x component type as JSON files under
data/profiling/{GPU_MODEL_NAME}/.  After the first load on a given GPU model,
all future sessions load from the cache instantly (~0ms).

Integration:
    Called from worker.py after model loading.
    Queried by worker._get_min_free_vram() for VRAM budget decisions.
"""

from __future__ import annotations

import json
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn as nn

import log

# ── CUDA fatal detection ─────────────────────────────────────────
# Mirrors _CUDA_FATAL_PATTERNS from worker.py — these errors mean
# the CUDA context is permanently corrupted and must be re-raised
# so the worker's error handling can properly mark GPUs as failed.

_CUDA_FATAL_PATTERNS = [
    "illegal memory access",
    "unspecified launch failure",
    "cuda error: an illegal instruction was encountered",
    "device-side assert",
    "unable to find an engine",
]


def _is_cuda_fatal(ex: Exception) -> bool:
    """Check if a CUDA error indicates permanent context corruption."""
    if isinstance(ex, torch.cuda.OutOfMemoryError):
        return False
    msg = str(ex).lower()
    return any(p in msg for p in _CUDA_FATAL_PATTERNS)


# ── Resolution grid ───────────────────────────────────────────────
#
# Covers 512 through 2048 in common SDXL aspect ratios.
# Profiling runs at each resolution in ascending pixel-count order
# so OOM at large sizes doesn't prevent caching smaller ones.

RESOLUTION_GRID: list[tuple[int, int]] = [
    (512, 512),
    (640, 640),
    (640, 768),
    (768, 768),
    (768, 1024),
    (832, 1216),
    (896, 1152),
    (1024, 1024),
    (1024, 1536),
    (1152, 896),
    (1216, 832),
    (1280, 1280),
    (1536, 1024),
    (1536, 1536),
    (1664, 1664),
    (1920, 1088),
    (2048, 2048),
]

# Sort by ascending pixel count for OOM-safe ordering
RESOLUTION_GRID.sort(key=lambda r: r[0] * r[1])

_CACHE_DIR = Path("data/profiling")
_lock = threading.Lock()

# In-memory caches: {(component_type, gpu_model): {resolution_key: bytes}}
_working_mem_caches: dict[tuple[str, str], dict[str, int]] = {}

# Guard against concurrent profiling of the same (component, gpu_model)
_profiling_in_progress: set[tuple[str, str]] = set()
_profiling_events: dict[tuple[str, str], threading.Event] = {}


# ── GPU model name ────────────────────────────────────────────────

def get_gpu_model_name(device: torch.device) -> str:
    """Get a filesystem-safe GPU identifier for cache directory naming.

    Uses the compute capability (SM architecture) since cuDNN algorithm
    selection — which determines workspace sizes — is architecture-dependent.
    All GPUs with the same SM version produce identical profiles.
    """
    major, minor = torch.cuda.get_device_capability(device)
    return f"sm_{major}{minor}"


# ── Cache I/O ─────────────────────────────────────────────────────

def _cache_path(component_type: str, gpu_model: str) -> Path:
    return _CACHE_DIR / gpu_model / f"{component_type}.json"


def _load_cache(component_type: str, gpu_model: str) -> dict[str, int] | None:
    """Load workspace cache from disk.  Returns None if not cached."""
    path = _cache_path(component_type, gpu_model)
    if not path.exists():
        return None
    try:
        with open(path) as f:
            data = json.load(f)
        resolutions = data.get("resolutions", {})
        if not resolutions:
            return None
        return {k: int(v) for k, v in resolutions.items()}
    except Exception as ex:
        log.warning(f"  WorkspaceProfiler: Failed to load cache {path}: {ex}")
        return None


def _save_cache(
    component_type: str,
    gpu_model: str,
    resolutions: dict[str, int],
    conv_workspaces: dict[str, int] | None = None,
) -> None:
    """Save workspace cache to disk."""
    path = _cache_path(component_type, gpu_model)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = {
        "component_type": component_type,
        "gpu_model": gpu_model,
        "profiled_at": datetime.now(timezone.utc).isoformat(),
        "torch_version": torch.__version__,
        "cudnn_version": str(torch.backends.cudnn.version()),
        "resolutions": resolutions,
    }
    if conv_workspaces:
        data["conv_workspaces"] = conv_workspaces

    # Atomic write: write to .tmp then rename to prevent corruption
    # if two threads or a crash interrupts the write.
    tmp_path = path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(data, f, indent=2)
    os.replace(tmp_path, path)
    log.debug(f"  WorkspaceProfiler: Saved cache → {path}")


# ── Dummy input creation ─────────────────────────────────────────

def _make_unet_input(
    w: int, h: int, device: torch.device, dtype: torch.dtype = torch.float16,
) -> dict:
    lat_h, lat_w = h // 8, w // 8
    return dict(
        sample=torch.randn(1, 4, lat_h, lat_w, device=device, dtype=dtype),
        timestep=torch.tensor([999.0], device=device),
        encoder_hidden_states=torch.randn(1, 77, 2048, device=device, dtype=dtype),
        added_cond_kwargs={
            "text_embeds": torch.randn(1, 1280, device=device, dtype=dtype),
            "time_ids": torch.zeros(1, 6, device=device, dtype=dtype),
        },
    )


def _run_model_forward(
    component_type: str, model: nn.Module,
    w: int, h: int, device: torch.device,
) -> None:
    """Run a single no_grad forward pass with dummy inputs."""
    # Infer dtype from model parameters — VAE runs in float32 while others
    # use float16.  Hardcoding float16 causes "Input type (c10::Half) and
    # bias type (float) should be the same" errors on float32 models.
    try:
        dtype = next(model.parameters()).dtype
    except StopIteration:
        dtype = torch.float16

    if component_type == "sdxl_unet":
        inputs = _make_unet_input(w, h, device, dtype)
        model(**inputs)

    elif component_type in ("sdxl_vae", "sdxl_vae_decoder"):
        lat_h, lat_w = h // 8, w // 8
        latent = torch.randn(1, 4, lat_h, lat_w, device=device, dtype=dtype)
        model.decode(latent)

    elif component_type in ("sdxl_vae_enc", "sdxl_vae_encoder"):
        pixel = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        model.encode(pixel)

    elif component_type == "upscale":
        # RRDBNet with scale=2 uses pixel_unshuffle: input [1,3,H,W]
        pixel = torch.randn(1, 3, h, w, device=device, dtype=dtype)
        model(pixel)

    elif component_type == "bgremove":
        # RMBG-2.0: always 1024x1024 input
        pixel = torch.randn(1, 3, 1024, 1024, device=device, dtype=dtype)
        model(pixel)

    elif component_type == "tagger":
        # SigLIP ViT: always 384x384 input
        pixel = torch.randn(1, 3, 384, 384, device=device, dtype=dtype)
        model(pixel)

    elif component_type in ("sdxl_te1", "sdxl_te2"):
        # Text encoders: fixed 77-token input
        ids = torch.randint(0, 49408, (1, 77), device=device)
        model(ids)

    else:
        raise ValueError(f"Unknown component type for profiling: {component_type}")


# ── cuDNN-frontend workspace query ───────────────────────────────

def _query_conv_workspace(
    input_shape: tuple[int, ...],
    weight_shape: tuple[int, ...],
    stride: tuple[int, ...],
    padding: tuple[int, ...],
    dilation: tuple[int, ...],
    groups: int = 1,
) -> int | None:
    """Query cuDNN-frontend for the workspace size of a single conv2d forward.

    Returns bytes, or None if the query fails.
    """
    try:
        import cudnn
    except ImportError:
        return None

    try:
        # Grouped convolutions need different tensor layout in cuDNN-frontend.
        # SDXL is almost entirely groups=1; skip grouped convs rather than
        # risk incorrect workspace estimates.  The primary VRAM prediction
        # (peak - baseline) is unaffected.
        if groups > 1:
            return None

        N, C_in, H_in, W_in = input_shape
        K, C_per_group, R, S = weight_shape

        graph = cudnn.pygraph(
            intermediate_data_type=cudnn.data_type.FLOAT,
            compute_data_type=cudnn.data_type.FLOAT,
        )

        X = graph.tensor(
            name="X",
            dim=[N, C_in, H_in, W_in],
            stride=[C_in * H_in * W_in, H_in * W_in, W_in, 1],
            data_type=cudnn.data_type.HALF,
        )
        W = graph.tensor(
            name="W",
            dim=[K, C_per_group, R, S],
            stride=[C_per_group * R * S, R * S, S, 1],
            data_type=cudnn.data_type.HALF,
        )

        Y = graph.conv_fprop(
            image=X, weight=W,
            padding=list(padding), stride=list(stride), dilation=list(dilation),
        )
        Y.set_output(True).set_data_type(cudnn.data_type.HALF)

        graph.validate()
        graph.build_operation_graph()
        graph.create_execution_plans([cudnn.heur_mode.A])
        graph.check_support()
        graph.build_plans()

        return graph.get_workspace_size()
    except Exception:
        return None


# ── Single-resolution profiling ──────────────────────────────────

def _profile_at_resolution(
    component_type: str, model: nn.Module,
    w: int, h: int, device: torch.device,
) -> tuple[int, int]:
    """Profile working memory and conv workspace at a single resolution.

    Returns (total_working_memory_bytes, max_conv_workspace_bytes).
    """
    # Register hooks on Conv2d layers to capture shapes for cuDNN queries
    conv_infos: list[dict] = []
    hook_handles: list[torch.utils.hooks.RemovableHook] = []

    def _make_hook(name: str):
        def hook(module, inp, out):
            conv_infos.append({
                "name": name,
                "input_shape": tuple(inp[0].shape),
                "weight_shape": tuple(module.weight.shape),
                "stride": tuple(module.stride),
                "padding": tuple(module.padding),
                "dilation": tuple(module.dilation),
                "groups": module.groups,
            })
        return hook

    for name, mod in model.named_modules():
        if isinstance(mod, nn.Conv2d):
            hook_handles.append(mod.register_forward_hook(_make_hook(name)))

    # Measure total working memory via peak tracking.
    # Hooks MUST be removed even if the forward pass OOMs, otherwise they
    # persist on the model's Conv2d modules and fire on every subsequent
    # inference call — leaking CPU memory into conv_infos unboundedly.
    with torch.cuda.device(device):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    baseline = torch.cuda.memory_allocated(device)

    try:
        with torch.no_grad():
            _run_model_forward(component_type, model, w, h, device)

        peak = torch.cuda.max_memory_allocated(device)
        working_mem = max(0, peak - baseline)
    finally:
        # Remove hooks unconditionally — this is the critical safety net
        for handle in hook_handles:
            handle.remove()

        # Release forward pass intermediates
        with torch.cuda.device(device):
            torch.cuda.empty_cache()

    # Query cuDNN workspace for each captured conv layer
    max_workspace = 0
    for info in conv_infos:
        ws = _query_conv_workspace(
            info["input_shape"], info["weight_shape"],
            info["stride"], info["padding"], info["dilation"],
            info["groups"],
        )
        if ws is not None and ws > max_workspace:
            max_workspace = ws

    return working_mem, max_workspace


# ── Full component profiling ─────────────────────────────────────

def _get_grid_for_component(component_type: str) -> list[tuple[int, int]]:
    """Return the resolution grid appropriate for a component type."""
    if component_type in ("sdxl_te1", "sdxl_te2"):
        return [(0, 0)]  # resolution-independent
    if component_type == "tagger":
        return [(384, 384)]  # fixed input
    if component_type == "bgremove":
        return [(1024, 1024)]  # always resized to this
    # Everything else uses the full grid
    return list(RESOLUTION_GRID)


def profile_component(
    component_type: str, model: nn.Module,
    device: torch.device, gpu_model: str,
) -> dict[str, int]:
    """Profile a model component at multiple resolutions.

    Runs instrumented forward passes in ascending pixel-count order.
    Stops gracefully on OOM (smaller resolutions are still cached).

    Returns {resolution_key: working_memory_bytes}.
    """
    grid = _get_grid_for_component(component_type)

    results: dict[str, int] = {}
    conv_ws: dict[str, int] = {}

    t_start = time.monotonic()
    log.debug(f"  WorkspaceProfiler: Profiling {component_type} on {gpu_model} "
             f"({len(grid)} resolutions)...")

    for w, h in grid:
        key = f"{w}x{h}" if w > 0 else "fixed"
        try:
            working_mem, max_workspace = _profile_at_resolution(
                component_type, model, w, h, device)
            results[key] = working_mem
            conv_ws[key] = max_workspace
            log.debug(f"    {key}: working={working_mem // (1024**2)}MB, "
                     f"conv_ws={max_workspace // (1024**2)}MB")
        except torch.cuda.OutOfMemoryError:
            log.warning(f"    {key}: OOM — stopping grid (will extrapolate)")
            with torch.cuda.device(device):
                torch.cuda.empty_cache()
            break
        except Exception as ex:
            if _is_cuda_fatal(ex):
                log.warning(f"    {key}: FATAL CUDA error — aborting profiling")
                raise  # Let worker's error handler mark GPUs as failed
            log.warning(f"    {key}: FAILED — {ex}")
            with torch.cuda.device(device):
                torch.cuda.empty_cache()

    elapsed = time.monotonic() - t_start
    log.debug(f"  WorkspaceProfiler: Done — {component_type}: "
             f"{len(results)} resolutions in {elapsed:.1f}s")

    _save_cache(component_type, gpu_model, results, conv_ws)
    return results


# ── Public API ───────────────────────────────────────────────────

def ensure_profiled(
    component_type: str, model: nn.Module,
    device: torch.device, gpu_model: str | None = None,
) -> dict[str, int]:
    """Load from cache or profile on first load.

    Call this after loading a model component to GPU.
    Returns {resolution_key: working_memory_bytes}.

    Thread-safe: if two workers call this simultaneously for the same
    (component_type, gpu_model), only one actually profiles — the other
    waits for the result.
    """
    if gpu_model is None:
        gpu_model = get_gpu_model_name(device)

    cache_key = (component_type, gpu_model)

    with _lock:
        if cache_key in _working_mem_caches:
            return _working_mem_caches[cache_key]

        # Another thread is already profiling this — wait for it
        if cache_key in _profiling_in_progress:
            event = _profiling_events.get(cache_key)
            if event is not None:
                _lock.release()
                try:
                    event.wait(timeout=120)
                finally:
                    _lock.acquire()
                # Check cache again after waiting
                if cache_key in _working_mem_caches:
                    return _working_mem_caches[cache_key]
                # If still not cached, fall through to disk check / re-profile

    # Try disk cache
    cached = _load_cache(component_type, gpu_model)
    if cached is not None:
        with _lock:
            _working_mem_caches[cache_key] = cached
        log.debug(f"  WorkspaceProfiler: Loaded {component_type} cache for {gpu_model} "
                 f"({len(cached)} resolutions from disk)")
        return cached

    # Mark as profiling-in-progress
    event = threading.Event()
    with _lock:
        # Double-check: another thread may have finished between our checks
        if cache_key in _working_mem_caches:
            return _working_mem_caches[cache_key]
        _profiling_in_progress.add(cache_key)
        _profiling_events[cache_key] = event

    try:
        result = profile_component(component_type, model, device, gpu_model)
        with _lock:
            _working_mem_caches[cache_key] = result
        return result
    finally:
        with _lock:
            _profiling_in_progress.discard(cache_key)
            _profiling_events.pop(cache_key, None)
        event.set()  # Wake any waiting threads


def get_working_memory(
    component_type: str, gpu_model: str,
    width: int, height: int,
) -> int | None:
    """Look up cached working memory for a resolution.

    Returns the exact profiled value for grid resolutions.
    For non-grid resolutions, returns the value from the nearest larger
    profiled resolution (conservative — working memory grows with size).
    If the target exceeds all profiled resolutions, linearly extrapolates
    from the largest with 20% headroom.

    Returns None if the component has not been profiled.
    """
    cache_key = (component_type, gpu_model)
    with _lock:
        cache = _working_mem_caches.get(cache_key)
    if cache is None:
        return None

    # Resolution-independent components
    if "fixed" in cache:
        return cache["fixed"]

    key = f"{width}x{height}"
    if key in cache:
        return cache[key]

    target_pixels = width * height

    # Find nearest larger profiled resolution (conservative)
    best_key = None
    best_pixels = float("inf")

    # Also track the largest profiled for extrapolation fallback
    largest_key = None
    largest_pixels = 0

    for res_key in cache:
        if res_key == "fixed":
            continue
        try:
            rw, rh = res_key.split("x")
            res_pixels = int(rw) * int(rh)
        except (ValueError, IndexError):
            continue

        if res_pixels >= target_pixels and res_pixels < best_pixels:
            best_pixels = res_pixels
            best_key = res_key

        if res_pixels > largest_pixels:
            largest_pixels = res_pixels
            largest_key = res_key

    if best_key is not None:
        return cache[best_key]

    # Target is larger than anything profiled — extrapolate
    if largest_key is not None and largest_pixels > 0:
        ratio = target_pixels / largest_pixels
        return int(cache[largest_key] * ratio * 1.2)

    return None


def invalidate_cache(component_type: str, gpu_model: str) -> None:
    """Remove cached data for a component (forces re-profiling on next load)."""
    cache_key = (component_type, gpu_model)
    with _lock:
        _working_mem_caches.pop(cache_key, None)
    path = _cache_path(component_type, gpu_model)
    if path.exists():
        path.unlink()
        log.debug(f"  WorkspaceProfiler: Invalidated cache {path}")
