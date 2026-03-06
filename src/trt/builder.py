"""TensorRT engine building from ONNX models.

Builds static + dynamic engines per model × component × GPU architecture:
  - Static engines for high-traffic resolutions (exact match, max perf)
  - Dynamic engines covering resolution ranges (fallback for arbitrary sizes)

Each dynamic engine writes a JSON sidecar ({label}.json) describing its
min/opt/max resolution range.  The runtime discovers available engines
by scanning these sidecars and picks the tightest-fitting one.

Uses maximum available workspace: GPU is drained (no inference) during
builds, so nearly all VRAM is available.  _get_workspace_gb() reserves
_BUILD_HEADROOM_GB (1 GB) for the CUDA context and TRT internals.
"""

from __future__ import annotations

import gc
import json
import os
import time as _time
from typing import Callable

import torch

import log


# ── Custom TRT logger ─────────────────────────────────────────────
# TensorRT's default trt.Logger prints directly to stderr with its own
# timestamp format.  Subclassing trt.ILogger routes all TRT and ONNX
# parser messages through our logging system instead.

_trt_logger = None


def _get_trt_logger():
    """Lazily create a singleton TRT logger (requires tensorrt import)."""
    global _trt_logger
    if _trt_logger is not None:
        return _trt_logger

    import tensorrt as trt

    class _FoxburrowTrtLogger(trt.ILogger):
        """Routes TensorRT log messages through foxburrow's log module.

        Only routes WARNING and above by default — TRT's INFO/VERBOSE
        output is extremely noisy during engine builds.
        """

        def log(self, severity, msg):
            msg = msg.strip()
            if not msg:
                return
            if severity == trt.ILogger.INTERNAL_ERROR or severity == trt.ILogger.ERROR:
                log.error(f"  TRT(native): {msg}")
            elif severity == trt.ILogger.WARNING:
                log.warning(f"  TRT(native): {msg}")
            # TRT INFO/VERBOSE are extremely verbose — skip them

    _trt_logger = _FoxburrowTrtLogger()
    return _trt_logger


# UNet static engines for the highest-traffic resolutions.
# The runtime picks the exact match when available; the dynamic engine covers the rest.
UNET_STATIC_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 768),     # Default / most common (55% of traffic)
    (1024, 1024),   # Square (25%)
    (1280, 1536),   # Hires
    (2048, 2048),   # Hires max
]

# VAE static engines — capped to avoid OOM during build.
# Resolutions above these use tiled VAE decode with a matching static engine.
VAE_STATIC_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 768),
    (1024, 1024),
    (1280, 1280),
]

# Dynamic engine profiles.  Each covers a resolution range; the runtime
# picks the tightest-fitting dynamic engine when no static match exists.
# UNet gets all three ranges; VAE only gets standard (hires VAE uses tiled decode).
DYNAMIC_PROFILES: list[dict] = [
    {"min": (512, 512), "opt": (768, 768), "max": (1024, 1024), "label": "dynamic-standard"},
    {"min": (1024, 1024), "opt": (1280, 1280), "max": (1536, 1536), "label": "dynamic-hires"},
    {"min": (1536, 1536), "opt": (1792, 1792), "max": (2048, 2048), "label": "dynamic-hires-xl"},
]

# Which dynamic profiles to build per component type.
_COMPONENT_DYNAMIC_PROFILES: dict[str, list[dict]] = {
    "unet": DYNAMIC_PROFILES,
    "vae": [DYNAMIC_PROFILES[0]],  # VAE: standard only (hires uses tiled decode)
}

# Backward compat alias
DYNAMIC_STANDARD = DYNAMIC_PROFILES[0]

# Headroom reserved during TRT builds (GB).  The GPU is drained (no inference)
# during builds, so we only need room for the CUDA context and TRT internals.
_BUILD_HEADROOM_GB = 1.0


def _get_workspace_gb(device_id: int) -> float:
    """Return max usable workspace for builds (GPU is drained, so use most VRAM)."""
    total_bytes = torch.cuda.get_device_properties(device_id).total_memory
    total_gb = total_bytes / (1024 ** 3)
    return max(total_gb - _BUILD_HEADROOM_GB, 1.0)


def get_arch_key(device_id: int) -> str:
    """Build architecture key for cache directory naming.

    Format: ``sm_{arch}_cu{cuda_major}{cuda_minor}``
    e.g. ``sm_120_cu128`` for Blackwell + CUDA 12.8
    """
    major, minor = torch.cuda.get_device_capability(device_id)
    arch = f"sm_{major * 10 + minor}"

    cuda_ver = torch.version.cuda or "0.0"
    cuda_parts = cuda_ver.split(".")
    cuda_major = cuda_parts[0]
    cuda_minor = cuda_parts[1] if len(cuda_parts) > 1 else "0"

    return f"{arch}_cu{cuda_major}{cuda_minor}"


def _latent_shape(width: int, height: int) -> tuple[int, int]:
    """Convert pixel resolution to latent spatial dimensions."""
    return height // 8, width // 8


def validate_onnx(onnx_path: str) -> bool:
    """Check that an ONNX file is valid and all external data files are accessible.

    Returns True if the ONNX file exists and all external data references
    can be resolved, False otherwise.  Does NOT load the full tensor data
    (fast check).
    """
    if not os.path.isfile(onnx_path):
        return False

    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
    except Exception as ex:
        log.warning(f"  TRT: Cannot load ONNX graph: {onnx_path}: {ex}")
        return False

    onnx_dir = os.path.dirname(onnx_path)
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    data_path = os.path.join(onnx_dir, entry.value)
                    if not os.path.isfile(data_path):
                        log.warning(f"  TRT: Missing external data file: "
                                    f"{entry.value} (referenced by {init.name})")
                        return False
    return True


def _remove_onnx_with_data(onnx_path: str) -> None:
    """Remove an ONNX file and its associated external data files."""
    if not os.path.isfile(onnx_path):
        return

    onnx_dir = os.path.dirname(onnx_path)
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=False)
        for init in model.graph.initializer:
            if init.data_location == onnx.TensorProto.EXTERNAL:
                for entry in init.external_data:
                    if entry.key == "location":
                        data_path = os.path.join(onnx_dir, entry.value)
                        if os.path.isfile(data_path):
                            os.remove(data_path)
    except Exception:
        pass  # Best-effort cleanup

    # Also remove the consolidated data file if present
    data_file = onnx_path + ".data"
    if os.path.isfile(data_file):
        os.remove(data_file)

    os.remove(onnx_path)


def _parse_onnx(parser, onnx_path: str) -> bool:
    """Parse an ONNX model, handling external data (weights > 2GB protobuf limit).

    Large models (e.g. SDXL UNet FP32 ~5GB) are exported by torch.onnx.export
    with external weight files.  The .onnx file contains only the graph
    structure.  TRT's ``parse_from_file()`` resolves these external references
    automatically; a raw ``parse(bytes)`` cannot.
    """
    # parse_from_file handles external data automatically (TRT 10+)
    try:
        if not parser.parse_from_file(onnx_path):
            for i in range(parser.num_errors):
                log.error(f"  TRT: ONNX parse error: {parser.get_error(i)}")
            return False
        return True
    except AttributeError:
        pass  # Older TRT without parse_from_file — fall through

    # Fallback: load with onnx library to resolve external data, then
    # serialize to bytes for the parser.  This only works for models
    # under the 2GB protobuf limit (i.e. VAE but not UNet).
    try:
        import onnx
        model = onnx.load(onnx_path, load_external_data=True)
        if not parser.parse(model.SerializeToString()):
            for i in range(parser.num_errors):
                log.error(f"  TRT: ONNX parse error: {parser.get_error(i)}")
            return False
        return True
    except Exception as ex:
        log.error(f"  TRT: Failed to load ONNX with external data: {ex}")
        return False


def build_te_engine(
    onnx_path: str,
    engine_path: str,
    component_type: str,
    device_id: int,
    max_workspace_gb: float = 0,
) -> bool:
    """Build a TensorRT engine for a text encoder (TE1 or TE2).

    Text encoders have no spatial dimensions — only the batch axis
    (number of 77-token chunks) is dynamic, ranging from 1 to 4.
    A single engine per text encoder covers all use cases.

    Args:
        onnx_path: Path to the ONNX model file.
        engine_path: Where to write the serialized engine.
        component_type: "te1" or "te2".
        device_id: CUDA device index to build on.
        max_workspace_gb: Maximum workspace size in GB.

    Returns:
        True if engine was built successfully, False on failure.
    """
    import tensorrt as trt

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    logger = _get_trt_logger()
    builder = trt.Builder(logger)
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

    config = None
    try:
        log.info(f"  TRT: Parsing ONNX for {component_type} (text encoder)...")

        if not _parse_onnx(parser, onnx_path):
            return False

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)
        config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()

        # Dynamic batch: 1 to 4 chunks, seq_len fixed at 77
        profile.set_shape("input_ids",      (1, 77), (1, 77), (4, 77))
        profile.set_shape("attention_mask",  (1, 77), (1, 77), (4, 77))

        config.add_optimization_profile(profile)

        return _do_build(builder, network, config, engine_path, component_type,
                         "text encoder", device_id, max_workspace_gb)
    finally:
        del parser, network, builder
        if config is not None:
            del config
        gc.collect()
        torch.cuda.empty_cache()


def build_static_engine(
    onnx_path: str,
    engine_path: str,
    component_type: str,
    width: int,
    height: int,
    device_id: int,
    max_workspace_gb: float = 0,
) -> bool:
    """Build a TensorRT engine with a fixed-resolution profile.

    The engine is optimized exclusively for the given resolution — TRT picks
    the single best kernel per layer.  Will not accept other resolutions at
    inference time.

    Args:
        onnx_path: Path to the ONNX model file.
        engine_path: Where to write the serialized engine.
        component_type: "unet" or "vae" — determines input shapes.
        width: Target image width in pixels.
        height: Target image height in pixels.
        device_id: CUDA device index to build on.
        max_workspace_gb: Maximum workspace size in GB.

    Returns:
        True if engine was built successfully, False on failure.
    """
    import tensorrt as trt

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    lat_h, lat_w = _latent_shape(width, height)

    logger = _get_trt_logger()
    builder = trt.Builder(logger)
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

    config = None
    try:
        log.info(f"  TRT: Parsing ONNX for {component_type} (static {width}x{height})...")

        if not _parse_onnx(parser, onnx_path):
            return False

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        if component_type == "unet":
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()

        if component_type == "unet":
            profile.set_shape("sample", (2, 4, lat_h, lat_w), (2, 4, lat_h, lat_w), (2, 4, lat_h, lat_w))
            profile.set_shape("timestep", (1,), (1,), (1,))
            profile.set_shape("encoder_hidden_states", (2, 77, 2048), (2, 77, 2048), (2, 77 * 4, 2048))
            profile.set_shape("text_embeds", (2, 1280), (2, 1280), (2, 1280))
            profile.set_shape("time_ids", (2, 6), (2, 6), (2, 6))
        elif component_type == "vae":
            profile.set_shape("latents", (1, 4, lat_h, lat_w), (1, 4, lat_h, lat_w), (1, 4, lat_h, lat_w))
        else:
            log.error(f"  TRT: Unknown component type: {component_type}")
            return False

        config.add_optimization_profile(profile)

        return _do_build(builder, network, config, engine_path, component_type,
                         f"static {width}x{height}", device_id, max_workspace_gb)
    finally:
        del parser, network, builder
        if config is not None:
            del config
        gc.collect()
        torch.cuda.empty_cache()


def build_dynamic_engine(
    onnx_path: str,
    engine_path: str,
    component_type: str,
    device_id: int,
    min_res: tuple[int, int],
    opt_res: tuple[int, int],
    max_res: tuple[int, int],
    max_workspace_gb: float = 0,
) -> bool:
    """Build a TensorRT engine with a dynamic-resolution profile.

    TRT optimizes for the OPT resolution but accepts any resolution within
    [MIN, MAX].  Slightly less optimal than a static engine (~5-15% slower
    at OPT) but covers arbitrary resolutions without PyTorch fallback.

    Args:
        onnx_path: Path to the ONNX model file.
        engine_path: Where to write the serialized engine.
        component_type: "unet" or "vae" — determines input shapes.
        device_id: CUDA device index to build on.
        min_res: Minimum (width, height) in pixels.
        opt_res: Optimal (width, height) — TRT tunes kernels for this.
        max_res: Maximum (width, height) in pixels.
        max_workspace_gb: Maximum workspace size in GB.

    Returns:
        True if engine was built successfully, False on failure.
    """
    import tensorrt as trt

    os.makedirs(os.path.dirname(engine_path), exist_ok=True)

    min_lat_h, min_lat_w = _latent_shape(*min_res)
    opt_lat_h, opt_lat_w = _latent_shape(*opt_res)
    max_lat_h, max_lat_w = _latent_shape(*max_res)

    logger = _get_trt_logger()
    builder = trt.Builder(logger)
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

    config = None
    try:
        log.info(f"  TRT: Parsing ONNX for {component_type} "
                 f"(dynamic {min_res[0]}x{min_res[1]}→{max_res[0]}x{max_res[1]}, "
                 f"opt {opt_res[0]}x{opt_res[1]})...")

        if not _parse_onnx(parser, onnx_path):
            return False

        config = builder.create_builder_config()
        config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

        if component_type == "unet":
            config.set_flag(trt.BuilderFlag.FP16)

        profile = builder.create_optimization_profile()

        if component_type == "unet":
            profile.set_shape("sample",
                              (2, 4, min_lat_h, min_lat_w),
                              (2, 4, opt_lat_h, opt_lat_w),
                              (2, 4, max_lat_h, max_lat_w))
            profile.set_shape("timestep", (1,), (1,), (1,))
            profile.set_shape("encoder_hidden_states",
                              (2, 77, 2048), (2, 77, 2048), (2, 77 * 4, 2048))
            profile.set_shape("text_embeds", (2, 1280), (2, 1280), (2, 1280))
            profile.set_shape("time_ids", (2, 6), (2, 6), (2, 6))
        elif component_type == "vae":
            profile.set_shape("latents",
                              (1, 4, min_lat_h, min_lat_w),
                              (1, 4, opt_lat_h, opt_lat_w),
                              (1, 4, max_lat_h, max_lat_w))
        else:
            log.error(f"  TRT: Unknown component type: {component_type}")
            return False

        config.add_optimization_profile(profile)

        label = f"dynamic {min_res[0]}x{min_res[1]}→{max_res[0]}x{max_res[1]}"
        ok = _do_build(builder, network, config, engine_path, component_type,
                       label, device_id, max_workspace_gb)
        if ok:
            _write_dynamic_sidecar(engine_path, min_res, opt_res, max_res)
        return ok
    finally:
        del parser, network, builder
        if config is not None:
            del config
        gc.collect()
        torch.cuda.empty_cache()


def _do_build(
    builder,
    network,
    config,
    engine_path: str,
    component_type: str,
    label: str,
    device_id: int,
    max_workspace_gb: float = 0,
) -> bool:
    """Shared build logic."""
    import tensorrt as trt

    workspace_gb = max_workspace_gb if max_workspace_gb > 0 else _get_workspace_gb(device_id)
    workspace_bytes = int(workspace_gb * 1024 * 1024 * 1024)
    config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

    log.info(f"  TRT: Building {component_type} engine ({label}) "
             f"(workspace={workspace_gb:.1f}GB, device={device_id})...")

    start = _time.monotonic()

    with torch.cuda.device(device_id):
        try:
            engine_bytes = builder.build_serialized_network(network, config)
        except Exception as ex:
            log.error(f"  TRT: Build failed for {component_type} ({label}): {ex}")
            return False

    if engine_bytes is None:
        log.error(f"  TRT: Build returned None for {component_type} ({label})")
        return False

    elapsed = _time.monotonic() - start
    log.info(f"  TRT: Engine built in {elapsed:.1f}s (workspace={workspace_gb:.1f}GB)")

    tmp_path = engine_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(engine_bytes)
    del engine_bytes  # Release serialized engine bytes from host RAM (can be multi-GB)
    os.replace(tmp_path, engine_path)

    engine_size_mb = os.path.getsize(engine_path) / (1024 * 1024)
    log.info(f"  TRT: Saved engine ({engine_size_mb:.0f}MB): {engine_path}")
    return True


def get_engine_path(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
    width: int,
    height: int,
) -> str:
    """Compute the filesystem path for a static TRT engine file.

    Layout: ``{cache_dir}/{hash}/{arch_key}/{component_type}/{WxH}.engine``
    """
    short_hash = model_hash[:16]
    return os.path.join(
        cache_dir, short_hash, arch_key, component_type,
        f"{width}x{height}.engine",
    )


def get_dynamic_engine_path(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
    label: str = "dynamic",
) -> str:
    """Compute the filesystem path for a dynamic TRT engine file.

    Layout: ``{cache_dir}/{hash}/{arch_key}/{component_type}/{label}.engine``
    """
    short_hash = model_hash[:16]
    return os.path.join(
        cache_dir, short_hash, arch_key, component_type,
        f"{label}.engine",
    )


def get_onnx_path(cache_dir: str, model_hash: str, component_type: str) -> str:
    """Compute the filesystem path for an ONNX file.

    Layout: ``{cache_dir}/{hash}/onnx/{component_type}.onnx``

    ONNX is arch-independent (shared across GPU architectures).
    Both UNet and VAE are exported as FP32 — TRT handles precision
    conversion (FP16 for UNet) during engine building via builder flags.
    """
    short_hash = model_hash[:16]
    return os.path.join(
        cache_dir, short_hash, "onnx",
        f"{component_type}.onnx",
    )


def _write_dynamic_sidecar(
    engine_path: str,
    min_res: tuple[int, int],
    opt_res: tuple[int, int],
    max_res: tuple[int, int],
) -> None:
    """Write a JSON sidecar alongside a dynamic engine describing its profile."""
    sidecar_path = engine_path.replace(".engine", ".json")
    data = {
        "min_res": list(min_res),
        "opt_res": list(opt_res),
        "max_res": list(max_res),
    }
    with open(sidecar_path, "w") as f:
        json.dump(data, f, indent=2)


def discover_dynamic_engines(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
) -> list[dict]:
    """Discover all dynamic engines for a component by scanning JSON sidecars.

    Returns a list of dicts with keys: path, label, min_res, opt_res, max_res.
    Sorted by range area (tightest first) for best-fit selection.
    """
    short_hash = model_hash[:16]
    comp_dir = os.path.join(cache_dir, short_hash, arch_key, component_type)
    if not os.path.isdir(comp_dir):
        return []

    results = []
    for name in os.listdir(comp_dir):
        if not name.endswith(".json"):
            continue
        json_path = os.path.join(comp_dir, name)
        engine_path = os.path.join(comp_dir, name.replace(".json", ".engine"))
        if not os.path.isfile(engine_path):
            continue
        try:
            with open(json_path) as f:
                data = json.load(f)
            results.append({
                "path": engine_path,
                "label": name[:-5],  # strip .json
                "min_res": tuple(data["min_res"]),
                "opt_res": tuple(data["opt_res"]),
                "max_res": tuple(data["max_res"]),
            })
        except (json.JSONDecodeError, KeyError, TypeError):
            continue

    # Sort by range area — tightest fit first
    def _range_area(d):
        w_range = d["max_res"][0] - d["min_res"][0]
        h_range = d["max_res"][1] - d["min_res"][1]
        return w_range * h_range

    results.sort(key=_range_area)
    return results


def find_best_dynamic_engine(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
    width: int,
    height: int,
) -> dict | None:
    """Find the best-fitting dynamic engine for a given resolution.

    Returns the tightest dynamic profile whose range covers (width, height),
    or None if no dynamic engine covers it.
    """
    for eng in discover_dynamic_engines(cache_dir, model_hash, component_type, arch_key):
        min_w, min_h = eng["min_res"]
        max_w, max_h = eng["max_res"]
        if min_w <= width <= max_w and min_h <= height <= max_h:
            return eng
    return None


def _get_static_resolutions(component_type: str) -> list[tuple[int, int]]:
    """Return the static resolution list for a component type."""
    if component_type == "vae":
        return VAE_STATIC_RESOLUTIONS
    return UNET_STATIC_RESOLUTIONS


def has_trt_coverage(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
    width: int,
    height: int,
) -> bool:
    """Check if a TRT engine can handle the given resolution.

    Returns True if a static engine exists for the exact resolution OR
    a dynamic engine exists whose range covers the resolution.
    For text encoders (te1/te2), resolution is irrelevant — just check
    if the single "default" engine exists.
    """
    # Text encoders: no spatial dimensions, single engine per component
    if component_type in ("te1", "te2"):
        return os.path.isfile(
            get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key, "default"))

    # Static exact match
    if os.path.isfile(get_engine_path(cache_dir, model_hash, component_type, arch_key, width, height)):
        return True
    # Dynamic range check — discover all available dynamic engines
    return find_best_dynamic_engine(cache_dir, model_hash, component_type, arch_key, width, height) is not None


def all_engines_exist(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
) -> bool:
    """Check if all required engines (static + dynamic) exist for a component."""
    # Text encoders: single "default" engine per component
    if component_type in ("te1", "te2"):
        return os.path.isfile(
            get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key, "default"))

    for w, h in _get_static_resolutions(component_type):
        path = get_engine_path(cache_dir, model_hash, component_type, arch_key, w, h)
        if not os.path.isfile(path):
            return False
    for profile in _COMPONENT_DYNAMIC_PROFILES.get(component_type, [DYNAMIC_STANDARD]):
        path = get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key, profile["label"])
        if not os.path.isfile(path):
            return False
    return True


def build_all_engines(
    model_hash: str,
    cache_dir: str,
    arch_key: str,
    device_id: int,
    max_workspace_gb: float = 0,
    progress_cb: Callable[[str, str], None] | None = None,
) -> dict[str, list[str]]:
    """Build all missing TRT engines for a model on a specific GPU architecture.

    For each component (unet, vae), builds:
      - One static engine per configured resolution
        (UNet: UNET_STATIC_RESOLUTIONS, VAE: VAE_STATIC_RESOLUTIONS)
      - Dynamic engines per _COMPONENT_DYNAMIC_PROFILES (UNet gets all three
        ranges: standard/hires/hires-xl; VAE gets standard only)

    Args:
        model_hash: Content fingerprint of the model (from registry).
        cache_dir: Root TRT cache directory.
        arch_key: GPU architecture key (e.g. "sm_120_cu128").
        device_id: CUDA device index to build on.
        progress_cb: Optional callback(component_type, engine_label) called
            before each engine build starts, for TUI progress display.

    Returns:
        Dict mapping component_type -> list of successfully built engine labels.
    """
    results: dict[str, list[str]] = {"unet": [], "vae": [], "te1": [], "te2": []}

    # Set CUDA device for the entire build pipeline — TRT's Builder and
    # OnnxParser bind to the current CUDA context at creation time.
    # Without this, parallel builds on different GPUs all target device 0,
    # causing memory contention and corrupt parser state.
    with torch.cuda.device(device_id):
        for component_type in ("te1", "te2", "unet", "vae"):
            onnx_path = get_onnx_path(cache_dir, model_hash, component_type)
            if not os.path.isfile(onnx_path):
                log.error(f"  TRT: ONNX not found for {component_type}: {onnx_path}")
                continue

            # Validate external data accessibility before attempting build
            if not validate_onnx(onnx_path):
                log.error(f"  TRT: ONNX validation failed for {component_type} — "
                          f"external data missing or corrupt: {onnx_path}")
                log.error(f"  TRT: Deleting invalid ONNX so it will be re-exported "
                          f"on next run: {onnx_path}")
                _remove_onnx_with_data(onnx_path)
                continue

            # Text encoders: single engine per component (dynamic batch only)
            if component_type in ("te1", "te2"):
                te_path = get_dynamic_engine_path(
                    cache_dir, model_hash, component_type, arch_key, "default")
                if os.path.isfile(te_path):
                    log.debug(f"  TRT: TE engine exists, skipping: {component_type}")
                    results[component_type].append("default")
                else:
                    if progress_cb:
                        progress_cb(component_type, "default")
                    ok = build_te_engine(
                        onnx_path=onnx_path,
                        engine_path=te_path,
                        component_type=component_type,
                        device_id=device_id,
                        max_workspace_gb=max_workspace_gb,
                    )
                    if ok:
                        results[component_type].append("default")
                continue

            # Static engines for all configured resolutions
            for sw, sh in _get_static_resolutions(component_type):
                static_path = get_engine_path(cache_dir, model_hash, component_type, arch_key, sw, sh)
                if os.path.isfile(static_path):
                    log.debug(f"  TRT: Static engine exists, skipping: {component_type} {sw}x{sh}")
                    results[component_type].append(f"static-{sw}x{sh}")
                else:
                    if progress_cb:
                        progress_cb(component_type, f"static-{sw}x{sh}")
                    ok = build_static_engine(
                        onnx_path=onnx_path,
                        engine_path=static_path,
                        component_type=component_type,
                        width=sw,
                        height=sh,
                        device_id=device_id,
                        max_workspace_gb=max_workspace_gb,
                    )
                    if ok:
                        results[component_type].append(f"static-{sw}x{sh}")

            # Dynamic engines — build all profiles configured for this component
            for dyn in _COMPONENT_DYNAMIC_PROFILES.get(component_type, [DYNAMIC_STANDARD]):
                label = dyn["label"]
                dyn_path = get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key, label)
                if os.path.isfile(dyn_path):
                    log.debug(f"  TRT: Dynamic engine exists, skipping: {component_type} {label}")
                    results[component_type].append(label)
                else:
                    if progress_cb:
                        progress_cb(component_type, label)
                    ok = build_dynamic_engine(
                        onnx_path=onnx_path,
                        engine_path=dyn_path,
                        component_type=component_type,
                        device_id=device_id,
                        min_res=dyn["min"],
                        opt_res=dyn["opt"],
                        max_res=dyn["max"],
                        max_workspace_gb=max_workspace_gb,
                    )
                    if ok:
                        results[component_type].append(label)

    return results
