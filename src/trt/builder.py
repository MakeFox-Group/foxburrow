"""TensorRT engine building from ONNX models.

Builds two engine types per model × component × GPU architecture:
  1. Static engine for 640×768 (80% of traffic, maximum kernel optimization)
  2. Dynamic engine covering 512×512 → 2048×2048 (optimized at 1024×1024)

Uses workspace stepping: starts at a conservative workspace size and
steps up on OOM failure.
"""

from __future__ import annotations

import os
import time as _time

import torch

import log


# Static engine: exact-match resolution for the overwhelming majority of traffic.
STATIC_RESOLUTION: tuple[int, int] = (640, 768)

# Dynamic engine: covers all other resolutions up to 2048×2048.
# TRT optimizes kernels for the OPT shape but handles any shape in [MIN, MAX].
DYNAMIC_MIN: tuple[int, int] = (512, 512)
DYNAMIC_OPT: tuple[int, int] = (1024, 1024)
DYNAMIC_MAX: tuple[int, int] = (2048, 2048)

# Workspace stepping parameters (GB)
_WORKSPACE_START_GB = 2.0
_WORKSPACE_STEP_GB = 1.0
_WORKSPACE_HEADROOM_GB = 1.0  # Reserved for CUDA context + TRT overhead


def _device_max_workspace_gb(device_id: int) -> float:
    """Query the GPU's total VRAM and return max usable workspace in GB."""
    total_bytes = torch.cuda.get_device_properties(device_id).total_memory
    total_gb = total_bytes / (1024 ** 3)
    return max(total_gb - _WORKSPACE_HEADROOM_GB, _WORKSPACE_START_GB)


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


def _parse_onnx(parser, onnx_path: str) -> bool:
    """Parse an ONNX model, handling external data (weights > 2GB protobuf limit).

    Large models (e.g. SDXL UNet FP16 ~5GB) are exported by torch.onnx.export
    with external weight files.  The 4MB .onnx file contains only the graph
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

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

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


def build_dynamic_engine(
    onnx_path: str,
    engine_path: str,
    component_type: str,
    device_id: int,
    min_res: tuple[int, int] = DYNAMIC_MIN,
    opt_res: tuple[int, int] = DYNAMIC_OPT,
    max_res: tuple[int, int] = DYNAMIC_MAX,
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

    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

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
    return _do_build(builder, network, config, engine_path, component_type,
                     label, device_id, max_workspace_gb)


def _do_build(
    builder,
    network,
    config,
    engine_path: str,
    component_type: str,
    label: str,
    device_id: int,
    max_workspace_gb: float,
) -> bool:
    """Shared build logic with workspace stepping."""
    import tensorrt as trt

    if max_workspace_gb <= 0:
        max_workspace_gb = _device_max_workspace_gb(device_id)

    workspace_gb = _WORKSPACE_START_GB
    engine_bytes = None

    with torch.cuda.device(device_id):
        while workspace_gb <= max_workspace_gb:
            workspace_bytes = int(workspace_gb * 1024 * 1024 * 1024)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

            log.info(f"  TRT: Building {component_type} engine ({label}) "
                     f"(workspace={workspace_gb:.1f}GB, device={device_id})...")

            start = _time.monotonic()

            try:
                serialized = builder.build_serialized_network(network, config)
            except Exception as ex:
                log.warning(f"  TRT: Build failed with workspace={workspace_gb:.1f}GB: {ex}")
                workspace_gb += _WORKSPACE_STEP_GB
                continue

            if serialized is None:
                log.warning(f"  TRT: Build returned None with workspace={workspace_gb:.1f}GB "
                            f"— stepping up")
                workspace_gb += _WORKSPACE_STEP_GB
                continue

            engine_bytes = serialized
            elapsed = _time.monotonic() - start
            log.info(f"  TRT: Engine built in {elapsed:.1f}s (workspace={workspace_gb:.1f}GB)")
            break

    if engine_bytes is None:
        log.error(f"  TRT: Failed to build {component_type} engine ({label}) "
                  f"after all workspace attempts (up to {max_workspace_gb:.1f}GB)")
        return False

    tmp_path = engine_path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(engine_bytes)
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
) -> str:
    """Compute the filesystem path for a dynamic TRT engine file.

    Layout: ``{cache_dir}/{hash}/{arch_key}/{component_type}/dynamic.engine``
    """
    short_hash = model_hash[:16]
    return os.path.join(
        cache_dir, short_hash, arch_key, component_type,
        "dynamic.engine",
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


def static_engine_exists(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
) -> bool:
    """Check if the static TRT engine file exists on disk."""
    w, h = STATIC_RESOLUTION
    path = get_engine_path(cache_dir, model_hash, component_type, arch_key, w, h)
    return os.path.isfile(path)


def dynamic_engine_exists(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
) -> bool:
    """Check if the dynamic TRT engine file exists on disk."""
    path = get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key)
    return os.path.isfile(path)


def all_engines_exist(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
) -> bool:
    """Check if all required engines exist for a component.

    UNet: static + dynamic (dynamic covers non-standard resolutions).
    VAE: static only (VAE runs once per image; dynamic builds fail due to
    the wide conv/upsample kernel range, and PyTorch fallback is acceptable).
    """
    if not static_engine_exists(cache_dir, model_hash, component_type, arch_key):
        return False
    if component_type == "unet":
        return dynamic_engine_exists(cache_dir, model_hash, component_type, arch_key)
    return True


def build_all_engines(
    model_hash: str,
    cache_dir: str,
    arch_key: str,
    device_id: int,
) -> dict[str, list[str]]:
    """Build all missing TRT engines for a model on a specific GPU architecture.

    Builds two engines per component:
      1. Static engine for 640×768 (exact match, maximum optimization)
      2. Dynamic engine for 512×512 → 2048×2048 (optimized at 1024×1024)

    Args:
        model_hash: Content fingerprint of the model (from registry).
        cache_dir: Root TRT cache directory.
        arch_key: GPU architecture key (e.g. "sm_120_cu128").
        device_id: CUDA device index to build on.

    Returns:
        Dict mapping component_type -> list of successfully built engine labels.
    """
    results: dict[str, list[str]] = {"unet": [], "vae": []}

    # Set CUDA device for the entire build pipeline — TRT's Builder and
    # OnnxParser bind to the current CUDA context at creation time.
    # Without this, parallel builds on different GPUs all target device 0,
    # causing memory contention and corrupt parser state.
    with torch.cuda.device(device_id):
        for component_type in ("unet", "vae"):
            onnx_path = get_onnx_path(cache_dir, model_hash, component_type)
            if not os.path.isfile(onnx_path):
                log.error(f"  TRT: ONNX not found for {component_type}: {onnx_path}")
                continue

            # Static engine for 640×768
            sw, sh = STATIC_RESOLUTION
            static_path = get_engine_path(cache_dir, model_hash, component_type, arch_key, sw, sh)
            if os.path.isfile(static_path):
                log.debug(f"  TRT: Static engine exists, skipping: {component_type} {sw}x{sh}")
                results[component_type].append(f"static-{sw}x{sh}")
            else:
                ok = build_static_engine(
                    onnx_path=onnx_path,
                    engine_path=static_path,
                    component_type=component_type,
                    width=sw,
                    height=sh,
                    device_id=device_id,
                )
                if ok:
                    results[component_type].append(f"static-{sw}x{sh}")

            # Dynamic engine for UNet only — covers non-standard resolutions.
            # VAE skipped: dynamic range (512→2048) is too wide for conv/upsample
            # kernels, and VAE runs once per image so PyTorch fallback is fine.
            if component_type == "unet":
                dynamic_path = get_dynamic_engine_path(cache_dir, model_hash, component_type, arch_key)
                if os.path.isfile(dynamic_path):
                    log.debug(f"  TRT: Dynamic engine exists, skipping: {component_type}")
                    results[component_type].append("dynamic")
                else:
                    ok = build_dynamic_engine(
                        onnx_path=onnx_path,
                        engine_path=dynamic_path,
                        component_type=component_type,
                        device_id=device_id,
                    )
                    if ok:
                        results[component_type].append("dynamic")

    return results
