"""TensorRT engine building from ONNX models.

Builds optimized TRT engines per resolution with workspace stepping:
starts at a conservative workspace size and steps up on OOM failure.
"""

from __future__ import annotations

import os
import time as _time

import torch

import log


# Target resolutions: (width, height) in pixels
# These cover ~99% of traffic. Unsupported resolutions fall back to PyTorch.
TARGET_RESOLUTIONS: list[tuple[int, int]] = [
    (640, 768),     # 80% of traffic — primary optimization target
    (768, 768),     # Square
    (1024, 1024),   # Max non-tiled; also MultiDiffusion tile size
]

# Workspace stepping parameters (GB)
_WORKSPACE_START_GB = 2.0
_WORKSPACE_STEP_GB = 1.0
_WORKSPACE_MAX_GB = 8.0


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


def build_engine(
    onnx_path: str,
    engine_path: str,
    component_type: str,
    width: int,
    height: int,
    device_id: int,
    max_workspace_gb: float = _WORKSPACE_MAX_GB,
) -> bool:
    """Build a TensorRT engine from an ONNX model for a specific resolution.

    Uses workspace stepping: starts at _WORKSPACE_START_GB, steps up by
    _WORKSPACE_STEP_GB on build failure, until max_workspace_gb.

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
    # EXPLICIT_BATCH was removed in TRT 10.0 (all networks are explicit-batch).
    # Guard for compatibility with both TRT 8.x/9.x and TRT 10.x+.
    try:
        explicit_batch_flag = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    except AttributeError:
        explicit_batch_flag = 0  # TRT 10+: flag removed, pass 0
    network = builder.create_network(explicit_batch_flag)
    parser = trt.OnnxParser(network, logger)

    log.info(f"  TRT: Parsing ONNX for {component_type} ({width}x{height})...")

    with open(onnx_path, "rb") as f:
        if not parser.parse(f.read()):
            for i in range(parser.num_errors):
                log.error(f"  TRT: ONNX parse error: {parser.get_error(i)}")
            return False

    config = builder.create_builder_config()

    # Set device for building
    config.set_flag(trt.BuilderFlag.GPU_FALLBACK)

    if component_type == "unet":
        config.set_flag(trt.BuilderFlag.FP16)
    # VAE stays FP32 — no FP16 flag

    # Create optimization profile for the target resolution
    profile = builder.create_optimization_profile()

    if component_type == "unet":
        # Fixed spatial shapes for this specific resolution.
        # Batch=2 for CFG (uncond + cond concatenated).
        # encoder_hidden_states seq_len is 77*N where N = prompt chunk count
        # (most prompts are 1 chunk = 77, long prompts can be 2-4 chunks).
        profile.set_shape("sample", (2, 4, lat_h, lat_w), (2, 4, lat_h, lat_w), (2, 4, lat_h, lat_w))
        profile.set_shape("timestep", (1,), (1,), (1,))
        profile.set_shape("encoder_hidden_states", (2, 77, 2048), (2, 77, 2048), (2, 77 * 4, 2048))
        profile.set_shape("text_embeds", (2, 1280), (2, 1280), (2, 1280))
        profile.set_shape("time_ids", (2, 6), (2, 6), (2, 6))
    elif component_type == "vae":
        # Batch=1 for VAE decode
        profile.set_shape("latents", (1, 4, lat_h, lat_w), (1, 4, lat_h, lat_w), (1, 4, lat_h, lat_w))
    else:
        log.error(f"  TRT: Unknown component type: {component_type}")
        return False

    config.add_optimization_profile(profile)

    # Set the active CUDA device so TRT builds on the correct GPU
    # (important on multi-GPU systems where device_id may not be 0).
    # Workspace stepping: try increasing sizes until build succeeds.
    workspace_gb = _WORKSPACE_START_GB
    engine_bytes = None

    with torch.cuda.device(device_id):
        while workspace_gb <= max_workspace_gb:
            workspace_bytes = int(workspace_gb * 1024 * 1024 * 1024)
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_bytes)

            log.info(f"  TRT: Building {component_type} engine for {width}x{height} "
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
        log.error(f"  TRT: Failed to build {component_type} engine for {width}x{height} "
                  f"after all workspace attempts (up to {max_workspace_gb:.1f}GB)")
        return False

    # Serialize to disk atomically — write to .tmp then rename to prevent
    # corrupt partial engine files on crash during multi-GB writes.
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
    """Compute the filesystem path for a TRT engine file.

    Layout: ``{cache_dir}/{model_hash}/{component_type}/{arch_key}/{WxH}.engine``
    """
    short_hash = model_hash[:16]
    return os.path.join(
        cache_dir, short_hash, component_type, arch_key,
        f"{width}x{height}.engine",
    )


def get_onnx_path(cache_dir: str, model_hash: str, component_type: str) -> str:
    """Compute the filesystem path for an ONNX file.

    Layout: ``{cache_dir}/{model_hash}/onnx/{component_type}_fp{precision}.onnx``
    """
    short_hash = model_hash[:16]
    precision = "fp16" if component_type == "unet" else "fp32"
    return os.path.join(
        cache_dir, short_hash, "onnx",
        f"{component_type}_{precision}.onnx",
    )


def engine_exists(
    cache_dir: str,
    model_hash: str,
    component_type: str,
    arch_key: str,
    width: int,
    height: int,
) -> bool:
    """Check if a TRT engine file exists on disk."""
    path = get_engine_path(cache_dir, model_hash, component_type, arch_key, width, height)
    return os.path.isfile(path)


def build_all_engines(
    model_hash: str,
    cache_dir: str,
    arch_key: str,
    device_id: int,
    resolutions: list[tuple[int, int]] | None = None,
) -> dict[str, list[tuple[int, int]]]:
    """Build all missing TRT engines for a model on a specific GPU architecture.

    Args:
        model_hash: Content fingerprint of the model (from registry).
        cache_dir: Root TRT cache directory.
        arch_key: GPU architecture key (e.g. "sm_120_cu128").
        device_id: CUDA device index to build on.
        resolutions: Target resolutions to build. Defaults to TARGET_RESOLUTIONS.

    Returns:
        Dict mapping component_type -> list of successfully built (width, height).
    """
    if resolutions is None:
        resolutions = TARGET_RESOLUTIONS

    results: dict[str, list[tuple[int, int]]] = {"unet": [], "vae": []}

    for component_type in ("unet", "vae"):
        onnx_path = get_onnx_path(cache_dir, model_hash, component_type)
        if not os.path.isfile(onnx_path):
            log.error(f"  TRT: ONNX not found for {component_type}: {onnx_path}")
            continue

        for width, height in resolutions:
            ep = get_engine_path(cache_dir, model_hash, component_type, arch_key, width, height)
            if os.path.isfile(ep):
                log.debug(f"  TRT: Engine exists, skipping: {component_type} {width}x{height}")
                results[component_type].append((width, height))
                continue

            ok = build_engine(
                onnx_path=onnx_path,
                engine_path=ep,
                component_type=component_type,
                width=width,
                height=height,
                device_id=device_id,
            )
            if ok:
                results[component_type].append((width, height))

    return results
