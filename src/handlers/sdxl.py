"""SDXL handler: text encoding, denoising, VAE decode/encode, hires fix.

Ports all logic from the C# SdxlHandler using PyTorch + diffusers.
"""

from __future__ import annotations

import math
import os
import re
import threading
import time
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import AutoencoderKL, EulerAncestralDiscreteScheduler
from PIL import Image
from transformers import CLIPTextModel, CLIPTextModelWithProjection, CLIPTokenizer

import log
from gpu import nvml
from gpu.pool import fix_meta_tensors, repair_accelerate_leak
from scheduling.job import InferenceJob, SdxlTokenizeResult, SdxlEncodeResult, SdxlRegionalEncodeResult

if TYPE_CHECKING:
    from gpu.pool import GpuInstance

def _fix_from_pretrained(model: torch.nn.Module, label: str) -> None:
    """Fix meta tensors left behind by from_pretrained()/accelerate.

    Also repairs any leaked accelerate init_empty_weights() context so
    that subsequent model construction (on any thread) isn't poisoned.
    """
    repair_accelerate_leak()
    n = fix_meta_tensors(model)
    if n:
        log.debug(f"  SDXL: Fixed {n} meta tensor(s) in {label}")


def _get_model_name(job: InferenceJob) -> str | None:
    """Extract clean model name from job's sdxl_input.model_dir."""
    inp = job.sdxl_input
    if not inp or not inp.model_dir:
        return None
    name = os.path.basename(inp.model_dir)
    for ext in (".safetensors", ".ckpt"):
        if name.endswith(ext):
            return name[:-len(ext)]
    return name


# Constants
VAE_SCALE_FACTOR = 0.13025
VAE_TILE_THRESHOLD = 1025    # force tiled mode when any dimension exceeds 1024 (pixels)
VAE_TILE_MAX = 768           # max tile size per axis when tiling (pixels)
LATENT_TILE_OVERLAP = 16     # latent overlap for VAE encode/decode tiles

def _auto_vae_tile(img_w: int, img_h: int, max_tile: int) -> tuple[int, int]:
    """Return (tile_w, tile_h) in pixels that evenly distribute the image
    into tiles <= max_tile. Each axis is computed independently, so tiles
    may be non-square for optimal coverage."""
    def _axis(dim: int) -> int:
        if dim <= max_tile:
            return dim             # fits in one tile — no splitting needed
        n = math.ceil(dim / max_tile)
        return math.ceil(math.ceil(dim / n) / 8) * 8  # round up to multiple of 8
    return _axis(img_w), _axis(img_h)


def _optimal_tile_for_axis(dim: int, min_tile: int, max_tile: int) -> int | None:
    """Find the optimal tile size for one axis within [min_tile, max_tile].

    Always uses max_tile when tiling is needed.  Tile overlap is a fixed
    cost per tile boundary, so larger tiles = fewer boundaries = fewer
    total tiles.  Dividing evenly into smaller tiles is always worse
    because each extra boundary adds a full overlap region.
    """
    if dim < min_tile:
        return None  # image dimension below engine minimum
    if dim <= max_tile:
        return dim  # fits in one tile — no splitting needed
    return max_tile  # largest tile minimizes overlap-inflated tile count


def _pick_vae_tile_size(img_w: int, img_h: int, gpu: "GpuInstance",
                        job: "InferenceJob",
                        component_type: str = "vae") -> tuple[int, int]:
    """Pick VAE tile size, TRT-aware with smart engine selection.

    Args:
        component_type: "vae" for decoder, "vae_enc" for encoder.

    Priority (respects dynamic_only — skips all static engines when set):
    1. Static TRT engine already cached on GPU (zero load cost)
    2. Static TRT engine on disk (load once, exact match)
    3. Dynamic TRT engine — choose optimal tile within engine's range
       to minimize tile count and overlap
    4. PyTorch fallback (auto-calc with VAE_TILE_MAX)
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)

    from trt.builder import (VAE_STATIC_RESOLUTIONS, get_arch_key, get_engine_path,
                             discover_dynamic_engines)

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae")
    if not fp:
        log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
                  f"PyTorch fallback (no VAE fingerprint)")
        return _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache
    dynamic_only = app_state.config.tensorrt.dynamic_only

    # --- Phase 1: Static engines (skip entirely if dynamic_only) ---
    if not dynamic_only:
        cached_static = []
        disk_static = []

        for ew, eh in VAE_STATIC_RESOLUTIONS:
            if ew > img_w or eh > img_h:
                continue
            trt_fp = f"{fp}:{component_type}_trt:{ew}x{eh}"
            if gpu.is_component_loaded(trt_fp):
                cached_static.append((ew, eh))
            else:
                engine_path = get_engine_path(cache_dir, fp, component_type, arch_key, ew, eh)
                if os.path.isfile(engine_path):
                    disk_static.append((ew, eh))

        if cached_static:
            best = max(cached_static, key=lambda r: r[0] * r[1])
            log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
                      f"{best[0]}x{best[1]} (cached static TRT {component_type})")
            return best

        if disk_static:
            best = max(disk_static, key=lambda r: r[0] * r[1])
            log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
                      f"{best[0]}x{best[1]} (on-disk static TRT {component_type})")
            return best

    # --- Phase 2: Dynamic engines ---
    dynamic_engines = discover_dynamic_engines(cache_dir, fp, component_type, arch_key)

    best_cached_tile: tuple[int, int] | None = None
    best_cached_tiles = float('inf')
    best_cached_eng: dict | None = None
    best_disk_tile: tuple[int, int] | None = None
    best_disk_tiles = float('inf')
    best_disk_eng: dict | None = None

    for eng in dynamic_engines:
        min_w, min_h = eng["min_res"]
        max_w, max_h = eng["max_res"]

        tile_w = _optimal_tile_for_axis(img_w, min_w, max_w)
        tile_h = _optimal_tile_for_axis(img_h, min_h, max_h)
        if tile_w is None or tile_h is None:
            continue

        # Estimate actual tile count accounting for overlap
        ovl = LATENT_TILE_OVERLAP * 8  # pixel overlap
        stride_w = max(1, tile_w - ovl)
        stride_h = max(1, tile_h - ovl)
        n_w = max(1, math.ceil(max(1, img_w - ovl) / stride_w))
        n_h = max(1, math.ceil(max(1, img_h - ovl) / stride_h))
        total = n_w * n_h

        trt_fp = f"{fp}:{component_type}_trt:{eng['label']}"
        is_cached = gpu.is_component_loaded(trt_fp)

        if is_cached and total < best_cached_tiles:
            best_cached_tile = (tile_w, tile_h)
            best_cached_tiles = total
            best_cached_eng = eng
        elif not is_cached and total < best_disk_tiles:
            best_disk_tile = (tile_w, tile_h)
            best_disk_tiles = total
            best_disk_eng = eng

    # Prefer cached dynamic engine (zero load cost)
    if best_cached_tile is not None:
        log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
                  f"{best_cached_tile[0]}x{best_cached_tile[1]} "
                  f"(cached dynamic TRT '{best_cached_eng['label']}', "
                  f"range {best_cached_eng['min_res'][0]}x{best_cached_eng['min_res'][1]}"
                  f"-{best_cached_eng['max_res'][0]}x{best_cached_eng['max_res'][1]}, "
                  f"~{best_cached_tiles} tiles)")
        return best_cached_tile

    # On-disk dynamic engine
    if best_disk_tile is not None:
        log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
                  f"{best_disk_tile[0]}x{best_disk_tile[1]} "
                  f"(dynamic TRT '{best_disk_eng['label']}', "
                  f"range {best_disk_eng['min_res'][0]}x{best_disk_eng['min_res'][1]}"
                  f"-{best_disk_eng['max_res'][0]}x{best_disk_eng['max_res'][1]}, "
                  f"~{best_disk_tiles} tiles)")
        return best_disk_tile

    # --- Phase 3: PyTorch fallback ---
    result = _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)
    log.debug(f"  SDXL: _pick_vae_tile_size {img_w}x{img_h} → "
              f"{result[0]}x{result[1]} (PyTorch fallback, no TRT engine available)")
    return result


# Tokenizer instances (loaded once, CPU-only)
_tokenizer_1: CLIPTokenizer | None = None
_tokenizer_2: CLIPTokenizer | None = None

# Cache for components extracted from single-file checkpoints.
# Maps checkpoint_path -> {category: model}.  Models are moved between
# CPU and GPU by the worker/pool — the same object reference is shared,
# so eviction (.to("cpu")) keeps the reference alive here for fast reload.
_checkpoint_cache: dict[str, dict[str, object]] = {}
_extraction_lock = threading.Lock()
# NOTE: _extraction_lock is held for the entire from_single_file() call.
# PyTorch's torch.fx.traceback.annotate uses a module-global dict that is
# not thread-safe — concurrent model construction causes KeyError crashes.
# Serializing extraction avoids this without monkey-patching torch internals.

# Prompt emphasis regex — A1111/Forge compatible
_RE_ATTENTION = re.compile(
    r"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[\d.]+)\s*\)|\)|]|[^\\()\[\]:]+|:"
)
# BREAK keyword regex — word-boundary match to avoid matching BREAKFAST, BREAKDOWN, etc.
_RE_BREAK = re.compile(r'\bBREAK\b')


# ====================================================================
# Model Loading
# ====================================================================

def _is_single_file(path: str) -> bool:
    """Check if path is a single-file safetensors checkpoint."""
    return os.path.isfile(path) and path.endswith(".safetensors")


def _ensure_checkpoint_extracted(checkpoint_path: str) -> dict[str, object]:
    """Extract all components from a single-file SDXL checkpoint.

    Thread-safe: extraction is serialized via _extraction_lock because
    PyTorch's torch.fx.traceback uses a module-global dict that crashes
    when two from_single_file() calls run concurrently (KeyError in annotate).

    Components are cached on CPU.  The same model objects are shared with
    the GPU pool cache — when the pool evicts a model (.to("cpu")), the
    reference here still holds, allowing fast reload via .to(device).
    """
    # Fast path — already extracted (no lock needed)
    if checkpoint_path in _checkpoint_cache:
        return _checkpoint_cache[checkpoint_path]

    with _extraction_lock:
        # Double-check under lock
        if checkpoint_path in _checkpoint_cache:
            return _checkpoint_cache[checkpoint_path]

        log.debug(f"  SDXL: Extracting components from "
                 f"{os.path.basename(checkpoint_path)} (this may take a moment)...")

        from diffusers import StableDiffusionXLPipeline

        pipe = StableDiffusionXLPipeline.from_single_file(
            checkpoint_path,
            torch_dtype=torch.float16,
        )

        components: dict[str, object] = {
            "sdxl_te1": pipe.text_encoder,
            "sdxl_te2": pipe.text_encoder_2,
            "sdxl_unet": pipe.unet,
            "sdxl_vae": pipe.vae,
            "tokenizer_1": pipe.tokenizer,
            "tokenizer_2": pipe.tokenizer_2,
        }

        # Convert UNet to channels_last (NHWC) memory format — cuDNN convolution
        # kernels are optimized for NHWC and avoid implicit format transposes.
        components["sdxl_unet"].to(memory_format=torch.channels_last)

        # SDPA (AttnProcessor2_0) is the default — uses F.scaled_dot_product_attention
        # which fuses Q/K/V and dispatches to flash-attention or memory-efficient kernels.
        # Previously forced math-only AttnProcessor due to Triton JIT GIL contention
        # in threaded workers; multiprocessing eliminates that issue.

        # Detect prediction type: safetensors header is the authoritative source
        # (checks v_pred marker tensor, ModelSpec metadata, kohya metadata,
        # companion YAML).  Diffusers' from_single_file() often misdetects
        # v-prediction checkpoints as epsilon.
        from utils.checkpoint import detect_prediction_type
        sched_config = dict(pipe.scheduler.config)
        diffusers_pred = sched_config.get("prediction_type", "epsilon")
        safetensors_pred = detect_prediction_type(checkpoint_path)

        if safetensors_pred is not None:
            pred_type = safetensors_pred
            source = "safetensors header"
            if safetensors_pred != diffusers_pred:
                log.info(f"  SDXL: Overriding diffusers prediction_type={diffusers_pred} "
                         f"→ {safetensors_pred} (from safetensors header)")
        else:
            pred_type = diffusers_pred
            source = "diffusers"

        sched_config["prediction_type"] = pred_type
        components["_scheduler_config"] = sched_config
        log.info(f"  SDXL: Scheduler prediction_type={pred_type} "
                 f"(from {source}: {os.path.basename(checkpoint_path)})")

        # Persist prediction_type to .fpcache so it survives process restarts
        from utils import fingerprint as fp_util
        fp_util.set_extra(checkpoint_path, prediction_type=pred_type)

        # Fix any meta tensors left by from_pretrained/accelerate, then
        # move neural-net components to CPU and set eval mode.
        # Skip None components — some checkpoints don't contain all parts
        # (e.g. missing text_encoder_2). They'll fail at load time instead.
        for key in ("sdxl_te1", "sdxl_te2", "sdxl_unet", "sdxl_vae"):
            if components[key] is None:
                log.warning(f"  SDXL: Component {key} is None in "
                           f"{os.path.basename(checkpoint_path)} — will be unavailable")
                continue
            _fix_from_pretrained(components[key], key)
            components[key].to("cpu")
            components[key].eval()

        del pipe
        torch.cuda.empty_cache()

        _checkpoint_cache[checkpoint_path] = components

        log.debug(f"  SDXL: All components extracted to CPU cache")
        return components


def init_tokenizers(model_path: str) -> None:
    """Load CLIP tokenizers from an SDXL checkpoint (directory or single-file)."""
    global _tokenizer_1, _tokenizer_2
    if _tokenizer_1 is not None:
        return

    if _is_single_file(model_path):
        components = _ensure_checkpoint_extracted(model_path)
        _tokenizer_1 = components["tokenizer_1"]
        _tokenizer_2 = components["tokenizer_2"]
    else:
        _tokenizer_1 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer")
        _tokenizer_2 = CLIPTokenizer.from_pretrained(model_path, subfolder="tokenizer_2")

    log.debug("  SDXL: Tokenizers loaded")


def load_component(category: str, model_dir: str | None, device: torch.device) -> object:
    """Load a single SDXL sub-model component to the given device.

    Supports both diffusers-format directories and single-file .safetensors.
    """
    if model_dir is None:
        raise ValueError(f"model_dir is required to load component {category}")

    if _is_single_file(model_dir):
        return _load_component_from_single_file(category, model_dir, device)

    # --- Diffusers-format directory loading ---
    dtype = torch.float16

    if category == "sdxl_te1":
        model = CLIPTextModel.from_pretrained(
            model_dir, subfolder="text_encoder", torch_dtype=dtype)
        _fix_from_pretrained(model, "text_encoder (CLIP-L)")
        model.to(device)
        model.eval()
        log.debug(f"  SDXL: Loaded text_encoder (CLIP-L) to {device}")
        return model

    elif category == "sdxl_te2":
        model = CLIPTextModelWithProjection.from_pretrained(
            model_dir, subfolder="text_encoder_2", torch_dtype=dtype)
        _fix_from_pretrained(model, "text_encoder_2 (CLIP-bigG)")
        model.to(device)
        model.eval()
        log.debug(f"  SDXL: Loaded text_encoder_2 (CLIP-bigG) to {device}")
        return model

    elif category == "sdxl_unet":
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder="unet", torch_dtype=dtype)
        _fix_from_pretrained(model, "UNet")
        model.to(device, memory_format=torch.channels_last)
        model.eval()
        log.debug(f"  SDXL: Loaded UNet (SDPA) to {device}")
        return model

    elif category in ("sdxl_vae", "sdxl_vae_enc"):
        # VAE must run in float32 — float16 causes NaN overflow in the decoder
        # for certain latent distributions (a well-known SDXL VAE issue).
        model = AutoencoderKL.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=torch.float32)
        _fix_from_pretrained(model, "VAE")
        model.to(device)
        model.eval()
        log.debug(f"  SDXL: Loaded VAE (float32) to {device}")
        return model

    else:
        raise ValueError(f"Unknown SDXL component category: {category}")


def _load_component_from_single_file(
    category: str, checkpoint_path: str, device: torch.device
) -> object:
    """Load a component from a single-file checkpoint via the extraction cache.

    Each GPU gets its own copy of the model to avoid cross-device conflicts
    when multiple GPUs load the same checkpoint.
    """
    import copy

    components = _ensure_checkpoint_extracted(checkpoint_path)

    # vae_enc shares the vae model object
    cache_key = "sdxl_vae" if category == "sdxl_vae_enc" else category
    if cache_key not in components:
        raise ValueError(f"Unknown SDXL component category: {category}")

    if components[cache_key] is None:
        raise RuntimeError(
            f"Component {cache_key} is not available in "
            f"{os.path.basename(checkpoint_path)} (from_single_file returned None)")

    # Deep-copy the CPU-resident model so each GPU gets an independent copy.
    # Without this, .to(device) would move the shared reference and break
    # any other GPU that was using the same model object.
    model = copy.deepcopy(components[cache_key])

    # VAE must run in float32 — float16 causes NaN overflow in the decoder
    # for certain latent distributions (a well-known SDXL VAE issue).
    if category in ("sdxl_vae", "sdxl_vae_enc"):
        model.to(device=device, dtype=torch.float32)
    elif category == "sdxl_unet":
        model.to(device, memory_format=torch.channels_last)
        model.eval()
    else:
        model.to(device)

    log.debug(f"  SDXL: Moved {category} to {device} (from checkpoint cache)")
    return model


# ====================================================================
# Pipeline Stages
# ====================================================================

def tokenize(job: InferenceJob) -> None:
    """Stage 1: CPU-only tokenization with emphasis weights."""
    inp = job.sdxl_input
    if inp is None:
        raise RuntimeError("SdxlInput is required for tokenization.")
    if _tokenizer_1 is None or _tokenizer_2 is None:
        if inp.model_dir:
            init_tokenizers(inp.model_dir)
        else:
            raise RuntimeError("SDXL tokenizers not initialized and no model_dir available.")

    # CLIP-L pads with 49407 (EOS), CLIP-G pads with 0
    CLIP_L_PAD = 49407
    CLIP_G_PAD = 0

    # Check for regional prompting keywords (only if enabled on the request)
    from utils.regional import detect_regional_keywords, parse_regional_prompt
    if inp.regional_prompting and detect_regional_keywords(inp.prompt):
        regional = parse_regional_prompt(inp.prompt, inp.negative_prompt)
        job.regional_info = regional
        log.debug(f"  SDXL: Regional prompting detected — {len(regional.regions)} regions"
                 + (f" + base" if regional.base_prompt else ""))

        # Tokenize each region's prompt + per-region negative
        region_toks: list[SdxlTokenizeResult] = []
        for i, region in enumerate(regional.regions):
            p1_ids, p1_weights, p1_mask = _tokenize_weighted(_tokenizer_1, region.prompt, CLIP_L_PAD)
            n1_ids, n1_weights, n1_mask = _tokenize_weighted(_tokenizer_1, region.negative, CLIP_L_PAD)
            p2_ids, p2_weights, p2_mask = _tokenize_weighted(_tokenizer_2, region.prompt, CLIP_G_PAD)
            n2_ids, n2_weights, n2_mask = _tokenize_weighted(_tokenizer_2, region.negative, CLIP_G_PAD)
            region_toks.append(SdxlTokenizeResult(
                prompt_tokens_1=p1_ids, prompt_weights_1=p1_weights,
                neg_tokens_1=n1_ids, neg_weights_1=n1_weights,
                prompt_tokens_2=p2_ids, prompt_weights_2=p2_weights,
                neg_tokens_2=n2_ids, neg_weights_2=n2_weights,
                prompt_mask_1=p1_mask, neg_mask_1=n1_mask,
                prompt_mask_2=p2_mask, neg_mask_2=n2_mask,
            ))

        # Tokenize the global/shared negative (used for pooled, ADDBASE neg, and fallback)
        gn1_ids, gn1_weights, gn1_mask = _tokenize_weighted(_tokenizer_1, regional.negative_prompt, CLIP_L_PAD)
        gn2_ids, gn2_weights, gn2_mask = _tokenize_weighted(_tokenizer_2, regional.negative_prompt, CLIP_G_PAD)

        # Tokenize base prompt if present
        bp1_ids, bp1_weights, bp1_mask = None, None, None
        bp2_ids, bp2_weights, bp2_mask = None, None, None
        if regional.base_prompt:
            bp1_ids, bp1_weights, bp1_mask = _tokenize_weighted(_tokenizer_1, regional.base_prompt, CLIP_L_PAD)
            bp2_ids, bp2_weights, bp2_mask = _tokenize_weighted(_tokenizer_2, regional.base_prompt, CLIP_G_PAD)

        # Align all chunk counts: all regions + shared negative + base must match per encoder
        max_1 = max(len(gn1_ids), *(len(rt.prompt_tokens_1) for rt in region_toks),
                    *(len(rt.neg_tokens_1) for rt in region_toks))
        max_2 = max(len(gn2_ids), *(len(rt.prompt_tokens_2) for rt in region_toks),
                    *(len(rt.neg_tokens_2) for rt in region_toks))
        if bp1_ids is not None:
            max_1 = max(max_1, len(bp1_ids))
            max_2 = max(max_2, len(bp2_ids))

        _pad_to_chunk_count(gn1_ids, gn1_weights, gn1_mask, max_1, CLIP_L_PAD)
        _pad_to_chunk_count(gn2_ids, gn2_weights, gn2_mask, max_2, CLIP_G_PAD)
        for rt in region_toks:
            _pad_to_chunk_count(rt.prompt_tokens_1, rt.prompt_weights_1, rt.prompt_mask_1, max_1, CLIP_L_PAD)
            _pad_to_chunk_count(rt.neg_tokens_1, rt.neg_weights_1, rt.neg_mask_1, max_1, CLIP_L_PAD)
            _pad_to_chunk_count(rt.prompt_tokens_2, rt.prompt_weights_2, rt.prompt_mask_2, max_2, CLIP_G_PAD)
            _pad_to_chunk_count(rt.neg_tokens_2, rt.neg_weights_2, rt.neg_mask_2, max_2, CLIP_G_PAD)
        if bp1_ids is not None:
            _pad_to_chunk_count(bp1_ids, bp1_weights, bp1_mask, max_1, CLIP_L_PAD)
            _pad_to_chunk_count(bp2_ids, bp2_weights, bp2_mask, max_2, CLIP_G_PAD)

        job.regional_tokenize_results = region_toks

        job.regional_shared_neg_tokenize = SdxlTokenizeResult(
            prompt_tokens_1=[], prompt_weights_1=[],
            neg_tokens_1=gn1_ids, neg_weights_1=gn1_weights,
            prompt_tokens_2=[], prompt_weights_2=[],
            neg_tokens_2=gn2_ids, neg_weights_2=gn2_weights,
            prompt_mask_1=[], neg_mask_1=gn1_mask,
            prompt_mask_2=[], neg_mask_2=gn2_mask,
        )

        if regional.base_prompt:
            job.regional_base_tokenize = SdxlTokenizeResult(
                prompt_tokens_1=bp1_ids, prompt_weights_1=bp1_weights,
                neg_tokens_1=gn1_ids, neg_weights_1=gn1_weights,
                prompt_tokens_2=bp2_ids, prompt_weights_2=bp2_weights,
                neg_tokens_2=gn2_ids, neg_weights_2=gn2_weights,
                prompt_mask_1=bp1_mask, neg_mask_1=gn1_mask,
                prompt_mask_2=bp2_mask, neg_mask_2=gn2_mask,
            )

        # Also tokenize first region as the "main" tokenize_result for non-regional stages
        # (pooled embeddings, etc.)
        first = region_toks[0]
        job.tokenize_result = first
        return

    p1_ids, p1_weights, p1_mask = _tokenize_weighted(_tokenizer_1, inp.prompt, CLIP_L_PAD)
    n1_ids, n1_weights, n1_mask = _tokenize_weighted(_tokenizer_1, inp.negative_prompt, CLIP_L_PAD)
    p2_ids, p2_weights, p2_mask = _tokenize_weighted(_tokenizer_2, inp.prompt, CLIP_G_PAD)
    n2_ids, n2_weights, n2_mask = _tokenize_weighted(_tokenizer_2, inp.negative_prompt, CLIP_G_PAD)

    # Align chunk counts: prompt and negative must have same number per encoder
    max_chunks_1 = max(len(p1_ids), len(n1_ids))
    _pad_to_chunk_count(p1_ids, p1_weights, p1_mask, max_chunks_1, CLIP_L_PAD)
    _pad_to_chunk_count(n1_ids, n1_weights, n1_mask, max_chunks_1, CLIP_L_PAD)
    max_chunks_2 = max(len(p2_ids), len(n2_ids))
    _pad_to_chunk_count(p2_ids, p2_weights, p2_mask, max_chunks_2, CLIP_G_PAD)
    _pad_to_chunk_count(n2_ids, n2_weights, n2_mask, max_chunks_2, CLIP_G_PAD)

    job.tokenize_result = SdxlTokenizeResult(
        prompt_tokens_1=p1_ids, prompt_weights_1=p1_weights,
        neg_tokens_1=n1_ids, neg_weights_1=n1_weights,
        prompt_tokens_2=p2_ids, prompt_weights_2=p2_weights,
        neg_tokens_2=n2_ids, neg_weights_2=n2_weights,
        prompt_mask_1=p1_mask, neg_mask_1=n1_mask,
        prompt_mask_2=p2_mask, neg_mask_2=n2_mask,
    )


def text_encode(job: InferenceJob, gpu: GpuInstance) -> None:
    """Stage 2: GPU text encoding with both CLIP encoders."""
    tok = job.tokenize_result
    if tok is None:
        raise RuntimeError("TokenizeResult is required for text encoding.")
    inp = job.sdxl_input
    if inp is None:
        raise RuntimeError("SdxlInput is required for text encoding.")

    # Branch to regional encoding if regional prompting is active
    if job.regional_info is not None:
        _text_encode_regional(job, gpu)
        return

    import time as _time
    from profiling.tracer import get_current_tracer

    n_chunks = len(tok.prompt_tokens_1)
    n_neg_chunks_1 = len(tok.neg_tokens_1)
    n_neg_chunks_2 = len(tok.neg_tokens_2)
    log.debug(f"  SDXL: Running text encoders ({n_chunks} chunk(s))...")

    _tracer = get_current_tracer()
    _model_name = _get_model_name(job)

    device = gpu.device

    # Check if any LoRA has TE weights — if so, must use PyTorch TEs
    has_te_lora = False
    if inp.loras:
        from state import app_state
        has_te_lora = _any_lora_has_te(inp.loras, app_state.lora_index)

    # Try TRT text encoders first (but not when LoRA has TE weights)
    use_trt = False
    if not has_te_lora:
        trt_te1 = _get_trt_te1_runner(gpu, job)
        trt_te2 = _get_trt_te2_runner(gpu, job)
        use_trt = trt_te1 is not None and trt_te2 is not None

    if use_trt:
        log.debug(f"  SDXL: Using TRT text encoders")
        run_te1 = lambda ids, masks: _run_trt_te1(trt_te1, ids, masks, device)
        run_te2 = lambda ids, masks: _run_trt_te2(trt_te2, ids, masks, device)
    else:
        # Fall back to PyTorch models
        if has_te_lora:
            log.debug(f"  SDXL: Using PyTorch text encoders (LoRA has TE weights)")
        te1 = _get_cached_model_optional(gpu, "sdxl_te1", job)
        te2 = _get_cached_model_optional(gpu, "sdxl_te2", job)
        if te1 is None or te2 is None:
            # PyTorch models not cached (TRT was expected but failed, or
            # LoRA has TE weights and TRT was skipped) — load fallback
            _reason = "LoRA has TE weights" if has_te_lora else "TRT unavailable"
            log.warning(f"  SDXL: {_reason} and TE not cached — loading PyTorch TEs")
            if te1 is None:
                te1 = load_component("sdxl_te1", inp.model_dir, device)
                _te1_fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te1", "sdxl_te1_fallback")
                gpu.cache_model(_te1_fp, "sdxl_te1", te1,
                                estimated_vram=250 * 1024 * 1024, source="PyTorch-fallback")
            if te2 is None:
                te2 = load_component("sdxl_te2", inp.model_dir, device)
                _te2_fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te2", "sdxl_te2_fallback")
                gpu.cache_model(_te2_fp, "sdxl_te2", te2,
                                estimated_vram=1400 * 1024 * 1024, source="PyTorch-fallback")
        # Apply TE LoRA adapters if present
        if has_te_lora:
            _ensure_te_loras(te1, te2, inp.loras, gpu, app_state.lora_index)
        else:
            if hasattr(te1, 'peft_config') and te1.peft_config:
                te1.disable_adapters()
            if hasattr(te2, 'peft_config') and te2.peft_config:
                te2.disable_adapters()
        run_te1 = lambda ids, masks: _run_text_encoder_1(te1, ids, masks, device)
        run_te2 = lambda ids, masks: _run_text_encoder_2(te2, ids, masks, device)

    _te_start = _time.monotonic()

    # TextEncoder1: hidden states [1, 77*N, 768]
    _t = _time.monotonic()
    p_h1 = run_te1(tok.prompt_tokens_1, tok.prompt_mask_1)
    if _tracer:
        _tracer.text_encode(job.job_id, _model_name, "te1", n_chunks, "prompt", _time.monotonic() - _t)
    _t = _time.monotonic()
    n_h1 = run_te1(tok.neg_tokens_1, tok.neg_mask_1)
    if _tracer:
        _tracer.text_encode(job.job_id, _model_name, "te1", n_neg_chunks_1, "negative", _time.monotonic() - _t)

    # TextEncoder2: hidden states [1, 77*N, 1280] + pooled [1, 1280]
    _t = _time.monotonic()
    p_h2, p_pooled = run_te2(tok.prompt_tokens_2, tok.prompt_mask_2)
    if _tracer:
        _tracer.text_encode(job.job_id, _model_name, "te2", n_chunks, "prompt", _time.monotonic() - _t)
    _t = _time.monotonic()
    n_h2, n_pooled = run_te2(tok.neg_tokens_2, tok.neg_mask_2)
    if _tracer:
        _tracer.text_encode(job.job_id, _model_name, "te2", n_neg_chunks_2, "negative", _time.monotonic() - _t)

    _te_elapsed = _time.monotonic() - _te_start
    _backend = "TRT" if use_trt else "PyTorch"
    log.debug(f"  SDXL: Encoder forward passes took {_te_elapsed:.3f}s "
              f"({n_chunks} chunks, 4 passes, {_backend})")

    # Apply emphasis weights to hidden states (not pooled) — flatten chunk weights
    _apply_token_weights(p_h1, _flatten_chunks(tok.prompt_weights_1))
    _apply_token_weights(n_h1, _flatten_chunks(tok.neg_weights_1))
    _apply_token_weights(p_h2, _flatten_chunks(tok.prompt_weights_2))
    _apply_token_weights(n_h2, _flatten_chunks(tok.neg_weights_2))

    # Concatenate hidden states along dim 2: [1,77*N,768]+[1,77*N,1280] = [1,77*N,2048]
    prompt_embeds = torch.cat([p_h1, p_h2], dim=2)
    neg_prompt_embeds = torch.cat([n_h1, n_h2], dim=2)

    # Pooled comes from TE2 only (NOT affected by emphasis)
    pooled_prompt_embeds = p_pooled
    neg_pooled_prompt_embeds = n_pooled

    # Forge: zero out all negative embeddings when negative prompt is empty
    if not inp.negative_prompt or not inp.negative_prompt.strip():
        log.debug("  SDXL: Empty negative prompt — zeroing embeddings (Forge behavior)")
        neg_prompt_embeds.zero_()
        neg_pooled_prompt_embeds.zero_()

    log.debug(f"  SDXL: Text encoding complete. embeds={list(prompt_embeds.shape)} "
             f"pooled={list(pooled_prompt_embeds.shape)}")

    job.encode_result = SdxlEncodeResult(
        prompt_embeds=prompt_embeds,
        neg_prompt_embeds=neg_prompt_embeds,
        pooled_prompt_embeds=pooled_prompt_embeds,
        neg_pooled_prompt_embeds=neg_pooled_prompt_embeds,
    )


def _text_encode_regional(job: InferenceJob, gpu: GpuInstance) -> None:
    """Encode each region's prompt separately through both CLIP encoders."""
    regional = job.regional_info
    region_toks = job.regional_tokenize_results
    inp = job.sdxl_input

    log.debug(f"  SDXL: Running regional text encoders ({len(region_toks)} regions)...")

    device = gpu.device

    # Check if any LoRA has TE weights — if so, must use PyTorch TEs
    has_te_lora = False
    if inp.loras:
        from state import app_state
        has_te_lora = _any_lora_has_te(inp.loras, app_state.lora_index)

    # Try TRT text encoders first (but not when LoRA has TE weights)
    use_trt = False
    if not has_te_lora:
        trt_te1 = _get_trt_te1_runner(gpu, job)
        trt_te2 = _get_trt_te2_runner(gpu, job)
        use_trt = trt_te1 is not None and trt_te2 is not None

    if use_trt:
        log.debug(f"  SDXL: Using TRT text encoders (regional)")
        run_te1 = lambda ids, masks: _run_trt_te1(trt_te1, ids, masks, device)
        run_te2 = lambda ids, masks: _run_trt_te2(trt_te2, ids, masks, device)
    else:
        if has_te_lora:
            log.debug(f"  SDXL: Using PyTorch text encoders (regional, LoRA has TE weights)")
        te1 = _get_cached_model_optional(gpu, "sdxl_te1", job)
        te2 = _get_cached_model_optional(gpu, "sdxl_te2", job)
        if te1 is None or te2 is None:
            _reason = "LoRA has TE weights" if has_te_lora else "TRT unavailable"
            log.warning(f"  SDXL: {_reason} and TE not cached — loading PyTorch TEs (regional)")
            if te1 is None:
                te1 = load_component("sdxl_te1", inp.model_dir, device)
                _te1_fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te1", "sdxl_te1_fallback")
                gpu.cache_model(_te1_fp, "sdxl_te1", te1,
                                estimated_vram=250 * 1024 * 1024, source="PyTorch-fallback")
            if te2 is None:
                te2 = load_component("sdxl_te2", inp.model_dir, device)
                _te2_fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te2", "sdxl_te2_fallback")
                gpu.cache_model(_te2_fp, "sdxl_te2", te2,
                                estimated_vram=1400 * 1024 * 1024, source="PyTorch-fallback")
        if has_te_lora:
            _ensure_te_loras(te1, te2, inp.loras, gpu, app_state.lora_index)
        else:
            if hasattr(te1, 'peft_config') and te1.peft_config:
                te1.disable_adapters()
            if hasattr(te2, 'peft_config') and te2.peft_config:
                te2.disable_adapters()
        run_te1 = lambda ids, masks: _run_text_encoder_1(te1, ids, masks, device)
        run_te2 = lambda ids, masks: _run_text_encoder_2(te2, ids, masks, device)

    region_embeds: list[torch.Tensor] = []
    first_pooled = None

    for i, tok in enumerate(region_toks):
        # Encode this region's prompt
        p_h1 = run_te1(tok.prompt_tokens_1, tok.prompt_mask_1)
        p_h2, p_pooled = run_te2(tok.prompt_tokens_2, tok.prompt_mask_2)

        _apply_token_weights(p_h1, _flatten_chunks(tok.prompt_weights_1))
        _apply_token_weights(p_h2, _flatten_chunks(tok.prompt_weights_2))

        embeds = torch.cat([p_h1, p_h2], dim=2)  # [1, 77*N, 2048]
        region_embeds.append(embeds)

        if i == 0:
            first_pooled = p_pooled

    # Always encode the global/shared negative (for pooled, ADDBASE neg, and non-regional fallback)
    shared_neg_tok = job.regional_shared_neg_tokenize
    if shared_neg_tok is None:
        raise RuntimeError("regional_shared_neg_tokenize is required for regional text encoding.")
    n_h1 = run_te1(shared_neg_tok.neg_tokens_1, shared_neg_tok.neg_mask_1)
    n_h2, n_pooled = run_te2(shared_neg_tok.neg_tokens_2, shared_neg_tok.neg_mask_2)
    _apply_token_weights(n_h1, _flatten_chunks(shared_neg_tok.neg_weights_1))
    _apply_token_weights(n_h2, _flatten_chunks(shared_neg_tok.neg_weights_2))
    neg_embeds = torch.cat([n_h1, n_h2], dim=2)

    # Encode per-region negatives if they differ across regions
    neg_region_embeds: list[torch.Tensor] | None = None
    if regional.has_per_region_neg:
        neg_region_embeds = []
        for tok in region_toks:
            nr_h1 = run_te1(tok.neg_tokens_1, tok.neg_mask_1)
            nr_h2, _ = run_te2(tok.neg_tokens_2, tok.neg_mask_2)
            _apply_token_weights(nr_h1, _flatten_chunks(tok.neg_weights_1))
            _apply_token_weights(nr_h2, _flatten_chunks(tok.neg_weights_2))
            neg_region_embeds.append(torch.cat([nr_h1, nr_h2], dim=2))
        log.debug(f"  SDXL: Encoded {len(neg_region_embeds)} per-region negatives")

    # Encode base prompt if present
    base_embeds = None
    base_pooled = None
    if regional.base_prompt and job.regional_base_tokenize is not None:
        bt = job.regional_base_tokenize
        b_h1 = run_te1(bt.prompt_tokens_1, bt.prompt_mask_1)
        b_h2, b_pooled = run_te2(bt.prompt_tokens_2, bt.prompt_mask_2)
        _apply_token_weights(b_h1, _flatten_chunks(bt.prompt_weights_1))
        _apply_token_weights(b_h2, _flatten_chunks(bt.prompt_weights_2))
        base_embeds = torch.cat([b_h1, b_h2], dim=2)
        base_pooled = b_pooled

    # Pooled: use base prompt's pooled if ADDBASE, otherwise first region's
    pooled = base_pooled if base_pooled is not None else first_pooled

    # Forge: zero out negative embeddings when negative prompt is empty
    neg_text = regional.negative_prompt
    if not neg_text or not neg_text.strip():
        log.debug("  SDXL: Empty negative prompt — zeroing embeddings (Forge behavior)")
        neg_embeds.zero_()
        n_pooled.zero_()
        if neg_region_embeds is not None:
            for nre in neg_region_embeds:
                nre.zero_()

    log.debug(f"  SDXL: Regional text encoding complete. "
             f"{len(region_embeds)} region embeds"
             f"{', per-region negatives' if neg_region_embeds else ''}"
             f", pooled={list(pooled.shape)}")

    job.regional_encode_result = SdxlRegionalEncodeResult(
        region_embeds=region_embeds,
        neg_prompt_embeds=neg_embeds,
        neg_region_embeds=neg_region_embeds,
        pooled_prompt_embeds=pooled,
        neg_pooled_prompt_embeds=n_pooled,
        base_embeds=base_embeds,
        base_ratio=regional.base_ratio,
    )


_progress_callback = None


def set_progress_callback(cb) -> None:
    """Set a callback for broadcasting denoise/stage progress.

    Called by the worker subprocess to redirect progress updates
    through IPC instead of the in-process WebSocket streamer.
    Signature: cb(job: InferenceJob) -> None
    """
    global _progress_callback
    _progress_callback = cb


def _broadcast_ws_progress(job: InferenceJob) -> None:
    """Broadcast denoise/stage progress.

    In the worker subprocess, this sends ProgressUpdate via IPC.
    Falls back to direct WebSocket broadcast for in-process execution.
    """
    try:
        if _progress_callback is not None:
            _progress_callback(job)
        else:
            from api.websocket import streamer as _ws_streamer
            import asyncio as _asyncio
            _asyncio.run_coroutine_threadsafe(_ws_streamer.broadcast_progress(job), job._loop)
    except Exception:
        pass


def _get_prediction_type(model_dir: str) -> str:
    """Detect prediction_type for an SDXL checkpoint (epsilon or v_prediction).

    Checks (in order):
    1. Checkpoint extraction cache (_scheduler_config captured from pipe)
    2. Safetensors header (v_pred marker tensor, ModelSpec metadata)
    3. Diffusers-format scheduler_config.json
    4. .fpcache metadata (persisted from previous extractions)
    5. Default: "epsilon"
    """
    # 1. Extraction cache (single-file checkpoints — already corrected by safetensors check)
    if model_dir in _checkpoint_cache:
        sched = _checkpoint_cache[model_dir].get("_scheduler_config")
        if sched:
            return sched.get("prediction_type", "epsilon")

    # 2. Safetensors header (most reliable for single-file checkpoints)
    if _is_single_file(model_dir):
        from utils.checkpoint import detect_prediction_type
        safetensors_pred = detect_prediction_type(model_dir)
        if safetensors_pred is not None:
            return safetensors_pred

    # 3. Diffusers-format directory
    import json
    sched_file = os.path.join(model_dir, "scheduler", "scheduler_config.json")
    if os.path.isfile(sched_file):
        try:
            with open(sched_file) as f:
                config = json.load(f)
            return config.get("prediction_type", "epsilon")
        except (OSError, json.JSONDecodeError):
            pass

    # 4. .fpcache metadata (persisted from previous extraction)
    from utils import fingerprint as fp_util
    cached = fp_util.get_extra(model_dir, "prediction_type")
    if cached:
        return cached

    return "epsilon"


def _create_scheduler(
    model_dir: str, steps: int, device: torch.device,
) -> EulerAncestralDiscreteScheduler:
    """Create an EulerAncestralDiscreteScheduler with the correct prediction_type.

    For v-prediction models:
      - prediction_type="v_prediction"
      - rescale_betas_zero_snr=True (zero terminal SNR)
      - timestep_spacing="trailing" (matches training schedule)

    For epsilon models (default):
      - prediction_type="epsilon"
      - timestep_spacing="linspace"
    """
    prediction_type = _get_prediction_type(model_dir)

    kwargs = {
        "beta_start": 0.00085,
        "beta_end": 0.012,
        "beta_schedule": "scaled_linear",
        "num_train_timesteps": 1000,
        "prediction_type": prediction_type,
    }

    if prediction_type == "v_prediction":
        kwargs["rescale_betas_zero_snr"] = True
        kwargs["timestep_spacing"] = "trailing"
        log.debug(f"  SDXL: Using v-prediction scheduler (zero-SNR, trailing spacing)")
    else:
        kwargs["timestep_spacing"] = "linspace"

    scheduler = EulerAncestralDiscreteScheduler(**kwargs)
    scheduler.set_timesteps(steps, device=device)
    return scheduler


def denoise(job: InferenceJob, gpu: GpuInstance) -> None:
    """Stage 3: GPU denoising with UNet + EulerAncestral scheduler."""
    import time as _time
    from profiling.tracer import get_current_tracer
    _tracer = get_current_tracer()
    _model_name = _get_model_name(job)

    inp = job.sdxl_input
    if inp is None:
        raise RuntimeError("SdxlInput is required for denoising.")

    # Branch to regional denoising if regional encoding was performed
    if job.regional_encode_result is not None:
        _denoise_regional(job, gpu)
        return

    enc = job.encode_result
    if enc is None:
        raise RuntimeError("EncodeResult is required for denoising.")

    is_hires = job.is_hires_pass and job.hires_input is not None
    hires = job.hires_input

    width = hires.hires_width if is_hires else inp.width
    height = hires.hires_height if is_hires else inp.height

    if is_hires:
        # hires_steps = desired active denoising steps.
        # Scale total scheduler steps so that int(total * strength) ≈ hires_steps.
        _active = hires.hires_steps
        _str = hires.denoising_strength
        steps = max(round(_active / max(_str, 0.001)), _active)
    else:
        steps = inp.steps

    # Different seed for hires to avoid correlated noise
    scheduler_seed = (inp.seed ^ 0x9E3779B9) if is_hires else inp.seed

    _setup_start = _time.monotonic()
    device = gpu.device

    # TRT path: use pre-compiled TensorRT engine when available (no LoRA)
    trt_runner = None
    if not inp.loras:
        trt_runner = _get_trt_unet_runner(gpu, job, width, height)

    if trt_runner is not None:
        unet = None  # not needed — TRT handles forward pass
        log.debug(f"  SDXL: Using TRT UNet for {width}x{height}")
    else:
        unet = _get_cached_model_optional(gpu, "sdxl_unet", job)
        if unet is None:
            # PyTorch model wasn't pre-loaded (TRT was expected to cover this
            # resolution but the engine failed to load).  Load PyTorch now as
            # a fallback — same cost as the old always-preload path.
            log.warning(f"  SDXL: TRT unavailable and UNet not cached — loading PyTorch fallback")
            unet = load_component("sdxl_unet", inp.model_dir, gpu.device)
            gpu.cache_model(
                getattr(job, '_stage_model_fps', {}).get("sdxl_unet", "sdxl_unet_fallback"),
                "sdxl_unet", unet,
                estimated_vram=5200 * 1024 * 1024,
                source="PyTorch-fallback",
            )

        # Apply LoRA adapters if requested
        if inp.loras:
            from state import app_state
            _ensure_loras(unet, inp.loras, gpu, app_state.lora_index)
        elif hasattr(unet, 'peft_config') and unet.peft_config:
            unet.disable_adapters()  # clean UNet for non-LoRA jobs

    _setup_elapsed = _time.monotonic() - _setup_start
    log.debug(f"  SDXL: Denoise setup (model+LoRA) took {_setup_elapsed:.3f}s")

    if _tracer:
        _has_lora = bool(inp.loras)
        _lora_name = inp.loras[0].name if inp.loras else None
        _tracer.denoise_setup(job.job_id, _model_name, _setup_elapsed, _has_lora, _lora_name)

    # Create scheduler with correct prediction_type for this model
    scheduler = _create_scheduler(inp.model_dir, steps, device)
    timesteps = scheduler.timesteps

    latent_h = height // 8
    latent_w = width // 8

    generator = torch.Generator(device=device).manual_seed(scheduler_seed)

    if is_hires:
        # Hires pass: start from upscaled latents with noise
        upscaled_latents = job.latents
        if upscaled_latents is None:
            raise RuntimeError("Latents are required for hires denoising pass.")

        strength = hires.denoising_strength
        # _active was computed above as the desired active step count
        init_timestep = min(_active, steps)
        start_step = max(steps - init_timestep, 0)

        if start_step >= len(timesteps):
            log.debug(f"  SDXL: Hires denoising skipped (strength={strength:.2f}, no active steps)")
            return

        # Add noise at the start timestep
        upscaled_latents = upscaled_latents.to(device=device, dtype=torch.float16)
        noise = torch.randn(upscaled_latents.shape, generator=generator,
                            device=device, dtype=upscaled_latents.dtype)
        latents = scheduler.add_noise(upscaled_latents, noise,
                                      timesteps[start_step:start_step+1])

        active_count = len(timesteps) - start_step
        log.debug(f"  SDXL: Hires denoising (strength={strength:.2f}, {active_count} active steps "
                 f"of {len(timesteps)} total, startStep={start_step})")
    else:
        # Base pass: start from pure noise
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator, device=device, dtype=torch.float16,
        ) * scheduler.init_noise_sigma
        start_step = 0

        log.debug(f"  SDXL: Denoising ({len(timesteps)} steps, "
                 f"latent=[1,4,{latent_h},{latent_w}])...")

    # Time conditioning: [orig_h, orig_w, crop_top, crop_left, target_h, target_w]
    orig_h = inp.height if is_hires else height
    orig_w = inp.width if is_hires else width
    add_time_ids = torch.tensor(
        [[orig_h, orig_w, 0, 0, height, width]],
        dtype=torch.float16, device=device,
    )

    active_step_count = len(timesteps) - start_step
    job.denoise_step = 0
    job.denoise_total_steps = active_step_count

    prompt_embeds = enc.prompt_embeds.to(device=device, dtype=torch.float16)
    neg_prompt_embeds = enc.neg_prompt_embeds.to(device=device, dtype=torch.float16)
    pooled_prompt_embeds = enc.pooled_prompt_embeds.to(device=device, dtype=torch.float16)
    neg_pooled_prompt_embeds = enc.neg_pooled_prompt_embeds.to(device=device, dtype=torch.float16)

    # Pre-compute static CFG tensors (these don't change between steps)
    batched_embeds = torch.cat([neg_prompt_embeds, prompt_embeds])
    batched_pooled = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
    batched_time_ids = torch.cat([add_time_ids, add_time_ids])

    _loop_start = _time.monotonic()
    _first_step_time = None

    for i in range(start_step, len(timesteps)):
        _step_start = _time.monotonic()
        job.denoise_step = i - start_step + 1
        _broadcast_ws_progress(job)
        t = timesteps[i]

        latent_input = scheduler.scale_model_input(latents, t)

        if trt_runner is not None:
            # TRT path — direct tensor I/O, no Python overhead, GIL-free
            latent_in = torch.cat([latent_input, latent_input])
            out = trt_runner.run(latent_in, t.unsqueeze(0), batched_embeds,
                                 batched_pooled, batched_time_ids)
            noise_pred_uncond, noise_pred_cond = out.chunk(2)
            noise_pred = noise_pred_uncond + inp.cfg_scale * (noise_pred_cond - noise_pred_uncond)
        else:
            with torch.no_grad():
                # Batched CFG: run uncond + cond in a single UNet forward pass
                latent_in = torch.cat([latent_input, latent_input])
                out = unet(
                    latent_in, t,
                    encoder_hidden_states=batched_embeds,
                    added_cond_kwargs={
                        "text_embeds": batched_pooled,
                        "time_ids": batched_time_ids,
                    },
                ).sample
                noise_pred_uncond, noise_pred_cond = out.chunk(2)
            noise_pred = noise_pred_uncond + inp.cfg_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

        _step_dur = _time.monotonic() - _step_start
        if _tracer:
            _tracer.denoise_step(
                job.job_id, _model_name, i - start_step + 1,
                active_step_count, int(t), _step_dur, False)

        if _first_step_time is None:
            _first_step_time = _time.monotonic() - _loop_start

    _loop_elapsed = _time.monotonic() - _loop_start
    _per_step = _loop_elapsed / max(active_step_count, 1)
    log.debug(f"  SDXL: Denoise loop took {_loop_elapsed:.3f}s "
              f"({active_step_count} steps, {_per_step:.3f}s/step, "
              f"first={_first_step_time:.3f}s)")

    # Diagnostic: check latent health after denoising
    lat_min = latents.min().item()
    lat_max = latents.max().item()
    lat_has_nan = torch.isnan(latents).any().item()
    if lat_has_nan or (lat_min == 0 and lat_max == 0):
        log.warning(f"  SDXL: Denoise — BAD latents after denoising: "
                    f"min={lat_min:.4f} max={lat_max:.4f} has_nan={lat_has_nan}")
    else:
        log.debug(f"  SDXL: Denoise complete. latent_range=[{lat_min:.4f}, {lat_max:.4f}]")

    job.latents = latents


def _denoise_regional(job: InferenceJob, gpu: GpuInstance) -> None:
    """Regional denoising: custom attention processors composite per-region text conditioning."""
    from utils.regional import build_region_masks
    from handlers.regional_attn import RegionalAttnState, install_regional_processors

    inp = job.sdxl_input
    rer = job.regional_encode_result
    regional = job.regional_info

    is_hires = job.is_hires_pass and job.hires_input is not None
    hires = job.hires_input

    width = hires.hires_width if is_hires else inp.width
    height = hires.hires_height if is_hires else inp.height

    if is_hires:
        _active = hires.hires_steps
        _str = hires.denoising_strength
        steps = max(round(_active / max(_str, 0.001)), _active)
    else:
        steps = inp.steps

    scheduler_seed = (inp.seed ^ 0x9E3779B9) if is_hires else inp.seed

    unet = _get_cached_model(gpu, "sdxl_unet", job)
    device = gpu.device

    # Apply LoRA adapters if requested
    if inp.loras:
        from state import app_state
        _ensure_loras(unet, inp.loras, gpu, app_state.lora_index)
    elif hasattr(unet, 'peft_config') and unet.peft_config:
        unet.disable_adapters()

    # Create scheduler with correct prediction_type for this model
    scheduler = _create_scheduler(inp.model_dir, steps, device)
    timesteps = scheduler.timesteps

    latent_h = height // 8
    latent_w = width // 8

    generator = torch.Generator(device=device).manual_seed(scheduler_seed)

    if is_hires:
        upscaled_latents = job.latents
        if upscaled_latents is None:
            raise RuntimeError("Latents are required for hires denoising pass.")

        strength = hires.denoising_strength
        init_timestep = min(_active, steps)
        start_step = max(steps - init_timestep, 0)

        if start_step >= len(timesteps):
            log.debug(f"  SDXL: Regional hires denoising skipped (strength={strength:.2f}, no active steps)")
            return

        upscaled_latents = upscaled_latents.to(device=device, dtype=torch.float16)
        noise = torch.randn(upscaled_latents.shape, generator=generator,
                            device=device, dtype=upscaled_latents.dtype)
        latents = scheduler.add_noise(upscaled_latents, noise,
                                      timesteps[start_step:start_step+1])

        active_count = len(timesteps) - start_step
        log.debug(f"  SDXL: Regional hires denoising (strength={strength:.2f}, {active_count} active steps "
                 f"of {len(timesteps)} total, startStep={start_step})")
    else:
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator, device=device, dtype=torch.float16,
        ) * scheduler.init_noise_sigma
        start_step = 0

        log.debug(f"  SDXL: Regional denoising ({len(timesteps)} steps, "
                 f"{len(regional.regions)} regions, latent=[1,4,{latent_h},{latent_w}])...")

    # Time conditioning
    orig_h = inp.height if is_hires else height
    orig_w = inp.width if is_hires else width
    add_time_ids = torch.tensor(
        [[orig_h, orig_w, 0, 0, height, width]],
        dtype=torch.float16, device=device,
    )

    # Build region masks
    region_masks = build_region_masks(
        regional.regions, latent_h, latent_w, device, torch.float16,
    )

    # Stack region embeddings: [N, 77*C, 2048]
    all_embeds = torch.cat(rer.region_embeds, dim=0).to(device=device, dtype=torch.float16)

    # Build per-region negative stack if negatives differ across regions
    neg_stacked: torch.Tensor | None = None
    if rer.neg_region_embeds is not None:
        neg_stacked = torch.cat(rer.neg_region_embeds, dim=0).to(device=device, dtype=torch.float16)

    # If ADDBASE: prepend base_embeds and full-coverage mask, then renormalize
    if rer.base_embeds is not None:
        base_ratio = rer.base_ratio
        base_mask = torch.ones(1, 1, latent_h, latent_w, device=device, dtype=torch.float16) * base_ratio
        region_masks = region_masks * (1 - base_ratio)
        region_masks = torch.cat([base_mask, region_masks], dim=0)
        all_embeds = torch.cat([rer.base_embeds.to(device=device, dtype=torch.float16), all_embeds], dim=0)
        # For per-region negatives, prepend shared neg as the base region's negative
        if neg_stacked is not None:
            neg_stacked = torch.cat([rer.neg_prompt_embeds.to(device=device, dtype=torch.float16), neg_stacked], dim=0)
        # Renormalize combined masks to sum to 1.0 at every spatial position
        mask_sum = region_masks.sum(dim=0, keepdim=True).clamp(min=1e-8)
        region_masks = region_masks / mask_sum

    # Create attention state and install processors
    state = RegionalAttnState()
    state.set_regions(all_embeds, region_masks)
    original_procs = install_regional_processors(unet, state)

    neg_prompt_embeds = rer.neg_prompt_embeds.to(device=device, dtype=torch.float16)
    pooled_prompt_embeds = rer.pooled_prompt_embeds.to(device=device, dtype=torch.float16)
    neg_pooled_prompt_embeds = rer.neg_pooled_prompt_embeds.to(device=device, dtype=torch.float16)

    # Use first region's embeds as placeholder for encoder_hidden_states
    placeholder_embeds = all_embeds[0:1]

    active_step_count = len(timesteps) - start_step
    job.denoise_step = 0
    job.denoise_total_steps = active_step_count

    # Pre-compute static CFG tensors for batched regional denoise
    uncond_enc = neg_stacked[0:1] if neg_stacked is not None else neg_prompt_embeds
    batched_regional_embeds = torch.cat([uncond_enc, placeholder_embeds])
    batched_regional_pooled = torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds])
    batched_regional_time_ids = torch.cat([add_time_ids, add_time_ids])

    try:
        for i in range(start_step, len(timesteps)):
            job.denoise_step = i - start_step + 1
            _broadcast_ws_progress(job)
            t = timesteps[i]

            latent_input = scheduler.scale_model_input(latents, t)

            # Batched CFG: uncond + cond in a single UNet forward pass.
            # The RegionalAttnProcessor splits the batch internally —
            # cond gets regional attention, uncond gets regional (per-region
            # negs) or standard SDPA (shared neg).
            state.region_embeds = all_embeds
            state.active = True
            if neg_stacked is not None:
                state.uncond_region_embeds = neg_stacked
                state.uncond_text_embeds = None
            else:
                state.uncond_region_embeds = None
                state.uncond_text_embeds = neg_prompt_embeds

            with torch.no_grad():
                latent_in = torch.cat([latent_input, latent_input])
                out = unet(
                    latent_in, t,
                    encoder_hidden_states=batched_regional_embeds,
                    added_cond_kwargs={
                        "text_embeds": batched_regional_pooled,
                        "time_ids": batched_regional_time_ids,
                    },
                ).sample
                noise_pred_uncond, noise_pred_cond = out.chunk(2)

            noise_pred = noise_pred_uncond + inp.cfg_scale * (noise_pred_cond - noise_pred_uncond)

            latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample
    finally:
        # Restore original attention processors and clear batched CFG state
        state.active = False
        state.uncond_region_embeds = None
        state.uncond_text_embeds = None
        unet.set_attn_processor(original_procs)

    # Diagnostic: check latent health after regional denoising
    lat_min = latents.min().item()
    lat_max = latents.max().item()
    lat_has_nan = torch.isnan(latents).any().item()
    if lat_has_nan or (lat_min == 0 and lat_max == 0):
        log.warning(f"  SDXL: Regional denoise — BAD latents: "
                    f"min={lat_min:.4f} max={lat_max:.4f} has_nan={lat_has_nan}")
    else:
        log.debug(f"  SDXL: Regional denoise complete. latent_range=[{lat_min:.4f}, {lat_max:.4f}]")

    job.latents = latents


def vae_decode(job: InferenceJob, gpu: GpuInstance) -> Image.Image:
    """Stage 4: GPU VAE decoding. Returns decoded PIL Image."""
    latents = job.latents
    if latents is None:
        raise RuntimeError("Latents are required for VAE decoding.")

    device = gpu.device
    vae = None  # loaded below only when TRT is unavailable

    # Determine image dimensions from latent shape
    lat_h = latents.shape[2]
    lat_w = latents.shape[3]
    img_w = lat_w * 8
    img_h = lat_h * 8

    # Handle explicit user overrides first
    has_user_override = job.vae_tile_width > 0 and job.vae_tile_height > 0
    if has_user_override:
        tile_w = (job.vae_tile_width  // 8) * 8
        tile_h = (job.vae_tile_height // 8) * 8
    else:
        tile_w, tile_h = img_w, img_h  # default: full image

    # Try TRT for full image first (may bypass tiling for 1280x1280, 1280x1536)
    log.debug(f"  SDXL: Selecting VAE for {img_w}x{img_h} decode")
    trt_vae = None if has_user_override else _get_trt_vae_runner(gpu, job, img_w, img_h)

    if trt_vae is not None:
        # TRT handles full image — skip tiling even if >= threshold
        tile_w, tile_h = img_w, img_h
        log.debug(f"  SDXL: TRT VAE covers full {img_w}x{img_h}")
    elif not has_user_override:
        # No explicit override and no full-image TRT — check if tiling needed
        if img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
            tile_w, tile_h = _pick_vae_tile_size(img_w, img_h, gpu, job)

    lat_tile_w = tile_w // 8
    lat_tile_h = tile_h // 8
    use_tiled = lat_w > lat_tile_w or lat_h > lat_tile_h

    # Try TRT for tile size (if tiling and no full-image TRT)
    if use_tiled and trt_vae is None:
        trt_vae = _get_trt_vae_runner(gpu, job, tile_w, tile_h)
        log.debug(f"  SDXL: Tiled VAE decode {img_w}x{img_h} → tile {tile_w}x{tile_h}, "
                  f"backend={'TRT' if trt_vae else 'PyTorch'}")

    # Prepare latents (dtype depends on TRT vs PyTorch)
    if trt_vae is not None:
        scaled_latents = latents.to(device=device, dtype=torch.float32) / VAE_SCALE_FACTOR
        log.debug(f"  SDXL: Using TRT VAE for {img_w}x{img_h}"
                  f"{f' (tiled {tile_w}x{tile_h})' if use_tiled else ''}")
    else:
        vae = _get_cached_model_optional(gpu, "sdxl_vae", job)
        if vae is None:
            # PyTorch VAE wasn't pre-loaded (TRT was expected but unavailable).
            log.warning(f"  SDXL: TRT VAE unavailable and VAE not cached — loading PyTorch fallback")
            inp = job.sdxl_input
            vae = load_component("sdxl_vae", inp.model_dir if inp else None, device)
            fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae", "sdxl_vae_fallback")
            gpu.cache_model(fp, "sdxl_vae", vae, estimated_vram=400 * 1024 * 1024,
                            source="PyTorch-fallback")
        scaled_latents = latents.to(device=device, dtype=vae.dtype) / VAE_SCALE_FACTOR

    # Diagnostic: check latent stats before decode
    lat_min = scaled_latents.min().item()
    lat_max = scaled_latents.max().item()
    lat_has_nan = torch.isnan(scaled_latents).any().item()
    if lat_has_nan or (lat_min == 0 and lat_max == 0):
        log.warning(f"  SDXL: VAE decode — BAD latents before decode: "
                    f"min={lat_min:.4f} max={lat_max:.4f} has_nan={lat_has_nan}")

    # Execute decode
    if use_tiled:
        image = _vae_decode_tiled(scaled_latents, vae, lat_tile_w, lat_tile_h,
                                  job=job, trt_runner=trt_vae)
    elif trt_vae is not None:
        # TRT VAE decode — GIL-free
        decoded = trt_vae.run(scaled_latents)

        # Diagnostic: check decoded tensor stats
        dec_min = decoded.min().item()
        dec_max = decoded.max().item()
        dec_has_nan = torch.isnan(decoded).any().item()
        dec_mean = decoded.mean().item()
        if dec_has_nan or dec_max - dec_min < 0.01:
            log.warning(f"  SDXL: VAE decode (TRT) — BAD decoded tensor: "
                        f"min={dec_min:.4f} max={dec_max:.4f} mean={dec_mean:.4f} "
                        f"has_nan={dec_has_nan}")

        image = _tensor_to_pil(decoded)
    else:
        with torch.no_grad():
            decoded = vae.decode(scaled_latents.to(device)).sample

        # Diagnostic: check decoded tensor stats
        dec_min = decoded.min().item()
        dec_max = decoded.max().item()
        dec_has_nan = torch.isnan(decoded).any().item()
        dec_mean = decoded.mean().item()
        if dec_has_nan or dec_max - dec_min < 0.01:
            log.warning(f"  SDXL: VAE decode — BAD decoded tensor: "
                        f"min={dec_min:.4f} max={dec_max:.4f} mean={dec_mean:.4f} "
                        f"has_nan={dec_has_nan}")

        image = _tensor_to_pil(decoded)

    # Detect suspiciously dark output
    arr = np.array(image)
    mean_pixel = arr.mean()
    if mean_pixel < 5.0:
        log.warning(f"  SDXL: VAE decode — BLACK IMAGE detected! "
                    f"mean_pixel={mean_pixel:.2f} "
                    f"latent_range=[{lat_min:.4f}, {lat_max:.4f}] "
                    f"has_nan={lat_has_nan}")

    log.debug(f"  SDXL: VAE decode complete. Image={image.width}x{image.height} "
             f"mean_px={mean_pixel:.1f}")
    return image


def vae_encode(job: InferenceJob, gpu: GpuInstance) -> None:
    """Stage: GPU VAE encode. Encodes job.input_image → job.latents.

    TRT-first pattern mirrors vae_decode():
    1. Try TRT encoder for full image
    2. If not, check tiling threshold with TRT-aware tile selection
    3. Fall back to PyTorch if no TRT coverage
    """
    if job.input_image is None:
        raise RuntimeError("input_image is required for VAE encode.")

    device = gpu.device
    img_w, img_h = job.input_image.size

    # Try TRT encoder for full image first
    trt_enc = _get_trt_vae_enc_runner(gpu, job, img_w, img_h)

    # Handle explicit user overrides first
    has_user_override = job.vae_tile_width > 0 and job.vae_tile_height > 0
    if has_user_override:
        tile_w = (job.vae_tile_width // 8) * 8
        tile_h = (job.vae_tile_height // 8) * 8
        if trt_enc is None or img_w > tile_w or img_h > tile_h:
            # User override forces tiling — try TRT for tile size
            trt_enc = _get_trt_vae_enc_runner(gpu, job, tile_w, tile_h)
    elif trt_enc is not None:
        # TRT covers full image — no tiling needed
        tile_w, tile_h = img_w, img_h
    elif img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
        # Use TRT-aware tile selection for encoder
        tile_w, tile_h = _pick_vae_tile_size(img_w, img_h, gpu, job, component_type="vae_enc")
        trt_enc = _get_trt_vae_enc_runner(gpu, job, tile_w, tile_h)
    else:
        tile_w, tile_h = img_w, img_h

    # Only load PyTorch VAE if TRT is not available
    vae = None
    if trt_enc is None:
        vae = _get_cached_model_optional(gpu, "sdxl_vae", job)
        if vae is None:
            log.warning(f"  SDXL: VAE not cached for encode — loading PyTorch fallback")
            inp = job.sdxl_input
            vae = load_component("sdxl_vae", inp.model_dir if inp else None, device)
            fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae", "sdxl_vae_fallback")
            gpu.cache_model(fp, "sdxl_vae", vae, estimated_vram=400 * 1024 * 1024,
                            source="PyTorch-fallback")

    if img_w > tile_w or img_h > tile_h:
        log.debug(f"  SDXL: Tiled VAE encode {img_w}x{img_h} → tile {tile_w}x{tile_h}"
                  f" ({'TRT' if trt_enc else 'PyTorch'})")
    latents = _vae_encode(job.input_image, vae, device, tile_w=tile_w, tile_h=tile_h,
                          job=job, trt_runner=trt_enc)
    job.latents = latents
    log.debug(f"  SDXL: VAE encode complete. shape={list(latents.shape)}"
              f" ({'TRT' if trt_enc else 'PyTorch'})")


def latent_upscale(job: InferenceJob) -> None:
    """CPU stage: Lanczos-3 upscale of latents for hires fix."""
    hires = job.hires_input
    if hires is None:
        raise RuntimeError("HiresInput is required for latent upscale.")
    latents = job.latents
    if latents is None:
        raise RuntimeError("Latents are required for latent upscale.")

    dst_h = hires.hires_height // 8
    dst_w = hires.hires_width // 8

    log.debug(f"  SDXL: Latent upscale (Lanczos3) "
             f"{list(latents.shape)} → [1,4,{dst_h},{dst_w}]")

    # Use torch interpolation (bicubic is closest to Lanczos for latents)
    job.latents = torch.nn.functional.interpolate(
        latents.float(), size=(dst_h, dst_w), mode="bicubic", align_corners=False,
    ).to(latents.dtype)
    job.is_hires_pass = True


def hires_transform(job: InferenceJob, gpu: GpuInstance) -> None:
    """Composite GPU stage: VAE decode → RealESRGAN 2x → VAE encode → hires denoise."""
    hires = job.hires_input
    if hires is None:
        raise RuntimeError("HiresInput is required for hires transform.")
    latents = job.latents
    if latents is None:
        raise RuntimeError("Latents are required for hires transform.")

    device = gpu.device
    t0 = time.monotonic()
    target_w = hires.hires_width
    target_h = hires.hires_height

    # Always load PyTorch VAE (needed for encode step later)
    vae = _get_cached_model_optional(gpu, "sdxl_vae", job)
    if vae is None:
        log.warning(f"  SDXL: VAE not cached for hires transform — loading PyTorch fallback")
        inp = job.sdxl_input
        vae = load_component("sdxl_vae", inp.model_dir if inp else None, device)
        fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae", "sdxl_vae_fallback")
        gpu.cache_model(fp, "sdxl_vae", vae, estimated_vram=400 * 1024 * 1024,
                        source="PyTorch-fallback")

    hires_lat_h = latents.shape[2]
    hires_lat_w = latents.shape[3]
    hires_img_w = hires_lat_w * 8
    hires_img_h = hires_lat_h * 8

    # Try TRT-aware tile selection for decode
    log.debug(f"  HiresTransform: Selecting VAE for {hires_img_w}x{hires_img_h} decode")
    trt_vae = _get_trt_vae_runner(gpu, job, hires_img_w, hires_img_h)
    if trt_vae is not None:
        tile_w, tile_h = hires_img_w, hires_img_h
        log.debug(f"  HiresTransform: TRT VAE covers full {hires_img_w}x{hires_img_h}")
    elif hires_img_w >= VAE_TILE_THRESHOLD or hires_img_h >= VAE_TILE_THRESHOLD:
        tile_w, tile_h = _pick_vae_tile_size(hires_img_w, hires_img_h, gpu, job)
        trt_vae = _get_trt_vae_runner(gpu, job, tile_w, tile_h)
        log.debug(f"  HiresTransform: Tiled {hires_img_w}x{hires_img_h} → "
                  f"tile {tile_w}x{tile_h}, backend={'TRT' if trt_vae else 'PyTorch'}")
    else:
        tile_w, tile_h = hires_img_w, hires_img_h

    # Scale latents with correct dtype
    if trt_vae is not None:
        scaled = latents.to(device=device, dtype=torch.float32) / VAE_SCALE_FACTOR
    else:
        scaled = latents.to(device=device, dtype=vae.dtype) / VAE_SCALE_FACTOR

    lat_tile_w = tile_w // 8
    lat_tile_h = tile_h // 8
    use_tiled = hires_lat_w > lat_tile_w or hires_lat_h > lat_tile_h

    if use_tiled:
        intermediate = _vae_decode_tiled(scaled, vae if trt_vae is None else None,
                                         lat_tile_w, lat_tile_h, job=job,
                                         trt_runner=trt_vae)
    elif trt_vae is not None:
        intermediate = _tensor_to_pil(trt_vae.run(scaled))
    else:
        with torch.no_grad():
            intermediate = _tensor_to_pil(vae.decode(scaled).sample)
    # Clear tile progress before upscale phase
    job.stage_step = 0
    job.stage_total_steps = 0
    log.debug(f"  HiresTransform: VAE decode → {intermediate.width}x{intermediate.height} "
             f"({(time.monotonic()-t0)*1000:.0f}ms)")

    from handlers.upscale import upscale_image
    upscaled = upscale_image(intermediate, gpu)
    log.debug(f"  HiresTransform: RealESRGAN 2x → {upscaled.width}x{upscaled.height} "
             f"({(time.monotonic()-t0)*1000:.0f}ms)")
    intermediate.close()

    if upscaled.width != target_w or upscaled.height != target_h:
        upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        log.debug(f"  HiresTransform: Resize → {target_w}x{target_h} "
                 f"({(time.monotonic()-t0)*1000:.0f}ms)")

    # Try TRT encoder for the encode step
    trt_enc = _get_trt_vae_enc_runner(gpu, job, target_w, target_h)
    if trt_enc is not None:
        # TRT covers full image — no tiling needed
        enc_tile_w, enc_tile_h = 0, 0
    else:
        enc_tile_w = job.vae_tile_width if job.vae_tile_width > 0 else 0
        enc_tile_h = job.vae_tile_height if job.vae_tile_height > 0 else 0
        if target_w >= VAE_TILE_THRESHOLD or target_h >= VAE_TILE_THRESHOLD:
            enc_tile_w, enc_tile_h = _pick_vae_tile_size(target_w, target_h, gpu, job,
                                                          component_type="vae_enc")
            trt_enc = _get_trt_vae_enc_runner(gpu, job, enc_tile_w, enc_tile_h)

    # Only load PyTorch VAE for encode if TRT is not available and not already loaded
    enc_vae = None if trt_enc is not None else vae
    encoded = _vae_encode(upscaled, enc_vae, device, tile_w=enc_tile_w, tile_h=enc_tile_h,
                          job=job, trt_runner=trt_enc)
    log.debug(f"  HiresTransform: VAE encode → {list(encoded.shape)} "
             f"({'TRT' if trt_enc else 'PyTorch'}, "
             f"{(time.monotonic()-t0)*1000:.0f}ms)")
    upscaled.close()

    job.latents = encoded
    job.is_hires_pass = True
    log.debug(f"  HiresTransform: Complete ({(time.monotonic()-t0)*1000:.0f}ms total)")


def calculate_base_resolution(target_w: int, target_h: int) -> tuple[int, int]:
    """Calculate base resolution for hires fix. ceil_to_64(target/2)."""
    base_w = ((target_w // 2 + 63) // 64) * 64
    base_h = ((target_h // 2 + 63) // 64) * 64
    return max(64, base_w), max(64, base_h)


# ====================================================================
# Tokenization with Emphasis Weighting
# ====================================================================

def _parse_prompt_attention(text: str) -> list[tuple[str, float]]:
    """Parse A1111/Forge-style prompt emphasis syntax.
    (word) = weight * 1.1, ((word)) = weight * 1.21, (word:1.5) = explicit weight,
    [word] = weight / 1.1"""
    result: list[tuple[str, float]] = []
    round_brackets: list[int] = []
    square_brackets: list[int] = []

    ROUND_MUL = 1.1
    SQUARE_MUL = 1.0 / 1.1

    for m in _RE_ATTENTION.finditer(text):
        token = m.group(0)
        weight_str = m.group(1) if m.group(1) else None

        if token.startswith("\\") and len(token) > 1:
            result.append((token[1:], 1.0))
        elif token == "(":
            round_brackets.append(len(result))
        elif token == "[":
            square_brackets.append(len(result))
        elif weight_str is not None and round_brackets:
            try:
                w = float(weight_str)
                _multiply_range(result, round_brackets.pop(), w)
            except ValueError:
                if round_brackets:
                    round_brackets.pop()
        elif token == ")" and round_brackets:
            _multiply_range(result, round_brackets.pop(), ROUND_MUL)
        elif token == "]" and square_brackets:
            _multiply_range(result, square_brackets.pop(), SQUARE_MUL)
        else:
            # Split on BREAK keyword (word-boundary) to emit chunk-boundary sentinels
            parts = _RE_BREAK.split(token)
            for j, part in enumerate(parts):
                if part:
                    result.append((part, 1.0))
                if j < len(parts) - 1:
                    result.append(("BREAK", -1.0))  # sentinel

    # Handle unbalanced brackets
    while round_brackets:
        _multiply_range(result, round_brackets.pop(), ROUND_MUL)
    while square_brackets:
        _multiply_range(result, square_brackets.pop(), SQUARE_MUL)

    if not result:
        result.append(("", 1.0))

    # Merge consecutive items with same weight (skip BREAK sentinels)
    i = 0
    while i < len(result) - 1:
        if (result[i][0] != "BREAK" and result[i + 1][0] != "BREAK"
                and abs(result[i][1] - result[i + 1][1]) < 1e-6):
            result[i] = (result[i][0] + result[i + 1][0], result[i][1])
            result.pop(i + 1)
        else:
            i += 1

    return result


def _multiply_range(lst: list[tuple[str, float]], start_idx: int, multiplier: float) -> None:
    for i in range(start_idx, len(lst)):
        if lst[i][0] == "BREAK":
            continue  # don't mutate BREAK sentinels
        lst[i] = (lst[i][0], lst[i][1] * multiplier)


def _tokenize_weighted(
    tokenizer: CLIPTokenizer, text: str, pad_token_id: int = 49407
) -> tuple[list[list[int]], list[list[float]], list[list[int]]]:
    """Tokenize with emphasis weights + multi-chunk support.
    Returns (chunks_of_token_ids, chunks_of_weights, chunks_of_masks).
    Each chunk is a 77-element list: [BOS] + 75 content + [EOS]."""
    BOS = 49406
    EOS = 49407
    CHUNK_SIZE = 77
    CONTENT_SLOTS = 75

    fragments = _parse_prompt_attention(text)

    # Tokenize all fragments into a flat (token_id, weight) list
    all_tokens: list[tuple[int, float]] = []
    for frag_text, weight in fragments:
        if frag_text == "BREAK":
            all_tokens.append((-1, -1.0))  # BREAK sentinel
            continue
        if not frag_text.strip():
            continue
        encoded = tokenizer(frag_text, add_special_tokens=False, return_tensors=None)
        for tid in encoded["input_ids"]:
            all_tokens.append((tid, weight))

    # Split into 75-token content chunks
    chunks_ids: list[list[int]] = []
    chunks_weights: list[list[float]] = []
    chunks_masks: list[list[int]] = []

    current_ids: list[int] = []
    current_weights: list[float] = []

    def _finalize_chunk() -> None:
        """Pad current content to 75 tokens, wrap with BOS/EOS -> 77-token chunk."""
        ids = [BOS] + current_ids[:CONTENT_SLOTS]
        wts = [1.0] + current_weights[:CONTENT_SLOTS]
        msk = [1] * (1 + len(current_ids[:CONTENT_SLOTS]))
        # EOS
        ids.append(EOS)
        wts.append(1.0)
        msk.append(1)
        # Pad to 77
        while len(ids) < CHUNK_SIZE:
            ids.append(pad_token_id)
            wts.append(1.0)
            msk.append(0)
        chunks_ids.append(ids)
        chunks_weights.append(wts)
        chunks_masks.append(msk)

    for tid, weight in all_tokens:
        if tid == -1 and weight == -1.0:
            # BREAK: finalize current chunk (even if partially filled)
            _finalize_chunk()
            current_ids = []
            current_weights = []
            continue
        current_ids.append(tid)
        current_weights.append(weight)
        if len(current_ids) >= CONTENT_SLOTS:
            _finalize_chunk()
            current_ids = []
            current_weights = []

    # Finalize remaining tokens, or ensure at least one chunk for empty prompts
    if current_ids or not chunks_ids:
        _finalize_chunk()

    # Each chunk's mask has 1s for BOS + content + EOS; subtract 2 per chunk for BOS/EOS
    content_count = sum(sum(mask) - 2 for mask in chunks_masks)
    log.debug(f"    Tokenized: {content_count} content tokens across {len(chunks_ids)} chunk(s), "
              f"pad={pad_token_id}")

    return chunks_ids, chunks_weights, chunks_masks


def _pad_to_chunk_count(
    chunks_ids: list[list[int]], chunks_weights: list[list[float]],
    chunks_masks: list[list[int]], target_count: int, pad_token_id: int,
) -> None:
    """Pad chunk lists in-place to reach target_count with empty chunks."""
    BOS, EOS, CHUNK_SIZE = 49406, 49407, 77
    while len(chunks_ids) < target_count:
        ids = [BOS, EOS] + [pad_token_id] * (CHUNK_SIZE - 2)
        wts = [1.0] * CHUNK_SIZE
        msk = [1, 1] + [0] * (CHUNK_SIZE - 2)
        chunks_ids.append(ids)
        chunks_weights.append(wts)
        chunks_masks.append(msk)


def _flatten_chunks(chunks: list[list[float]]) -> list[float]:
    """Flatten list of chunk weights into a single flat list."""
    return [w for chunk in chunks for w in chunk]


# ====================================================================
# Text Encoding Helpers
# ====================================================================

def _run_text_encoder_1(
    model: CLIPTextModel, chunks_ids: list[list[int]], chunks_masks: list[list[int]],
    device: torch.device
) -> torch.Tensor:
    """Run CLIP-L text encoder over multiple chunks (batched).
    Returns hidden states [1, 77*N, 768]."""
    input_ids = torch.tensor(chunks_ids, dtype=torch.long, device=device)        # [N, 77]
    attention_mask = torch.tensor(chunks_masks, dtype=torch.long, device=device)  # [N, 77]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-2]  # [N, 77, 768]
    return hidden.reshape(1, -1, hidden.shape[-1])  # [1, 77*N, 768]


def _run_text_encoder_2(
    model: CLIPTextModelWithProjection, chunks_ids: list[list[int]],
    chunks_masks: list[list[int]], device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CLIP-bigG text encoder over multiple chunks (batched).
    Returns (hidden_states [1,77*N,1280], pooled [1,1280] from first chunk)."""
    input_ids = torch.tensor(chunks_ids, dtype=torch.long, device=device)        # [N, 77]
    attention_mask = torch.tensor(chunks_masks, dtype=torch.long, device=device)  # [N, 77]
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-2]  # [N, 77, 1280]
        pooled = outputs.text_embeds[0:1]   # [1, 1280] from first chunk only
    return hidden.reshape(1, -1, hidden.shape[-1]), pooled  # [1, 77*N, 1280], [1, 1280]


def _apply_token_weights(hidden_states: torch.Tensor, weights: list[float]) -> None:
    """Apply per-token emphasis weights to hidden states [1, seq_len, hidden_dim].
    Simple multiplication, no mean normalization (SDXL "No norm" mode)."""
    seq_len = hidden_states.shape[1]
    for pos in range(min(seq_len, len(weights))):
        w = weights[pos]
        if abs(w - 1.0) < 1e-6:
            continue
        hidden_states[0, pos, :] *= w


# ====================================================================
# VAE Helpers
# ====================================================================

def _tensor_to_pil(tensor: torch.Tensor) -> Image.Image:
    """Convert decoded VAE tensor [1, 3, H, W] in [-1, 1] to PIL RGB image."""
    t = tensor.squeeze(0).clamp(-1, 1)
    t = ((t + 1) / 2 * 255).byte()
    arr = t.permute(1, 2, 0).cpu().numpy()
    return Image.fromarray(arr, "RGB")


def _vae_encode(image: Image.Image, vae: "AutoencoderKL | None", device: torch.device,
                tile_w: int = 0, tile_h: int = 0,
                job: InferenceJob | None = None,
                trt_runner=None) -> torch.Tensor:
    """VAE-encode a PIL image to latent space. Returns [1,4,H/8,W/8].
    Forces tiled encoding when any dimension >= VAE_TILE_THRESHOLD, with
    non-square tiles auto-computed per axis for optimal coverage.

    When trt_runner is provided, uses TRT for encoding (output is already
    scaled by VAE_SCALE_FACTOR). Either vae or trt_runner must be provided.
    """
    if vae is None and trt_runner is None:
        raise ValueError("_vae_encode requires either vae or trt_runner")

    img_w, img_h = image.size

    # When trt_runner is provided with tile_w=0/tile_h=0, the caller guarantees
    # this engine covers the full image — skip auto-tiling entirely.
    if trt_runner is not None and tile_w == 0 and tile_h == 0:
        eff_w, eff_h = img_w, img_h
    elif tile_w > 0 and tile_h > 0:
        eff_w = (tile_w // 8) * 8
        eff_h = (tile_h // 8) * 8
    elif img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
        eff_w, eff_h = _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)
    else:
        eff_w, eff_h = img_w, img_h

    if img_w > eff_w or img_h > eff_h:
        return _vae_encode_tiled(image, vae, device, eff_w, eff_h, job=job,
                                 trt_runner=trt_runner)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)

    if trt_runner is not None:
        t = (t / 127.5 - 1.0).to(dtype=torch.float32, device=device)
        with torch.no_grad():
            latents = trt_runner.run(t)  # already scaled by VAE_SCALE_FACTOR
    else:
        t = (t / 127.5 - 1.0).to(dtype=vae.dtype, device=device)
        with torch.no_grad():
            dist = vae.encode(t).latent_dist
            latents = dist.mean * VAE_SCALE_FACTOR

    return latents


def _vae_encode_tiled(image: Image.Image, vae: "AutoencoderKL | None",
                      device: torch.device,
                      tile_w_px: int, tile_h_px: int,
                      *, job: InferenceJob | None = None,
                      trt_runner=None) -> torch.Tensor:
    """Tiled VAE encode for large images. Supports non-square tiles.
    Tiles in pixel space, blends in latent space with linear feathering.
    Either vae or trt_runner must be provided."""
    import time as _time
    from profiling.tracer import get_current_tracer
    _tracer = get_current_tracer()
    _model_name = _get_model_name(job) if job else None
    min_tile_px = LATENT_TILE_OVERLAP * 8 + 8  # 136px minimum
    if tile_w_px < min_tile_px or tile_h_px < min_tile_px:
        raise ValueError(
            f"VAE encode tile {tile_w_px}x{tile_h_px}px is smaller than minimum "
            f"{min_tile_px}px (overlap={LATENT_TILE_OVERLAP * 8}px). Increase tile size.")

    img_w, img_h = image.size
    lat_h = img_h // 8
    lat_w = img_w // 8

    overlap_px = LATENT_TILE_OVERLAP * 8   # same overlap as decode
    overlap_lat = LATENT_TILE_OVERLAP

    stride_w_px = tile_w_px - overlap_px
    stride_h_px = tile_h_px - overlap_px

    tiles_y = max(1, (img_h - overlap_px + stride_h_px - 1) // stride_h_px)
    tiles_x = max(1, (img_w - overlap_px + stride_w_px - 1) // stride_w_px)

    total_tiles = tiles_x * tiles_y
    log.debug(f"  SDXL: Tiled VAE encode {img_w}x{img_h} — "
             f"{tiles_x}x{tiles_y} grid ({total_tiles} tiles, "
             f"tile={tile_w_px}x{tile_h_px})")

    if job is not None:
        job.stage_step = 0
        job.stage_total_steps = total_tiles

    lat_sum = np.zeros((1, 4, lat_h, lat_w), dtype=np.float32)
    weights  = np.zeros((1, 1, lat_h, lat_w), dtype=np.float32)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    tile_count = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            py = min(ty * stride_h_px, img_h - tile_h_px)
            px = min(tx * stride_w_px, img_w - tile_w_px)
            py, px = max(0, py), max(0, px)
            py2 = min(py + tile_h_px, img_h)
            px2 = min(px + tile_w_px, img_w)

            tile_np = arr[py:py2, px:px2]
            t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0)

            _tile_start = _time.monotonic()
            if trt_runner is not None:
                t = (t / 127.5 - 1.0).to(dtype=torch.float32, device=device)
                with torch.no_grad():
                    tile_lat_t = trt_runner.run(t)  # already scaled
            else:
                t = (t / 127.5 - 1.0).to(dtype=vae.dtype, device=device)
                with torch.no_grad():
                    tile_lat_t = vae.encode(t).latent_dist.mean * VAE_SCALE_FACTOR
            _tile_dur = _time.monotonic() - _tile_start

            tile_lat_np = tile_lat_t.squeeze(0).cpu().float().numpy()  # [4, th, tw]
            th_lat = tile_lat_np.shape[1]
            tw_lat = tile_lat_np.shape[2]

            ly = py // 8
            lx = px // 8

            # Linear feathering weight mask in latent space
            w_mask = np.ones((th_lat, tw_lat), dtype=np.float32)
            for y in range(th_lat):
                wy = 1.0
                if y < overlap_lat:
                    wy = (y + 0.5) / overlap_lat
                elif y >= th_lat - overlap_lat:
                    wy = (th_lat - y - 0.5) / overlap_lat
                for x in range(tw_lat):
                    wx = 1.0
                    if x < overlap_lat:
                        wx = (x + 0.5) / overlap_lat
                    elif x >= tw_lat - overlap_lat:
                        wx = (tw_lat - x - 0.5) / overlap_lat
                    w_mask[y, x] = wy * wx

            lat_sum[0, :, ly:ly+th_lat, lx:lx+tw_lat] += tile_lat_np * w_mask[np.newaxis]
            weights [0, 0, ly:ly+th_lat, lx:lx+tw_lat] += w_mask

            tile_count += 1
            if _tracer and job:
                _tracer.vae_tile(job.job_id, _model_name, tile_count, total_tiles,
                                 px2 - px, py2 - py, "encode", _tile_dur)
            if job is not None:
                job.stage_step = tile_count

        log.debug(f"  SDXL: Tiled VAE encode row {ty+1}/{tiles_y}")

    latents = torch.from_numpy(lat_sum / weights).to(dtype=torch.float16, device=device)
    log.debug(f"  SDXL: Tiled VAE encode complete. shape={list(latents.shape)}")
    return latents


def _vae_decode_tiled(
    latents: torch.Tensor, vae: AutoencoderKL | None,
    lat_tile_w: int, lat_tile_h: int,
    *, job: InferenceJob | None = None,
    trt_runner=None,
) -> Image.Image:
    """Tiled VAE decode for large images with linear feathering blend.
    Uses numpy broadcasting for weight accumulation (no Python pixel loops)."""
    if trt_runner is None and vae is None:
        raise ValueError("_vae_decode_tiled requires either vae or trt_runner")
    import time as _time
    from profiling.tracer import get_current_tracer
    _tracer = get_current_tracer()
    _model_name = _get_model_name(job) if job else None
    min_lat = LATENT_TILE_OVERLAP + 1
    if lat_tile_w < min_lat or lat_tile_h < min_lat:
        raise ValueError(
            f"VAE decode tile {lat_tile_w*8}x{lat_tile_h*8}px is smaller than minimum "
            f"{min_lat*8}px (overlap={LATENT_TILE_OVERLAP*8}px). Increase tile size.")

    lat_h = latents.shape[2]
    lat_w = latents.shape[3]
    img_h = lat_h * 8
    img_w = lat_w * 8

    stride_w = lat_tile_w - LATENT_TILE_OVERLAP
    stride_h = lat_tile_h - LATENT_TILE_OVERLAP

    tiles_y = max(1, (lat_h - LATENT_TILE_OVERLAP + stride_h - 1) // stride_h)
    tiles_x = max(1, (lat_w - LATENT_TILE_OVERLAP + stride_w - 1) // stride_w)

    total_tiles = tiles_x * tiles_y
    log.debug(f"  SDXL: Tiled VAE decode {img_w}x{img_h} — "
             f"{tiles_x}x{tiles_y} grid ({total_tiles} tiles, "
             f"tile={lat_tile_w*8}x{lat_tile_h*8}, "
             f"backend={'TRT' if trt_runner else 'PyTorch'})")

    if job is not None:
        job.stage_step = 0
        job.stage_total_steps = total_tiles

    # Accumulation buffers: [3, H, W] for RGB + [H, W] for weights
    rgb_sum = np.zeros((3, img_h, img_w), dtype=np.float32)
    weights = np.zeros((img_h, img_w), dtype=np.float32)

    overlap_px = LATENT_TILE_OVERLAP * 8
    tile_count = 0

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            lat_y = min(ty * stride_h, lat_h - lat_tile_h)
            lat_x = min(tx * stride_w, lat_w - lat_tile_w)
            lat_y = max(0, lat_y)
            lat_x = max(0, lat_x)
            tile_h = min(lat_tile_h, lat_h - lat_y)
            tile_w = min(lat_tile_w, lat_w - lat_x)

            tile = latents[:, :, lat_y:lat_y+tile_h, lat_x:lat_x+tile_w]

            _tile_start = _time.monotonic()
            if trt_runner is not None:
                decoded = trt_runner.run(tile)
            else:
                with torch.no_grad():
                    decoded = vae.decode(tile).sample
            _tile_dur = _time.monotonic() - _tile_start

            decoded_np = decoded.squeeze(0).cpu().float().numpy()  # [3, H, W]
            tile_img_h = tile_h * 8
            tile_img_w = tile_w * 8
            img_x = lat_x * 8
            img_y = lat_y * 8

            # Build feathering weights via numpy broadcasting (not pixel loops)
            wy = np.ones(tile_img_h, dtype=np.float32)
            wy[:overlap_px] = (np.arange(overlap_px, dtype=np.float32) + 0.5) / overlap_px
            wy[tile_img_h - overlap_px:] = (np.arange(overlap_px, 0, -1, dtype=np.float32) - 0.5) / overlap_px

            wx = np.ones(tile_img_w, dtype=np.float32)
            wx[:overlap_px] = (np.arange(overlap_px, dtype=np.float32) + 0.5) / overlap_px
            wx[tile_img_w - overlap_px:] = (np.arange(overlap_px, 0, -1, dtype=np.float32) - 0.5) / overlap_px

            w_mask = wy[:, np.newaxis] * wx[np.newaxis, :]  # [H, W]

            # Convert [-1, 1] → [0, 1] and accumulate
            rgb = decoded_np / 2.0 + 0.5  # [3, H, W]
            rgb_sum[:, img_y:img_y+tile_img_h, img_x:img_x+tile_img_w] += rgb * w_mask[np.newaxis]
            weights[img_y:img_y+tile_img_h, img_x:img_x+tile_img_w] += w_mask

            tile_count += 1
            if _tracer and job:
                _tracer.vae_tile(job.job_id, _model_name, tile_count, total_tiles,
                                 tile_img_w, tile_img_h, "decode", _tile_dur)
            if job is not None:
                job.stage_step = tile_count

    # Build final image
    w_safe = np.maximum(weights, 1e-8)
    rgb_out = np.clip(rgb_sum / w_safe[np.newaxis] * 255, 0, 255).astype(np.uint8)  # [3, H, W]
    image = Image.fromarray(rgb_out.transpose(1, 2, 0), "RGB")

    log.debug(f"  SDXL: Tiled VAE decode complete. Image={img_w}x{img_h} ({tile_count} tiles)")
    return image


# ====================================================================
# Helpers
# ====================================================================

# ====================================================================
# LoRA / PEFT Support
# ====================================================================

def _ensure_loras(unet, lora_specs: list, gpu: GpuInstance, lora_index: dict) -> None:
    """Load/activate PEFT LoRA adapters on the UNet.

    Uses PEFT adapter mode: adapters are injected as separate matrices
    without modifying base weights, so the base UNet stays clean on GPU.
    Incompatible LoRAs (e.g. SD1.5 on SDXL) are silently skipped.
    """
    adapter_names = []
    adapter_weights = []

    for spec in lora_specs:
        entry = lora_index.get(spec.name)
        if entry is None:
            log.warning(f"  LoRA not found in index: {spec.name}")
            continue

        # PEFT registers adapters as nn.Module children — Python module names
        # can't contain periods.  Replace with underscores for PEFT compatibility.
        adapter_name = spec.name.replace(".", "_")

        # Check if adapter already loaded on this UNet
        if hasattr(unet, 'peft_config') and adapter_name in unet.peft_config:
            # Already loaded — just activate
            pass
        else:
            try:
                _load_lora_adapter(unet, entry.path, adapter_name, gpu)
            except _LoraSkipped:
                continue
            except Exception as ex:
                log.warning(f"  LoRA: Failed to load '{adapter_name}': {ex}")
                continue

        adapter_names.append(adapter_name)
        adapter_weights.append(spec.weight)

    if adapter_names:
        unet.set_adapters(adapter_names, adapter_weights)
        log.debug(f"  LoRA: Activated {len(adapter_names)} adapter(s): "
                 f"{', '.join(f'{n}={w:.2f}' for n, w in zip(adapter_names, adapter_weights))}")
    elif hasattr(unet, 'peft_config') and unet.peft_config:
        unet.disable_adapters()


class _LoraSkipped(Exception):
    """Raised when a LoRA is intentionally skipped (incompatible, too large, etc.)."""
    pass


def _detect_has_te_keys(raw_sd: dict) -> bool:
    """Check if a raw LoRA state dict has text encoder keys (A1111/Kohya or diffusers)."""
    for k in raw_sd:
        if k.startswith("lora_te") or k.startswith("text_encoder"):
            return True
    return False


def _detect_lora_arch(state_dict: dict) -> str:
    """Detect LoRA architecture from weight shapes.

    Checks cross-attention key projection dimensions:
      - SD 1.5: text encoder dim = 768
      - SDXL:   text encoder dim = 2048

    Returns ``"sd15"``, ``"sdxl"``, or ``"unknown"``.
    """
    for key, tensor in state_dict.items():
        # Cross-attention to_k captures text encoder dimension.
        # Works for both A1111 keys (attn2_to_k.lora_down) and
        # diffusers keys (attn2.to_k.lora_A).
        is_cross_attn_key = ("attn2" in key and "to_k" in key)
        is_down_weight = ("lora_down" in key or "lora_A" in key or "lora_a" in key)
        if is_cross_attn_key and is_down_weight and tensor.dim() == 2:
            in_dim = tensor.shape[1]  # [rank, in_features]
            if in_dim == 768:
                return "sd15"
            elif in_dim == 2048:
                return "sdxl"
    return "unknown"


# The UNet architecture that foxburrow serves (SDXL).
_EXPECTED_LORA_ARCH = "sdxl"

# SGM/Kohya LoRA key → diffusers module path for non-block UNet layers.
# These keys crash diffusers' _maybe_map_sgm_blocks_to_diffusers because they
# don't contain input_blocks/middle_block/output_blocks patterns.
_SDXL_NONBLOCK_LORA_MAP = {
    "label_emb_0_0": "add_embedding.linear_1",
    "label_emb_0_2": "add_embedding.linear_2",
    "time_embed_0": "time_embedding.linear_1",
    "time_embed_2": "time_embedding.linear_2",
    # out_0 → conv_norm_out is GroupNorm — PEFT cannot inject LoRA into GroupNorm
    "out_2": "conv_out",
}


def _extract_nonblock_lora_keys(state_dict: dict) -> tuple[dict, dict]:
    """Extract and convert non-block lora_unet_* keys that crash diffusers' SGM mapper.

    Pops matching keys from state_dict in-place and returns them converted to
    the same format that _convert_non_diffusers_lora_to_diffusers produces.

    Returns:
        (converted_state_dict, network_alphas)
    """
    converted = {}
    alphas = {}
    to_pop = []

    for key in list(state_dict.keys()):
        if not key.startswith("lora_unet_"):
            continue
        lora_name = key.split(".")[0]
        module_key = lora_name[len("lora_unet_"):]

        diffusers_module = _SDXL_NONBLOCK_LORA_MAP.get(module_key)
        if diffusers_module is None:
            continue

        suffix = key[len(lora_name):]

        if suffix == ".alpha":
            alphas[f"unet.{diffusers_module}.alpha"] = state_dict[key].item()
            to_pop.append(key)
        elif suffix == ".lora_down.weight":
            converted[f"unet.{diffusers_module}.lora.down.weight"] = state_dict[key]
            to_pop.append(key)
        elif suffix == ".lora_up.weight":
            converted[f"unet.{diffusers_module}.lora.up.weight"] = state_dict[key]
            to_pop.append(key)
        elif suffix == ".dora_scale":
            converted[f"unet.{diffusers_module}.lora_magnitude_vector.weight"] = state_dict[key]
            to_pop.append(key)

    for key in to_pop:
        state_dict.pop(key)

    return converted, alphas


def _load_lora_adapter(unet, lora_path: str, adapter_name: str, gpu: GpuInstance) -> None:
    """Load a LoRA safetensors/pt file as a PEFT adapter on the UNet.

    Uses diffusers' built-in LoRA conversion to handle A1111/Forge/Kohya
    key naming conventions, then loads via UNet's native load_lora_adapter.

    Raises ``_LoraSkipped`` if the LoRA is incompatible or won't fit in VRAM.
    """
    from safetensors.torch import load_file as load_safetensors
    from diffusers.loaders.lora_pipeline import _convert_non_diffusers_lora_to_diffusers
    from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers
    from utils import fingerprint as fp_cache

    t0 = time.monotonic()

    # ── Quick architecture check from cache ─────────────────────────
    cached_arch = fp_cache.get_extra(lora_path, "lora_arch")
    if cached_arch is not None and cached_arch != _EXPECTED_LORA_ARCH and cached_arch != "unknown":
        log.warning(f"  LoRA: Skipping '{adapter_name}' — architecture mismatch "
                    f"(LoRA is {cached_arch}, model expects {_EXPECTED_LORA_ARCH})")
        raise _LoraSkipped(f"Architecture mismatch: {cached_arch}")

    # ── VRAM pre-check (file size as upper bound) ─────────────────
    lora_size = os.path.getsize(lora_path)
    _, _, free = nvml.get_memory_info(gpu.nvml_handle)
    if lora_size > free:
        log.warning(f"  LoRA: Skipping '{adapter_name}' — insufficient VRAM "
                    f"(file {lora_size // (1024*1024)}MB, free {free // (1024*1024)}MB)")
        raise _LoraSkipped(f"Insufficient VRAM: {lora_size} > {free}")

    # ── Load raw state dict ─────────────────────────────────────────
    # Load to CPU first when arch detection is needed (avoids wasting GPU
    # VRAM on incompatible LoRAs). Load directly to GPU when cache confirms
    # compatibility — saves the CPU→GPU transfer overhead.
    need_detect = cached_arch is None
    load_device = "cpu" if need_detect else str(gpu.device)

    if lora_path.endswith(".safetensors"):
        raw_sd = load_safetensors(lora_path, device=load_device)
    else:
        raw_sd = torch.load(lora_path, map_location=load_device, weights_only=True)

    # ── Detect and cache architecture + TE presence if not yet cached ──
    if need_detect:
        detected_arch = _detect_lora_arch(raw_sd)
        has_te = _detect_has_te_keys(raw_sd)
        fp_cache.set_extra(lora_path, lora_arch=detected_arch,
                           has_te_lora="1" if has_te else "0")
        if detected_arch != _EXPECTED_LORA_ARCH and detected_arch != "unknown":
            log.warning(f"  LoRA: Skipping '{adapter_name}' — architecture mismatch "
                        f"(LoRA is {detected_arch}, model expects {_EXPECTED_LORA_ARCH})")
            del raw_sd
            raise _LoraSkipped(f"Architecture mismatch: {detected_arch}")
        # Arch OK — move tensors to GPU
        raw_sd = {k: v.to(gpu.device) for k, v in raw_sd.items()}

    # ── Convert key format ──────────────────────────────────────────
    # Detect format: A1111/Kohya keys start with "lora_unet_" / "lora_te_",
    # diffusers keys contain "lora_A" / "lora_B" already
    is_a1111 = any(k.startswith("lora_unet_") or k.startswith("lora_te") for k in raw_sd)

    if is_a1111:
        # Extract non-block UNet keys (label_emb, time_embed, etc.) BEFORE the
        # SGM mapper — diffusers' mapper crashes on keys that aren't block-level.
        nonblock_sd, nonblock_alphas = _extract_nonblock_lora_keys(raw_sd)
        # Remap SGM/LDM block indices to diffusers structure (input_blocks_4 → down_blocks.1.attentions.0)
        # This must happen BEFORE the key format conversion
        raw_sd = _maybe_map_sgm_blocks_to_diffusers(raw_sd, unet.config)
        # Convert A1111/Kohya → diffusers format (handles remaining key renaming)
        converted_sd, network_alphas = _convert_non_diffusers_lora_to_diffusers(raw_sd)
        # Merge manually-converted non-block keys back
        if nonblock_sd:
            converted_sd.update(nonblock_sd)
            if network_alphas is None:
                network_alphas = nonblock_alphas
            elif nonblock_alphas:
                network_alphas.update(nonblock_alphas)
    else:
        converted_sd = raw_sd
        network_alphas = None

    # Repair accelerate leak BEFORE injection — if the leak is active, PEFT
    # creates LoRA parameters on meta device, and load_state_dict's copy_()
    # into meta tensors is a silent no-op (weights are lost, not loaded).
    repair_accelerate_leak()

    # Use diffusers' built-in UNet LoRA loading (handles diffusers→PEFT conversion,
    # LoraConfig creation, adapter injection, and weight loading)
    unet.load_lora_adapter(
        converted_sd,
        prefix="unet",
        adapter_name=adapter_name,
        network_alphas=network_alphas,
    )

    # Post-injection check: if a concurrent thread re-activated the leak
    # during injection, LoRA weights are silently corrupted (zeroed).
    repair_accelerate_leak()
    n = fix_meta_tensors(unet)
    if n:
        log.error(f"  LoRA: {n} meta tensor(s) in UNet after '{adapter_name}' injection — "
                  f"LoRA weights are CORRUPTED (zeroed). Accelerate leak race condition.")

    # Track loaded adapter on the GPU
    if hasattr(gpu, '_loaded_lora_adapters'):
        gpu._loaded_lora_adapters[adapter_name] = lora_path

    elapsed_ms = (time.monotonic() - t0) * 1000
    n_unet_keys = sum(1 for k in converted_sd if k.startswith("unet."))
    n_te_keys = sum(1 for k in converted_sd if k.startswith("text_encoder"))
    parts = [f"{n_unet_keys} unet"]
    if n_te_keys:
        parts.append(f"{n_te_keys} te")
    log.debug(f"  LoRA: Loaded adapter '{adapter_name}' from {os.path.basename(lora_path)} "
             f"({' + '.join(parts)} params, {lora_size // (1024*1024)}MB, {elapsed_ms:.0f}ms)")


# ── TE LoRA Support ───────────────────────────────────────────────

def _any_lora_has_te(lora_specs: list, lora_index: dict) -> bool:
    """Check if any requested LoRA has text encoder weights (cached check)."""
    from utils import fingerprint as fp_cache
    for spec in lora_specs:
        entry = lora_index.get(spec.name)
        if entry is None:
            continue
        cached = fp_cache.get_extra(entry.path, "has_te_lora")
        if cached == "1":
            return True
        if cached is None:
            # Not yet scanned — peek at safetensors header keys
            try:
                if entry.path.endswith(".safetensors"):
                    from safetensors import safe_open
                    with safe_open(entry.path, framework="pt") as f:
                        keys = f.keys()
                    has_te = any(k.startswith("lora_te") or k.startswith("text_encoder") for k in keys)
                else:
                    # For .pt files we can't peek without loading — assume no TE
                    has_te = False
                fp_cache.set_extra(entry.path, has_te_lora="1" if has_te else "0")
                if has_te:
                    return True
            except Exception as ex:
                log.warning(f"  LoRA: Failed to peek TE keys in {entry.path}: {ex}")
    return False


def _ensure_te_loras(te1, te2, lora_specs: list, gpu, lora_index: dict) -> None:
    """Load/activate PEFT LoRA adapters on the text encoders.

    Mirrors _ensure_loras() for UNet but operates on TE1 (CLIP-L) and TE2 (CLIP-bigG).
    Only loads TE weights from LoRA files that actually contain them.
    """
    from diffusers.loaders.lora_base import _load_lora_into_text_encoder
    from diffusers.loaders.lora_pipeline import _convert_non_diffusers_lora_to_diffusers
    from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers
    from safetensors.torch import load_file as load_safetensors
    from utils import fingerprint as fp_cache

    te1_adapter_names: list[str] = []
    te1_adapter_weights: list[float] = []
    te2_adapter_names: list[str] = []
    te2_adapter_weights: list[float] = []

    for spec in lora_specs:
        entry = lora_index.get(spec.name)
        if entry is None:
            continue

        adapter_name = spec.name.replace(".", "_")

        # Check if this LoRA has TE weights
        cached_te = fp_cache.get_extra(entry.path, "has_te_lora")
        if cached_te == "0":
            continue  # No TE weights in this LoRA

        # Check if already loaded on these TEs
        te1_loaded = hasattr(te1, 'peft_config') and adapter_name in te1.peft_config
        te2_loaded = hasattr(te2, 'peft_config') and adapter_name in te2.peft_config
        if te1_loaded or te2_loaded:
            # Already injected — just track for set_adapters activation
            if te1_loaded:
                te1_adapter_names.append(adapter_name)
                te1_adapter_weights.append(spec.weight)
            if te2_loaded:
                te2_adapter_names.append(adapter_name)
                te2_adapter_weights.append(spec.weight)
            if te1_loaded and te2_loaded:
                continue
            # One TE loaded but not the other — still need to load the missing one

        # Load the LoRA file and convert keys
        try:
            lora_path = entry.path
            if lora_path.endswith(".safetensors"):
                raw_sd = load_safetensors(lora_path, device=str(gpu.device))
            else:
                raw_sd = torch.load(lora_path, map_location=str(gpu.device), weights_only=True)

            is_a1111 = any(k.startswith("lora_unet_") or k.startswith("lora_te") for k in raw_sd)
            if is_a1111:
                # Extract non-block UNet keys before SGM mapper (prevents crash)
                _extract_nonblock_lora_keys(raw_sd)
                # UNet config needed for SGM block mapping — get from cache if available
                unet = _get_cached_model_optional(gpu, "sdxl_unet", None)
                if unet is not None:
                    raw_sd = _maybe_map_sgm_blocks_to_diffusers(raw_sd, unet.config)
                # Snapshot before conversion — _convert_non_diffusers_lora_to_diffusers
                # mutates the input dict via .pop(), so the fallback path needs a copy
                raw_sd_snapshot = dict(raw_sd)
                try:
                    converted_sd, network_alphas = _convert_non_diffusers_lora_to_diffusers(raw_sd)
                except ValueError:
                    # Conversion may fail if UNet block mapping was skipped — we only
                    # need TE keys, so extract them manually before conversion
                    te_keys = {k: v for k, v in raw_sd_snapshot.items() if k.startswith("lora_te")}
                    if not te_keys:
                        continue
                    # Minimal conversion: just TE keys
                    converted_sd, network_alphas = _convert_non_diffusers_lora_to_diffusers(te_keys)
            else:
                converted_sd = raw_sd
                network_alphas = None

            # Check if there are actually TE keys after conversion
            has_te1 = any(k.startswith("text_encoder.") for k in converted_sd)
            has_te2 = any(k.startswith("text_encoder_2.") for k in converted_sd)

            if not has_te1 and not has_te2:
                fp_cache.set_extra(lora_path, has_te_lora="0")
                continue

            t0 = time.monotonic()

            # Repair accelerate leak BEFORE injection — if active, PEFT creates
            # LoRA params on meta device and load_state_dict copy_() is a no-op.
            repair_accelerate_leak()

            if has_te1 and not te1_loaded:
                _load_lora_into_text_encoder(
                    converted_sd, network_alphas, te1,
                    prefix="text_encoder",
                    text_encoder_name="text_encoder",
                    adapter_name=adapter_name,
                )
                repair_accelerate_leak()
                n = fix_meta_tensors(te1)
                if n:
                    log.error(f"  LoRA: {n} meta tensor(s) in TE1 after '{adapter_name}' injection — "
                              f"LoRA weights are CORRUPTED (zeroed). Accelerate leak race condition.")

            if has_te2 and not te2_loaded:
                _load_lora_into_text_encoder(
                    converted_sd, network_alphas, te2,
                    prefix="text_encoder_2",
                    text_encoder_name="text_encoder_2",
                    adapter_name=adapter_name,
                )
                repair_accelerate_leak()
                n = fix_meta_tensors(te2)
                if n:
                    log.error(f"  LoRA: {n} meta tensor(s) in TE2 after '{adapter_name}' injection — "
                              f"LoRA weights are CORRUPTED (zeroed). Accelerate leak race condition.")

            elapsed_ms = (time.monotonic() - t0) * 1000
            te_parts = []
            if has_te1:
                te_parts.append("TE1")
            if has_te2:
                te_parts.append("TE2")
            log.debug(f"  LoRA: Loaded TE adapter '{adapter_name}' ({'+'.join(te_parts)}, {elapsed_ms:.0f}ms)")

            # Track per-TE adapter lists for set_adapters activation
            if has_te1:
                if adapter_name not in te1_adapter_names:
                    te1_adapter_names.append(adapter_name)
                    te1_adapter_weights.append(spec.weight)
            if has_te2:
                if adapter_name not in te2_adapter_names:
                    te2_adapter_names.append(adapter_name)
                    te2_adapter_weights.append(spec.weight)

        except _LoraSkipped:
            continue
        except Exception as ex:
            log.warning(f"  LoRA: Failed to load TE adapter '{adapter_name}': {ex}")
            continue

    # Activate loaded adapters with correct weights, or disable if none
    if te1_adapter_names:
        if hasattr(te1, 'set_adapters'):
            te1.set_adapters(te1_adapter_names, te1_adapter_weights)
    else:
        if hasattr(te1, 'disable_adapters'):
            te1.disable_adapters()
    if te2_adapter_names:
        if hasattr(te2, 'set_adapters'):
            te2.set_adapters(te2_adapter_names, te2_adapter_weights)
    else:
        if hasattr(te2, 'disable_adapters'):
            te2.disable_adapters()


# ====================================================================
# TRT Runner Retrieval
# ====================================================================

def _get_trt_unet_runner(gpu: GpuInstance, job: InferenceJob,
                          width: int, height: int):
    """Try to get a TRT UNet runner for the given resolution.

    Tries static engine (exact resolution match) first for maximum performance,
    then falls back to the best-fitting dynamic engine discovered via JSON sidecars.
    Returns a TrtUNetRunner or None to fall back to PyTorch.
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return None

    from trt.builder import get_arch_key, get_engine_path
    from trt.runner import TrtUNetRunner

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_unet")
    if not fp:
        return None

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache
    dynamic_only = app_state.config.tensorrt.dynamic_only

    # 1. Try static engine for exact resolution match (skip if dynamic_only)
    if not dynamic_only:
        static_path = get_engine_path(cache_dir, fp, "unet", arch_key, width, height)
        if os.path.isfile(static_path):
            trt_fp = f"{fp}:unet_trt:{width}x{height}"
            cached = gpu.get_cached_model(trt_fp)
            if cached is not None:
                log.debug(f"  TRT: Using cached UNet static runner {width}x{height}")
                return cached.model
            try:
                engine_size = os.path.getsize(static_path)
                if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                    raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
                unet_getter = lambda min_bytes: gpu.get_trt_shared_memory("unet", min_bytes)
                runner = TrtUNetRunner(static_path, gpu.device, shared_memory_getter=unet_getter)
                gpu.cache_model(
                    trt_fp, "sdxl_unet_trt", runner,
                    estimated_vram=runner.vram_usage,
                    source=f"TRT-UNet-static-{width}x{height}",
                    evict_callback=runner.unload,
                )
                log.debug(f"  TRT: Loaded UNet static runner {width}x{height} on GPU [{gpu.uuid}]")
                return runner
            except Exception as ex:
                log.warning(f"  TRT: Failed to load UNet static engine {width}x{height}: {ex}")

    # 2. Fall back to best-fitting dynamic engine (discovered via JSON sidecars)
    from trt.builder import find_best_dynamic_engine

    dyn = find_best_dynamic_engine(cache_dir, fp, "unet", arch_key, width, height)
    if dyn is not None:
        label = dyn["label"]
        dynamic_path = dyn["path"]
        trt_fp = f"{fp}:unet_trt:{label}"
        cached = gpu.get_cached_model(trt_fp)
        if cached is not None:
            log.debug(f"  TRT: Using cached UNet {label} runner for {width}x{height}")
            return cached.model
        try:
            engine_size = os.path.getsize(dynamic_path)
            if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
            unet_getter = lambda min_bytes: gpu.get_trt_shared_memory("unet", min_bytes)
            runner = TrtUNetRunner(dynamic_path, gpu.device, shared_memory_getter=unet_getter)
            gpu.cache_model(
                trt_fp, "sdxl_unet_trt", runner,
                estimated_vram=runner.vram_usage,
                source=f"TRT-UNet-{label}",
                evict_callback=runner.unload,
            )
            log.debug(f"  TRT: Loaded UNet {label} runner for {width}x{height} on GPU [{gpu.uuid}]")
            return runner
        except Exception as ex:
            log.warning(f"  TRT: Failed to load UNet {label} engine for {width}x{height}: {ex}")
    else:
        log.debug(f"  TRT: No dynamic UNet engine covers {width}x{height} "
                  f"(profiles are per-axis min/max, rectangular aspect ratios may fall in gaps)")

    return None


def _get_trt_vae_runner(gpu: GpuInstance, job: InferenceJob,
                         width: int, height: int):
    """Try to get a TRT VAE runner for the given resolution.

    Tries static engine first, then best-fitting dynamic engine.
    Returns a TrtVaeRunner or None.
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return None

    from trt.builder import get_arch_key, get_engine_path
    from trt.runner import TrtVaeRunner

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae")
    if not fp:
        return None

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache
    dynamic_only = app_state.config.tensorrt.dynamic_only

    # 1. Try static engine for exact resolution match (skip if dynamic_only)
    if not dynamic_only:
        static_path = get_engine_path(cache_dir, fp, "vae", arch_key, width, height)
        if os.path.isfile(static_path):
            trt_fp = f"{fp}:vae_trt:{width}x{height}"
            cached = gpu.get_cached_model(trt_fp)
            if cached is not None:
                log.debug(f"  TRT: Using cached VAE static runner {width}x{height}")
                return cached.model
            try:
                engine_size = os.path.getsize(static_path)
                if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                    raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
                vae_getter = lambda min_bytes: gpu.get_trt_shared_memory("vae", min_bytes)
                runner = TrtVaeRunner(static_path, gpu.device, shared_memory_getter=vae_getter)
                gpu.cache_model(
                    trt_fp, "sdxl_vae_trt", runner,
                    estimated_vram=runner.vram_usage,
                    source=f"TRT-VAE-static-{width}x{height}",
                    evict_callback=runner.unload,
                )
                log.debug(f"  TRT: Loaded VAE static runner {width}x{height} on GPU [{gpu.uuid}]")
                return runner
            except Exception as ex:
                log.warning(f"  TRT: Failed to load VAE static engine {width}x{height}: {ex}")

    # 2. Fall back to best-fitting dynamic engine (discovered via JSON sidecars)
    from trt.builder import find_best_dynamic_engine

    dyn = find_best_dynamic_engine(cache_dir, fp, "vae", arch_key, width, height)
    if dyn is not None:
        label = dyn["label"]
        dynamic_path = dyn["path"]
        trt_fp = f"{fp}:vae_trt:{label}"
        cached = gpu.get_cached_model(trt_fp)
        if cached is not None:
            log.debug(f"  TRT: Using cached VAE {label} runner for {width}x{height}")
            return cached.model
        try:
            engine_size = os.path.getsize(dynamic_path)
            if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
            vae_getter = lambda min_bytes: gpu.get_trt_shared_memory("vae", min_bytes)
            runner = TrtVaeRunner(dynamic_path, gpu.device, shared_memory_getter=vae_getter)
            gpu.cache_model(
                trt_fp, "sdxl_vae_trt", runner,
                estimated_vram=runner.vram_usage,
                source=f"TRT-VAE-{label}",
                evict_callback=runner.unload,
            )
            log.debug(f"  TRT: Loaded VAE {label} runner for {width}x{height} on GPU [{gpu.uuid}]")
            return runner
        except Exception as ex:
            log.warning(f"  TRT: Failed to load VAE {label} engine for {width}x{height}: {ex}")
    else:
        log.debug(f"  TRT: No dynamic VAE engine covers {width}x{height}")

    return None


def _get_trt_vae_enc_runner(gpu: "GpuInstance", job: InferenceJob,
                             width: int, height: int):
    """Try to get a TRT VAE encoder runner for the given resolution.

    Mirrors _get_trt_vae_runner() but for the encoder component type.
    Uses the same VAE model fingerprint (same weights, different ONNX graph).
    Returns a TrtVaeEncoderRunner or None.
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return None

    from trt.builder import get_arch_key, get_engine_path
    from trt.runner import TrtVaeEncoderRunner

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_vae")
    if not fp:
        return None

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache
    dynamic_only = app_state.config.tensorrt.dynamic_only

    # 1. Try static engine for exact resolution match (skip if dynamic_only)
    if not dynamic_only:
        static_path = get_engine_path(cache_dir, fp, "vae_enc", arch_key, width, height)
        if os.path.isfile(static_path):
            trt_fp = f"{fp}:vae_enc_trt:{width}x{height}"
            cached = gpu.get_cached_model(trt_fp)
            if cached is not None:
                log.debug(f"  TRT: Using cached VAE encoder static runner {width}x{height}")
                return cached.model
            try:
                engine_size = os.path.getsize(static_path)
                if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                    raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
                enc_getter = lambda min_bytes: gpu.get_trt_shared_memory("vae_enc", min_bytes)
                runner = TrtVaeEncoderRunner(static_path, gpu.device, shared_memory_getter=enc_getter)
                gpu.cache_model(
                    trt_fp, "sdxl_vae_enc_trt", runner,
                    estimated_vram=runner.vram_usage,
                    source=f"TRT-VAE-Enc-static-{width}x{height}",
                    evict_callback=runner.unload,
                )
                log.debug(f"  TRT: Loaded VAE encoder static runner {width}x{height} on GPU [{gpu.uuid}]")
                return runner
            except Exception as ex:
                log.warning(f"  TRT: Failed to load VAE encoder static engine {width}x{height}: {ex}")

    # 2. Fall back to best-fitting dynamic engine
    from trt.builder import find_best_dynamic_engine

    dyn = find_best_dynamic_engine(cache_dir, fp, "vae_enc", arch_key, width, height)
    if dyn is not None:
        label = dyn["label"]
        dynamic_path = dyn["path"]
        trt_fp = f"{fp}:vae_enc_trt:{label}"
        cached = gpu.get_cached_model(trt_fp)
        if cached is not None:
            log.debug(f"  TRT: Using cached VAE encoder {label} runner for {width}x{height}")
            return cached.model
        try:
            engine_size = os.path.getsize(dynamic_path)
            if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
                raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
            enc_getter = lambda min_bytes: gpu.get_trt_shared_memory("vae_enc", min_bytes)
            runner = TrtVaeEncoderRunner(dynamic_path, gpu.device, shared_memory_getter=enc_getter)
            gpu.cache_model(
                trt_fp, "sdxl_vae_enc_trt", runner,
                estimated_vram=runner.vram_usage,
                source=f"TRT-VAE-Enc-{label}",
                evict_callback=runner.unload,
            )
            log.debug(f"  TRT: Loaded VAE encoder {label} runner for {width}x{height} on GPU [{gpu.uuid}]")
            return runner
        except Exception as ex:
            log.warning(f"  TRT: Failed to load VAE encoder {label} engine for {width}x{height}: {ex}")
    else:
        log.debug(f"  TRT: No dynamic VAE encoder engine covers {width}x{height}")

    return None


def _get_trt_te1_runner(gpu: "GpuInstance", job: InferenceJob):
    """Try to get a TRT TE1 (CLIP-L) runner.

    Text encoders have a single "default" engine (no resolution variants).
    Returns a TrtTe1Runner or None to fall back to PyTorch.
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return None

    from trt.builder import get_arch_key, get_dynamic_engine_path
    from trt.runner import TrtTe1Runner

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te1")
    if not fp:
        return None

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache

    engine_path = get_dynamic_engine_path(cache_dir, fp, "te1", arch_key, "default")
    if not os.path.isfile(engine_path):
        return None

    trt_fp = f"{fp}:te1_trt:default"
    cached = gpu.get_cached_model(trt_fp)
    if cached is not None:
        return cached.model

    try:
        engine_size = os.path.getsize(engine_path)
        if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
            raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
        te1_getter = lambda min_bytes: gpu.get_trt_shared_memory("te1", min_bytes)
        runner = TrtTe1Runner(engine_path, gpu.device, shared_memory_getter=te1_getter)
        gpu.cache_model(
            trt_fp, "sdxl_te1_trt", runner,
            estimated_vram=runner.vram_usage,
            source="TRT-TE1",
            evict_callback=runner.unload,
        )
        log.debug(f"  TRT: Loaded TE1 runner on GPU [{gpu.uuid}]")
        return runner
    except Exception as ex:
        log.warning(f"  TRT: Failed to load TE1 engine: {ex}")
        return None


def _get_trt_te2_runner(gpu: "GpuInstance", job: InferenceJob):
    """Try to get a TRT TE2 (CLIP-bigG) runner.

    Returns a TrtTe2Runner or None to fall back to PyTorch.
    """
    from state import app_state
    if not app_state.config.tensorrt.enabled:
        return None

    from trt.builder import get_arch_key, get_dynamic_engine_path
    from trt.runner import TrtTe2Runner

    fp = getattr(job, '_stage_model_fps', {}).get("sdxl_te2")
    if not fp:
        return None

    arch_key = get_arch_key(gpu.device_id)

    cache_dir = app_state.config.server.tensorrt_cache

    engine_path = get_dynamic_engine_path(cache_dir, fp, "te2", arch_key, "default")
    if not os.path.isfile(engine_path):
        return None

    trt_fp = f"{fp}:te2_trt:default"
    cached = gpu.get_cached_model(trt_fp)
    if cached is not None:
        return cached.model

    try:
        engine_size = os.path.getsize(engine_path)
        if not gpu.ensure_free_vram(engine_size, protect={trt_fp}):
            raise RuntimeError(f"Insufficient VRAM for TRT engine ({engine_size // (1024*1024)}MB)")
        te2_getter = lambda min_bytes: gpu.get_trt_shared_memory("te2", min_bytes)
        runner = TrtTe2Runner(engine_path, gpu.device, shared_memory_getter=te2_getter)
        gpu.cache_model(
            trt_fp, "sdxl_te2_trt", runner,
            estimated_vram=runner.vram_usage,
            source="TRT-TE2",
            evict_callback=runner.unload,
        )
        log.debug(f"  TRT: Loaded TE2 runner on GPU [{gpu.uuid}]")
        return runner
    except Exception as ex:
        log.warning(f"  TRT: Failed to load TE2 engine: {ex}")
        return None


def _run_trt_te1(
    runner, chunks_ids: list[list[int]], chunks_masks: list[list[int]],
    device: torch.device,
) -> torch.Tensor:
    """Run TRT TE1 over multiple chunks (batched).
    Returns hidden states [1, 77*N, 768]."""
    input_ids = torch.tensor(chunks_ids, dtype=torch.long, device=device)        # [N, 77]
    attention_mask = torch.tensor(chunks_masks, dtype=torch.long, device=device)  # [N, 77]
    hidden = runner.run(input_ids, attention_mask)  # [N, 77, 768]
    return hidden.reshape(1, -1, hidden.shape[-1])  # [1, 77*N, 768]


def _run_trt_te2(
    runner, chunks_ids: list[list[int]], chunks_masks: list[list[int]],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run TRT TE2 over multiple chunks (batched).
    Returns (hidden_states [1,77*N,1280], pooled [1,1280] from first chunk)."""
    input_ids = torch.tensor(chunks_ids, dtype=torch.long, device=device)        # [N, 77]
    attention_mask = torch.tensor(chunks_masks, dtype=torch.long, device=device)  # [N, 77]
    hidden, text_embeds = runner.run(input_ids, attention_mask)  # [N,77,1280], [N,1280]
    return hidden.reshape(1, -1, hidden.shape[-1]), text_embeds[0:1]  # [1,77*N,1280], [1,1280]


def _get_cached_model(gpu: GpuInstance, category: str,
                      job: "InferenceJob | None" = None) -> object:
    """Retrieve a loaded model from the GPU cache.

    When *job* is provided and carries ``_stage_model_fps`` (set by the worker
    before execution), the exact fingerprint is used for lookup.  This prevents
    returning the wrong model when multiple instances of the same category are
    cached (e.g. two sdxl_te1 from different checkpoints) — the LRU entry might
    be unprotected by active fingerprints and get evicted mid-inference.
    """
    # Exact fingerprint lookup (preferred — race-safe)
    fp = getattr(job, '_stage_model_fps', {}).get(category) if job else None
    if fp:
        cached = gpu.get_cached_model(fp)
        if cached:
            return cached.model
        log.warning(f"  _get_cached_model: fingerprint lookup FAILED for "
                    f"{category} fp={fp[:16]}… on GPU [{gpu.uuid}] — "
                    f"falling back to category scan")

    # Fallback: category scan (for callers without fingerprint context)
    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == category:
                if fp:
                    log.warning(f"  _get_cached_model: category fallback found "
                                f"{category} fp={cached.fingerprint[:16]}… "
                                f"(wanted {fp[:16]}…)")
                return cached.model
    raise RuntimeError(f"Model {category} not found in GPU [{gpu.uuid}] cache")


def _get_cached_model_optional(gpu: GpuInstance, category: str,
                               job: "InferenceJob | None" = None) -> object | None:
    """Retrieve a loaded model from the GPU cache, or None if not loaded."""
    fp = getattr(job, '_stage_model_fps', {}).get(category) if job else None
    if fp:
        cached = gpu.get_cached_model(fp)
        if cached:
            return cached.model
        log.warning(f"  _get_cached_model_optional: fingerprint lookup FAILED for "
                    f"{category} fp={fp[:16]}… on GPU [{gpu.uuid}] — "
                    f"falling back to category scan")

    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == category:
                if fp:
                    log.warning(f"  _get_cached_model_optional: category fallback found "
                                f"{category} fp={cached.fingerprint[:16]}… "
                                f"(wanted {fp[:16]}…)")
                return cached.model
    return None
