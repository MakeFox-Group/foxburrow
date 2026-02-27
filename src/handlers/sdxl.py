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
from scheduling.job import InferenceJob, SdxlTokenizeResult, SdxlEncodeResult, SdxlRegionalEncodeResult

if TYPE_CHECKING:
    from gpu.pool import GpuInstance

# Constants
VAE_SCALE_FACTOR = 0.13025
VAE_TILE_THRESHOLD = 1024    # force tiled mode when any image dimension >= this (pixels)
VAE_TILE_MAX = 768           # max tile size per axis when tiling (pixels)
LATENT_TILE_OVERLAP = 16     # latent overlap for VAE encode/decode tiles

# MultiDiffusion (tiled UNet) constants
UNET_TILE_MAX = 1024          # default max UNet tile size (pixels) when auto-calculating
UNET_TILE_OVERLAP = 48        # latent overlap (= 384px) for smooth blending
UNET_TILE_THRESHOLD = 256     # auto-enable tiling when latent dim > this (> 2048px)


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


def _auto_unet_tile(lat_w: int, lat_h: int, max_tile_px: int) -> tuple[int, int]:
    """Return (tile_lat_w, tile_lat_h) in latent units, each ≤ max_tile_px // 8."""
    def _axis(dim: int) -> int:
        max_lat = max_tile_px // 8
        if dim <= max_lat:
            return max_lat
        n = math.ceil(dim / max_lat)
        return math.ceil(dim / n)
    return _axis(lat_w), _axis(lat_h)


# Tokenizer instances (loaded once, CPU-only)
_tokenizer_1: CLIPTokenizer | None = None
_tokenizer_2: CLIPTokenizer | None = None

# Cache for components extracted from single-file checkpoints.
# Maps checkpoint_path -> {category: model}.  Models are moved between
# CPU and GPU by the worker/pool — the same object reference is shared,
# so eviction (.to("cpu")) keeps the reference alive here for fast reload.
_checkpoint_cache: dict[str, dict[str, object]] = {}
_extraction_lock = threading.Lock()
# Per-path events: threads waiting on the same checkpoint being extracted
# by another thread block on the event, not the global lock.
_extraction_events: dict[str, threading.Event] = {}

# Prompt emphasis regex — A1111/Forge compatible
_RE_ATTENTION = re.compile(
    r"\\\(|\\\)|\\\[|\\]|\\\\|\\|\(|\[|:\s*([+-]?[\d.]+)\s*\)|\)|]|[^\\()\[\]:]+|:"
)


# ====================================================================
# Model Loading
# ====================================================================

def _is_single_file(path: str) -> bool:
    """Check if path is a single-file safetensors checkpoint."""
    return os.path.isfile(path) and path.endswith(".safetensors")


def _ensure_checkpoint_extracted(checkpoint_path: str) -> dict[str, object]:
    """Extract all components from a single-file SDXL checkpoint.

    Thread-safe: different checkpoints extract in parallel.  If two threads
    request the *same* checkpoint, the second waits for the first to finish.

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

        # Another thread already extracting this exact checkpoint?
        if checkpoint_path in _extraction_events:
            wait_event = _extraction_events[checkpoint_path]
        else:
            # We claim this checkpoint — create event for future waiters
            wait_event = None
            _extraction_events[checkpoint_path] = threading.Event()

    if wait_event is not None:
        # Wait for the other thread that's extracting the same checkpoint
        wait_event.wait()
        result = _checkpoint_cache.get(checkpoint_path)
        if result is None:
            raise RuntimeError(
                f"Checkpoint extraction failed for {os.path.basename(checkpoint_path)}")
        return result

    # We're the extractor — heavy work runs lock-free (parallel with other checkpoints)
    try:
        log.info(f"  SDXL: Extracting components from "
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

        # Enable xformers on UNet if available (must be done before moving to CPU —
        # some diffusers versions require the model on a CUDA device for this call)
        try:
            components["sdxl_unet"].enable_xformers_memory_efficient_attention()
            log.info("  SDXL: xformers enabled on UNet")
        except Exception as ex:
            log.info(f"  SDXL: xformers not available, using default attention ({ex})")

        # Move neural-net components to CPU and set eval mode
        for key in ("sdxl_te1", "sdxl_te2", "sdxl_unet", "sdxl_vae"):
            components[key].to("cpu")
            components[key].eval()

        del pipe
        torch.cuda.empty_cache()

        with _extraction_lock:
            _checkpoint_cache[checkpoint_path] = components

        log.info(f"  SDXL: All components extracted to CPU cache")
        return components
    finally:
        # Wake any threads waiting on this checkpoint (success or failure)
        with _extraction_lock:
            done_event = _extraction_events.pop(checkpoint_path, None)
        if done_event is not None:
            done_event.set()


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

    log.info("  SDXL: Tokenizers loaded")


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
        model.to(device)
        model.eval()
        log.info(f"  SDXL: Loaded text_encoder (CLIP-L) to {device}")
        return model

    elif category == "sdxl_te2":
        model = CLIPTextModelWithProjection.from_pretrained(
            model_dir, subfolder="text_encoder_2", torch_dtype=dtype)
        model.to(device)
        model.eval()
        log.info(f"  SDXL: Loaded text_encoder_2 (CLIP-bigG) to {device}")
        return model

    elif category == "sdxl_unet":
        from diffusers import UNet2DConditionModel
        model = UNet2DConditionModel.from_pretrained(
            model_dir, subfolder="unet", torch_dtype=dtype)
        model.to(device)
        model.eval()
        try:
            model.enable_xformers_memory_efficient_attention()
            log.info(f"  SDXL: Loaded UNet with xformers to {device}")
        except Exception:
            log.info(f"  SDXL: Loaded UNet (no xformers) to {device}")
        return model

    elif category in ("sdxl_vae", "sdxl_vae_enc"):
        model = AutoencoderKL.from_pretrained(
            model_dir, subfolder="vae", torch_dtype=dtype)
        model.to(device)
        model.eval()
        log.info(f"  SDXL: Loaded VAE to {device}")
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

    # Deep-copy the CPU-resident model so each GPU gets an independent copy.
    # Without this, .to(device) would move the shared reference and break
    # any other GPU that was using the same model object.
    model = copy.deepcopy(components[cache_key])
    model.to(device)
    log.info(f"  SDXL: Moved {category} to {device} (from checkpoint cache)")
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
        log.info(f"  SDXL: Regional prompting detected — {len(regional.regions)} regions"
                 + (f" + base" if regional.base_prompt else ""))

        # Tokenize each region's prompt + shared negative
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
        job.regional_tokenize_results = region_toks

        # Tokenize the global/shared negative (used for pooled, ADDBASE neg, and fallback)
        gn1_ids, gn1_weights, gn1_mask = _tokenize_weighted(_tokenizer_1, regional.negative_prompt, CLIP_L_PAD)
        gn2_ids, gn2_weights, gn2_mask = _tokenize_weighted(_tokenizer_2, regional.negative_prompt, CLIP_G_PAD)
        job.regional_shared_neg_tokenize = SdxlTokenizeResult(
            prompt_tokens_1=[], prompt_weights_1=[],
            neg_tokens_1=gn1_ids, neg_weights_1=gn1_weights,
            prompt_tokens_2=[], prompt_weights_2=[],
            neg_tokens_2=gn2_ids, neg_weights_2=gn2_weights,
            prompt_mask_1=[], neg_mask_1=gn1_mask,
            prompt_mask_2=[], neg_mask_2=gn2_mask,
        )

        # Tokenize base prompt if present
        if regional.base_prompt:
            bp1_ids, bp1_weights, bp1_mask = _tokenize_weighted(_tokenizer_1, regional.base_prompt, CLIP_L_PAD)
            bp2_ids, bp2_weights, bp2_mask = _tokenize_weighted(_tokenizer_2, regional.base_prompt, CLIP_G_PAD)
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

    log.info("  SDXL: Running text encoders...")

    # Get models from GPU cache
    te1 = _get_cached_model(gpu, "sdxl_te1")
    te2 = _get_cached_model(gpu, "sdxl_te2")

    device = gpu.device

    # TextEncoder1: hidden states [1, 77, 768]
    p_h1 = _run_text_encoder_1(te1, tok.prompt_tokens_1, tok.prompt_mask_1, device)
    n_h1 = _run_text_encoder_1(te1, tok.neg_tokens_1, tok.neg_mask_1, device)

    # TextEncoder2: hidden states [1, 77, 1280] + pooled [1, 1280]
    p_h2, p_pooled = _run_text_encoder_2(te2, tok.prompt_tokens_2, tok.prompt_mask_2, device)
    n_h2, n_pooled = _run_text_encoder_2(te2, tok.neg_tokens_2, tok.neg_mask_2, device)

    # Apply emphasis weights to hidden states (not pooled)
    _apply_token_weights(p_h1, tok.prompt_weights_1)
    _apply_token_weights(n_h1, tok.neg_weights_1)
    _apply_token_weights(p_h2, tok.prompt_weights_2)
    _apply_token_weights(n_h2, tok.neg_weights_2)

    # Concatenate hidden states along dim 2: [1,77,768]+[1,77,1280] = [1,77,2048]
    prompt_embeds = torch.cat([p_h1, p_h2], dim=2)
    neg_prompt_embeds = torch.cat([n_h1, n_h2], dim=2)

    # Pooled comes from TE2 only (NOT affected by emphasis)
    pooled_prompt_embeds = p_pooled
    neg_pooled_prompt_embeds = n_pooled

    # Forge: zero out all negative embeddings when negative prompt is empty
    if not inp.negative_prompt or not inp.negative_prompt.strip():
        log.info("  SDXL: Empty negative prompt — zeroing embeddings (Forge behavior)")
        neg_prompt_embeds.zero_()
        neg_pooled_prompt_embeds.zero_()

    log.info(f"  SDXL: Text encoding complete. embeds={list(prompt_embeds.shape)} "
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

    log.info(f"  SDXL: Running regional text encoders ({len(region_toks)} regions)...")

    te1 = _get_cached_model(gpu, "sdxl_te1")
    te2 = _get_cached_model(gpu, "sdxl_te2")
    device = gpu.device

    region_embeds: list[torch.Tensor] = []
    first_pooled = None

    for i, tok in enumerate(region_toks):
        # Encode this region's prompt
        p_h1 = _run_text_encoder_1(te1, tok.prompt_tokens_1, tok.prompt_mask_1, device)
        p_h2, p_pooled = _run_text_encoder_2(te2, tok.prompt_tokens_2, tok.prompt_mask_2, device)

        _apply_token_weights(p_h1, tok.prompt_weights_1)
        _apply_token_weights(p_h2, tok.prompt_weights_2)

        embeds = torch.cat([p_h1, p_h2], dim=2)  # [1, 77, 2048]
        region_embeds.append(embeds)

        if i == 0:
            first_pooled = p_pooled

    # Always encode the global/shared negative (for pooled, ADDBASE neg, and non-regional fallback)
    shared_neg_tok = job.regional_shared_neg_tokenize
    if shared_neg_tok is None:
        raise RuntimeError("regional_shared_neg_tokenize is required for regional text encoding.")
    n_h1 = _run_text_encoder_1(te1, shared_neg_tok.neg_tokens_1, shared_neg_tok.neg_mask_1, device)
    n_h2, n_pooled = _run_text_encoder_2(te2, shared_neg_tok.neg_tokens_2, shared_neg_tok.neg_mask_2, device)
    _apply_token_weights(n_h1, shared_neg_tok.neg_weights_1)
    _apply_token_weights(n_h2, shared_neg_tok.neg_weights_2)
    neg_embeds = torch.cat([n_h1, n_h2], dim=2)

    # Encode per-region negatives if they differ across regions
    neg_region_embeds: list[torch.Tensor] | None = None
    if regional.has_per_region_neg:
        neg_region_embeds = []
        for tok in region_toks:
            nr_h1 = _run_text_encoder_1(te1, tok.neg_tokens_1, tok.neg_mask_1, device)
            nr_h2, _ = _run_text_encoder_2(te2, tok.neg_tokens_2, tok.neg_mask_2, device)
            _apply_token_weights(nr_h1, tok.neg_weights_1)
            _apply_token_weights(nr_h2, tok.neg_weights_2)
            neg_region_embeds.append(torch.cat([nr_h1, nr_h2], dim=2))
        log.info(f"  SDXL: Encoded {len(neg_region_embeds)} per-region negatives")

    # Encode base prompt if present
    base_embeds = None
    base_pooled = None
    if regional.base_prompt and job.regional_base_tokenize is not None:
        bt = job.regional_base_tokenize
        b_h1 = _run_text_encoder_1(te1, bt.prompt_tokens_1, bt.prompt_mask_1, device)
        b_h2, b_pooled = _run_text_encoder_2(te2, bt.prompt_tokens_2, bt.prompt_mask_2, device)
        _apply_token_weights(b_h1, bt.prompt_weights_1)
        _apply_token_weights(b_h2, bt.prompt_weights_2)
        base_embeds = torch.cat([b_h1, b_h2], dim=2)
        base_pooled = b_pooled

    # Pooled: use base prompt's pooled if ADDBASE, otherwise first region's
    pooled = base_pooled if base_pooled is not None else first_pooled

    # Forge: zero out negative embeddings when negative prompt is empty
    neg_text = regional.negative_prompt
    if not neg_text or not neg_text.strip():
        log.info("  SDXL: Empty negative prompt — zeroing embeddings (Forge behavior)")
        neg_embeds.zero_()
        n_pooled.zero_()
        if neg_region_embeds is not None:
            for nre in neg_region_embeds:
                nre.zero_()

    log.info(f"  SDXL: Regional text encoding complete. "
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


def _gaussian_weights(h: int, w: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """2D Gaussian weight mask [1,1,h,w] for MultiDiffusion tile blending.
    Peaks at center, near-zero at edges — ensures seamless tile boundaries."""
    sigma = 0.5
    y = torch.linspace(-1.0, 1.0, h, device=device, dtype=dtype)
    x = torch.linspace(-1.0, 1.0, w, device=device, dtype=dtype)
    gy = torch.exp(-y ** 2 / (2 * sigma ** 2))
    gx = torch.exp(-x ** 2 / (2 * sigma ** 2))
    return (gy.unsqueeze(1) * gx.unsqueeze(0)).unsqueeze(0).unsqueeze(0)  # [1,1,h,w]


def _unet_tiled(
    unet, latent_input: torch.Tensor, t,
    prompt_embeds: torch.Tensor, neg_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor, neg_pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor, cfg_scale: float,
    tile_w: int, tile_h: int, tile_overlap: int,
) -> torch.Tensor:
    """MultiDiffusion: tile the UNet forward pass and blend noise predictions.

    Runs the UNet on overlapping tile_w×tile_h latent tiles, accumulates noise
    predictions weighted by a 2D Gaussian, then normalises.
    The full-resolution add_time_ids are passed to every tile so SDXL's size
    conditioning always sees the intended output dimensions.
    """
    device = latent_input.device
    lat_h, lat_w = latent_input.shape[2], latent_input.shape[3]
    stride_x = tile_w - tile_overlap
    stride_y = tile_h - tile_overlap
    tiles_y = max(1, math.ceil((lat_h - tile_overlap) / stride_y))
    tiles_x = max(1, math.ceil((lat_w - tile_overlap) / stride_x))

    noise_sum  = torch.zeros_like(latent_input)
    weight_sum = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=latent_input.dtype)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = min(ty * stride_y, lat_h - tile_h)
            x0 = min(tx * stride_x, lat_w - tile_w)
            y0, x0 = max(0, y0), max(0, x0)
            y1 = min(y0 + tile_h, lat_h)
            x1 = min(x0 + tile_w, lat_w)

            tile = latent_input[:, :, y0:y1, x0:x1]

            with torch.no_grad():
                # Batch [uncond, cond] for CFG in a single UNet pass
                tile_in = torch.cat([tile, tile])
                out = unet(
                    tile_in, t,
                    encoder_hidden_states=torch.cat([neg_prompt_embeds, prompt_embeds]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds]),
                        "time_ids": torch.cat([add_time_ids, add_time_ids]),
                    },
                ).sample
                e_uncond, e_cond = out.chunk(2)

            noise_pred = e_uncond + cfg_scale * (e_cond - e_uncond)
            w = _gaussian_weights(y1 - y0, x1 - x0, device, latent_input.dtype)
            noise_sum [:, :, y0:y1, x0:x1] += noise_pred * w
            weight_sum[:, :, y0:y1, x0:x1] += w

    return noise_sum / weight_sum


def _broadcast_ws_progress(job: InferenceJob) -> None:
    """Broadcast denoise progress via WebSocket (thread-safe — called from executor)."""
    try:
        from api.websocket import streamer as _ws_streamer
        import asyncio as _asyncio
        _asyncio.run_coroutine_threadsafe(_ws_streamer.broadcast_progress(job), job._loop)
    except Exception:
        pass


def denoise(job: InferenceJob, gpu: GpuInstance) -> None:
    """Stage 3: GPU denoising with UNet + EulerAncestral scheduler."""
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

    unet = _get_cached_model(gpu, "sdxl_unet")
    device = gpu.device

    # Apply LoRA adapters if requested
    if inp.loras:
        from state import app_state
        _ensure_loras(unet, inp.loras, gpu, app_state.lora_index)
    elif hasattr(unet, 'peft_config') and unet.peft_config:
        unet.disable_adapters()  # clean UNet for non-LoRA jobs

    # Create scheduler
    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        timestep_spacing="linspace",
    )
    scheduler.set_timesteps(steps, device=device)
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
            log.info(f"  SDXL: Hires denoising skipped (strength={strength:.2f}, no active steps)")
            return

        # Add noise at the start timestep
        upscaled_latents = upscaled_latents.to(device=device, dtype=torch.float16)
        noise = torch.randn(upscaled_latents.shape, generator=generator,
                            device=device, dtype=upscaled_latents.dtype)
        latents = scheduler.add_noise(upscaled_latents, noise,
                                      timesteps[start_step:start_step+1])

        active_count = len(timesteps) - start_step
        log.info(f"  SDXL: Hires denoising (strength={strength:.2f}, {active_count} active steps "
                 f"of {len(timesteps)} total, startStep={start_step})")
    else:
        # Base pass: start from pure noise
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator, device=device, dtype=torch.float16,
        ) * scheduler.init_noise_sigma
        start_step = 0

        log.info(f"  SDXL: Denoising ({len(timesteps)} steps, "
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

    prompt_embeds = enc.prompt_embeds
    neg_prompt_embeds = enc.neg_prompt_embeds
    pooled_prompt_embeds = enc.pooled_prompt_embeds
    neg_pooled_prompt_embeds = enc.neg_pooled_prompt_embeds

    # MultiDiffusion: tile the UNet when latent exceeds threshold
    # unet_tile_width/height from job (pixels) → latent units; 0 = auto (≤ UNET_TILE_MAX)
    if job.unet_tile_width > 0 and job.unet_tile_height > 0:
        unet_tile_w = job.unet_tile_width  // 8
        unet_tile_h = job.unet_tile_height // 8
    else:
        unet_tile_w, unet_tile_h = _auto_unet_tile(latent_w, latent_h, UNET_TILE_MAX)
    use_tiled = latent_h > UNET_TILE_THRESHOLD or latent_w > UNET_TILE_THRESHOLD
    if use_tiled:
        _stride_x = unet_tile_w - UNET_TILE_OVERLAP
        _stride_y = unet_tile_h - UNET_TILE_OVERLAP
        _tiles_y = max(1, math.ceil((latent_h - UNET_TILE_OVERLAP) / _stride_y))
        _tiles_x = max(1, math.ceil((latent_w - UNET_TILE_OVERLAP) / _stride_x))
        log.info(f"  SDXL: MultiDiffusion enabled — "
                 f"{_tiles_x}x{_tiles_y} tiles "
                 f"({unet_tile_w*8}x{unet_tile_h*8}px, "
                 f"{UNET_TILE_OVERLAP*8}px overlap)")

    for i in range(start_step, len(timesteps)):
        job.denoise_step = i - start_step + 1
        _broadcast_ws_progress(job)
        t = timesteps[i]

        latent_input = scheduler.scale_model_input(latents, t)

        if use_tiled:
            noise_pred = _unet_tiled(
                unet, latent_input, t,
                prompt_embeds, neg_prompt_embeds,
                pooled_prompt_embeds, neg_pooled_prompt_embeds,
                add_time_ids, inp.cfg_scale,
                unet_tile_w, unet_tile_h, UNET_TILE_OVERLAP,
            )
        else:
            with torch.no_grad():
                # Batched CFG: run uncond + cond in a single UNet forward pass
                latent_in = torch.cat([latent_input, latent_input])
                out = unet(
                    latent_in, t,
                    encoder_hidden_states=torch.cat([neg_prompt_embeds, prompt_embeds]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds]),
                        "time_ids": torch.cat([add_time_ids, add_time_ids]),
                    },
                ).sample
                noise_pred_uncond, noise_pred_cond = out.chunk(2)
            noise_pred = noise_pred_uncond + inp.cfg_scale * (noise_pred_cond - noise_pred_uncond)

        latents = scheduler.step(noise_pred, t, latents, generator=generator).prev_sample

    job.latents = latents


def _unet_tiled_regional(
    unet, latent_input: torch.Tensor, t,
    neg_prompt_embeds: torch.Tensor,
    pooled_prompt_embeds: torch.Tensor, neg_pooled_prompt_embeds: torch.Tensor,
    add_time_ids: torch.Tensor, cfg_scale: float,
    tile_w: int, tile_h: int, tile_overlap: int,
    state,  # RegionalAttnState
    full_masks: torch.Tensor,  # [N, 1, latent_h, latent_w]
    pos_stacked: torch.Tensor,  # [N, 77, dim] — positive region embeds
    neg_stacked: torch.Tensor | None = None,  # [N, 77, dim] — per-region neg embeds, or None
) -> torch.Tensor:
    """MultiDiffusion with regional prompting: tile the UNet and blend noise predictions.

    For each tile, crops the region masks and updates the attention state.
    If neg_stacked is provided, the uncond pass also uses regional attention.
    """
    device = latent_input.device
    lat_h, lat_w = latent_input.shape[2], latent_input.shape[3]
    stride_x = tile_w - tile_overlap
    stride_y = tile_h - tile_overlap
    tiles_y = max(1, math.ceil((lat_h - tile_overlap) / stride_y))
    tiles_x = max(1, math.ceil((lat_w - tile_overlap) / stride_x))

    noise_sum  = torch.zeros_like(latent_input)
    weight_sum = torch.zeros(1, 1, lat_h, lat_w, device=device, dtype=latent_input.dtype)

    # Set up batched CFG state once — processor handles per-element routing
    state.region_embeds = pos_stacked
    state.active = True
    if neg_stacked is not None:
        state.uncond_region_embeds = neg_stacked
        state.uncond_text_embeds = None
    else:
        state.uncond_region_embeds = None
        state.uncond_text_embeds = neg_prompt_embeds

    uncond_enc = neg_stacked[0:1] if neg_stacked is not None else neg_prompt_embeds

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            y0 = min(ty * stride_y, lat_h - tile_h)
            x0 = min(tx * stride_x, lat_w - tile_w)
            y0, x0 = max(0, y0), max(0, x0)
            y1 = min(y0 + tile_h, lat_h)
            x1 = min(x0 + tile_w, lat_w)

            tile = latent_input[:, :, y0:y1, x0:x1]

            # Crop region masks to this tile's bounds
            tile_masks = full_masks[:, :, y0:y1, x0:x1]
            # Renormalize tile masks
            tile_mask_sum = tile_masks.sum(dim=0, keepdim=True).clamp(min=1e-8)
            tile_masks = tile_masks / tile_mask_sum
            state.set_tile_masks(tile_masks, tile_bounds=(y0, x0, y1, x1))

            with torch.no_grad():
                # Batched CFG: uncond + cond in a single UNet pass per tile
                tile_in = torch.cat([tile, tile])
                out = unet(
                    tile_in, t,
                    encoder_hidden_states=torch.cat([uncond_enc, pos_stacked[0:1]]),
                    added_cond_kwargs={
                        "text_embeds": torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds]),
                        "time_ids": torch.cat([add_time_ids, add_time_ids]),
                    },
                ).sample
                noise_pred_uncond, noise_pred_cond = out.chunk(2)

            noise_pred = noise_pred_uncond + cfg_scale * (noise_pred_cond - noise_pred_uncond)
            w = _gaussian_weights(y1 - y0, x1 - x0, device, latent_input.dtype)
            noise_sum [:, :, y0:y1, x0:x1] += noise_pred * w
            weight_sum[:, :, y0:y1, x0:x1] += w

    state.active = False
    state.uncond_region_embeds = None
    state.uncond_text_embeds = None
    return noise_sum / weight_sum


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

    unet = _get_cached_model(gpu, "sdxl_unet")
    device = gpu.device

    # Apply LoRA adapters if requested
    if inp.loras:
        from state import app_state
        _ensure_loras(unet, inp.loras, gpu, app_state.lora_index)
    elif hasattr(unet, 'peft_config') and unet.peft_config:
        unet.disable_adapters()

    # Create scheduler
    scheduler = EulerAncestralDiscreteScheduler(
        beta_start=0.00085,
        beta_end=0.012,
        beta_schedule="scaled_linear",
        num_train_timesteps=1000,
        prediction_type="epsilon",
        timestep_spacing="linspace",
    )
    scheduler.set_timesteps(steps, device=device)
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
            log.info(f"  SDXL: Regional hires denoising skipped (strength={strength:.2f}, no active steps)")
            return

        upscaled_latents = upscaled_latents.to(device=device, dtype=torch.float16)
        noise = torch.randn(upscaled_latents.shape, generator=generator,
                            device=device, dtype=upscaled_latents.dtype)
        latents = scheduler.add_noise(upscaled_latents, noise,
                                      timesteps[start_step:start_step+1])

        active_count = len(timesteps) - start_step
        log.info(f"  SDXL: Regional hires denoising (strength={strength:.2f}, {active_count} active steps "
                 f"of {len(timesteps)} total, startStep={start_step})")
    else:
        latents = torch.randn(
            (1, 4, latent_h, latent_w),
            generator=generator, device=device, dtype=torch.float16,
        ) * scheduler.init_noise_sigma
        start_step = 0

        log.info(f"  SDXL: Regional denoising ({len(timesteps)} steps, "
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

    # Stack region embeddings: [N, 77, 2048]
    all_embeds = torch.cat(rer.region_embeds, dim=0)  # each is [1, 77, 2048] → [N, 77, 2048]

    # Build per-region negative stack if negatives differ across regions
    neg_stacked: torch.Tensor | None = None
    if rer.neg_region_embeds is not None:
        neg_stacked = torch.cat(rer.neg_region_embeds, dim=0)  # [N, 77, 2048]

    # If ADDBASE: prepend base_embeds and full-coverage mask, then renormalize
    if rer.base_embeds is not None:
        base_ratio = rer.base_ratio
        base_mask = torch.ones(1, 1, latent_h, latent_w, device=device, dtype=torch.float16) * base_ratio
        region_masks = region_masks * (1 - base_ratio)
        region_masks = torch.cat([base_mask, region_masks], dim=0)
        all_embeds = torch.cat([rer.base_embeds, all_embeds], dim=0)
        # For per-region negatives, prepend shared neg as the base region's negative
        if neg_stacked is not None:
            neg_stacked = torch.cat([rer.neg_prompt_embeds, neg_stacked], dim=0)
        # Renormalize combined masks to sum to 1.0 at every spatial position
        mask_sum = region_masks.sum(dim=0, keepdim=True).clamp(min=1e-8)
        region_masks = region_masks / mask_sum

    # Create attention state and install processors
    state = RegionalAttnState()
    state.set_regions(all_embeds, region_masks)
    original_procs = install_regional_processors(unet, state)

    neg_prompt_embeds = rer.neg_prompt_embeds
    pooled_prompt_embeds = rer.pooled_prompt_embeds
    neg_pooled_prompt_embeds = rer.neg_pooled_prompt_embeds

    # Use first region's embeds as placeholder for encoder_hidden_states
    placeholder_embeds = all_embeds[0:1]

    active_step_count = len(timesteps) - start_step
    job.denoise_step = 0
    job.denoise_total_steps = active_step_count

    # MultiDiffusion tiling setup
    if job.unet_tile_width > 0 and job.unet_tile_height > 0:
        unet_tile_w = job.unet_tile_width  // 8
        unet_tile_h = job.unet_tile_height // 8
    else:
        unet_tile_w, unet_tile_h = _auto_unet_tile(latent_w, latent_h, UNET_TILE_MAX)
    use_tiled = latent_h > UNET_TILE_THRESHOLD or latent_w > UNET_TILE_THRESHOLD
    if use_tiled:
        _stride_x = unet_tile_w - UNET_TILE_OVERLAP
        _stride_y = unet_tile_h - UNET_TILE_OVERLAP
        _tiles_y = max(1, math.ceil((latent_h - UNET_TILE_OVERLAP) / _stride_y))
        _tiles_x = max(1, math.ceil((latent_w - UNET_TILE_OVERLAP) / _stride_x))
        log.info(f"  SDXL: Regional MultiDiffusion enabled — "
                 f"{_tiles_x}x{_tiles_y} tiles "
                 f"({unet_tile_w*8}x{unet_tile_h*8}px, "
                 f"{UNET_TILE_OVERLAP*8}px overlap)")

    try:
        for i in range(start_step, len(timesteps)):
            job.denoise_step = i - start_step + 1
            _broadcast_ws_progress(job)
            t = timesteps[i]

            latent_input = scheduler.scale_model_input(latents, t)

            if use_tiled:
                noise_pred = _unet_tiled_regional(
                    unet, latent_input, t,
                    neg_prompt_embeds,
                    pooled_prompt_embeds, neg_pooled_prompt_embeds,
                    add_time_ids, inp.cfg_scale,
                    unet_tile_w, unet_tile_h, UNET_TILE_OVERLAP,
                    state, region_masks,
                    pos_stacked=all_embeds,
                    neg_stacked=neg_stacked,
                )
            else:
                with torch.no_grad():
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

                    latent_in = torch.cat([latent_input, latent_input])
                    uncond_enc = neg_stacked[0:1] if neg_stacked is not None else neg_prompt_embeds
                    out = unet(
                        latent_in, t,
                        encoder_hidden_states=torch.cat([uncond_enc, placeholder_embeds]),
                        added_cond_kwargs={
                            "text_embeds": torch.cat([neg_pooled_prompt_embeds, pooled_prompt_embeds]),
                            "time_ids": torch.cat([add_time_ids, add_time_ids]),
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

    job.latents = latents


def vae_decode(job: InferenceJob, gpu: GpuInstance) -> Image.Image:
    """Stage 4: GPU VAE decoding. Returns decoded PIL Image."""
    latents = job.latents
    if latents is None:
        raise RuntimeError("Latents are required for VAE decoding.")

    vae = _get_cached_model(gpu, "sdxl_vae")
    device = gpu.device

    # Scale latent and match VAE dtype/device
    latents = latents.to(device=device, dtype=vae.dtype)
    scaled_latents = latents / VAE_SCALE_FACTOR

    lat_h = scaled_latents.shape[2]
    lat_w = scaled_latents.shape[3]
    img_w = lat_w * 8
    img_h = lat_h * 8

    # Resolve tile dimensions — explicit overrides, auto-calculate, or full-image
    if job.vae_tile_width > 0 and job.vae_tile_height > 0:
        tile_w = (job.vae_tile_width  // 8) * 8
        tile_h = (job.vae_tile_height // 8) * 8
    elif img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
        tile_w, tile_h = _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)
    else:
        tile_w, tile_h = img_w, img_h

    lat_tile_w = tile_w // 8
    lat_tile_h = tile_h // 8

    # Use tiled decode if image needs splitting
    if lat_w > lat_tile_w or lat_h > lat_tile_h:
        image = _vae_decode_tiled(scaled_latents, vae, lat_tile_w, lat_tile_h)
    else:
        with torch.no_grad():
            decoded = vae.decode(scaled_latents.to(device)).sample
        image = _tensor_to_pil(decoded)

    log.info(f"  SDXL: VAE decode complete. Image={image.width}x{image.height}")
    return image


def vae_encode(job: InferenceJob, gpu: GpuInstance) -> None:
    """Stage: GPU VAE encode. Encodes job.input_image → job.latents."""
    if job.input_image is None:
        raise RuntimeError("input_image is required for VAE encode.")

    vae = _get_cached_model(gpu, "sdxl_vae")
    device = gpu.device

    tile_w = job.vae_tile_width if job.vae_tile_width > 0 else 0
    tile_h = job.vae_tile_height if job.vae_tile_height > 0 else 0
    latents = _vae_encode(job.input_image, vae, device, tile_w=tile_w, tile_h=tile_h)
    job.latents = latents
    log.info(f"  SDXL: VAE encode complete. shape={list(latents.shape)}")


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

    log.info(f"  SDXL: Latent upscale (Lanczos3) "
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

    vae = _get_cached_model(gpu, "sdxl_vae")

    scaled = latents.to(device=device, dtype=vae.dtype) / VAE_SCALE_FACTOR
    hires_lat_h = scaled.shape[2]
    hires_lat_w = scaled.shape[3]
    hires_img_w = hires_lat_w * 8
    hires_img_h = hires_lat_h * 8

    if hires_img_w >= VAE_TILE_THRESHOLD or hires_img_h >= VAE_TILE_THRESHOLD:
        dtw, dth = _auto_vae_tile(hires_img_w, hires_img_h, VAE_TILE_MAX)
        intermediate = _vae_decode_tiled(scaled, vae, dtw // 8, dth // 8)
    else:
        with torch.no_grad():
            decoded = vae.decode(scaled).sample
        intermediate = _tensor_to_pil(decoded)
    log.info(f"  HiresTransform: VAE decode → {intermediate.width}x{intermediate.height} "
             f"({(time.monotonic()-t0)*1000:.0f}ms)")

    from handlers.upscale import upscale_image
    upscaled = upscale_image(intermediate, gpu)
    log.info(f"  HiresTransform: RealESRGAN 2x → {upscaled.width}x{upscaled.height} "
             f"({(time.monotonic()-t0)*1000:.0f}ms)")
    intermediate.close()

    if upscaled.width != target_w or upscaled.height != target_h:
        upscaled = upscaled.resize((target_w, target_h), Image.LANCZOS)
        log.info(f"  HiresTransform: Resize → {target_w}x{target_h} "
                 f"({(time.monotonic()-t0)*1000:.0f}ms)")

    tile_w = job.vae_tile_width if job.vae_tile_width > 0 else 0
    tile_h = job.vae_tile_height if job.vae_tile_height > 0 else 0
    encoded = _vae_encode(upscaled, vae, device, tile_w=tile_w, tile_h=tile_h)
    log.info(f"  HiresTransform: VAE encode → {list(encoded.shape)} "
             f"({(time.monotonic()-t0)*1000:.0f}ms)")
    upscaled.close()

    job.latents = encoded
    job.is_hires_pass = True
    log.info(f"  HiresTransform: Complete ({(time.monotonic()-t0)*1000:.0f}ms total)")


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
            result.append((token, 1.0))

    # Handle unbalanced brackets
    while round_brackets:
        _multiply_range(result, round_brackets.pop(), ROUND_MUL)
    while square_brackets:
        _multiply_range(result, square_brackets.pop(), SQUARE_MUL)

    if not result:
        result.append(("", 1.0))

    # Merge consecutive items with same weight
    i = 0
    while i < len(result) - 1:
        if abs(result[i][1] - result[i + 1][1]) < 1e-6:
            result[i] = (result[i][0] + result[i + 1][0], result[i][1])
            result.pop(i + 1)
        else:
            i += 1

    return result


def _multiply_range(lst: list[tuple[str, float]], start_idx: int, multiplier: float) -> None:
    for i in range(start_idx, len(lst)):
        lst[i] = (lst[i][0], lst[i][1] * multiplier)


def _tokenize_weighted(
    tokenizer: CLIPTokenizer, text: str, pad_token_id: int = 49407
) -> tuple[list[int], list[float], list[int]]:
    """Tokenize with emphasis weights. Returns (token_ids, weights, attention_mask)."""
    chunks = _parse_prompt_attention(text)
    limit = 77  # CLIP sequence length

    # Tokenize each chunk, strip BOS/EOS
    all_tokens: list[tuple[int, float]] = []
    for chunk_text, weight in chunks:
        if not chunk_text.strip():
            continue
        encoded = tokenizer(chunk_text, add_special_tokens=True, return_tensors=None)
        input_ids = encoded["input_ids"]
        # Strip BOS (first) and EOS (last)
        for tid in input_ids[1:-1]:
            all_tokens.append((tid, weight))

    # Build 77-token sequence: BOS + up to 75 content + EOS + padding
    content_slots = limit - 2  # 75
    tokens = [0] * limit
    weights = [1.0] * limit
    mask = [0] * limit

    # BOS
    tokens[0] = 49406
    weights[0] = 1.0
    mask[0] = 1

    # Content tokens (truncate if needed)
    count = min(len(all_tokens), content_slots)
    for i in range(count):
        tokens[i + 1] = all_tokens[i][0]
        weights[i + 1] = all_tokens[i][1]
        mask[i + 1] = 1

    # EOS
    tokens[count + 1] = 49407
    weights[count + 1] = 1.0
    mask[count + 1] = 1

    # Pad remaining with specified pad token
    for i in range(count + 2, limit):
        tokens[i] = pad_token_id
        weights[i] = 1.0

    log.info(f"    Tokenized: {count} content tokens, pad={pad_token_id}, total={limit}")

    return tokens, weights, mask


# ====================================================================
# Text Encoding Helpers
# ====================================================================

def _run_text_encoder_1(
    model: CLIPTextModel, token_ids: list[int], mask: list[int], device: torch.device
) -> torch.Tensor:
    """Run CLIP-L text encoder. Returns hidden states [1, 77, 768]."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        # Use penultimate hidden state (SDXL standard)
        hidden = outputs.hidden_states[-2]

    return hidden


def _run_text_encoder_2(
    model: CLIPTextModelWithProjection, token_ids: list[int], mask: list[int],
    device: torch.device
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run CLIP-bigG text encoder. Returns (hidden_states [1,77,1280], pooled [1,1280])."""
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    attention_mask = torch.tensor([mask], dtype=torch.long, device=device)

    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        output_hidden_states=True)
        hidden = outputs.hidden_states[-2]
        pooled = outputs.text_embeds

    return hidden, pooled


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


def _vae_encode(image: Image.Image, vae: AutoencoderKL, device: torch.device,
                tile_w: int = 0, tile_h: int = 0) -> torch.Tensor:
    """VAE-encode a PIL image to latent space. Returns [1,4,H/8,W/8].
    Forces tiled encoding when any dimension >= VAE_TILE_THRESHOLD, with
    non-square tiles auto-computed per axis for optimal coverage."""
    img_w, img_h = image.size

    if tile_w > 0 and tile_h > 0:
        eff_w = (tile_w // 8) * 8
        eff_h = (tile_h // 8) * 8
    elif img_w >= VAE_TILE_THRESHOLD or img_h >= VAE_TILE_THRESHOLD:
        eff_w, eff_h = _auto_vae_tile(img_w, img_h, VAE_TILE_MAX)
    else:
        eff_w, eff_h = img_w, img_h

    if img_w > eff_w or img_h > eff_h:
        return _vae_encode_tiled(image, vae, device, eff_w, eff_h)

    arr = np.array(image.convert("RGB"), dtype=np.float32)
    t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
    t = (t / 127.5 - 1.0).to(dtype=torch.float16, device=device)

    with torch.no_grad():
        dist = vae.encode(t).latent_dist
        latents = dist.mean * VAE_SCALE_FACTOR

    return latents


def _vae_encode_tiled(image: Image.Image, vae: AutoencoderKL,
                      device: torch.device,
                      tile_w_px: int, tile_h_px: int) -> torch.Tensor:
    """Tiled VAE encode for large images. Supports non-square tiles.
    Tiles in pixel space, blends in latent space with linear feathering."""
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

    log.info(f"  SDXL: Tiled VAE encode {img_w}x{img_h} — "
             f"{tiles_x}x{tiles_y} grid ({tiles_x * tiles_y} tiles, "
             f"tile={tile_w_px}x{tile_h_px})")

    lat_sum = np.zeros((1, 4, lat_h, lat_w), dtype=np.float32)
    weights  = np.zeros((1, 1, lat_h, lat_w), dtype=np.float32)

    arr = np.array(image.convert("RGB"), dtype=np.float32)

    for ty in range(tiles_y):
        for tx in range(tiles_x):
            py = min(ty * stride_h_px, img_h - tile_h_px)
            px = min(tx * stride_w_px, img_w - tile_w_px)
            py, px = max(0, py), max(0, px)
            py2 = min(py + tile_h_px, img_h)
            px2 = min(px + tile_w_px, img_w)

            tile_np = arr[py:py2, px:px2]
            t = torch.from_numpy(tile_np).permute(2, 0, 1).unsqueeze(0)
            t = (t / 127.5 - 1.0).to(dtype=torch.float16, device=device)

            with torch.no_grad():
                tile_lat_t = vae.encode(t).latent_dist.mean * VAE_SCALE_FACTOR

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

        log.info(f"  SDXL: Tiled VAE encode row {ty+1}/{tiles_y}")

    latents = torch.from_numpy(lat_sum / weights).to(dtype=torch.float16, device=device)
    log.info(f"  SDXL: Tiled VAE encode complete. shape={list(latents.shape)}")
    return latents


def _vae_decode_tiled(
    latents: torch.Tensor, vae: AutoencoderKL,
    lat_tile_w: int, lat_tile_h: int,
) -> Image.Image:
    """Tiled VAE decode for large images with linear feathering blend.
    Uses numpy broadcasting for weight accumulation (no Python pixel loops)."""
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

    log.info(f"  SDXL: Tiled VAE decode {img_w}x{img_h} — "
             f"{tiles_x}x{tiles_y} grid ({tiles_x * tiles_y} tiles, "
             f"tile={lat_tile_w*8}x{lat_tile_h*8})")

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

            with torch.no_grad():
                decoded = vae.decode(tile).sample

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

    # Build final image
    w_safe = np.maximum(weights, 1e-8)
    rgb_out = np.clip(rgb_sum / w_safe[np.newaxis] * 255, 0, 255).astype(np.uint8)  # [3, H, W]
    image = Image.fromarray(rgb_out.transpose(1, 2, 0), "RGB")

    log.info(f"  SDXL: Tiled VAE decode complete. Image={img_w}x{img_h} ({tile_count} tiles)")
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
    """
    adapter_names = []
    adapter_weights = []

    for spec in lora_specs:
        entry = lora_index.get(spec.name)
        if entry is None:
            log.warning(f"  LoRA not found in index: {spec.name}")
            continue

        adapter_name = spec.name

        # Check if adapter already loaded on this UNet
        if hasattr(unet, 'peft_config') and adapter_name in unet.peft_config:
            # Already loaded — just activate
            pass
        else:
            _load_lora_adapter(unet, entry.path, adapter_name, gpu)

        adapter_names.append(adapter_name)
        adapter_weights.append(spec.weight)

    if adapter_names:
        unet.set_adapters(adapter_names, adapter_weights)
        log.info(f"  LoRA: Activated {len(adapter_names)} adapter(s): "
                 f"{', '.join(f'{n}={w:.2f}' for n, w in zip(adapter_names, adapter_weights))}")
    elif hasattr(unet, 'peft_config') and unet.peft_config:
        unet.disable_adapters()


def _load_lora_adapter(unet, lora_path: str, adapter_name: str, gpu: GpuInstance) -> None:
    """Load a LoRA safetensors/pt file as a PEFT adapter on the UNet.

    Uses diffusers' built-in LoRA conversion to handle A1111/Forge/Kohya
    key naming conventions, then loads via UNet's native load_lora_adapter.
    """
    from safetensors.torch import load_file as load_safetensors
    from diffusers.loaders.lora_pipeline import _convert_non_diffusers_lora_to_diffusers
    from diffusers.loaders.lora_conversion_utils import _maybe_map_sgm_blocks_to_diffusers

    t0 = time.monotonic()

    # Load raw state dict
    if lora_path.endswith(".safetensors"):
        raw_sd = load_safetensors(lora_path, device=str(gpu.device))
    else:
        raw_sd = torch.load(lora_path, map_location=gpu.device, weights_only=True)

    # Detect format: A1111/Kohya keys start with "lora_unet_" / "lora_te_",
    # diffusers keys contain "lora_A" / "lora_B" already
    is_a1111 = any(k.startswith("lora_unet_") or k.startswith("lora_te") for k in raw_sd)

    if is_a1111:
        # Remap SGM/LDM block indices to diffusers structure (input_blocks_4 → down_blocks.1.attentions.0)
        # This must happen BEFORE the key format conversion
        raw_sd = _maybe_map_sgm_blocks_to_diffusers(raw_sd, unet.config)
        # Convert A1111/Kohya → diffusers format (handles remaining key renaming)
        converted_sd, network_alphas = _convert_non_diffusers_lora_to_diffusers(raw_sd)
    else:
        converted_sd = raw_sd
        network_alphas = None

    # Use diffusers' built-in UNet LoRA loading (handles diffusers→PEFT conversion,
    # LoraConfig creation, adapter injection, and weight loading)
    unet.load_lora_adapter(
        converted_sd,
        prefix="unet",
        adapter_name=adapter_name,
        network_alphas=network_alphas,
    )

    # Track loaded adapter on the GPU
    if hasattr(gpu, '_loaded_lora_adapters'):
        gpu._loaded_lora_adapters[adapter_name] = lora_path

    elapsed_ms = (time.monotonic() - t0) * 1000
    n_unet_keys = sum(1 for k in converted_sd if k.startswith("unet."))
    log.info(f"  LoRA: Loaded adapter '{adapter_name}' from {os.path.basename(lora_path)} "
             f"({n_unet_keys} unet params, {elapsed_ms:.0f}ms)")


def _get_cached_model(gpu: GpuInstance, category: str) -> object:
    """Retrieve a loaded model from the GPU cache by category."""
    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == category:
                return cached.model
    raise RuntimeError(f"Model {category} not found in GPU [{gpu.uuid}] cache")


def _get_cached_model_optional(gpu: GpuInstance, category: str) -> object | None:
    """Retrieve a loaded model from the GPU cache by category, or None if not loaded."""
    with gpu._cache_lock:
        for cached in gpu._cache.values():
            if cached.category == category:
                return cached.model
    return None
