"""Fingerprint-based model deduplication registry."""

from __future__ import annotations

import os
import threading

import log
from scheduling.job import ModelComponentId
from utils import fingerprint as fp_util


class VramEstimates:
    """Estimated VRAM usage per component type (bytes) for fp16 models."""
    SDXL_TEXT_ENCODER_1 = 250 * 1024 * 1024    # ~250 MB (fp16 CLIP-L)
    SDXL_TEXT_ENCODER_2 = 700 * 1024 * 1024    # ~700 MB (fp16 CLIP-bigG)
    SDXL_UNET = 2500 * 1024 * 1024             # ~2.5 GB (fp16 UNet)
    SDXL_VAE_DECODER = 100 * 1024 * 1024       # ~100 MB (fp16 VAE dec)
    SDXL_VAE_ENCODER = 100 * 1024 * 1024       # ~100 MB (fp16 VAE enc)
    UPSCALE = 400 * 1024 * 1024                # ~400 MB
    BGREMOVE = 300 * 1024 * 1024               # ~300 MB
    TAGGER = 150 * 1024 * 1024                 # ~150 MB


# Diffusers safetensors filenames by component
_SDXL_COMPONENT_FILES = {
    "text_encoder": "model.safetensors",
    "text_encoder_2": "model.safetensors",
    "unet": "diffusion_pytorch_model.safetensors",
    "vae": "diffusion_pytorch_model.safetensors",
}

_SDXL_COMPONENT_VRAM = {
    "text_encoder": VramEstimates.SDXL_TEXT_ENCODER_1,
    "text_encoder_2": VramEstimates.SDXL_TEXT_ENCODER_2,
    "unet": VramEstimates.SDXL_UNET,
    "vae_decoder": VramEstimates.SDXL_VAE_DECODER,
    "vae_encoder": VramEstimates.SDXL_VAE_ENCODER,
}

_SDXL_COMPONENT_CATEGORIES = {
    "text_encoder": "sdxl_te1",
    "text_encoder_2": "sdxl_te2",
    "unet": "sdxl_unet",
    "vae_decoder": "sdxl_vae",
    "vae_encoder": "sdxl_vae_enc",
}


class ModelRegistry:
    """Maps model directories and paths to ModelComponentId sets.

    Detects shared components across checkpoints using content fingerprinting.
    """

    def __init__(self):
        self._lock = threading.Lock()
        # fingerprint -> ModelComponentId
        self._components: dict[str, ModelComponentId] = {}
        # model_dir -> list of component IDs [te1, te2, unet, vae_dec, vae_enc]
        self._sdxl_checkpoints: dict[str, list[ModelComponentId]] = {}
        self._upscale_component: ModelComponentId | None = None
        self._bgremove_component: ModelComponentId | None = None

    def register_sdxl_checkpoint(self, model_path: str) -> None:
        """Register an SDXL checkpoint (directory or single-file safetensors).

        Thread-safe: can be called concurrently from background scanning threads.
        """
        model_path = os.path.realpath(model_path)
        with self._lock:
            if model_path in self._sdxl_checkpoints:
                return

        if os.path.isfile(model_path) and model_path.endswith(".safetensors"):
            self._register_single_file(model_path)
        else:
            self._register_diffusers_dir(model_path)

    def _register_single_file(self, checkpoint_path: str) -> None:
        """Register a single-file SDXL safetensors checkpoint.

        Computes SHA256 fingerprint via the standard fingerprint module,
        which stores results in a per-directory .fpcache file. First run
        hashes the file; subsequent runs hit the cache instantly.
        """
        base_fp = fp_util.compute(checkpoint_path)

        components = []
        for suffix, category, vram in [
            ("te1", "sdxl_te1", VramEstimates.SDXL_TEXT_ENCODER_1),
            ("te2", "sdxl_te2", VramEstimates.SDXL_TEXT_ENCODER_2),
            ("unet", "sdxl_unet", VramEstimates.SDXL_UNET),
            ("vae", "sdxl_vae", VramEstimates.SDXL_VAE_DECODER),
        ]:
            fp = f"{base_fp}:{suffix}"
            with self._lock:
                if fp in self._components:
                    comp = self._components[fp]
                else:
                    comp = ModelComponentId(
                        fingerprint=fp, category=category,
                        estimated_vram_bytes=vram,
                    )
                    self._components[fp] = comp
            components.append(comp)

        # VAE encoder = same model as VAE decoder
        components.append(components[3])

        with self._lock:
            if checkpoint_path in self._sdxl_checkpoints:
                return  # another thread registered it while we were hashing
            self._sdxl_checkpoints[checkpoint_path] = components
        log.info(f"  ModelRegistry: Registered single-file SDXL checkpoint "
                 f"{os.path.basename(checkpoint_path)} ({len(components)} components)")

    def _register_diffusers_dir(self, model_dir: str) -> None:
        """Register a diffusers-format SDXL checkpoint directory."""
        components = []
        for subdir, category in [
            ("text_encoder", "sdxl_te1"),
            ("text_encoder_2", "sdxl_te2"),
            ("unet", "sdxl_unet"),
            ("vae", "sdxl_vae"),       # VAE decoder
        ]:
            subdir_path = os.path.join(model_dir, subdir)
            model_file = _find_model_file(subdir_path)
            if model_file is None:
                raise FileNotFoundError(
                    f"No model file found in {subdir_path} for SDXL checkpoint {model_dir}")

            vram = _SDXL_COMPONENT_VRAM.get(subdir if subdir != "vae" else "vae_decoder",
                                             VramEstimates.SDXL_VAE_DECODER)
            comp = self._get_or_create(model_file, category, vram)
            components.append(comp)

        # VAE encoder — same model file as VAE decoder in diffusers format
        vae_dir = os.path.join(model_dir, "vae")
        vae_file = _find_model_file(vae_dir)
        if vae_file:
            vae_enc = self._get_or_create(vae_file, "sdxl_vae_enc", VramEstimates.SDXL_VAE_ENCODER)
        else:
            vae_enc = components[3]
        components.append(vae_enc)

        with self._lock:
            if model_dir in self._sdxl_checkpoints:
                return  # another thread registered it while we were hashing
            self._sdxl_checkpoints[model_dir] = components

            # Count shared components (snapshot under lock)
            shared_count = 0
            for comp in components:
                for other_dir, other_comps in self._sdxl_checkpoints.items():
                    if other_dir == model_dir:
                        continue
                    if any(c.fingerprint == comp.fingerprint for c in other_comps):
                        shared_count += 1
                        break

        log.info(f"  ModelRegistry: Registered SDXL checkpoint {os.path.basename(model_dir)} "
                 f"({len(components)} components, {shared_count} shared)")

    def register_upscale_model(self, model_path: str) -> None:
        self._upscale_component = self._get_or_create(
            model_path, "upscale", VramEstimates.UPSCALE)
        log.info(f"  ModelRegistry: Registered upscale model {model_path}")

    def register_bgremove_model(self, model_path: str) -> None:
        self._bgremove_component = self._get_or_create(
            model_path, "bgremove", VramEstimates.BGREMOVE)
        log.info(f"  ModelRegistry: Registered BGRemove model {model_path}")

    def is_sdxl_registered(self, model_path: str) -> bool:
        """Check if an SDXL checkpoint is already registered."""
        model_path = os.path.realpath(model_path)
        with self._lock:
            return model_path in self._sdxl_checkpoints

    def get_sdxl_components(self, model_dir: str) -> list[ModelComponentId]:
        model_dir = os.path.realpath(model_dir)
        with self._lock:
            if model_dir not in self._sdxl_checkpoints:
                raise KeyError(f"SDXL checkpoint not registered: {model_dir}")
            return list(self._sdxl_checkpoints[model_dir])

    def get_sdxl_te_components(self, model_dir: str) -> list[ModelComponentId]:
        comps = self.get_sdxl_components(model_dir)
        return [comps[0], comps[1]]  # TE1, TE2

    def get_sdxl_unet_component(self, model_dir: str) -> ModelComponentId:
        return self.get_sdxl_components(model_dir)[2]

    def get_sdxl_vae_component(self, model_dir: str) -> ModelComponentId:
        return self.get_sdxl_components(model_dir)[3]

    def get_sdxl_vae_encoder_component(self, model_dir: str) -> ModelComponentId:
        return self.get_sdxl_components(model_dir)[4]

    def get_upscale_component(self) -> ModelComponentId:
        if self._upscale_component is None:
            raise RuntimeError("Upscale model not registered.")
        return self._upscale_component

    def get_bgremove_component(self) -> ModelComponentId:
        if self._bgremove_component is None:
            raise RuntimeError("BGRemove model not registered.")
        return self._bgremove_component

    @property
    def sdxl_checkpoints(self) -> dict[str, list[ModelComponentId]]:
        with self._lock:
            return dict(self._sdxl_checkpoints)

    def _get_or_create(self, model_path: str, category: str, estimated_vram: int) -> ModelComponentId:
        fingerprint = fp_util.compute(model_path)  # I/O — no lock held
        with self._lock:
            if fingerprint in self._components:
                existing = self._components[fingerprint]
                log.info(f"    ModelRegistry: {category} shares content with "
                         f"existing {existing.category} (fingerprint match)")
                return existing
            comp = ModelComponentId(
                fingerprint=fingerprint,
                category=category,
                estimated_vram_bytes=estimated_vram,
            )
            self._components[fingerprint] = comp
            return comp


def _find_model_file(directory: str) -> str | None:
    """Find the model file in a diffusers component directory."""
    if not os.path.isdir(directory):
        return None
    # Prefer safetensors
    for name in ["diffusion_pytorch_model.safetensors", "model.safetensors",
                 "diffusion_pytorch_model.fp16.safetensors", "model.fp16.safetensors",
                 "diffusion_pytorch_model.bin", "model.bin",
                 "pytorch_model.bin"]:
        path = os.path.join(directory, name)
        if os.path.isfile(path):
            return path
    # Fallback: first .safetensors file
    for f in sorted(os.listdir(directory)):
        if f.endswith(".safetensors"):
            return os.path.join(directory, f)
    return None
