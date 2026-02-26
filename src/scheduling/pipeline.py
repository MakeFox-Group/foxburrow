"""Pipeline factories — creates WorkStage[] sequences for each job type."""

from __future__ import annotations

from scheduling.job import StageType, WorkStage
from scheduling.model_registry import ModelRegistry


class PipelineFactory:
    """Creates WorkStage[] pipelines for each job type using the ModelRegistry."""

    def __init__(self, registry: ModelRegistry):
        self._registry = registry

    def create_sdxl_pipeline(self, model_dir: str) -> list[WorkStage]:
        """4-stage SDXL: Tokenize → TextEncode → Denoise → VaeDecode."""
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)
        vae_comp = self._registry.get_sdxl_vae_component(model_dir)

        return [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_VAE_DECODE,
                      required_components=[vae_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_hires_pipeline(self, model_dir: str) -> list[WorkStage]:
        """6-stage SDXL hires fix:
        Tokenize → TextEncode → Denoise(base) → HiresTransform → Denoise(hires) → VaeDecode."""
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)
        vae_comp = self._registry.get_sdxl_vae_component(model_dir)
        vae_enc_comp = self._registry.get_sdxl_vae_encoder_component(model_dir)

        return [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_HIRES_TRANSFORM,
                      required_components=[vae_comp, vae_enc_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_VAE_DECODE,
                      required_components=[vae_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_latents_pipeline(self, model_dir: str) -> list[WorkStage]:
        """3-stage SDXL stopping at latents (no VAE decode)."""
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)

        return [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_hires_denoise_pipeline(self, model_dir: str) -> list[WorkStage]:
        """3-stage hires denoise on pre-upscaled latents (no transform):
        Tokenize → TextEncode → Denoise(hires)."""
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)

        return [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_encode_latents_pipeline(self, model_dir: str) -> list[WorkStage]:
        """1-stage: VAE encode only. Input image → latents."""
        vae_comp = self._registry.get_sdxl_vae_encoder_component(model_dir)
        return [
            WorkStage(type=StageType.GPU_VAE_ENCODE,
                      required_components=[vae_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_hires_latents_pipeline(self, model_dir: str) -> list[WorkStage]:
        """4-stage hires fix starting from latents, outputting latents:
        Tokenize → TextEncode → HiresTransform → Denoise(hires)."""
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)
        vae_comp = self._registry.get_sdxl_vae_component(model_dir)
        vae_enc_comp = self._registry.get_sdxl_vae_encoder_component(model_dir)

        return [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_HIRES_TRANSFORM,
                      required_components=[vae_comp, vae_enc_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
        ]

    def create_sdxl_decode_latents_pipeline(self, model_dir: str) -> list[WorkStage]:
        """1-stage: VAE decode only."""
        vae_comp = self._registry.get_sdxl_vae_component(model_dir)

        return [
            WorkStage(type=StageType.GPU_VAE_DECODE,
                      required_components=[vae_comp],
                      required_capability="sdxl"),
        ]

    def create_enhance_pipeline(self, model_dir: str, needs_upscale: bool = True) -> list[WorkStage]:
        """Enhance pipeline: Tokenize → TextEncode → [Upscale →] VaeEncode → Denoise(hires) → VaeDecode.

        Takes an input image, optionally upscales it (if needs_upscale),
        then runs a hires denoise pass to add detail, then decodes to a final image.
        """
        te_comps = self._registry.get_sdxl_te_components(model_dir)
        unet_comp = self._registry.get_sdxl_unet_component(model_dir)
        vae_comp = self._registry.get_sdxl_vae_component(model_dir)
        vae_enc_comp = self._registry.get_sdxl_vae_encoder_component(model_dir)

        stages = [
            WorkStage(type=StageType.CPU_TOKENIZE),
            WorkStage(type=StageType.GPU_TEXT_ENCODE,
                      required_components=te_comps,
                      required_capability="sdxl"),
        ]

        if needs_upscale:
            upscale_comp = self._registry.get_upscale_component()
            stages.append(WorkStage(type=StageType.GPU_UPSCALE,
                                    required_components=[upscale_comp],
                                    required_capability="upscale"))

        stages.extend([
            WorkStage(type=StageType.GPU_VAE_ENCODE,
                      required_components=[vae_enc_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_DENOISE,
                      required_components=[unet_comp],
                      required_capability="sdxl"),
            WorkStage(type=StageType.GPU_VAE_DECODE,
                      required_components=[vae_comp],
                      required_capability="sdxl"),
        ])

        return stages

    def create_upscale_pipeline(self) -> list[WorkStage]:
        component = self._registry.get_upscale_component()
        return [
            WorkStage(type=StageType.GPU_UPSCALE,
                      required_components=[component],
                      required_capability="upscale"),
        ]

    def create_bgremove_pipeline(self) -> list[WorkStage]:
        component = self._registry.get_bgremove_component()
        return [
            WorkStage(type=StageType.GPU_BGREMOVE,
                      required_components=[component],
                      required_capability="bgremove"),
        ]
