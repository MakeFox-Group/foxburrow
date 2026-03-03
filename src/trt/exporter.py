"""ONNX export for SDXL UNet and VAE decoder.

Exports PyTorch models to ONNX format with dynamic spatial axes,
enabling a single ONNX graph to serve multiple resolutions via
TensorRT optimization profiles.

torch.onnx.export() is NOT thread-safe — it uses a global flag
(GLOBALS.in_onnx_export) that asserts False on entry.  All exports
must be serialized via _onnx_lock.
"""

from __future__ import annotations

import os
import threading

import torch
import torch.nn as nn

import log

# Serializes all torch.onnx.export() calls — the TorchScript exporter
# uses a module-global GLOBALS.in_onnx_export flag that is not thread-safe.
_onnx_lock = threading.Lock()


class _UNetOnnxWrapper(nn.Module):
    """Wrapper that flattens SDXL UNet's dict kwargs into positional args.

    The SDXL UNet expects ``added_cond_kwargs={"text_embeds": ..., "time_ids": ...}``
    as a dict, but ``torch.onnx.export()`` only supports flat positional/keyword
    tensor arguments.  This wrapper unpacks the dict so the ONNX graph has
    five clean named inputs.
    """

    def __init__(self, unet: nn.Module):
        super().__init__()
        self.unet = unet

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        return self.unet(
            sample,
            timestep,
            encoder_hidden_states=encoder_hidden_states,
            added_cond_kwargs={
                "text_embeds": text_embeds,
                "time_ids": time_ids,
            },
        ).sample


class _VaeDecoderOnnxWrapper(nn.Module):
    """Wrapper for VAE decoder-only export.

    The full AutoencoderKL has both encode and decode paths.  For txt2img
    inference we only need decode, so this wrapper calls ``vae.decode()``
    and extracts ``.sample`` from the result.
    """

    def __init__(self, vae: nn.Module):
        super().__init__()
        self.vae = vae

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        return self.vae.decode(latents).sample


class _Te1OnnxWrapper(nn.Module):
    """Wrapper for CLIP-L text encoder (TE1) ONNX export.

    Extracts ``hidden_states[-2]`` (penultimate layer) which is what SDXL
    uses for the 768-dim component of the prompt embedding.  Config is
    patched to always output hidden states so the traced graph includes
    all transformer layers.
    """

    def __init__(self, te1: nn.Module):
        super().__init__()
        self.te1 = te1
        te1.config.output_hidden_states = True

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.te1(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.hidden_states[-2]  # [N, 77, 768]


class _Te2OnnxWrapper(nn.Module):
    """Wrapper for CLIP-bigG text encoder (TE2) ONNX export.

    Returns both ``hidden_states[-2]`` (1280-dim component of prompt embedding)
    and ``text_embeds`` (pooled projection used for SDXL conditioning).
    """

    def __init__(self, te2: nn.Module):
        super().__init__()
        self.te2 = te2
        te2.config.output_hidden_states = True

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        outputs = self.te2(input_ids=input_ids, attention_mask=attention_mask)
        return outputs.hidden_states[-2], outputs.text_embeds  # [N,77,1280], [N,1280]


def export_te1_onnx(
    te1: nn.Module,
    output_path: str,
    opset: int = 17,
) -> None:
    """Export CLIP-L text encoder to ONNX in float32.

    Args:
        te1: A loaded CLIPTextModel (can be on CPU or GPU).
        output_path: Where to write the .onnx file.
        opset: ONNX opset version.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device("cpu")
    te1_cpu = te1.to(device, dtype=torch.float32)
    te1_cpu.eval()

    wrapper = _Te1OnnxWrapper(te1_cpu)

    dummy_input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long, device=device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["hidden_states"]

    dynamic_axes = {
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "hidden_states": {0: "batch"},
    }

    log.info(f"  TRT: Exporting TE1 (CLIP-L) to ONNX (opset {opset})...")

    with _onnx_lock, torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  TRT: TE1 ONNX exported ({file_size_mb:.0f}MB): {output_path}")


def export_te2_onnx(
    te2: nn.Module,
    output_path: str,
    opset: int = 17,
) -> None:
    """Export CLIP-bigG text encoder to ONNX in float32.

    Args:
        te2: A loaded CLIPTextModelWithProjection (can be on CPU or GPU).
        output_path: Where to write the .onnx file.
        opset: ONNX opset version.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device("cpu")
    te2_cpu = te2.to(device, dtype=torch.float32)
    te2_cpu.eval()

    wrapper = _Te2OnnxWrapper(te2_cpu)

    dummy_input_ids = torch.zeros(1, 77, dtype=torch.long, device=device)
    dummy_attention_mask = torch.ones(1, 77, dtype=torch.long, device=device)

    input_names = ["input_ids", "attention_mask"]
    output_names = ["hidden_states", "text_embeds"]

    dynamic_axes = {
        "input_ids": {0: "batch"},
        "attention_mask": {0: "batch"},
        "hidden_states": {0: "batch"},
        "text_embeds": {0: "batch"},
    }

    log.info(f"  TRT: Exporting TE2 (CLIP-bigG) to ONNX (opset {opset})...")

    with _onnx_lock, torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_input_ids, dummy_attention_mask),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  TRT: TE2 ONNX exported ({file_size_mb:.0f}MB): {output_path}")


def export_unet_onnx(
    unet: nn.Module,
    output_path: str,
    opset: int = 17,
) -> None:
    """Export an SDXL UNet to ONNX in float32.

    Exported as FP32 because TRT's ONNX parser cannot convert FP16 weights
    stored in ONNX external data format (required for models > 2GB protobuf
    limit).  TRT's FP16 builder flag handles precision conversion during
    engine compilation.

    Args:
        unet: A loaded UNet2DConditionModel (can be on CPU or GPU).
        output_path: Where to write the .onnx file.
        opset: ONNX opset version (17 supports all SDXL ops).
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Move to CPU for export (avoids GPU memory pressure during build).
    # FP32 for ONNX — TRT handles FP16 conversion at engine build time.
    device = torch.device("cpu")
    unet_cpu = unet.to(device, dtype=torch.float32)
    unet_cpu.eval()

    wrapper = _UNetOnnxWrapper(unet_cpu)

    # Dummy inputs matching SDXL UNet signature (batch=2 for CFG)
    # Use a mid-range latent size; dynamic axes handle the rest.
    dummy_sample = torch.randn(2, 4, 96, 96, dtype=torch.float32, device=device)
    dummy_timestep = torch.tensor([999], dtype=torch.long, device=device)
    dummy_encoder_hidden_states = torch.randn(2, 77, 2048, dtype=torch.float32, device=device)
    dummy_text_embeds = torch.randn(2, 1280, dtype=torch.float32, device=device)
    dummy_time_ids = torch.randn(2, 6, dtype=torch.float32, device=device)

    dummy_args = (
        dummy_sample,
        dummy_timestep,
        dummy_encoder_hidden_states,
        dummy_text_embeds,
        dummy_time_ids,
    )

    input_names = ["sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"]
    output_names = ["noise_pred"]

    # Dynamic axes: spatial dims vary per resolution, seq_len varies per prompt
    # length (77*N where N = number of 75-token chunks needed for the prompt).
    dynamic_axes = {
        "sample": {0: "batch", 2: "latent_h", 3: "latent_w"},
        "encoder_hidden_states": {0: "batch", 1: "seq_len"},
        "text_embeds": {0: "batch"},
        "time_ids": {0: "batch"},
        "noise_pred": {0: "batch", 2: "latent_h", 3: "latent_w"},
    }

    log.info(f"  TRT: Exporting UNet to ONNX (opset {opset})...")

    with _onnx_lock, torch.no_grad():
        torch.onnx.export(
            wrapper,
            dummy_args,
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  TRT: UNet ONNX exported ({file_size_mb:.0f}MB): {output_path}")


def export_vae_onnx(
    vae: nn.Module,
    output_path: str,
    opset: int = 17,
) -> None:
    """Export an SDXL VAE decoder to ONNX in float32.

    VAE must remain float32 — FP16 causes NaN overflow in the decoder
    for certain latent distributions (well-known SDXL VAE issue).

    Args:
        vae: A loaded AutoencoderKL (can be on CPU or GPU).
        output_path: Where to write the .onnx file.
        opset: ONNX opset version.
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    device = torch.device("cpu")
    vae_cpu = vae.to(device, dtype=torch.float32)
    vae_cpu.eval()

    wrapper = _VaeDecoderOnnxWrapper(vae_cpu)

    # Dummy input: batch=1, 4 latent channels, mid-range spatial size
    dummy_latents = torch.randn(1, 4, 96, 96, dtype=torch.float32, device=device)

    input_names = ["latents"]
    output_names = ["image"]

    dynamic_axes = {
        "latents": {2: "latent_h", 3: "latent_w"},
        "image": {2: "image_h", 3: "image_w"},
    }

    log.info(f"  TRT: Exporting VAE decoder to ONNX (opset {opset})...")

    with _onnx_lock, torch.no_grad():
        torch.onnx.export(
            wrapper,
            (dummy_latents,),
            output_path,
            input_names=input_names,
            output_names=output_names,
            dynamic_axes=dynamic_axes,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )

    file_size_mb = os.path.getsize(output_path) / (1024 * 1024)
    log.info(f"  TRT: VAE ONNX exported ({file_size_mb:.0f}MB): {output_path}")
