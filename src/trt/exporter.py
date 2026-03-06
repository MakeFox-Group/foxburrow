"""ONNX export for SDXL UNet and VAE decoder.

Exports PyTorch models to ONNX format with dynamic spatial axes,
enabling a single ONNX graph to serve multiple resolutions via
TensorRT optimization profiles.

torch.onnx.export() is NOT thread-safe — it uses a global flag
(GLOBALS.in_onnx_export) that asserts False on entry.  All exports
must be serialized via _onnx_lock.

Large models (SDXL UNet ~5GB FP32) exceed the 2GB protobuf limit.
PyTorch's TorchScript exporter auto-detects this and writes hundreds
of individual per-tensor files alongside the .onnx graph stub.  After
export, consolidate_external_data() merges these into a single .data
file for reliable TRT parsing.
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


def consolidate_external_data(onnx_path: str, component_name: str) -> None:
    """Merge per-tensor external data files into a single .data file.

    PyTorch's TorchScript ONNX exporter (dynamo=False) creates one file per
    tensor when the model exceeds the 2GB protobuf limit.  This produces
    hundreds of individual files (e.g. ``unet.down_blocks.0.resnets.0.conv1.weight``).

    TRT's ``parse_from_file()`` can handle these, but they're fragile:
    if ANY file is missing (interrupted export, filesystem issue), the entire
    parse fails silently.  Consolidating into a single ``{component}.onnx.data``
    file makes validation trivial and is the standard ONNX convention.
    """
    import onnx
    from onnx.external_data_helper import convert_model_to_external_data

    file_size = os.path.getsize(onnx_path)
    # If the file is large (>500MB), it's self-contained — no external data
    if file_size > 500 * 1024 * 1024:
        return

    # Load the graph WITHOUT external data to check if external refs exist
    model = onnx.load(onnx_path, load_external_data=False)
    has_external = any(
        init.data_location == onnx.TensorProto.EXTERNAL
        for init in model.graph.initializer
    )
    if not has_external:
        return

    # Collect the list of old per-tensor external data files for cleanup
    onnx_dir = os.path.dirname(onnx_path)
    old_locations: set[str] = set()
    for init in model.graph.initializer:
        if init.data_location == onnx.TensorProto.EXTERNAL:
            for entry in init.external_data:
                if entry.key == "location":
                    old_locations.add(entry.value)

    data_filename = f"{component_name}.onnx.data"

    # Skip consolidation if already pointing to a single consolidated file
    if old_locations == {data_filename}:
        log.debug(f"  TRT: {component_name} ONNX already consolidated")
        return

    log.info(f"  TRT: Consolidating {len(old_locations)} external data files "
             f"for {component_name} into {data_filename}...")

    # Reload WITH external data (loads all weights into memory)
    del model
    model = onnx.load(onnx_path, load_external_data=True)

    # Convert all tensor data to reference a single external file
    convert_model_to_external_data(
        model,
        all_tensors_to_one_file=True,
        location=data_filename,
        size_threshold=1024,
        convert_attribute=False,
    )

    # Save the consolidated model (overwrites the .onnx + creates .data file)
    onnx.save(model, onnx_path)
    del model

    consolidated_data_path = os.path.join(onnx_dir, data_filename)
    if os.path.isfile(consolidated_data_path):
        data_size_mb = os.path.getsize(consolidated_data_path) / (1024 * 1024)
        log.info(f"  TRT: Consolidated {component_name} external data "
                 f"({data_size_mb:.0f}MB): {consolidated_data_path}")
    else:
        log.error(f"  TRT: Consolidation failed — data file not created: "
                  f"{consolidated_data_path}")
        return

    # Clean up old per-tensor files
    cleaned = 0
    for old_name in old_locations:
        if old_name == data_filename:
            continue
        old_path = os.path.join(onnx_dir, old_name)
        if os.path.isfile(old_path):
            os.remove(old_path)
            cleaned += 1
    if cleaned:
        log.debug(f"  TRT: Cleaned up {cleaned} old external data files "
                  f"for {component_name}")


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

    # NOTE: UNet FP32 (~5GB) exceeds the 2GB protobuf limit, so PyTorch
    # creates hundreds of individual per-tensor files.  The manager calls
    # consolidate_external_data() after export to merge them into a single
    # .data file before dispatching to TRT build queues.


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
