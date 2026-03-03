"""TensorRT inference runners for SDXL UNet and VAE.

Provides thin wrappers around TRT execution contexts that accept and
return PyTorch tensors directly via ``data_ptr()`` — zero-copy GPU
memory sharing.  The ``execute_async_v3`` API releases the Python GIL
during kernel execution, enabling true multi-GPU parallelism.

Shared device memory
~~~~~~~~~~~~~~~~~~~~
Runners accept an optional ``shared_memory_getter`` callable at construction.
After deserializing the engine (which reveals the exact ``device_memory_size``),
the runner calls ``shared_memory_getter(device_memory_size)`` to obtain a
shared buffer.  This avoids any need to peek at the engine or guess sizes.

The execution context is created with
``create_execution_context_without_device_memory()``, and the shared buffer
is assigned before each ``execute_async_v3()`` call.

This lets multiple cached engines of the same component type (e.g. three
UNet engines at different resolutions) share a single VRAM buffer for
workspace/activations.  Since per-session-key concurrency limits guarantee
at most one UNet (or one VAE) executes at a time on a given GPU,
timesharing is safe.

The shared buffer MUST be allocated on the same GPU as the engine — no
cross-device VRAM transfers over PCIe.
"""

from __future__ import annotations

import os
from typing import Callable

import torch

import log


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Convert a TensorRT DataType to the corresponding PyTorch dtype."""
    import tensorrt as trt
    _map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
        trt.int64: torch.int64,
        trt.int32: torch.int32,
        trt.int8: torch.int8,
        trt.bool: torch.bool,
    }
    # TRT 10+ adds bfloat16
    if hasattr(trt, "bfloat16"):
        _map[trt.bfloat16] = torch.bfloat16
    if trt_dtype not in _map:
        raise ValueError(f"Unsupported TRT dtype: {trt_dtype}")
    return _map[trt_dtype]


class TrtUNetRunner:
    """TensorRT inference runner for the SDXL UNet.

    Loads a serialized TRT engine, creates an execution context, and
    manages I/O tensor bindings.  Inputs and outputs are PyTorch tensors
    on the same GPU — no copies, no Python overhead during forward pass.

    Args:
        engine_path: Path to the serialized .engine file.
        device: Target torch CUDA device.
        shared_memory_getter: Optional callable ``(min_bytes) -> torch.Tensor``
            that returns a shared device memory buffer of at least *min_bytes*.
            Called during __init__ to ensure the buffer exists, and again on
            each ``run()`` to get the current (possibly grown) buffer.

    Lifecycle:
        getter = lambda n: gpu.get_trt_shared_memory("unet", n)
        runner = TrtUNetRunner(engine_path, device, shared_memory_getter=getter)
        out = runner.run(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
        runner.unload()
    """

    def __init__(self, engine_path: str, device: torch.device,
                 shared_memory_getter: Callable[[int], torch.Tensor] | None = None):
        import tensorrt as trt

        self.device = device
        self.engine_path = engine_path
        self._stream = torch.cuda.Stream(device=device)
        self._shared_memory_getter = shared_memory_getter

        # Deserialize engine from disk.  Runtime and logger must be kept alive
        # as long as the engine exists — TRT may reference them internally.
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        # device_memory_size is the VRAM TRT needs for workspace + activations.
        # Baked in at build time based on selected tactics.
        self._device_memory_size = self._engine.device_memory_size

        if shared_memory_getter is not None:
            # Ensure the shared buffer is allocated (or grown) at init time.
            # The getter is re-invoked on each run() to get the current buffer
            # (which may have grown if a larger engine was loaded after us).
            shared_memory_getter(self._device_memory_size)
            self._context = self._engine.create_execution_context_without_device_memory()
        else:
            self._context = self._engine.create_execution_context()

        if self._context is None:
            raise RuntimeError(f"Failed to create TRT execution context: {engine_path}")

        # Cache expected dtypes from the engine for each I/O tensor so we
        # can coerce inputs at runtime (e.g. scheduler gives float32 timestep
        # but the ONNX was traced with int64).
        self._input_dtypes: dict[str, torch.dtype] = {}
        for name in ("sample", "timestep", "encoder_hidden_states", "text_embeds", "time_ids"):
            self._input_dtypes[name] = _trt_dtype_to_torch(self._engine.get_tensor_dtype(name))
        self._output_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype("noise_pred"))

        # Pre-allocate output buffer (will be resized on first run if needed)
        self._output_buffer: torch.Tensor | None = None
        self._engine_size = os.path.getsize(engine_path)

        engine_mb = self._engine_size / (1024 * 1024)
        dev_mb = self._device_memory_size / (1024 * 1024)
        shared_tag = " [shared]" if self._shared_memory_getter is not None else ""
        log.debug(f"  TRT: UNet runner loaded ({engine_mb:.0f}MB on disk, "
                  f"{dev_mb:.0f}MB device mem{shared_tag}, "
                  f"output={self._output_dtype}, "
                  f"timestep={self._input_dtypes['timestep']}) on {device}")

    def run(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        text_embeds: torch.Tensor,
        time_ids: torch.Tensor,
    ) -> torch.Tensor:
        """Run UNet inference through TRT engine.

        All inputs must be contiguous CUDA tensors on self.device.
        Returns the noise prediction tensor [batch, 4, lat_h, lat_w].
        """
        ctx = self._context
        stream_ptr = self._stream.cuda_stream

        # Re-query the shared buffer on every call — the pool may have grown
        # it since this runner was created (e.g. a larger engine was loaded).
        # Keep buf alive as a local — data_ptr() is only valid while the tensor exists.
        if self._shared_memory_getter is not None:
            buf = self._shared_memory_getter(self._device_memory_size)
            ctx.device_memory = buf.data_ptr()

        # Prepare inputs: coerce dtypes to match engine expectations, ensure
        # contiguity, and fix alignment.  TRT reads raw bytes at data_ptr() —
        # a dtype mismatch (e.g. float32 timestep vs engine's int64) silently
        # reads wrong data.  Small view tensors (like timestep from scheduler
        # indexing) may also have non-8-byte-aligned addresses.
        inputs = {
            "sample": sample,
            "timestep": timestep,
            "encoder_hidden_states": encoder_hidden_states,
            "text_embeds": text_embeds,
            "time_ids": time_ids,
        }
        for name in inputs:
            t = inputs[name]
            expected = self._input_dtypes[name]
            if t.dtype != expected:
                t = t.to(expected)
            t = t.contiguous()
            if t.data_ptr() % 8 != 0:
                t = t.clone()
            inputs[name] = t

        # Set input shapes and bind tensor addresses
        for name, t in inputs.items():
            ctx.set_input_shape(name, tuple(t.shape))
            ctx.set_tensor_address(name, t.data_ptr())

        # Allocate or reuse output buffer
        output_shape = ctx.get_tensor_shape("noise_pred")
        if (self._output_buffer is None
                or list(self._output_buffer.shape) != list(output_shape)
                or self._output_buffer.device != self.device):
            self._output_buffer = torch.empty(
                tuple(output_shape), dtype=self._output_dtype, device=self.device)

        ctx.set_tensor_address("noise_pred", self._output_buffer.data_ptr())

        # Execute — this releases the GIL
        ok = ctx.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TRT UNet execute_async_v3 failed")

        self._stream.synchronize()

        # Clone to decouple from the internal buffer — callers (e.g. CFG
        # chunk+math) must not hold views into storage that gets overwritten
        # on the next call.  The clone cost (~0.3ms for a typical UNet output)
        # is negligible vs the ~67ms TRT execution.
        return self._output_buffer.clone()

    @property
    def vram_usage(self) -> int:
        """GPU memory used by this runner (engine weights + output buffer).

        When using shared device memory, the workspace/activation memory is
        NOT counted here — it's tracked by the shared pool instead.
        """
        buf_bytes = 0
        if self._output_buffer is not None:
            buf_bytes = self._output_buffer.nelement() * self._output_buffer.element_size()
        if self._shared_memory_getter is not None:
            # Shared mode: only engine weights (file size proxy) + output buffer
            return self._engine_size + buf_bytes
        # Standalone mode: engine weights + private device memory + output buffer
        return self._engine_size + self._device_memory_size + buf_bytes

    def unload(self) -> None:
        """Free TRT context, engine, runtime, and GPU buffers."""
        if self._context is not None:
            del self._context
            self._context = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._runtime = None
        self._trt_logger = None
        self._output_buffer = None
        self._stream = None
        self._shared_memory_getter = None
        torch.cuda.empty_cache()
        log.debug(f"  TRT: UNet runner unloaded from {self.device}")


class TrtVaeRunner:
    """TensorRT inference runner for the SDXL VAE decoder.

    Same pattern as TrtUNetRunner but for VAE decode:
    Input: latents [1, 4, lat_h, lat_w] (float32)
    Output: image [1, 3, img_h, img_w] (float32)
    """

    def __init__(self, engine_path: str, device: torch.device,
                 shared_memory_getter: Callable[[int], torch.Tensor] | None = None):
        import tensorrt as trt

        self.device = device
        self.engine_path = engine_path
        self._stream = torch.cuda.Stream(device=device)
        self._shared_memory_getter = shared_memory_getter

        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self._device_memory_size = self._engine.device_memory_size

        if shared_memory_getter is not None:
            # Ensure the shared buffer is allocated (or grown) at init time.
            # The getter is re-invoked on each run() to get the current buffer
            # (which may have grown if a larger engine was loaded after us).
            shared_memory_getter(self._device_memory_size)
            self._context = self._engine.create_execution_context_without_device_memory()
        else:
            self._context = self._engine.create_execution_context()

        if self._context is None:
            raise RuntimeError(f"Failed to create TRT execution context: {engine_path}")

        self._output_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype("image"))

        self._output_buffer: torch.Tensor | None = None
        self._engine_size = os.path.getsize(engine_path)

        engine_mb = self._engine_size / (1024 * 1024)
        dev_mb = self._device_memory_size / (1024 * 1024)
        shared_tag = " [shared]" if self._shared_memory_getter is not None else ""
        log.debug(f"  TRT: VAE runner loaded ({engine_mb:.0f}MB on disk, "
                  f"{dev_mb:.0f}MB device mem{shared_tag}, "
                  f"output={self._output_dtype}) on {device}")

    def run(self, latents: torch.Tensor) -> torch.Tensor:
        """Run VAE decoder inference through TRT engine.

        Input must be a contiguous float32 CUDA tensor on self.device.
        Returns the decoded image tensor [1, 3, img_h, img_w].
        """
        ctx = self._context
        stream_ptr = self._stream.cuda_stream

        # Re-query the shared buffer on every call — the pool may have grown
        # it since this runner was created (e.g. a larger engine was loaded).
        # Keep buf alive as a local — data_ptr() is only valid while the tensor exists.
        if self._shared_memory_getter is not None:
            buf = self._shared_memory_getter(self._device_memory_size)
            ctx.device_memory = buf.data_ptr()

        # Keep contiguous reference alive through async execution.
        # Clone if not 8-byte aligned (same defensive check as UNet runner).
        latents = latents.contiguous()
        if latents.data_ptr() % 8 != 0:
            latents = latents.clone()

        ctx.set_input_shape("latents", tuple(latents.shape))
        ctx.set_tensor_address("latents", latents.data_ptr())

        output_shape = ctx.get_tensor_shape("image")
        if (self._output_buffer is None
                or list(self._output_buffer.shape) != list(output_shape)
                or self._output_buffer.device != self.device):
            self._output_buffer = torch.empty(
                tuple(output_shape), dtype=self._output_dtype, device=self.device)

        ctx.set_tensor_address("image", self._output_buffer.data_ptr())

        ok = ctx.execute_async_v3(stream_ptr)
        if not ok:
            raise RuntimeError("TRT VAE execute_async_v3 failed")

        self._stream.synchronize()
        return self._output_buffer.clone()

    @property
    def vram_usage(self) -> int:
        """GPU memory used by this runner (engine weights + output buffer).

        When using shared device memory, the workspace/activation memory is
        NOT counted here — it's tracked by the shared pool instead.
        """
        buf_bytes = 0
        if self._output_buffer is not None:
            buf_bytes = self._output_buffer.nelement() * self._output_buffer.element_size()
        if self._shared_memory_getter is not None:
            # Shared mode: only engine weights (file size proxy) + output buffer
            return self._engine_size + buf_bytes
        # Standalone mode: engine weights + private device memory + output buffer
        return self._engine_size + self._device_memory_size + buf_bytes

    def unload(self) -> None:
        """Free TRT context, engine, runtime, and GPU buffers."""
        if self._context is not None:
            del self._context
            self._context = None
        if self._engine is not None:
            del self._engine
            self._engine = None
        self._runtime = None
        self._trt_logger = None
        self._output_buffer = None
        self._stream = None
        self._shared_memory_getter = None
        torch.cuda.empty_cache()
        log.debug(f"  TRT: VAE runner unloaded from {self.device}")
