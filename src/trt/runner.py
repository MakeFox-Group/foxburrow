"""TensorRT inference runners for SDXL UNet and VAE.

Provides thin wrappers around TRT execution contexts that accept and
return PyTorch tensors directly via ``data_ptr()`` — zero-copy GPU
memory sharing.  The ``execute_async_v3`` API releases the Python GIL
during kernel execution, enabling true multi-GPU parallelism.
"""

from __future__ import annotations

import os

import torch

import log


def _trt_dtype_to_torch(trt_dtype) -> torch.dtype:
    """Convert a TensorRT DataType to the corresponding PyTorch dtype."""
    import tensorrt as trt
    _map = {
        trt.float32: torch.float32,
        trt.float16: torch.float16,
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

    Lifecycle:
        runner = TrtUNetRunner(engine_path, device)
        out = runner.run(sample, timestep, encoder_hidden_states, text_embeds, time_ids)
        runner.unload()
    """

    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt

        self.device = device
        self.engine_path = engine_path
        self._stream = torch.cuda.Stream(device=device)

        # Deserialize engine from disk.  Runtime and logger must be kept alive
        # as long as the engine exists — TRT may reference them internally.
        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TRT execution context: {engine_path}")

        # Determine output dtype from the engine itself
        self._output_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype("noise_pred"))

        # Pre-allocate output buffer (will be resized on first run if needed)
        self._output_buffer: torch.Tensor | None = None
        self._engine_size = os.path.getsize(engine_path)

        engine_mb = self._engine_size / (1024 * 1024)
        log.debug(f"  TRT: UNet runner loaded ({engine_mb:.0f}MB, "
                  f"output={self._output_dtype}) on {device}")

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

        # Ensure inputs are contiguous and keep references alive through
        # async execution — .contiguous() may return a new temporary tensor
        # whose data_ptr() would be invalidated by GC if not referenced.
        sample = sample.contiguous()
        timestep = timestep.contiguous()
        encoder_hidden_states = encoder_hidden_states.contiguous()
        text_embeds = text_embeds.contiguous()
        time_ids = time_ids.contiguous()

        # Set input shapes (for dynamic dimensions)
        ctx.set_input_shape("sample", tuple(sample.shape))
        ctx.set_input_shape("timestep", tuple(timestep.shape))
        ctx.set_input_shape("encoder_hidden_states", tuple(encoder_hidden_states.shape))
        ctx.set_input_shape("text_embeds", tuple(text_embeds.shape))
        ctx.set_input_shape("time_ids", tuple(time_ids.shape))

        # Bind input tensor addresses
        ctx.set_tensor_address("sample", sample.data_ptr())
        ctx.set_tensor_address("timestep", timestep.data_ptr())
        ctx.set_tensor_address("encoder_hidden_states", encoder_hidden_states.data_ptr())
        ctx.set_tensor_address("text_embeds", text_embeds.data_ptr())
        ctx.set_tensor_address("time_ids", time_ids.data_ptr())

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
        return self._output_buffer

    @property
    def vram_usage(self) -> int:
        """Estimated GPU memory usage in bytes (engine + buffers)."""
        buf_bytes = 0
        if self._output_buffer is not None:
            buf_bytes = self._output_buffer.nelement() * self._output_buffer.element_size()
        # TRT engine device memory is harder to measure precisely;
        # use the serialized engine size as a rough proxy (typically close)
        return self._engine_size + buf_bytes

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
        torch.cuda.empty_cache()
        log.debug(f"  TRT: UNet runner unloaded from {self.device}")


class TrtVaeRunner:
    """TensorRT inference runner for the SDXL VAE decoder.

    Same pattern as TrtUNetRunner but for VAE decode:
    Input: latents [1, 4, lat_h, lat_w] (float32)
    Output: image [1, 3, img_h, img_w] (float32)
    """

    def __init__(self, engine_path: str, device: torch.device):
        import tensorrt as trt

        self.device = device
        self.engine_path = engine_path
        self._stream = torch.cuda.Stream(device=device)

        self._trt_logger = trt.Logger(trt.Logger.WARNING)
        self._runtime = trt.Runtime(self._trt_logger)

        with open(engine_path, "rb") as f:
            engine_bytes = f.read()

        self._engine = self._runtime.deserialize_cuda_engine(engine_bytes)
        if self._engine is None:
            raise RuntimeError(f"Failed to deserialize TRT engine: {engine_path}")

        self._context = self._engine.create_execution_context()
        if self._context is None:
            raise RuntimeError(f"Failed to create TRT execution context: {engine_path}")

        self._output_dtype = _trt_dtype_to_torch(self._engine.get_tensor_dtype("image"))

        self._output_buffer: torch.Tensor | None = None
        self._engine_size = os.path.getsize(engine_path)

        engine_mb = self._engine_size / (1024 * 1024)
        log.debug(f"  TRT: VAE runner loaded ({engine_mb:.0f}MB, "
                  f"output={self._output_dtype}) on {device}")

    def run(self, latents: torch.Tensor) -> torch.Tensor:
        """Run VAE decoder inference through TRT engine.

        Input must be a contiguous float32 CUDA tensor on self.device.
        Returns the decoded image tensor [1, 3, img_h, img_w].
        """
        ctx = self._context
        stream_ptr = self._stream.cuda_stream

        # Keep contiguous reference alive through async execution
        latents = latents.contiguous()

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
        return self._output_buffer

    @property
    def vram_usage(self) -> int:
        """Estimated GPU memory usage in bytes."""
        buf_bytes = 0
        if self._output_buffer is not None:
            buf_bytes = self._output_buffer.nelement() * self._output_buffer.element_size()
        return self._engine_size + buf_bytes

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
        torch.cuda.empty_cache()
        log.debug(f"  TRT: VAE runner unloaded from {self.device}")
