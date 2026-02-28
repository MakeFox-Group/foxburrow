"""TensorRT acceleration for SDXL UNet and VAE.

Provides ONNX export, TRT engine building, and inference runners that
bypass Python/Triton entirely — zero JIT variance and no GIL contention
during multi-GPU denoising.

Pipeline: PyTorch model → ONNX export → TRT engine build → TRT inference
"""

from trt.runner import TrtUNetRunner, TrtVaeRunner
from trt.exporter import export_unet_onnx, export_vae_onnx
from trt.builder import build_static_engine, build_dynamic_engine, build_all_engines
from trt.manager import TrtBuildManager

__all__ = [
    "TrtUNetRunner",
    "TrtVaeRunner",
    "export_unet_onnx",
    "export_vae_onnx",
    "build_static_engine",
    "build_dynamic_engine",
    "build_all_engines",
    "TrtBuildManager",
]
