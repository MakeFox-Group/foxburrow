"""GPU pool management: GpuInstance + GpuPool.

Tracks loaded models, VRAM usage, and provides model cache with LRU eviction.
"""

from __future__ import annotations

import ctypes
import itertools
import threading
import time as _time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Callable

import torch
import torch.nn as nn

import log
from api.websocket import streamer
from config import GpuConfig
from gpu import nvml

# Save originals at import time — before any accelerate context can patch them.
# accelerate's init_empty_weights() monkey-patches these to create meta tensors;
# if the context leaks, ALL subsequent model construction is broken process-wide.
_ORIG_REGISTER_PARAMETER = nn.Module.register_parameter
_ORIG_REGISTER_BUFFER = nn.Module.register_buffer


def repair_accelerate_leak() -> bool:
    """Detect and repair a leaked accelerate init_empty_weights() context.

    accelerate's init_empty_weights() monkey-patches nn.Module.register_parameter
    and nn.Module.register_buffer to create meta tensors.  If the context doesn't
    exit cleanly (threading, exceptions), these patches persist process-wide,
    breaking ALL subsequent model construction.

    Returns True if a leak was detected and repaired.
    """
    test = nn.Linear(1, 1, bias=False)
    if not test.weight.is_meta:
        return False
    log.warning("  Detected leaked accelerate init_empty_weights() context "
                "— restoring nn.Module")
    nn.Module.register_parameter = _ORIG_REGISTER_PARAMETER
    nn.Module.register_buffer = _ORIG_REGISTER_BUFFER
    # Verify
    test2 = nn.Linear(1, 1, bias=False)
    if test2.weight.is_meta:
        log.error("  Failed to repair accelerate leak!")
        return False
    return True


def patch_accelerate_thread_safety() -> None:
    """Monkey-patch accelerate's init_on_device() to be thread-safe.

    accelerate's init_on_device() saves nn.Module.register_parameter to a local
    variable on entry, patches it globally, then restores from the local on exit.
    This is NOT thread-safe:

        Thread A enters → saves ORIG, patches to PATCH_A
        Thread B enters → saves PATCH_A (not ORIG!), patches to PATCH_B
        Thread A exits  → restores ORIG  (correct)
        Thread B exits  → restores PATCH_A  (LEAKED!)

    Now register_parameter is permanently the meta-tensor version.  Every
    subsequent nn.Module construction creates meta tensors process-wide.

    Our fix: replace init_on_device's finally block to ALWAYS restore from our
    import-time originals (_ORIG_REGISTER_PARAMETER / _ORIG_REGISTER_BUFFER),
    never from stale local captures.  This makes concurrent entries/exits safe.
    """
    from contextlib import contextmanager

    import accelerate.big_modeling as accel_bm
    from accelerate.utils import parse_flag_from_env

    @contextmanager
    def _safe_init_on_device(device, include_buffers=None):
        if include_buffers is None:
            include_buffers = parse_flag_from_env(
                "ACCELERATE_INIT_INCLUDE_BUFFERS", False)

        # include_buffers=True uses PyTorch's device context manager (same as
        # accelerate's original lines 120-123 of big_modeling.py).  No
        # register_parameter patching needed — torch handles it natively.
        if include_buffers:
            with device:
                yield
            return

        def register_empty_parameter(module, name, param):
            _ORIG_REGISTER_PARAMETER(module, name, param)
            if param is not None:
                param_cls = type(module._parameters[name])
                kwargs = module._parameters[name].__dict__
                kwargs["requires_grad"] = param.requires_grad
                module._parameters[name] = param_cls(
                    module._parameters[name].to(device), **kwargs)

        try:
            nn.Module.register_parameter = register_empty_parameter
            yield
        finally:
            # ALWAYS restore from import-time originals — never from stale
            # locals that may contain another thread's patched version.
            nn.Module.register_parameter = _ORIG_REGISTER_PARAMETER
            # register_buffer isn't patched in this branch, but restore it
            # unconditionally as a safety net (idempotent no-op normally).
            nn.Module.register_buffer = _ORIG_REGISTER_BUFFER

    # Patch both the module-internal name (used by init_empty_weights via
    # module-global lookup) and the public re-export in accelerate.__init__
    # (used by any code doing `from accelerate import init_on_device`).
    accel_bm.init_on_device = _safe_init_on_device
    import accelerate
    accelerate.init_on_device = _safe_init_on_device
    log.debug("  Patched accelerate.init_on_device() for thread safety")


def fix_meta_tensors(model: nn.Module) -> int:
    """Replace any remaining meta tensors in a model with zero-filled tensors.

    ``from_pretrained()`` uses accelerate's ``init_empty_weights()`` internally
    to create the model skeleton on the ``meta`` device, then loads the state
    dict.  But some registered buffers (e.g. ``position_ids``, ``causal_mask``)
    aren't in the state dict and remain as meta tensors.  Inference works fine
    because these buffers are recomputed at runtime, but ``.to(device)`` walks
    ALL tensors and raises ``NotImplementedError: Cannot copy out of meta
    tensor``.

    This function surgically replaces only the meta tensors, leaving all
    properly loaded parameters/buffers intact.  Replacement tensors are created
    on the same device as the model's existing non-meta tensors (so LoRA
    parameters injected while the model is on CUDA land on the correct GPU).

    Returns the number of tensors fixed.
    """
    # Detect target device from the first non-meta parameter or buffer.
    # If the model is already on a CUDA device, replacements must go there too.
    target_device = torch.device("cpu")
    for t in itertools.chain(model.parameters(), model.buffers()):
        if not t.is_meta:
            target_device = t.device
            break

    fixed = 0

    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        new_param = nn.Parameter(
            torch.zeros(param.shape, dtype=param.dtype, device=target_device),
            requires_grad=param.requires_grad,
        )
        setattr(parent, parts[-1], new_param)
        fixed += 1

    for name, buf in list(model.named_buffers()):
        if not buf.is_meta:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device=target_device)
        parent.register_buffer(parts[-1], new_buf)
        fixed += 1

    return fixed


def _cuda_index_from_pci_bus_id(pci_bus_id: str) -> int | None:
    """Map a PCI bus ID string to a CUDA device index via the CUDA runtime.
    Returns None if the device is not visible to CUDA."""
    try:
        libcudart = ctypes.cdll.LoadLibrary("libcudart.so")
    except OSError:
        # Fallback: try versioned name
        try:
            libcudart = ctypes.cdll.LoadLibrary("libcudart.so.12")
        except OSError:
            return None
    device = ctypes.c_int(-1)
    # cudaDeviceGetByPCIBusId(int *device, const char *pciBusId)
    ret = libcudart.cudaDeviceGetByPCIBusId(
        ctypes.byref(device),
        pci_bus_id.encode("utf-8"),
    )
    if ret != 0:  # cudaSuccess = 0
        return None
    return device.value


@dataclass
class CachedModel:
    """A model component cached on a GPU."""
    fingerprint: str
    category: str  # e.g. "sdxl_te1", "sdxl_unet", "upscale"
    model: object  # PyTorch module
    estimated_vram: int  # bytes
    actual_vram: int = 0  # measured bytes (0 = not measured)
    source: str = ""  # human-readable source name, e.g. "xavier_v10"
    evict_callback: Callable[[], None] | None = None  # called on eviction for cleanup
    cached_at: float = 0.0  # time.monotonic() when cached
    last_used: float = 0.0  # time.monotonic() when last accessed
    use_count: int = 0  # how many times this model has been used


class GpuInstance:
    """Represents a single GPU with model cache and VRAM tracking."""

    # Default CPU cache: 64 GB per worker
    DEFAULT_CPU_CACHE_BYTES = 64 * 1024 * 1024 * 1024

    def __init__(self, config: GpuConfig, nvml_device: nvml.NvmlDeviceInfo, torch_device_id: int,
                 cpu_cache_bytes: int = 0):
        self.uuid = config.uuid
        self.name = config.name
        self.capabilities = config.capabilities
        self.device_id = torch_device_id
        self.device = torch.device(f"cuda:{torch_device_id}")
        self.nvml_handle = nvml_device.handle
        self.total_memory = nvml_device.total_memory

        # Busy flag for status reporting (one job at a time per GPU)
        self._busy_lock = threading.Lock()
        self._busy = False

        # Model cache: fingerprint -> CachedModel (LRU ordered)
        self._cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._cache_lock = threading.Lock()

        # CPU model cache: holds evicted models on CPU RAM for fast reload.
        # When a model is evicted from GPU VRAM, it's moved to CPU and stored
        # here.  Loading from CPU cache (~0.1-0.5s for .to(device)) is much
        # faster than loading from disk (~2-8s for from_pretrained + .to()).
        self._cpu_cache: OrderedDict[str, tuple[CachedModel, int]] = OrderedDict()
        self._cpu_cache_lock = threading.Lock()
        self._cpu_cache_bytes: int = 0  # current total estimated bytes in CPU cache
        self._cpu_cache_limit: int = cpu_cache_bytes if cpu_cache_bytes > 0 else self.DEFAULT_CPU_CACHE_BYTES

        # Active model fingerprints — simple set (one job at a time)
        self._active_fps: set[str] = set()
        self._active_fp_lock = threading.Lock()

        # Model loading lock — serializes model loading and eviction.
        self.model_load_lock = threading.Lock()

        # Per-GPU config: models to pre-load and never evict
        self.onload: set[str] = config.onload
        self.unevictable: set[str] = config.unevictable
        # Fingerprints of unevictable models (populated at load time)
        self._unevictable_fingerprints: set[str] = set()
        # Fingerprints of onload models — evictable but should be reloaded when VRAM frees up
        self._onload_fingerprints: set[str] = set()

        # LoRA adapter tracking: adapter_name -> lora_path
        # Adapters live inside the UNet's PEFT layers; cleared when UNet is evicted
        self._loaded_lora_adapters: dict[str, str] = {}

        # TRT shared device memory pools — one buffer per component type (e.g.
        # "unet", "vae").  All TRT engines of the same type on this GPU share a
        # single workspace/activation buffer.  Safe because per-session-key limits
        # guarantee at most one UNet (or one VAE) executes at a time per GPU.
        # Each buffer is a torch.Tensor allocated on THIS GPU — no cross-device.
        self._trt_shared_mem: dict[str, torch.Tensor] = {}
        self._trt_shared_mem_lock = threading.Lock()

        # GPU failure tracking — permanently disables the GPU when a fatal
        # CUDA error corrupts the context (e.g. cudaErrorIllegalAddress / Xid 31)
        self._failed = False
        self._fail_reason: str = ""
        self._consecutive_failures = 0

        # TRT build tracking — set by GpuWorkerProxy when drained for TRT.
        # AdmissionControl uses this to exclude drained GPUs from capacity.
        self._trt_building = False

    def get_vram_stats(self) -> dict:
        """Return actual VRAM usage from PyTorch + NVML."""
        allocated = torch.cuda.memory_allocated(self.device)
        reserved = torch.cuda.memory_reserved(self.device)
        total, used, free = nvml.get_memory_info(self.nvml_handle)
        stats = {
            "allocated": allocated,       # PyTorch tensor VRAM
            "reserved": reserved,         # PyTorch cached allocator
            "total": total,               # GPU total capacity (NVML)
            "used": used,                 # Total VRAM in use (NVML)
            "free": free,                 # NVML-reported free
        }
        from gpu.torch_ext import HAS_ALLOC_TAGS, ALLOC_TAG_MODEL_WEIGHTS, ALLOC_TAG_ACTIVATIONS
        if HAS_ALLOC_TAGS:
            stats["model_vram"] = torch.cuda.memory_allocated_by_tag(
                ALLOC_TAG_MODEL_WEIGHTS, self.device)
            stats["activation_vram"] = torch.cuda.memory_allocated_by_tag(
                ALLOC_TAG_ACTIVATIONS, self.device)
        return stats

    @property
    def is_busy(self) -> bool:
        with self._busy_lock:
            return self._busy

    def acquire(self) -> None:
        """Mark GPU as busy (one job at a time)."""
        with self._busy_lock:
            self._busy = True

    def release(self) -> None:
        with self._busy_lock:
            self._busy = False

    def supports_capability(self, cap: str) -> bool:
        return cap.lower() in self.capabilities

    # ---- Failure tracking ----

    @property
    def is_failed(self) -> bool:
        """Whether this GPU has been permanently marked as failed."""
        return self._failed

    def mark_failed(self, reason: str) -> None:
        """Permanently mark this GPU as failed. No further work will be dispatched."""
        if not self._failed:
            self._failed = True
            self._fail_reason = reason
            log.error(f"  GPU [{self.uuid}]: PERMANENTLY FAILED — {reason}")

    def record_success(self) -> None:
        """Record a successful job completion, resetting the consecutive failure counter."""
        self._consecutive_failures = 0

    def record_failure(self) -> bool:
        """Record a job failure. Returns True if GPU should be disabled (5 consecutive)."""
        self._consecutive_failures += 1
        if self._consecutive_failures >= 5:
            self.mark_failed(f"{self._consecutive_failures} consecutive job failures")
            return True
        return False

    def mark_unevictable(self, fingerprint: str) -> None:
        """Mark a cached model fingerprint as unevictable."""
        self._unevictable_fingerprints.add(fingerprint)

    def mark_onload(self, fingerprint: str) -> None:
        """Mark a cached model fingerprint as an onload model (auto-reload after eviction)."""
        self._onload_fingerprints.add(fingerprint)

    def get_evicted_onload_fingerprints(self) -> set[str]:
        """Return onload fingerprints that are not currently cached (were evicted)."""
        with self._cache_lock:
            return self._onload_fingerprints - set(self._cache.keys())

    # ---- Model cache ----

    def get_cached_model(self, fingerprint: str) -> CachedModel | None:
        with self._cache_lock:
            if fingerprint in self._cache:
                self._cache.move_to_end(fingerprint)
                m = self._cache[fingerprint]
                m.last_used = _time.monotonic()
                m.use_count += 1
                return m
            return None

    def cache_model(self, fingerprint: str, category: str, model: object, estimated_vram: int,
                    source: str = "", actual_vram: int = 0,
                    evict_callback: Callable[[], None] | None = None) -> None:
        now = _time.monotonic()
        with self._cache_lock:
            self._cache[fingerprint] = CachedModel(
                fingerprint=fingerprint,
                category=category,
                model=model,
                estimated_vram=estimated_vram,
                actual_vram=actual_vram,
                source=source,
                evict_callback=evict_callback,
                cached_at=now,
                last_used=now,
                use_count=0,
            )
            self._cache.move_to_end(fingerprint)

    def evict_other_sources(self, keep_source: str) -> int:
        """Evict SDXL-category cached models whose source differs from keep_source.

        One GPU = one safetensor model.  When a new checkpoint is loaded,
        evict the old checkpoint's components (UNet, TEs, VAE, TRT engines)
        so the new checkpoint fits.  Utility models (upscale, bgremove,
        tagger) are left alone — they're checkpoint-agnostic.

        Models are moved to CPU cache for fast reload if the same
        checkpoint is requested again later.

        Returns the number of models evicted.
        """
        if not keep_source:
            return 0

        # Collect fingerprints to evict (can't modify cache while iterating)
        to_evict: list[str] = []
        active_fps = self.get_active_fingerprints()
        with self._cache_lock:
            for fp, m in self._cache.items():
                # Only evict SDXL-category models (checkpoint-specific)
                if not m.category.startswith("sdxl"):
                    continue
                if not self._is_evictable(fp, None):
                    continue
                if fp in active_fps:
                    continue
                if m.source and m.source != keep_source:
                    to_evict.append(fp)

        evicted = 0
        for fp in to_evict:
            with self._cache_lock:
                if fp not in self._cache:
                    continue
                model_entry = self._cache.pop(fp)

            vram_mb = (model_entry.actual_vram if model_entry.actual_vram > 0
                       else model_entry.estimated_vram) // (1024 * 1024)
            if model_entry.evict_callback:
                model_entry.evict_callback()

            already_cached = self._is_in_cpu_cache(model_entry.fingerprint)
            if already_cached:
                model_entry.model = None
            else:
                self._safe_to_cpu(model_entry)

            self._maybe_release_trt_shared_memory(model_entry.category)
            if not already_cached:
                self._cpu_cache_store(model_entry)

            log.debug(f"  GPU [{self.uuid}]: Evicted {model_entry.category} "
                      f"({model_entry.source}, ~{vram_mb}MB) — "
                      f"switching to {keep_source}")
            evicted += 1

        if evicted > 0:
            torch.cuda.empty_cache()

        return evicted

    def is_component_loaded(self, fingerprint: str) -> bool:
        with self._cache_lock:
            return fingerprint in self._cache

    def get_cached_categories(self) -> list[str]:
        with self._cache_lock:
            return [m.category for m in self._cache.values()]

    def get_cached_models_info(self) -> list[dict]:
        """Return info about all cached models: category, source, VRAM usage."""
        with self._cache_lock:
            return [
                {
                    "category": m.category,
                    "source": m.source,
                    "vram": m.actual_vram if m.actual_vram > 0 else m.estimated_vram,
                    "estimated_vram": m.estimated_vram,
                    "actual_vram": m.actual_vram,
                }
                for m in self._cache.values()
            ]

    def get_evictable_vram(self, protect: set[str] | None = None) -> int:
        """Return total VRAM of cached models that could be evicted.

        Models protected by active fingerprints, the unevictable set, or the
        explicit *protect* set are excluded.

        NOTE: ``active_fps`` is snapshot separately from the cache iteration.
        If a fingerprint is removed from active between the snapshot and scan,
        we may undercount evictable VRAM (conservative/safe).  The actual
        eviction logic in ``ensure_free_vram`` re-checks under lock.
        """
        active_fps = self.get_active_fingerprints()
        with self._cache_lock:
            total = 0
            for fp, m in self._cache.items():
                if not self._is_evictable(fp, protect):
                    continue
                if fp in active_fps:
                    continue
                total += m.actual_vram if m.actual_vram > 0 else m.estimated_vram
            return total

    def get_loaded_models_vram(self) -> int:
        """Return total VRAM consumed by all cached models."""
        with self._cache_lock:
            return sum(
                m.actual_vram if m.actual_vram > 0 else m.estimated_vram
                for m in self._cache.values()
            )

    def add_active_fingerprints(self, fingerprints: set[str]) -> None:
        """Mark fingerprints as actively in use by the current job."""
        with self._active_fp_lock:
            self._active_fps.update(fingerprints)

    def remove_active_fingerprints(self, fingerprints: set[str]) -> None:
        """Remove fingerprints from active set when a job finishes."""
        with self._active_fp_lock:
            self._active_fps.difference_update(fingerprints)

    def clear_active_fingerprints(self) -> None:
        """Clear all active fingerprints (called when job completes)."""
        with self._active_fp_lock:
            self._active_fps.clear()

    def get_active_fingerprints(self) -> set[str]:
        """Return the set of all fingerprints currently in use."""
        with self._active_fp_lock:
            return set(self._active_fps)

    # ---- CPU model cache ----

    def get_cpu_cached_model(self, fingerprint: str) -> CachedModel | None:
        """Retrieve a model from the CPU cache, removing it.

        Returns the CachedModel entry (model on CPU) if found, else None.
        The caller is responsible for moving the model to GPU.
        """
        with self._cpu_cache_lock:
            if fingerprint not in self._cpu_cache:
                return None
            entry, stored_bytes = self._cpu_cache.pop(fingerprint)
            self._cpu_cache_bytes -= stored_bytes
        return entry

    def _cpu_cache_store(self, model_entry: CachedModel) -> None:
        """Store an evicted model in the CPU cache.

        Enforces the memory limit by evicting the oldest entries first.
        TRT runners and models without a .to() method are skipped.
        """
        # TRT runners can't be CPU-cached
        if model_entry.category.endswith("_trt"):
            return
        if not hasattr(model_entry.model, "to"):
            return

        vram = model_entry.actual_vram if model_entry.actual_vram > 0 else model_entry.estimated_vram
        if vram <= 0:
            return

        with self._cpu_cache_lock:
            # If this fingerprint is already in CPU cache, remove old entry first
            if model_entry.fingerprint in self._cpu_cache:
                _, old_bytes = self._cpu_cache.pop(model_entry.fingerprint)
                self._cpu_cache_bytes -= old_bytes

            # Evict oldest entries until we have room
            while (self._cpu_cache_bytes + vram > self._cpu_cache_limit
                   and self._cpu_cache):
                _, (evicted, ev_bytes) = self._cpu_cache.popitem(last=False)
                self._cpu_cache_bytes -= ev_bytes
                log.debug(f"  GPU [{self.uuid}]: CPU cache evicted {evicted.category} "
                         f"({evicted.source}, ~{ev_bytes // (1024*1024)}MB)")
                del evicted

            self._cpu_cache[model_entry.fingerprint] = (model_entry, vram)
            self._cpu_cache.move_to_end(model_entry.fingerprint)
            self._cpu_cache_bytes += vram
            _total = self._cpu_cache_bytes

        log.debug(f"  GPU [{self.uuid}]: CPU cache stored {model_entry.category} "
                 f"({model_entry.source}, ~{vram // (1024*1024)}MB, "
                 f"total {_total // (1024*1024)}MB / "
                 f"{self._cpu_cache_limit // (1024*1024)}MB)")

    def _is_in_cpu_cache(self, fingerprint: str) -> bool:
        """Check if a model fingerprint exists in the CPU cache (without removing it)."""
        with self._cpu_cache_lock:
            return fingerprint in self._cpu_cache

    def get_cpu_cache_info(self) -> dict:
        """Return CPU cache statistics."""
        with self._cpu_cache_lock:
            return {
                "count": len(self._cpu_cache),
                "bytes": self._cpu_cache_bytes,
                "limit": self._cpu_cache_limit,
                "models": [
                    {"category": m.category, "source": m.source, "bytes": stored_bytes}
                    for m, stored_bytes in self._cpu_cache.values()
                ],
            }

    def _is_evictable(self, fp: str, protect: set[str] | None) -> bool:
        """Check if a cached model can be evicted."""
        if protect and fp in protect:
            return False
        if fp in self._unevictable_fingerprints:
            return False
        return True

    def _safe_to_cpu(self, model_entry: CachedModel) -> None:
        """Move a model to CPU, fixing any meta tensors first.

        TRT runners cannot be moved to CPU — they are destroyed on eviction
        and reloaded from disk when needed again.  The evict_callback handles
        cleanup (freeing TRT context + device memory).
        """
        # TRT runners: no CPU offload possible — evict_callback handles cleanup
        if model_entry.category.endswith("_trt"):
            return
        if not hasattr(model_entry.model, "to"):
            return
        if isinstance(model_entry.model, nn.Module):
            n = fix_meta_tensors(model_entry.model)
            if n:
                log.warning(f"  GPU [{self.uuid}]: Fixed {n} meta tensor(s) "
                            f"in {model_entry.category} before eviction")
        model_entry.model.to("cpu")

    def evict_lru(self, protect: set[str] | None = None) -> CachedModel | None:
        """Evict the oldest non-protected cached model.

        One GPU = one checkpoint.  No scoring needed — just evict the
        first evictable model (LRU order from OrderedDict).
        """
        active_fps = self.get_active_fingerprints()
        with self._cache_lock:
            best_fp: str | None = None
            for fp in self._cache:
                if not self._is_evictable(fp, protect):
                    continue
                if fp in active_fps:
                    continue
                best_fp = fp
                break  # first evictable = oldest (LRU)

            if best_fp is None:
                return None

            model_entry = self._cache.pop(best_fp)

        # Release cache lock before calling callbacks and CPU transfer —
        # these can be slow (model.to("cpu") releases GIL during CUDA→CPU
        # transfer) and would block all other cache reads if held.
        vram_mb = (model_entry.actual_vram if model_entry.actual_vram > 0
                   else model_entry.estimated_vram) // (1024 * 1024)
        if model_entry.evict_callback:
            model_entry.evict_callback()

        # Check if this model is already in the CPU cache — if so, skip the
        # expensive GPU→CPU transfer and just free the GPU memory.  The CPU
        # cache already has a usable copy from a previous eviction.
        already_cached = self._is_in_cpu_cache(model_entry.fingerprint)
        if already_cached:
            # Just release the GPU model — CPU cache already has it
            model_entry.model = None
        else:
            self._safe_to_cpu(model_entry)

        torch.cuda.empty_cache()
        log.debug(f"  GPU [{self.uuid}]: Evicted {model_entry.category} "
                 f"({model_entry.source}, ~{vram_mb}MB)"
                 f"{' (already in CPU cache)' if already_cached else ''}")

        # Free TRT shared memory if this was the last runner of its type
        self._maybe_release_trt_shared_memory(model_entry.category)

        # Store evicted model in CPU cache for fast reload (skip if already cached)
        if not already_cached:
            self._cpu_cache_store(model_entry)

        return model_entry

    def ensure_free_vram(self, min_free_bytes: int, protect: set[str] | None = None) -> bool:
        """Evict cached models until at least min_free_bytes VRAM is free.

        Pre-checks whether eviction can possibly succeed before touching
        anything.  If the total reclaimable VRAM (evictable models + freeable
        TRT shared memory + current free) is less than what's needed, returns
        False immediately without evicting a single model.

        Escalating eviction strategy (only if pre-check passes):
        1. Normal LRU eviction (inactive, evictable models)
        2. Orphaned TRT shared memory (buffers whose runners were all evicted)
        3. Force-free ALL non-active TRT shared memory (buffers are re-created
           on next use — this reclaims 100s of MB per component type)

        Returns True if the requested VRAM is available, False if not.
        """
        _, _, free = nvml.get_memory_info(self.nvml_handle)
        if free >= min_free_bytes:
            return True

        # Pre-check: can we possibly free enough?  Avoid partially evicting
        # models only to discover a non-evictable model blocks us.
        evictable = self.get_evictable_vram(protect)
        trt_freeable = self._get_freeable_trt_shared_memory_vram()
        max_available = free + evictable + trt_freeable
        if max_available < min_free_bytes:
            log.warning(
                f"  GPU [{self.uuid}]: Cannot free {min_free_bytes // (1024*1024)}MB — "
                f"only {max_available // (1024*1024)}MB reclaimable "
                f"(free={free // (1024*1024)}MB, evictable={evictable // (1024*1024)}MB, "
                f"trt_freeable={trt_freeable // (1024*1024)}MB)")
            return False

        while True:
            _, _, free = nvml.get_memory_info(self.nvml_handle)
            if free >= min_free_bytes:
                return True
            evicted = self.evict_lru(protect)
            if evicted is None:
                # No more evictable models — try freeing orphaned TRT shared
                # memory pools (buffers whose runners were all evicted).
                freed_any = self._free_orphaned_trt_shared_memory()
                if freed_any:
                    continue
                # Last resort: force-free ALL TRT shared memory that isn't
                # actively executing.  The buffers are large (UNet ~438-738MB,
                # VAE ~2.5GB) and will be re-allocated on next runner use.
                freed_any = self._force_free_trt_shared_memory()
                if freed_any:
                    continue
                log.warning(f"  GPU [{self.uuid}]: Cannot free {min_free_bytes // (1024*1024)}MB — "
                            f"only {free // (1024*1024)}MB free, no evictable models")
                return False

    def _free_orphaned_trt_shared_memory(self) -> bool:
        """Free TRT shared memory pools whose runners are no longer cached.

        Returns True if any memory was freed.
        """
        # Categories that map to each component type
        comp_to_cat = {
            "unet": "sdxl_unet_trt",
            "vae": "sdxl_vae_trt",
            "vae_enc": "sdxl_vae_enc_trt",
            "te1": "sdxl_te1_trt",
            "te2": "sdxl_te2_trt",
        }

        with self._trt_shared_mem_lock:
            pool_types = list(self._trt_shared_mem.keys())

        if not pool_types:
            return False

        freed_any = False
        with self._cache_lock:
            cached_cats = {m.category for m in self._cache.values()}

        for comp_type in pool_types:
            required_cat = comp_to_cat.get(comp_type)
            if required_cat is None:
                log.warning(f"  GPU [{self.uuid}]: Unknown TRT pool key '{comp_type}' "
                            f"— cannot determine orphan status, skipping")
                continue
            if required_cat not in cached_cats:
                freed = self.release_trt_shared_memory(comp_type)
                if freed > 0:
                    freed_any = True

        return freed_any

    def _force_free_trt_shared_memory(self) -> bool:
        """Force-free TRT shared memory buffers not currently in active use.

        Unlike _free_orphaned_trt_shared_memory (which only frees buffers for
        evicted runners), this frees buffers even when runners remain cached —
        as long as no runner of that type is actively executing.  The buffer is
        re-allocated transparently on next runner use via get_trt_shared_memory.

        Returns True if any memory was freed.
        """
        cat_to_comp = {
            "sdxl_unet_trt": "unet",
            "sdxl_vae_trt": "vae",
            "sdxl_vae_enc_trt": "vae_enc",
            "sdxl_te1_trt": "te1",
            "sdxl_te2_trt": "te2",
        }

        with self._trt_shared_mem_lock:
            pool_types = list(self._trt_shared_mem.keys())

        if not pool_types:
            return False

        # Find which component types have actively-executing TRT runners.
        # These cannot be freed — the runner is using the buffer right now.
        active_fps = self.get_active_fingerprints()
        active_comp_types: set[str] = set()
        with self._cache_lock:
            for fp, m in self._cache.items():
                if not m.category.endswith("_trt"):
                    continue
                if fp in active_fps:
                    comp_type = cat_to_comp.get(m.category)
                    if comp_type:
                        active_comp_types.add(comp_type)

        freed_any = False
        known_comp_types = set(cat_to_comp.values())
        for comp_type in pool_types:
            if comp_type in active_comp_types:
                continue  # Runner actively executing — buffer in use
            if comp_type not in known_comp_types:
                continue  # Unknown pool key — can't verify active status
            freed = self.release_trt_shared_memory(comp_type)
            if freed > 0:
                log.debug(f"  GPU [{self.uuid}]: Force-freed TRT shared memory "
                          f"'{comp_type}' ({freed // (1024*1024)}MB)")
                freed_any = True

        return freed_any

    def _get_freeable_trt_shared_memory_vram(self) -> int:
        """Return TRT shared memory VRAM that could be freed.

        Excludes buffers for component types with actively-executing runners.
        """
        cat_to_comp = {
            "sdxl_unet_trt": "unet",
            "sdxl_vae_trt": "vae",
            "sdxl_vae_enc_trt": "vae_enc",
            "sdxl_te1_trt": "te1",
            "sdxl_te2_trt": "te2",
        }

        with self._trt_shared_mem_lock:
            pool_items = list(self._trt_shared_mem.items())

        if not pool_items:
            return 0

        active_fps = self.get_active_fingerprints()
        active_comp_types: set[str] = set()
        with self._cache_lock:
            for fp, m in self._cache.items():
                if not m.category.endswith("_trt"):
                    continue
                if fp in active_fps:
                    comp_type = cat_to_comp.get(m.category)
                    if comp_type:
                        active_comp_types.add(comp_type)

        known_comp_types = set(cat_to_comp.values())
        total = 0
        for comp_type, buf in pool_items:
            if comp_type not in active_comp_types and comp_type in known_comp_types:
                total += buf.nbytes
        return total

    @property
    def session_cache_count(self) -> int:
        with self._cache_lock:
            return len(self._cache)

    # ---- TRT shared device memory ----

    def get_trt_shared_memory(self, component_type: str, min_bytes: int) -> torch.Tensor:
        """Get or grow the shared device memory buffer for a TRT component type.

        The buffer is allocated on THIS GPU's device — no cross-GPU VRAM access.
        If the existing buffer is too small, it is reallocated (grown).

        Args:
            component_type: "unet" or "vae" — determines which shared pool.
            min_bytes: Minimum buffer size (engine.device_memory_size).

        Returns:
            A torch.Tensor of at least *min_bytes* bytes on this GPU.
        """
        with self._trt_shared_mem_lock:
            existing = self._trt_shared_mem.get(component_type)
            if existing is not None and existing.nbytes >= min_bytes:
                return existing

            # Allocate on this specific GPU.  torch.empty with uint8 gives us
            # a raw byte buffer.  device= ensures it lands on the correct card.
            buf = torch.empty(min_bytes, dtype=torch.uint8, device=self.device)
            self._trt_shared_mem[component_type] = buf

            old_mb = (existing.nbytes / (1024 * 1024)) if existing is not None else 0
            new_mb = min_bytes / (1024 * 1024)
            if existing is not None:
                log.debug(f"  GPU [{self.uuid}]: TRT shared memory '{component_type}' "
                          f"grown {old_mb:.0f}MB → {new_mb:.0f}MB")
            else:
                log.debug(f"  GPU [{self.uuid}]: TRT shared memory '{component_type}' "
                          f"allocated {new_mb:.0f}MB")
            return buf

    def release_trt_shared_memory(self, component_type: str) -> int:
        """Free the shared device memory buffer for a TRT component type.

        Called after the last TRT runner of this type is evicted from the
        cache.  Returns the number of bytes freed (0 if no buffer existed).
        """
        with self._trt_shared_mem_lock:
            buf = self._trt_shared_mem.pop(component_type, None)
            if buf is None:
                return 0
            freed = buf.nbytes
            del buf
        torch.cuda.empty_cache()
        log.debug(f"  GPU [{self.uuid}]: TRT shared memory '{component_type}' "
                  f"freed ({freed // (1024 * 1024)}MB)")
        return freed

    def _maybe_release_trt_shared_memory(self, evicted_category: str) -> None:
        """Release TRT shared memory if no other runners of this type remain cached.

        Must be called AFTER the model has been removed from _cache.
        Maps TRT cache categories (e.g. 'sdxl_unet_trt') to shared memory
        component types ('unet', 'vae', 'te1', 'te2').

        Thread safety: the check-then-act gap between the cache scan and
        release_trt_shared_memory is safe because this is called from
        evict_lru, which is called from ensure_free_vram, which is always
        called under gpu.model_load_lock.  No concurrent thread can load
        a new runner between the scan and the release.
        """
        if not evicted_category.endswith("_trt"):
            return

        # Map category → shared memory component type
        cat_to_comp = {
            "sdxl_unet_trt": "unet",
            "sdxl_vae_trt": "vae",
            "sdxl_vae_enc_trt": "vae_enc",
            "sdxl_te1_trt": "te1",
            "sdxl_te2_trt": "te2",
        }
        comp_type = cat_to_comp.get(evicted_category)
        if comp_type is None:
            return

        # Check if any other TRT runners of this type remain cached
        with self._cache_lock:
            for m in self._cache.values():
                if m.category == evicted_category:
                    return  # still have at least one runner

        self.release_trt_shared_memory(comp_type)

    def get_trt_shared_memory_vram(self) -> int:
        """Return total VRAM consumed by TRT shared device memory buffers."""
        with self._trt_shared_mem_lock:
            return sum(buf.nbytes for buf in self._trt_shared_mem.values())


class GpuPool:
    """Manages all available GPUs."""

    def __init__(self):
        self.gpus: list[GpuInstance] = []

    def initialize(self, gpu_configs: list[GpuConfig]) -> None:
        """Match config GPU entries against NVML devices and create GpuInstances.
        Uses PCI bus ID to reliably bridge NVML UUIDs to CUDA device indices."""
        nvml.init()
        devices = nvml.get_devices()

        log.debug(f"  GpuPool: Found {len(devices)} GPU(s) via NVML")
        for dev in devices:
            log.debug(f"    NVML[{dev.index}]: {dev.name} UUID={dev.uuid} "
                     f"PCI={dev.pci_bus_id}")

        # Build UUID lookup
        device_by_uuid: dict[str, nvml.NvmlDeviceInfo] = {}
        for dev in devices:
            device_by_uuid[dev.uuid.lower()] = dev

        for cfg in gpu_configs:
            if not cfg.enabled:
                log.debug(f"  GpuPool: GPU [{cfg.uuid}] ({cfg.name}) is disabled — skipping")
                continue

            # Config UUID is like "GPU-caaaa9d0-..." — match against NVML UUID
            config_uuid = cfg.uuid.lower()
            nvml_dev = device_by_uuid.get(config_uuid)
            if not nvml_dev:
                # Try stripping "gpu-" prefix for partial matching
                stripped = config_uuid
                if stripped.startswith("gpu-"):
                    stripped = stripped[4:]
                for nvml_uuid, dev in device_by_uuid.items():
                    nvml_stripped = nvml_uuid
                    if nvml_stripped.startswith("gpu-"):
                        nvml_stripped = nvml_stripped[4:]
                    if nvml_stripped == stripped or nvml_uuid.endswith(stripped):
                        nvml_dev = dev
                        break

            if nvml_dev is None:
                log.warning(f"  GpuPool: Config GPU [{cfg.uuid}] not found in "
                            f"NVML devices — skipping")
                continue

            # Use NVML index as placeholder — workers always use cuda:0 via
            # CUDA_VISIBLE_DEVICES, so the main process doesn't need to resolve
            # the real CUDA index (which would require touching libcudart).
            cuda_idx = nvml_dev.index

            gpu = GpuInstance(cfg, nvml_dev, cuda_idx)
            self.gpus.append(gpu)

            # NOTE: In the multiprocessing architecture, these GpuInstance objects
            # are config/NVML holders only.  Workers create their own GpuInstance
            # internally (with real cuda:0 device).  VRAM caps, runtime feature
            # checks, and model caching all happen inside worker subprocesses.

            log.debug(f"  GpuPool: Registered GPU [{cfg.uuid}] = {nvml_dev.name} "
                     f"(CUDA:{cuda_idx}, PCI={nvml_dev.pci_bus_id}, "
                     f"{nvml_dev.total_memory // (1024*1024)}MB, "
                     f"caps={cfg.capabilities})")
            streamer.fire_event("gpu_added", {
                "uuid": cfg.uuid,
                "name": cfg.name,
                "device_id": cuda_idx,
                "total_memory": nvml_dev.total_memory,
                "capabilities": sorted(cfg.capabilities),
            })

    def remove_gpu(self, uuid: str) -> bool:
        """Remove a GPU by UUID and fire a ``gpu_removed`` event.

        Returns True if found and removed, False if not found.
        """
        for i, gpu in enumerate(self.gpus):
            if gpu.uuid.lower() == uuid.lower():
                self.gpus.pop(i)
                log.debug(f"  GpuPool: Removed GPU [{gpu.uuid}] ({gpu.name})")
                streamer.fire_event("gpu_removed", {
                    "uuid": gpu.uuid,
                    "name": gpu.name,
                    "device_id": gpu.device_id,
                })
                return True
        return False

    def has_capability(self, cap: str) -> bool:
        return any(g.supports_capability(cap) for g in self.gpus)

    def find_with_capability(self, cap: str) -> GpuInstance | None:
        for g in self.gpus:
            if g.supports_capability(cap):
                return g
        return None

    def available_count(self, cap: str) -> int:
        return sum(1 for g in self.gpus
                   if not g.is_failed and not g._trt_building
                   and g.supports_capability(cap))

    def get_all_capabilities(self) -> set[str]:
        caps: set[str] = set()
        for g in self.gpus:
            caps.update(g.capabilities)
        return caps


def is_trt_category(category: str) -> bool:
    """Check if a cache category is a TRT runner (not a PyTorch model)."""
    return category.endswith("_trt")
