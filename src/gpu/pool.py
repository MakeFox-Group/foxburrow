"""GPU pool management: GpuInstance + GpuPool.

Tracks loaded models, VRAM usage, and provides model cache with LRU eviction.
"""

from __future__ import annotations

import ctypes
import threading
from collections import Counter, OrderedDict
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
    properly loaded parameters/buffers intact.

    Returns the number of tensors fixed.
    """
    fixed = 0

    for name, param in list(model.named_parameters()):
        if not param.is_meta:
            continue
        parts = name.split(".")
        parent = model
        for part in parts[:-1]:
            parent = getattr(parent, part)
        new_param = nn.Parameter(
            torch.zeros(param.shape, dtype=param.dtype, device="cpu"),
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
        new_buf = torch.zeros(buf.shape, dtype=buf.dtype, device="cpu")
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


class GpuInstance:
    """Represents a single GPU with model cache and VRAM tracking."""

    def __init__(self, config: GpuConfig, nvml_device: nvml.NvmlDeviceInfo, torch_device_id: int):
        self.uuid = config.uuid
        self.name = config.name
        self.capabilities = config.capabilities
        self.device_id = torch_device_id
        self.device = torch.device(f"cuda:{torch_device_id}")
        self.nvml_handle = nvml_device.handle
        self.total_memory = nvml_device.total_memory

        # Busy semaphore for status reporting
        self._busy_lock = threading.Lock()
        self._busy = False

        # Model cache: fingerprint -> CachedModel (LRU ordered)
        self._cache: OrderedDict[str, CachedModel] = OrderedDict()
        self._cache_lock = threading.Lock()

        # Active model pointers — ref-counted so concurrent jobs don't clobber each other
        self._active_fp_counts: Counter[str] = Counter()
        self._active_fp_lock = threading.Lock()

        # Session group tracking
        self._current_group: str | None = None

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

        # GPU failure tracking — permanently disables the GPU when a fatal
        # CUDA error corrupts the context (e.g. cudaErrorIllegalAddress / Xid 31)
        self._failed = False
        self._fail_reason: str = ""
        self._consecutive_failures = 0

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
        return self._busy

    def try_acquire(self) -> bool:
        with self._busy_lock:
            if self._busy:
                return False
            self._busy = True
            return True

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
                return self._cache[fingerprint]
            return None

    def cache_model(self, fingerprint: str, category: str, model: object, estimated_vram: int,
                    source: str = "", actual_vram: int = 0,
                    evict_callback: Callable[[], None] | None = None) -> None:
        with self._cache_lock:
            self._cache[fingerprint] = CachedModel(
                fingerprint=fingerprint,
                category=category,
                model=model,
                estimated_vram=estimated_vram,
                actual_vram=actual_vram,
                source=source,
                evict_callback=evict_callback,
            )
            self._cache.move_to_end(fingerprint)

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
        """Increment ref counts for fingerprints actively in use by a job."""
        with self._active_fp_lock:
            self._active_fp_counts.update(fingerprints)

    def remove_active_fingerprints(self, fingerprints: set[str]) -> None:
        """Decrement ref counts when a job finishes using these fingerprints."""
        with self._active_fp_lock:
            self._active_fp_counts.subtract(fingerprints)
            # Clean up zero/negative counts
            self._active_fp_counts += Counter()

    def get_active_fingerprints(self) -> set[str]:
        """Return the set of all fingerprints currently in use by any job."""
        with self._active_fp_lock:
            return {fp for fp, count in self._active_fp_counts.items() if count > 0}

    # Legacy compatibility
    def set_active_fingerprints(self, fingerprints: set[str]) -> None:
        self.add_active_fingerprints(fingerprints)

    def _is_evictable(self, fp: str, protect: set[str] | None) -> bool:
        """Check if a cached model can be evicted."""
        if protect and fp in protect:
            return False
        if fp in self._unevictable_fingerprints:
            return False
        return True

    def _safe_to_cpu(self, model_entry: CachedModel) -> None:
        """Move a model to CPU, fixing any meta tensors first."""
        if not hasattr(model_entry.model, "to"):
            return
        if isinstance(model_entry.model, nn.Module):
            n = fix_meta_tensors(model_entry.model)
            if n:
                log.warning(f"  GPU [{self.uuid}]: Fixed {n} meta tensor(s) "
                            f"in {model_entry.category} before eviction")
        model_entry.model.to("cpu")

    def evict_lru(self, protect: set[str] | None = None) -> CachedModel | None:
        """Evict the least-recently-used cached model not in the protected,
        unevictable, or actively-in-use sets. Prefers evicting models from
        non-current session groups first. Moves model to CPU.
        Returns evicted model or None."""
        # Snapshot active fingerprints BEFORE acquiring _cache_lock to maintain
        # consistent lock ordering (active_fp_lock → cache_lock), matching
        # get_evictable_vram(). Models actively being used by another thread's
        # forward pass must never be evicted — .to("cpu") mid-inference causes
        # device mismatch errors or silent corruption (black images).
        active_fps = self.get_active_fingerprints()
        with self._cache_lock:
            # First pass: try to evict from non-current groups (least valuable)
            if self._current_group:
                for fp in list(self._cache.keys()):
                    if not self._is_evictable(fp, protect):
                        continue
                    if fp in active_fps:
                        continue
                    m = self._cache[fp]
                    if _get_group_for_category(m.category) != self._current_group:
                        model_entry = self._cache.pop(fp)
                        if model_entry.evict_callback:
                            model_entry.evict_callback()
                        self._safe_to_cpu(model_entry)
                        torch.cuda.empty_cache()
                        log.debug(f"  GPU [{self.uuid}]: Evicted {model_entry.category} "
                                 f"(~{model_entry.estimated_vram // (1024*1024)}MB, "
                                 f"non-current group)")
                        return model_entry
            # Second pass: evict from any group (but still respect unevictable/active)
            for fp in list(self._cache.keys()):
                if not self._is_evictable(fp, protect):
                    continue
                if fp in active_fps:
                    continue
                model_entry = self._cache.pop(fp)
                if model_entry.evict_callback:
                    model_entry.evict_callback()
                self._safe_to_cpu(model_entry)
                torch.cuda.empty_cache()
                log.debug(f"  GPU [{self.uuid}]: Evicted {model_entry.category} "
                         f"(~{model_entry.estimated_vram // (1024*1024)}MB)")
                return model_entry
            return None

    def ensure_free_vram(self, min_free_bytes: int, protect: set[str] | None = None) -> None:
        """Evict cached models until at least min_free_bytes VRAM is free."""
        while True:
            _, _, free = nvml.get_memory_info(self.nvml_handle)
            if free >= min_free_bytes:
                return
            evicted = self.evict_lru(protect)
            if evicted is None:
                log.warning(f"  GPU [{self.uuid}]: Cannot free {min_free_bytes // (1024*1024)}MB — "
                            f"only {free // (1024*1024)}MB free, no evictable models")
                return

    def ensure_session_group(self, group: str) -> None:
        """Switch the current session group. Models from other groups are NOT
        evicted immediately — they stay cached and are only evicted when VRAM
        is needed (via ensure_free_vram/evict_lru, which prefers evicting
        non-current-group models first)."""
        if self._current_group != group:
            log.debug(f"  GPU [{self.uuid}]: Session group → {group}")
        self._current_group = group

    @property
    def session_cache_count(self) -> int:
        with self._cache_lock:
            return len(self._cache)


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

            # Map NVML device to CUDA index via PCI bus ID bridge
            cuda_idx = _cuda_index_from_pci_bus_id(nvml_dev.pci_bus_id)
            if cuda_idx is None:
                # Fallback: with CUDA_DEVICE_ORDER=PCI_BUS_ID, NVML index
                # should match CUDA index
                log.warning(f"  GpuPool: Could not resolve CUDA index for "
                            f"GPU [{cfg.uuid}] via PCI bus ID "
                            f"({nvml_dev.pci_bus_id}), falling back to "
                            f"NVML index {nvml_dev.index}")
                cuda_idx = nvml_dev.index

            gpu = GpuInstance(cfg, nvml_dev, cuda_idx)
            self.gpus.append(gpu)

            # Cap VRAM allocation at 98% — turns fatal Xid 31 MMU faults into
            # recoverable torch.cuda.OutOfMemoryError when cuDNN overcommits.
            try:
                torch.cuda.set_per_process_memory_fraction(0.98, gpu.device)
                cap_mb = int(nvml_dev.total_memory * 0.98 / (1024 * 1024))
                log.debug(f"  GpuPool: VRAM cap set to 98% ({cap_mb}MB) on CUDA:{cuda_idx}")
            except Exception as e:
                log.warning(f"  GpuPool: Could not set VRAM cap on CUDA:{cuda_idx}: {e}")

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

        # Probe fork features against the actual CUDA allocator backend.
        # Must run after at least one GPU is registered (CUDA context ready).
        if self.gpus:
            from gpu.torch_ext import check_runtime_support
            check_runtime_support(self.gpus[0].device)

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
        return sum(1 for g in self.gpus if not g.is_failed and g.supports_capability(cap))

    def get_all_capabilities(self) -> set[str]:
        caps: set[str] = set()
        for g in self.gpus:
            caps.update(g.capabilities)
        return caps


def _get_group_for_category(category: str) -> str:
    """Map model category to session group."""
    if category.startswith("sdxl") or category == "sdxl_lora":
        return "sdxl"
    if category == "upscale":
        return "upscale"
    if category == "bgremove":
        return "bgremove"
    if category == "tagger":
        return "tagger"
    return "unknown"
