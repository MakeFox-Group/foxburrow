"""Entry point: config, GPU init, model discovery, start server."""

from __future__ import annotations

import argparse
import asyncio
import os
import secrets
import signal
import sys
import threading

# Force CUDA to enumerate GPUs by PCI bus ID (matching NVML ordering).
# Must be set BEFORE importing torch or any CUDA library.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

# Use CUDA's native cudaMallocAsync allocator instead of PyTorch's caching allocator.
# This delegates all allocation/free to the CUDA driver's stream-ordered memory pools,
# which handle fragmentation natively and can release memory back to the OS.
# Matches Stable Diffusion WebUI Forge's --cuda-malloc flag.
os.environ.setdefault("PYTORCH_ALLOC_CONF", "backend:cudaMallocAsync")

# Make CUDA operations synchronous so errors are reported at the exact kernel that
# fails, not at some later unrelated API call.  Slower, but needed to diagnose
# the recurring illegal memory access crashes on RTX 5070 (Blackwell/sm_120).
# TODO: Remove once the root cause is found and fixed.
os.environ.setdefault("CUDA_LAUNCH_BLOCKING", "1")

# Keep HuggingFace downloads (config JSONs, tokenizer vocabs) in data/hf_cache/
# instead of ~/.cache/huggingface/. Must be set before any HF imports.
os.environ.setdefault("HF_HOME", os.path.join(os.path.abspath("data"), "hf_cache"))

import torch
import uvicorn

# Explicitly enable all Scaled Dot-Product Attention backends.
# This matches Forge's --attention-pytorch behaviour and ensures PyTorch picks
# the fastest available kernel (flash, mem-efficient, or math) per operation.
torch.backends.cuda.enable_math_sdp(True)
torch.backends.cuda.enable_flash_sdp(True)
torch.backends.cuda.enable_mem_efficient_sdp(True)

import log
from config import FoxBurrowConfig, _auto_threads
from gpu.pool import GpuPool
from scheduling.model_registry import ModelRegistry
from scheduling.pipeline import PipelineFactory
from scheduling.queue import JobQueue
from scheduling.scheduler import GpuScheduler
from scheduling.worker import GpuWorker
from state import app_state


_CONFIG_CANDIDATES = [
    "foxburrow.ini",
    "conf/foxburrow.ini",
]

_DEFAULT_CONFIG_PATH = "conf/foxburrow.ini"


def find_config() -> str:
    """Find foxburrow.ini relative to the working directory."""
    for c in _CONFIG_CANDIDATES:
        if os.path.isfile(c):
            return os.path.abspath(c)
    return ""


def _generate_secret() -> str:
    """Generate a 32-character alphanumeric secret (a-z, A-Z, 0-9)."""
    alphabet = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
    return "".join(secrets.choice(alphabet) for _ in range(32))


def generate_default_config() -> str:
    """Auto-detect GPUs via NVML and write a default foxburrow.ini.

    Returns the absolute path of the generated config file.
    """
    from gpu import nvml

    try:
        nvml.init()
        devices = nvml.get_devices()
    except Exception as ex:
        log.error(f"  Failed to detect GPUs: {ex}")
        log.error("  Ensure NVIDIA drivers are installed (nvidia-smi should work).")
        sys.exit(1)

    secret = _generate_secret()

    lines = [
        "[server]",
        "address=127.0.0.1",
        "port=15888",
        "models_dir=models/",
        "tensorrt_cache=data/tensorrt_cache/",
        f"secret={secret}",
        "",
        "# IMPORTANT: You must review this configuration and set enabled=true",
        "# before FoxBurrow will start. This is a safety measure to ensure you",
        "# have reviewed and customized settings for your environment.",
        "enabled=false",
        "",
    ]

    for dev in devices:
        vram_gb = dev.total_memory / (1024 ** 3)
        caps = "sdxl,upscale,bgremove,tag"

        # Always onload tagger; only mark unevictable for GPUs with >8GB VRAM
        if vram_gb > 8:
            onload = "tag"
            unevictable = "tag"
        else:
            onload = "tag"
            unevictable = ""

        lines.append(f"[{dev.uuid}]")
        lines.append(f"name={dev.name}")
        lines.append(f"capabilities={caps}")
        lines.append(f"onload={onload}")
        if unevictable:
            lines.append(f"unevictable={unevictable}")
        lines.append("")

    config_path = os.path.abspath(_DEFAULT_CONFIG_PATH)
    os.makedirs(os.path.dirname(config_path), exist_ok=True)
    with open(config_path, "w") as f:
        f.write("\n".join(lines))

    log.info(f"  Generated default config: {config_path}")
    log.info(f"  Detected {len(devices)} GPU(s)")
    for dev in devices:
        vram_gb = dev.total_memory / (1024 ** 3)
        log.info(f"    {dev.name} ({vram_gb:.1f} GB) — {dev.uuid}")

    return config_path


def discover_sdxl_models(models_dir: str) -> dict[str, str]:
    """Discover SDXL checkpoints under models_dir/sdxl/ recursively.

    Walks the directory tree to find:
    - Diffusers-format directories (with unet/, text_encoder/, text_encoder_2/, vae/)
    - Single-file .safetensors checkpoints (at any nesting depth)

    Model names:
    - Top-level: filename without extension (e.g. "mymodel")
    - Nested: "subfolder/filename" relative path (e.g. "anime/mymodel")
    """
    sdxl_dir = os.path.join(models_dir, "sdxl")
    if not os.path.isdir(sdxl_dir):
        log.warning(f"  SDXL models directory not found: {sdxl_dir}")
        return {}

    _DIFFUSERS_REQUIRED = {"unet", "text_encoder", "text_encoder_2", "vae"}
    models: dict[str, str] = {}

    for dirpath, dirnames, filenames in os.walk(sdxl_dir):
        # Check if current directory is a diffusers model
        subdirs_here = set(dirnames)
        if _DIFFUSERS_REQUIRED.issubset(subdirs_here):
            rel = os.path.relpath(dirpath, sdxl_dir)
            model_name = rel if rel != "." else os.path.basename(dirpath)
            if model_name in models:
                log.warning(f"  SDXL name collision: '{model_name}' already registered "
                            f"({models[model_name]}), skipping {dirpath}")
            else:
                models[model_name] = dirpath
                log.info(f"  Discovered SDXL model (diffusers): {model_name}")
            # Don't descend into diffusers model subdirectories
            dirnames.clear()
            continue

        # Check for single-file .safetensors in this directory
        for fname in sorted(filenames):
            if not fname.endswith(".safetensors"):
                continue
            fpath = os.path.join(dirpath, fname)
            base_name = fname.rsplit(".", 1)[0]
            rel_dir = os.path.relpath(dirpath, sdxl_dir)
            if rel_dir == ".":
                model_name = base_name
            else:
                model_name = f"{rel_dir}/{base_name}"
            if model_name in models:
                log.warning(f"  SDXL name collision: '{model_name}' already registered "
                            f"({models[model_name]}), skipping {fpath}")
            else:
                models[model_name] = fpath
                log.info(f"  Discovered SDXL model (single-file): {model_name}")

    return models


_SDXL_HF_CONFIGS = [
    "stabilityai/stable-diffusion-xl-base-1.0",
    "stabilityai/stable-diffusion-xl-refiner-1.0",
]

_HF_CONFIG_PATTERNS = ["**/*.json", "*.json", "*.txt", "**/*.txt", "**/*.model"]


def _prefetch_sdxl_configs() -> None:
    """Pre-download SDXL config/tokenizer files from HuggingFace.

    Downloads only config JSONs and tokenizer vocab files (~10-15MB),
    NOT model weights. Cached in data/hf_cache/ (via HF_HOME env var).
    Subsequent runs use local_files_only=True — no network access at all.
    """
    from huggingface_hub import snapshot_download

    for repo_id in _SDXL_HF_CONFIGS:
        # Try local cache first (no network, no progress bars)
        try:
            snapshot_download(
                repo_id,
                allow_patterns=_HF_CONFIG_PATTERNS,
                local_files_only=True,
            )
            log.info(f"  HF configs cached: {repo_id}")
            continue
        except Exception:
            pass  # Not cached yet — download below

        # First run: download from HuggingFace
        try:
            snapshot_download(
                repo_id,
                allow_patterns=_HF_CONFIG_PATTERNS,
            )
            log.info(f"  HF configs downloaded: {repo_id}")
        except Exception as ex:
            log.warning(f"  Failed to download HF configs for {repo_id}: {ex}")
            log.warning("  Single-file checkpoint extraction may fail without network access.")


def _background_model_init(
    config: FoxBurrowConfig,
    models_dir: str,
    all_sdxl: dict[str, str],
) -> None:
    """Heavy initialization that runs in a background thread after server starts.

    Handles: HF config prefetch, SDXL fingerprinting, utility model registration,
    LoRA hashing, and GPU onload/unevictable processing.
    """
    import time
    start_time = time.monotonic()
    log.info("  Background init: starting...")

    # ONE shared fingerprint pool — SDXL base models go first (priority),
    # then LoRA hashing reuses the same threads.
    from concurrent.futures import ThreadPoolExecutor
    fp_threads = _auto_threads(config.threads.fingerprint, 8)
    fp_pool = ThreadPoolExecutor(max_workers=fp_threads)
    gpus = app_state.gpu_pool.gpus

    # 1. Start model scanning FIRST — CPU/disk only, runs concurrently
    #    with all the warm-up work below.
    if all_sdxl:
        from utils.model_scanner import ModelScanner
        scanner = ModelScanner(app_state.registry, app_state, max_workers=fp_threads)
        scanner.start(all_sdxl, pool=fp_pool)
        app_state.model_scanner = scanner

    # 2. Pre-warm everything while scanner runs in background.
    #    Without this, parallel GPU onload threads all hit cold-start costs
    #    simultaneously: first `import timm` holds the GIL for ~5-10s (huge
    #    model registry), first CUDA op per device initializes the runtime
    #    (~2-5s each, driver-serialized). Paying these once upfront avoids
    #    serializing the parallel GPU onload.

    # HF config files (network on first run, then cached locally)
    _prefetch_sdxl_configs()

    # Heavy library imports (GIL-bound — do once before threads fan out)
    import timm              # noqa: F401 — massive model registry
    import safetensors.torch  # noqa: F401

    # CUDA context init: first op per device triggers expensive runtime init.
    # Driver serializes this internally so sequential is optimal.
    if gpus:
        log.info(f"  Pre-warming CUDA contexts for {len(gpus)} GPU(s)...")
        for gpu in gpus:
            torch.zeros(1, device=gpu.device)

    # 3. Register utility models (upscale, bgremove) — fast, single files.
    # Must happen BEFORE GPU onload so upscale/bgremove components exist.
    upscale_path = discover_model_file(models_dir, os.path.join("other", "upscale"),
                                       [".pth", ".safetensors", ".onnx"])
    if upscale_path:
        app_state.registry.register_upscale_model(upscale_path)

    bgremove_dir = os.path.join(models_dir, "other", "bgremove")
    bgremove_model = os.path.join(bgremove_dir, "model.safetensors")
    if os.path.isfile(bgremove_model):
        app_state.registry.register_bgremove_model(bgremove_model)
    else:
        bgremove_path = discover_model_file(models_dir, os.path.join("other", "bgremove"),
                                            [".safetensors", ".pth", ".onnx"])
        if bgremove_path:
            app_state.registry.register_bgremove_model(bgremove_path)

    # 4a. Quick GPU onloads (upscale, bgremove) — fast, <1s each.
    # These must finish before SDXL onloads so components exist in the cache.
    if gpus:
        with ThreadPoolExecutor(max_workers=len(gpus)) as gpu_pool:
            list(gpu_pool.map(
                lambda g: _process_gpu_onload(
                    g, app_state, types={"upscale", "bgremove"}),
                gpus))

    # 4b. Fire-and-forget tagger loading — slow (~30s) but independent.
    # Runs concurrently with model scanning, SDXL onloads, and LoRA hashing.
    # The tagger finishes whenever it finishes; /api/tag returns 404 until ready.
    tagger_threads: list[threading.Thread] = []
    for gpu in gpus:
        if "tag" in gpu.onload:
            t = threading.Thread(
                target=_process_gpu_onload,
                args=(gpu, app_state),
                kwargs={"types": {"tag"}},
                name=f"tagger-{gpu.uuid[:8]}",
                daemon=True,
            )
            t.start()
            tagger_threads.append(t)
    if tagger_threads:
        log.info(f"  Tagger loading on {len(tagger_threads)} GPU(s) (background)")

    # 5. Wait for model scanning to finish.
    # Base models get priority I/O — LoRA hashing starts after.
    scanner = app_state.model_scanner
    if scanner is not None and scanner._thread is not None:
        scanner._thread.join()

    # 6. Now load SDXL-specific onloads (model pre-loads into GPU VRAM)
    # and mark unevictable entries. Requires models to be registered.
    if gpus:
        with ThreadPoolExecutor(max_workers=len(gpus)) as gpu_pool:
            list(gpu_pool.map(
                lambda g: _process_gpu_onload(g, app_state, types={"sdxl"}),
                gpus))

        for gpu in gpus:
            _process_gpu_unevictable(gpu, app_state)

    # 7. LoRA hashing — same fingerprint pool, starts AFTER SDXL models.
    # Last consumer shuts down the shared pool when complete.
    if app_state.lora_index:
        from utils.lora_index import start_background_hashing
        start_background_hashing(app_state.lora_index, pool=fp_pool, shutdown_pool=True)
    else:
        fp_pool.shutdown(wait=False)

    elapsed = time.monotonic() - start_time
    log.info(f"  Background init: complete ({elapsed:.1f}s)")

    from api.websocket import streamer
    streamer.fire_event("init_complete", {"elapsed_s": round(elapsed, 1)})


def discover_model_file(models_dir: str, subdir: str, extensions: list[str]) -> str | None:
    """Find a model file in models_dir/subdir/ with one of the given extensions."""
    search_dir = os.path.join(models_dir, subdir)
    if not os.path.isdir(search_dir):
        return None
    for f in sorted(os.listdir(search_dir)):
        for ext in extensions:
            if f.endswith(ext):
                return os.path.join(search_dir, f)
    return None


def write_pid(path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(str(os.getpid()))


# Component alias map for onload/unevictable syntax:
#   "te1"  → ["sdxl_te1"]
#   "te2"  → ["sdxl_te2"]
#   "te"   → ["sdxl_te1", "sdxl_te2"]
#   "unet" → ["sdxl_unet"]
#   "vae"  → ["sdxl_vae"]
#   (no suffix = all)
_COMPONENT_ALIASES: dict[str, list[str]] = {
    "te1":  ["sdxl_te1"],
    "te2":  ["sdxl_te2"],
    "te":   ["sdxl_te1", "sdxl_te2"],
    "unet": ["sdxl_unet"],
    "vae":  ["sdxl_vae"],
}
_ALL_SDXL_CATEGORIES = ["sdxl_te1", "sdxl_te2", "sdxl_unet", "sdxl_vae"]

# Registry component index: [te1, te2, unet, vae_dec, vae_enc]
_CATEGORY_TO_REGISTRY_INDEX = {
    "sdxl_te1": 0,
    "sdxl_te2": 1,
    "sdxl_unet": 2,
    "sdxl_vae": 3,
}


def _resolve_onload_entry(entry: str, app_state) -> list[dict] | str:
    """Parse an onload/unevictable entry and resolve to load actions.

    Returns a list of dicts: [{"type": "tag"}, {"type": "sdxl", "model": ..., "categories": [...]}]
    Or a string error message on failure.
    """
    entry = entry.strip().lower()

    # Simple types
    if entry in ("tag", "upscale", "bgremove"):
        return [{"type": entry}]

    # SDXL component: "model_name" or "model_name:component"
    if ":" in entry:
        model_name, component = entry.rsplit(":", 1)
        if component not in _COMPONENT_ALIASES:
            return f"Unknown component '{component}' (valid: te1, te2, te, unet, vae)"
        categories = _COMPONENT_ALIASES[component]
    else:
        model_name = entry
        categories = list(_ALL_SDXL_CATEGORIES)

    # Look up model
    if model_name not in app_state.sdxl_models:
        return f"Unknown model '{model_name}'"

    return [{"type": "sdxl", "model": model_name, "categories": categories}]


def _process_gpu_onload(gpu, app_state, types: set[str] | None = None) -> None:
    """Pre-load models specified in GPU's onload config.

    If *types* is provided, only process onload entries of those types
    (e.g. ``{"tag", "upscale", "bgremove"}``). Others are silently skipped.
    Pass ``None`` to process all types.
    """
    from gpu.pool import GpuInstance

    for entry in gpu.onload:
        # Quick pre-filter: skip entries whose type doesn't match this phase.
        # Avoids resolving SDXL model names before they're registered.
        if types is not None:
            entry_lower = entry.strip().lower()
            if entry_lower in ("tag", "upscale", "bgremove"):
                if entry_lower not in types:
                    continue
            else:
                # Everything else is an SDXL entry (model name or model:component)
                if "sdxl" not in types:
                    continue
        actions = _resolve_onload_entry(entry, app_state)
        if isinstance(actions, str):
            log.warning(f"  GPU [{gpu.uuid}]: onload={entry}: {actions}")
            continue

        for action in actions:
            try:
                if action["type"] == "tag":
                    if not gpu.supports_capability("tag"):
                        log.warning(f"  GPU [{gpu.uuid}]: onload=tag but GPU lacks 'tag' capability")
                        continue
                    from handlers.tagger import init_tagger, unload_tagger
                    import torch
                    before = torch.cuda.memory_allocated(gpu.device)
                    init_tagger(gpu.device)
                    after = torch.cuda.memory_allocated(gpu.device)
                    actual_vram = after - before
                    # Evict callback: clean up _tagger_instances when cache evicts the tagger
                    _dev = gpu.device
                    def _tagger_evict_cb(_d=_dev):
                        unload_tagger(_d)
                    fp = f"tagger:{gpu.uuid}"
                    gpu.cache_model(
                        fp, "tagger", None,
                        estimated_vram=150 * 1024 * 1024,
                        source="JTP_PILOT2_SigLIP",
                        actual_vram=actual_vram,
                        evict_callback=_tagger_evict_cb,
                    )
                    gpu.mark_onload(fp)
                    log.info(f"  GPU [{gpu.uuid}]: Pre-loaded tagger ({actual_vram // (1024*1024)}MB)")

                elif action["type"] == "upscale":
                    from handlers.upscale import load_model as load_upscale
                    model = load_upscale(gpu.device)
                    comp = app_state.registry.get_upscale_component()
                    gpu.cache_model(comp.fingerprint, "upscale", model,
                                    comp.estimated_vram_bytes, source="realesrgan")
                    gpu.mark_onload(comp.fingerprint)
                    log.info(f"  GPU [{gpu.uuid}]: Pre-loaded upscale model")

                elif action["type"] == "bgremove":
                    from handlers.bgremove import load_model as load_bgremove
                    model = load_bgremove(gpu.device)
                    comp = app_state.registry.get_bgremove_component()
                    gpu.cache_model(comp.fingerprint, "bgremove", model,
                                    comp.estimated_vram_bytes, source="rmbg")
                    gpu.mark_onload(comp.fingerprint)
                    log.info(f"  GPU [{gpu.uuid}]: Pre-loaded bgremove model")

                elif action["type"] == "sdxl":
                    model_name = action["model"]
                    model_dir = app_state.sdxl_models[model_name]
                    registry_comps = app_state.registry.get_sdxl_components(model_dir)
                    from handlers.sdxl import load_component

                    for category in action["categories"]:
                        idx = _CATEGORY_TO_REGISTRY_INDEX.get(category)
                        if idx is None:
                            continue
                        comp = registry_comps[idx]
                        if gpu.is_component_loaded(comp.fingerprint):
                            gpu.mark_onload(comp.fingerprint)
                            continue
                        model = load_component(category, model_dir, gpu.device)
                        # Extract display name
                        source = os.path.basename(model_dir)
                        for ext in (".safetensors", ".ckpt"):
                            if source.endswith(ext):
                                source = source[:-len(ext)]
                                break
                        gpu.cache_model(comp.fingerprint, category, model,
                                        comp.estimated_vram_bytes, source=source)
                        gpu.mark_onload(comp.fingerprint)
                        log.info(f"  GPU [{gpu.uuid}]: Pre-loaded {category} from {source}")

            except Exception as ex:
                log.log_exception(ex, f"GPU [{gpu.uuid}]: onload={entry} failed")


def _process_gpu_unevictable(gpu, app_state) -> None:
    """Mark fingerprints as unevictable based on GPU's unevictable config."""
    for entry in gpu.unevictable:
        actions = _resolve_onload_entry(entry, app_state)
        if isinstance(actions, str):
            log.warning(f"  GPU [{gpu.uuid}]: unevictable={entry}: {actions}")
            continue

        for action in actions:
            try:
                if action["type"] == "tag":
                    # Tagger isn't in the GPU model cache (it's managed separately),
                    # so it's inherently unevictable. Nothing to do.
                    pass

                elif action["type"] == "upscale":
                    comp = app_state.registry.get_upscale_component()
                    gpu.mark_unevictable(comp.fingerprint)

                elif action["type"] == "bgremove":
                    comp = app_state.registry.get_bgremove_component()
                    gpu.mark_unevictable(comp.fingerprint)

                elif action["type"] == "sdxl":
                    model_name = action["model"]
                    model_dir = app_state.sdxl_models[model_name]
                    registry_comps = app_state.registry.get_sdxl_components(model_dir)

                    for category in action["categories"]:
                        idx = _CATEGORY_TO_REGISTRY_INDEX.get(category)
                        if idx is None:
                            continue
                        comp = registry_comps[idx]
                        gpu.mark_unevictable(comp.fingerprint)
                        log.info(f"  GPU [{gpu.uuid}]: Marked {category} ({model_name}) as unevictable")

            except Exception as ex:
                log.log_exception(ex, f"GPU [{gpu.uuid}]: unevictable={entry} failed")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="foxburrow — GPU inference server")
    parser.add_argument(
        "--no-tui", action="store_true",
        help="Disable the TUI console (headless mode — logs to stdout)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    # Start file logging before anything else so the full startup is captured.
    log.init_file(os.path.abspath("logs/foxburrow.log"))
    log.info("foxburrow starting (Python/PyTorch)")

    # Write PID file
    pid_path = os.path.abspath("data/foxburrow.pid")
    write_pid(pid_path)
    log.info(f"  PID {os.getpid()} written to {pid_path}")

    # Find or generate config
    config_path = find_config()
    first_start = False
    if not config_path:
        log.info("  No foxburrow.ini found — generating default configuration...")
        config_path = generate_default_config()
        first_start = True

    log.info(f"  Config: {config_path}")
    config = FoxBurrowConfig.load_from_file(config_path)
    app_state.config = config

    # Set HF token if configured (enables faster downloads and private repos)
    if config.server.hf_token:
        os.environ.setdefault("HF_TOKEN", config.server.hf_token)

    # Restrict CUDA visibility to only enabled GPUs.
    # Without this, PyTorch/CUDA runtime creates driver contexts on ALL GPUs
    # (visible in nvidia-smi), and third-party libs like diffusers can leak
    # VRAM onto disabled GPUs via the default device (cuda:0).
    # CUDA accepts GPU UUIDs directly — no index mapping needed.
    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        enabled_uuids = [cfg.uuid for cfg in config.gpus if cfg.enabled]
        if enabled_uuids:
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(enabled_uuids)
            log.info(f"  CUDA visibility: {len(enabled_uuids)} enabled GPU(s)")
        elif config.gpus:
            # All GPUs disabled — hide everything
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
            log.warning("  All GPUs are disabled in config — CUDA has no visible devices")
    else:
        log.info(f"  CUDA_VISIBLE_DEVICES already set: {os.environ['CUDA_VISIBLE_DEVICES']}")

    # Safety gate: server won't start unless enabled=true
    if not config.server.enabled:
        log.info("")
        log.info("=" * 60)
        log.info("  FoxBurrow is NOT ENABLED.")
        log.info("")
        if first_start:
            log.info("  A default configuration has been generated at:")
            log.info(f"    {config_path}")
            log.info("")
            log.info("  Please review your GPU settings, network address,")
            log.info("  and API secret, then set enabled=true to start.")
        else:
            log.info("  Set enabled=true in your foxburrow.ini to start.")
        log.info("=" * 60)
        sys.exit(0)

    # Root user warning
    if os.getuid() == 0:
        log.warning("!!! Running as root !!!")
        log.warning("  This is not recommended. Consider running as a non-root user.")

    # Network exposure warning
    if config.server.address in ("0.0.0.0", "::"):
        log.warning(f"  Listening on {config.server.address} — this may expose FoxBurrow")
        log.warning("  to ALL network interfaces, including public ones.")
        log.warning("  Use 127.0.0.1 to restrict to local connections only.")

    # API secret status
    if config.server.secret:
        log.info(f"  API authentication: ENABLED")
        log.info("    Header:  Authorization: Bearer <secret>")
        log.info("    URL:     ?apikey=<secret>")
        log.info("    See 'secret=' in foxburrow.ini for your token")
        if len(config.server.secret) < 12:
            log.warning("  Your API secret is very short — consider using at least 16 characters.")
    else:
        log.info("")
        log.warning("!!! WARNING: No API secret configured !!!")
        log.warning("  All API endpoints are PUBLICLY ACCESSIBLE.")
        log.warning("  Anyone who can reach this server can use it.")
        log.warning("  Add 'secret=<your_token>' to [server] in foxburrow.ini")
        log.info("")
    # Resolve models directory relative to working directory (project root)
    models_dir = os.path.normpath(os.path.abspath(config.server.models_dir))
    log.info(f"  Models dir: {models_dir}")

    # ── Model discovery (fast — stat only, no hashing) ──────────────
    available_capabilities: set[str] = set()

    # Discover SDXL models (recursive walk)
    all_sdxl = discover_sdxl_models(models_dir)
    if all_sdxl:
        available_capabilities.add("sdxl")

    # Discover LoRA files (filename scan only, no hashing yet)
    loras_dir = os.path.join(models_dir, "loras")
    app_state.loras_dir = loras_dir
    if os.path.isdir(loras_dir):
        from utils.lora_index import discover_loras
        app_state.lora_index = discover_loras(loras_dir)
        log.info(f"  Discovered {len(app_state.lora_index)} LoRA files")
    else:
        log.info(f"  LoRA directory not found: {loras_dir} (skipping)")

    # Discover upscale model (in models/other/)
    upscale_path = discover_model_file(models_dir, os.path.join("other", "upscale"),
                                       [".pth", ".safetensors", ".onnx"])
    if upscale_path:
        available_capabilities.add("upscale")
        from handlers.upscale import set_model_path
        set_model_path(upscale_path)
        log.info(f"  Upscale model: {upscale_path}")

    # Discover bgremove model (in models/other/)
    bgremove_found = False
    bgremove_dir = os.path.join(models_dir, "other", "bgremove")
    if os.path.isfile(os.path.join(bgremove_dir, "config.json")):
        bgremove_found = True
        from handlers.bgremove import set_model_path
        set_model_path(bgremove_dir)
        log.info(f"  BGRemove model: {bgremove_dir}")
    else:
        bgremove_path = discover_model_file(models_dir, os.path.join("other", "bgremove"),
                                            [".safetensors", ".pth", ".onnx"])
        if bgremove_path:
            bgremove_found = True
            from handlers.bgremove import set_model_path
            set_model_path(bgremove_path)
            log.info(f"  BGRemove model: {bgremove_path}")
    if bgremove_found:
        available_capabilities.add("bgremove")

    # Discover tagger model (in models/other/)
    tagger_found = False
    tagger_dir = os.path.join(models_dir, "other", "tagger")
    tagger_path = discover_model_file(models_dir, os.path.join("other", "tagger"), [".onnx"])
    if tagger_path:
        tagger_found = True
        from handlers.tagger import set_model_path
        set_model_path(os.path.dirname(tagger_path))
    elif os.path.isdir(tagger_dir):
        has_model = any(f.endswith(".safetensors") for f in os.listdir(tagger_dir))
        has_tags = os.path.isfile(os.path.join(tagger_dir, "tags.json"))
        if has_model and has_tags:
            tagger_found = True
            from handlers.tagger import set_model_path
            set_model_path(tagger_dir)
    if tagger_found:
        available_capabilities.add("tag")

    # Report missing models
    missing = {
        "sdxl":     "No SDXL checkpoints found. Place .safetensors files in models/sdxl/",
        "upscale":  "No upscale model found. Place a RealESRGAN model in models/other/upscale/",
        "bgremove": "No background removal model found. Place RMBG-2.0 in models/other/bgremove/",
        "tag":      "No tagger model found. Place JTP PILOT2 SigLIP files in models/other/tagger/",
    }
    for cap, msg in missing.items():
        if cap not in available_capabilities:
            log.warning(f"  {msg}")
            log.warning(f"    '{cap}' capability has been disabled.")

    if not available_capabilities:
        log.error("No models found — nothing to do. Install at least one model and try again.")
        sys.exit(1)

    # ── Initialize GPU pool ────────────────────────────────────────
    app_state.gpu_pool.initialize(config.gpus)

    if not app_state.gpu_pool.gpus:
        log.error("No GPUs matched config. Check [GPU-<uuid>] sections in foxburrow.ini.")
        sys.exit(1)

    # Strip unavailable capabilities from all GPUs
    for gpu_inst in app_state.gpu_pool.gpus:
        removed = gpu_inst.capabilities - available_capabilities
        if removed:
            gpu_inst.capabilities &= available_capabilities
            gpu_inst.onload -= removed
            gpu_inst.unevictable -= removed

    # Create pipeline factory
    app_state.pipeline_factory = PipelineFactory(app_state.registry)

    # Create scheduler and workers
    scheduler = GpuScheduler(app_state.queue)
    workers = []
    for gpu in app_state.gpu_pool.gpus:
        worker = GpuWorker(gpu, app_state.queue, scheduler._wake)
        workers.append(worker)
    scheduler.set_workers(workers)
    app_state.scheduler = scheduler

    # Create FastAPI app
    from api.server import create_app

    # Shared startup logic for both TUI and headless modes.
    # tui_mode: when True, skip SIGINT override (Textual handles Ctrl+C)
    #           and signal the ready_event so the main thread can launch the TUI.
    ready_event = threading.Event() if not args.no_tui else None

    async def on_startup():
        if args.no_tui:
            # Headless: override uvicorn's SIGINT handler — first Ctrl+C kills immediately.
            import signal
            def _force_exit(signum, frame):
                log.info("  SIGINT — exiting")
                os._exit(0)
            signal.signal(signal.SIGINT, _force_exit)
            signal.signal(signal.SIGTERM, _force_exit)

        # Store the event loop so background threads can fire WebSocket events
        from api.websocket import streamer
        streamer.set_loop(asyncio.get_running_loop())

        # Start workers and scheduler immediately so the server can accept jobs
        for w in workers:
            w.start()
        scheduler.start()
        log.info(f"  Scheduler started with {len(workers)} worker(s)")

        # Start filesystem watcher — auto-detect model additions/removals.
        # Runs async tasks in the event loop, zero CPU when idle (inotify).
        from utils.fs_watcher import FileSystemWatcher
        watcher = FileSystemWatcher(app_state)
        watcher.start()
        app_state.fs_watcher = watcher

        # Kick off ALL heavy initialization in a background thread:
        # HF config prefetch, model fingerprinting, LoRA hashing, GPU onload.
        # The server is already listening and /api/status reports scan progress.
        t = threading.Thread(
            target=_background_model_init,
            args=(config, models_dir, all_sdxl),
            name="bg-init", daemon=True,
        )
        t.start()

        # Signal TUI that the server is ready
        if ready_event is not None:
            ready_event.set()

    app = create_app(on_startup=on_startup)

    url = f"{config.server.address}:{config.server.port}"
    log.info(f"  Starting server on {url}")

    if args.no_tui:
        # ── Headless mode: uvicorn blocks on main thread (original behavior) ──
        uvicorn.run(
            app,
            host=config.server.address,
            port=config.server.port,
            log_level="warning",
        )
    else:
        # ── TUI mode: uvicorn in daemon thread, Textual on main thread ──
        uvi_config = uvicorn.Config(
            app,
            host=config.server.address,
            port=config.server.port,
            log_level="warning",
        )
        server = uvicorn.Server(uvi_config)

        def _run_uvicorn_in_thread():
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            loop.run_until_complete(server.serve())

        uvi_thread = threading.Thread(
            target=_run_uvicorn_in_thread,
            name="uvicorn",
            daemon=True,
        )
        uvi_thread.start()

        # Wait for the server to be ready (on_startup fires ready_event)
        if not ready_event.wait(timeout=30):
            log.error("  Server failed to start within 30s — aborting")
            server.should_exit = True
            sys.exit(1)

        # Launch TUI on main thread
        from tui.app import FoxburrowApp
        tui = FoxburrowApp(uvicorn_server=server)
        tui.run()

        # TUI exited — shut down uvicorn
        server.should_exit = True
        uvi_thread.join(timeout=5)
        if uvi_thread.is_alive():
            log.warning("  uvicorn did not stop cleanly within 5s")


if __name__ == "__main__":
    main()
