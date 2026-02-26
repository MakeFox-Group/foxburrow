"""Entry point: config, GPU init, model discovery, start server."""

from __future__ import annotations

import asyncio
import os
import secrets
import signal
import sys

# Force CUDA to enumerate GPUs by PCI bus ID (matching NVML ordering).
# Must be set BEFORE importing torch or any CUDA library.
os.environ.setdefault("CUDA_DEVICE_ORDER", "PCI_BUS_ID")

import torch
import uvicorn

import log
from config import FoxBurrowConfig
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
    """Discover SDXL checkpoints under models_dir/sdxl/.

    Supports both diffusers-format directories (with unet/, text_encoder/, etc.)
    and single-file .safetensors checkpoints.
    """
    sdxl_dir = os.path.join(models_dir, "sdxl")
    if not os.path.isdir(sdxl_dir):
        log.warning(f"  SDXL models directory not found: {sdxl_dir}")
        return {}

    models: dict[str, str] = {}
    for name in sorted(os.listdir(sdxl_dir)):
        path = os.path.join(sdxl_dir, name)
        if os.path.isdir(path):
            # Diffusers format: directory with required subdirectories
            required = ["unet", "text_encoder", "text_encoder_2", "vae"]
            if all(os.path.isdir(os.path.join(path, d)) for d in required):
                models[name] = path
                log.info(f"  Discovered SDXL model (diffusers): {name}")
            else:
                log.warning(f"  Skipping {name}: missing subdirectories "
                            f"(need {', '.join(required)})")
        elif name.endswith(".safetensors"):
            # Single-file SDXL checkpoint
            model_name = name.rsplit(".", 1)[0]
            models[model_name] = path
            log.info(f"  Discovered SDXL model (single-file): {model_name}")
    return models


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


def _process_gpu_onload(gpu, app_state) -> None:
    """Pre-load models specified in GPU's onload config."""
    from gpu.pool import GpuInstance

    for entry in gpu.onload:
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


def main() -> None:
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

    # ── Model discovery ─────────────────────────────────────────────
    # Scan for models BEFORE GPU init — no point initializing GPUs if
    # there's nothing to serve.
    available_capabilities: set[str] = set()

    # Discover SDXL models
    sdxl_models = discover_sdxl_models(models_dir)
    app_state.sdxl_models = sdxl_models

    if sdxl_models:
        available_capabilities.add("sdxl")

        # Set default model
        if config.server.default_sdxl_model and config.server.default_sdxl_model in sdxl_models:
            app_state.default_sdxl_model_dir = sdxl_models[config.server.default_sdxl_model]
        else:
            first_name = next(iter(sdxl_models))
            app_state.default_sdxl_model_dir = sdxl_models[first_name]
            if config.server.default_sdxl_model:
                log.warning(f"  Default model '{config.server.default_sdxl_model}' not found, "
                            f"using {first_name}")

        log.info(f"  Default SDXL model: {os.path.basename(app_state.default_sdxl_model_dir)}")

        # Initialize tokenizers from first model
        from handlers.sdxl import init_tokenizers
        init_tokenizers(app_state.default_sdxl_model_dir)

        # Register SDXL checkpoints
        for name, path in sdxl_models.items():
            app_state.registry.register_sdxl_checkpoint(path)

    # Discover LoRA files
    loras_dir = os.path.join(models_dir, "loras")
    app_state.loras_dir = loras_dir
    if os.path.isdir(loras_dir):
        from utils.lora_index import discover_loras, start_background_hashing
        app_state.lora_index = discover_loras(loras_dir)
        log.info(f"  Discovered {len(app_state.lora_index)} LoRA files")
        _lora_hash_thread = start_background_hashing(app_state.lora_index)
    else:
        log.info(f"  LoRA directory not found: {loras_dir} (skipping)")

    # Discover upscale model (in models/other/)
    upscale_path = discover_model_file(models_dir, os.path.join("other", "upscale"),
                                       [".pth", ".safetensors", ".onnx"])
    if upscale_path:
        available_capabilities.add("upscale")
        from handlers.upscale import set_model_path
        set_model_path(upscale_path)
        app_state.registry.register_upscale_model(upscale_path)
        log.info(f"  Upscale model: {upscale_path}")

    # Discover bgremove model (in models/other/)
    bgremove_found = False
    bgremove_dir = os.path.join(models_dir, "other", "bgremove")
    if os.path.isfile(os.path.join(bgremove_dir, "config.json")):
        bgremove_found = True
        from handlers.bgremove import set_model_path
        set_model_path(bgremove_dir)
        model_file = os.path.join(bgremove_dir, "model.safetensors")
        if os.path.isfile(model_file):
            app_state.registry.register_bgremove_model(model_file)
        log.info(f"  BGRemove model: {bgremove_dir}")
    else:
        bgremove_path = discover_model_file(models_dir, os.path.join("other", "bgremove"),
                                            [".safetensors", ".pth", ".onnx"])
        if bgremove_path:
            bgremove_found = True
            from handlers.bgremove import set_model_path
            set_model_path(bgremove_path)
            app_state.registry.register_bgremove_model(bgremove_path)
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
        # Check for safetensors + tags.json (JTP PILOT2 SigLIP format)
        has_model = any(f.endswith(".safetensors") for f in os.listdir(tagger_dir))
        has_tags = os.path.isfile(os.path.join(tagger_dir, "tags.json"))
        if has_model and has_tags:
            tagger_found = True
            from handlers.tagger import set_model_path
            set_model_path(tagger_dir)
    if tagger_found:
        available_capabilities.add("tag")

    # ── Report missing models ──────────────────────────────────────
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

    # Pre-load models specified in per-GPU onload config
    for gpu in app_state.gpu_pool.gpus:
        _process_gpu_onload(gpu, app_state)

    # Mark unevictable fingerprints for each GPU
    for gpu in app_state.gpu_pool.gpus:
        _process_gpu_unevictable(gpu, app_state)

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

    async def on_startup():
        for w in workers:
            w.start()
        scheduler.start()
        log.info(f"  Scheduler started with {len(workers)} worker(s)")

    app = create_app(on_startup=on_startup)

    url = f"{config.server.address}:{config.server.port}"
    log.info(f"  Starting server on {url}")

    uvicorn.run(
        app,
        host=config.server.address,
        port=config.server.port,
        log_level="warning",
    )


if __name__ == "__main__":
    main()
