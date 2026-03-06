"""Global application state — import from here, not from main."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gpu.pool import GpuPool
from scheduling.model_registry import ModelRegistry
from scheduling.queue import AdmissionControl, JobQueue

if TYPE_CHECKING:
    from scheduling.job import InferenceJob
    from utils.fs_watcher import FileSystemWatcher
    from utils.model_scanner import ModelScanner


class AppState:
    """Global application state — accessible from routes via `from state import app_state`."""

    def __init__(self):
        self.config = None
        self.gpu_pool: GpuPool = GpuPool()
        self.registry: ModelRegistry = ModelRegistry()
        self.pipeline_factory = None
        self.queue: JobQueue = JobQueue()
        self.admission: AdmissionControl | None = None
        self.scheduler = None
        self.sdxl_models: dict[str, str] = {}
        self.model_scanner: ModelScanner | None = None
        self.lora_index: dict = {}  # name -> LoraEntry (from utils.lora_index)
        self.loras_dir: str | None = None  # path to loras directory for rescans
        self.fs_watcher: FileSystemWatcher | None = None
        self.trt_manager = None  # TrtBuildManager (set in main.py on_startup)

        # Job registry for queue-based API
        self.jobs: dict[str, "InferenceJob"] = {}          # job_id → job
        # Atomic result storage: job_id → (bytes, media_type)
        self.job_results: dict[str, tuple[bytes, str]] = {}


app_state = AppState()


def propagate_lora_index() -> None:
    """Push the current LoRA index to all GPU worker processes."""
    scheduler = app_state.scheduler
    if scheduler and scheduler.workers:
        snapshot = dict(app_state.lora_index)
        for w in scheduler.workers:
            w.update_lora_index(snapshot)


def propagate_sdxl_models() -> None:
    """Push the current SDXL model map to all GPU worker processes."""
    scheduler = app_state.scheduler
    if scheduler and scheduler.workers:
        snapshot = dict(app_state.sdxl_models)
        for w in scheduler.workers:
            w.update_sdxl_models(snapshot)
