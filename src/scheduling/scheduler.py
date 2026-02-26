"""GPU scheduler: dispatch loop with model affinity scoring."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

import log
from scheduling.job import (
    InferenceJob, JobResult, StageType, WorkStage,
)
from scheduling.queue import JobQueue, WorkGroup

if TYPE_CHECKING:
    from scheduling.worker import GpuWorker


# Session key/group mappings
_SESSION_MAP: dict[StageType, tuple[str, str]] = {
    StageType.GPU_TEXT_ENCODE:     ("sdxl_te",          "sdxl"),
    StageType.GPU_DENOISE:        ("sdxl_unet",        "sdxl"),
    StageType.GPU_VAE_DECODE:     ("sdxl_vae",         "sdxl"),
    StageType.GPU_VAE_ENCODE:     ("sdxl_vae",         "sdxl"),
    StageType.GPU_HIRES_TRANSFORM:("sdxl_hires_xform", "sdxl"),
    StageType.GPU_UPSCALE:        ("upscale",          "upscale"),
    StageType.GPU_BGREMOVE:       ("bgremove",         "bgremove"),
}

_KEY_TO_GROUP: dict[str, str] = {v[0]: v[1] for v in _SESSION_MAP.values()}


def get_session_key(stage_type: StageType) -> str | None:
    entry = _SESSION_MAP.get(stage_type)
    return entry[0] if entry else None


def get_session_group(stage_type: StageType) -> str | None:
    entry = _SESSION_MAP.get(stage_type)
    return entry[1] if entry else None


def get_session_group_from_key(key: str) -> str | None:
    return _KEY_TO_GROUP.get(key)


class GpuScheduler:
    """Central scheduling loop. Dispatches CPU stages to thread pool,
    GPU stages to the best-scoring worker."""

    def __init__(self, queue: JobQueue):
        self._queue = queue
        self._workers: list[GpuWorker] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._cpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)

        # Wire queue to wake us
        self._queue.scheduler_wake = self._wake

    @property
    def workers(self) -> list[GpuWorker]:
        return self._workers

    def set_workers(self, workers: list[GpuWorker]) -> None:
        self._workers = workers

    def start(self) -> None:
        self._task = asyncio.get_running_loop().create_task(self._run_loop())

    async def _run_loop(self) -> None:
        log.info("  GpuScheduler: Started")
        try:
            while True:
                self._wake.clear()
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=1.0)
                except asyncio.TimeoutError:
                    pass

                try:
                    self._run_scheduling_round()
                except Exception as ex:
                    log.log_exception(ex, "GpuScheduler: Error in scheduling round")
        except asyncio.CancelledError:
            log.info("  GpuScheduler: Stopped")

    def _run_scheduling_round(self) -> None:
        groups = self._queue.get_work_groups()
        if not groups:
            return

        cpu_groups = [g for g in groups if g.stage.is_cpu_only]
        gpu_groups = [g for g in groups if not g.stage.is_cpu_only]

        # Dispatch CPU stages to thread pool
        for group in cpu_groups:
            jobs = list(group.jobs)
            self._queue.remove(jobs)
            for job in jobs:
                self._cpu_pool.submit(self._execute_cpu_stage, job)

        # Dispatch GPU stages using scoring — one job at a time
        dispatched = True
        while dispatched and gpu_groups:
            dispatched = False
            best_score = -2**31
            best_worker: GpuWorker | None = None
            best_group: WorkGroup | None = None

            for worker in self._workers:
                for group in gpu_groups:
                    if not group.jobs:
                        continue
                    score = self._score_worker(worker, group)
                    if score > best_score:
                        best_score = score
                        best_worker = worker
                        best_group = group

            if best_worker is None or best_group is None or best_score == -2**31:
                break

            job = best_group.jobs.pop(0)
            self._queue.remove([job])

            log.info(f"  GpuScheduler: Dispatching 1 job [{best_group.stage}] "
                     f"to GPU [{best_worker.gpu.uuid}] "
                     f"(score={best_score}, active={best_worker.active_count})")

            best_worker.dispatch(best_group.stage, job)
            dispatched = True

            if not best_group.jobs:
                gpu_groups.remove(best_group)

    def _score_worker(self, worker: "GpuWorker", group: WorkGroup) -> int:
        """Score how well a worker matches a work group. Higher is better."""
        stage = group.stage

        if not worker.can_accept_work(stage):
            return -2**31

        score = 0
        loaded_cats = worker.get_loaded_categories()
        required_group = get_session_group(stage.type)

        # Model affinity
        if stage.required_components:
            loaded_count = sum(
                1 for c in stage.required_components
                if worker.gpu.is_component_loaded(c.fingerprint)
            )
            if loaded_count == len(stage.required_components):
                score += 1000
            elif loaded_count > 0:
                score += 300
            else:
                score -= 500

        # Cross-group penalty
        if required_group:
            for cat in loaded_cats:
                cat_group = get_session_group_from_key(cat)
                if cat_group and cat_group != required_group:
                    score -= 300

        # Batch size bonus
        score += 50 * len(group.jobs)

        # Starvation prevention
        score += int(10 * group.oldest_age_seconds)

        # Prefer idle workers
        if worker.is_idle:
            score += 100

        return score

    def _execute_cpu_stage(self, job: InferenceJob) -> None:
        """Execute a CPU-only stage on the thread pool."""
        try:
            stage = job.current_stage
            if stage is None:
                return

            if job.started_at is None:
                from datetime import datetime
                job.started_at = datetime.utcnow()

            if stage.type == StageType.CPU_TOKENIZE:
                from handlers.sdxl import tokenize
                tokenize(job)
            else:
                raise RuntimeError(f"Not a CPU stage: {stage.type}")

            job.current_stage_index += 1

            if job.is_complete:
                from datetime import datetime
                job.completed_at = datetime.utcnow()
                job.set_result(JobResult(success=True))
            else:
                self._queue.re_enqueue(job)

        except Exception as ex:
            log.log_exception(ex, f"CPU stage failed for {job}")
            from datetime import datetime
            job.completed_at = datetime.utcnow()
            job.set_result(JobResult(success=False, error=str(ex)))
