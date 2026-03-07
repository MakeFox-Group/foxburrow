"""GPU scheduler: dispatch loop with model affinity scoring.

Each GPU runs one job at a time (entire pipeline start-to-finish).
No concurrent stages, no cross-GPU stage dispatch.
"""

from __future__ import annotations

import asyncio
import concurrent.futures
import time as _time
from datetime import datetime
from typing import TYPE_CHECKING

import log
from config import SchedulerConfig
from scheduling.job import (
    InferenceJob, JobResult, StageType,
)
from scheduling.pipeline import get_all_components
if TYPE_CHECKING:
    from scheduling.worker_proxy import GpuWorkerProxy as GpuWorker


class GpuScheduler:
    """Central scheduling loop. Dispatches CPU stages to thread pool,
    GPU jobs to the best-scoring worker (entire pipeline per GPU)."""

    def __init__(self, queue, config: SchedulerConfig | None = None):
        self._queue = queue
        self._config = config or SchedulerConfig()
        self._workers: list[GpuWorker] = []
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None
        self._cpu_pool = concurrent.futures.ThreadPoolExecutor(max_workers=4)
        self._cpu_submitted: set[str] = set()  # job_ids submitted to CPU pool

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
            log.debug("  GpuScheduler: Stopped")

    def _run_scheduling_round(self) -> None:
        # Kick off CPU preprocessing for any new jobs
        self._process_pending_cpu_jobs()

        jobs = self._queue.get_ready_jobs()
        if not jobs:
            return

        # Fail jobs that have OOM'd on every healthy GPU
        all_gpu_uuids = {w.gpu.uuid for w in self._workers if not w.gpu.is_failed}
        if not all_gpu_uuids:
            return
        for job in list(jobs):
            if job.oom_gpu_ids >= all_gpu_uuids:
                self._queue.remove(job)
                jobs.remove(job)
                log.error(f"  GpuScheduler: Job[{job.job_id}] OOM on all "
                          f"{len(all_gpu_uuids)} GPUs — failing")
                job.completed_at = datetime.utcnow()
                # Release admission slot before setting result
                from state import app_state
                if app_state.admission is not None:
                    app_state.admission.release(job.type)
                job.set_result(JobResult(
                    success=False,
                    error="Out of memory on all available GPUs"))

        if not jobs:
            return

        # Greedy dispatch: find best (worker, job) match and dispatch,
        # repeating until no more valid matches.
        dispatched = True
        while dispatched and jobs:
            dispatched = False
            best_score = -2**31
            best_worker: GpuWorker | None = None
            best_job: InferenceJob | None = None

            for job in jobs:
                for worker in self._workers:
                    score = self._score_worker(worker, job)
                    if score > best_score:
                        best_score = score
                        best_worker = worker
                        best_job = job

            if best_worker is None or best_job is None or best_score == -2**31:
                break

            self._queue.remove(best_job)
            jobs.remove(best_job)

            log.debug(f"  GpuScheduler: Dispatching {best_job} "
                      f"to GPU [{best_worker.gpu.uuid}] "
                      f"(score={best_score})")

            best_worker.dispatch(best_job)
            dispatched = True

    def _score_worker(self, worker: "GpuWorker", job: InferenceJob) -> int:
        """Score how well a worker matches a job. Higher is better."""
        if worker.gpu.is_failed:
            return -2**31

        if not worker.can_accept_work():
            return -2**31

        # OOM avoidance
        if worker.gpu.uuid in job.oom_gpu_ids:
            return -2**31

        # Capability check — all GPU stages must be supported
        for stage in job.pipeline:
            if stage.required_capability and not worker.gpu.supports_capability(stage.required_capability):
                return -2**31

        # VRAM budget gate
        if not worker.check_vram_budget(job):
            return -2**31

        from scheduling.worker_proxy import _global_measured_vram

        score = 0

        # ── Model affinity (DOMINANT) ─────────────────────────────
        all_components = get_all_components(job.pipeline)
        component_weights = {
            "sdxl_unet": 500,
            "sdxl_te2": 150,
            "sdxl_te1": 50,
            "sdxl_vae": 30,
            "sdxl_vae_enc": 30,
            "upscale": 100,
            "bgremove": 100,
        }

        loaded_count = 0
        missing_vram = 0
        num_missing = 0
        trt_covered_fps: set[str] = set()

        # ── TRT coverage detection ────────────────────────────────
        trt_affinity_score = 0
        inp = job.sdxl_input
        if inp is not None:
            w, h = inp.width, inp.height
            # Check hires resolution for hires jobs
            if job.hires_input:
                hi = job.hires_input
                if hi.hires_width > 0 and hi.hires_height > 0:
                    w, h = max(w, hi.hires_width), max(h, hi.hires_height)

            for c in all_components:
                if c.category == "sdxl_unet":
                    aff = worker.gpu.check_trt_affinity(c.fingerprint, "unet", w, h)
                    if aff >= 1:
                        trt_covered_fps.add(c.fingerprint)
                    if aff == 2:
                        trt_affinity_score += 200
                    elif aff == 1:
                        trt_affinity_score += 150
                    elif aff == -1:
                        trt_affinity_score -= 150
                elif c.category in ("sdxl_vae", "sdxl_vae_enc"):
                    # Check both VAE decode and encode TRT
                    for trt_comp in ("vae", "vae_enc"):
                        aff = worker.gpu.check_trt_affinity(c.fingerprint, trt_comp, w, h)
                        if aff >= 1:
                            trt_covered_fps.add(c.fingerprint)
                            trt_affinity_score += 200 if aff == 2 else 150
                            break
                        elif aff == -1:
                            trt_affinity_score -= 150
                            break
                elif c.category == "sdxl_te1":
                    te_fp = f"{c.fingerprint}:te1_trt:default"
                    if worker.gpu.is_component_loaded(te_fp):
                        trt_covered_fps.add(c.fingerprint)
                        trt_affinity_score += 100
                elif c.category == "sdxl_te2":
                    te_fp = f"{c.fingerprint}:te2_trt:default"
                    if worker.gpu.is_component_loaded(te_fp):
                        trt_covered_fps.add(c.fingerprint)
                        trt_affinity_score += 100

        score += trt_affinity_score

        # ── Component affinity scoring ────────────────────────────
        for c in all_components:
            if worker.gpu.is_component_loaded(c.fingerprint):
                loaded_count += 1
                score += component_weights.get(c.category, 30)
            elif c.fingerprint in trt_covered_fps:
                loaded_count += 1
            else:
                measured = _global_measured_vram.get(c.fingerprint)
                missing_vram += measured if measured is not None else c.estimated_vram_bytes
                num_missing += 1

        # Full pipeline loaded bonus
        if loaded_count == len(all_components) and all_components:
            score += 300

        # ── Time-to-ready ─────────────────────────────────────────
        # GPU is always idle when dispatching (one job at a time, checked by can_accept_work)
        load_rate = self._config.load_rate_mb_s * 1024 * 1024
        load_s = (missing_vram / load_rate + 0.5 * num_missing) if num_missing > 0 else 0.0
        score -= int(load_s * 50)

        # Tiebreakers
        if all_components:
            if loaded_count == len(all_components):
                score += 100
            elif loaded_count > 0:
                score += 50

        score += 20  # idle bonus (GPU is always idle when scoring)

        # Diversification: prefer GPUs that haven't been dispatched recently
        idle_since = _time.monotonic() - worker._last_dispatch_time
        score += min(int(idle_since * 2), 30)

        # ── Starvation prevention ─────────────────────────────────
        # 3-phase ramp: linear → quadratic → hard override
        age = (datetime.utcnow() - job.created_at).total_seconds()
        linear_s = self._config.starvation_linear_s
        hard_s = self._config.starvation_hard_s
        base_mult = 25
        if age <= linear_s:
            starvation = int(base_mult * age)
        elif age <= hard_s:
            linear_base = base_mult * linear_s
            excess = age - linear_s
            starvation = int(linear_base + base_mult * excess + 0.5 * excess * excess)
        else:
            starvation = 50000
        score += starvation

        return score

    def _process_pending_cpu_jobs(self) -> None:
        """Submit CPU preprocessing for jobs that need it."""
        pending = self._queue.get_pending_cpu_jobs()
        for job in pending:
            if job.job_id in self._cpu_submitted:
                continue
            stage = job.current_stage
            if stage is not None and stage.is_cpu_only:
                self._cpu_submitted.add(job.job_id)
                self._cpu_pool.submit(self._execute_cpu_stage, job)
            elif stage is not None and not stage.is_cpu_only:
                # No CPU stage needed — mark as ready immediately
                job._cpu_ready = True

    def estimate_available_slots(self) -> dict[str, int]:
        """Estimate available job slots per capability.

        With one-job-per-GPU, each idle GPU = 1 slot for each of its capabilities.
        """
        from scheduling.worker import estimate_gpu_slots

        totals: dict[str, int] = {}
        for w in self._workers:
            for cap, count in estimate_gpu_slots(w).items():
                totals[cap] = totals.get(cap, 0) + count
        return totals

    def _execute_cpu_stage(self, job: InferenceJob) -> None:
        """Execute a CPU-only stage on the thread pool."""
        try:
            stage = job.current_stage
            if stage is None:
                return

            if job.started_at is None:
                job.started_at = datetime.utcnow()

            if stage.type == StageType.CPU_TOKENIZE:
                from handlers.sdxl import tokenize
                tokenize(job)
            else:
                raise RuntimeError(f"Not a CPU stage: {stage.type}")

            job.current_stage_index += 1
            job._cpu_ready = True

            # Wake scheduler so it picks up the now-ready job
            self._wake.set()

        except Exception as ex:
            log.log_exception(ex, f"CPU stage failed for {job}")
            job.completed_at = datetime.utcnow()
            job.set_result(JobResult(success=False, error=str(ex)))
        finally:
            self._cpu_submitted.discard(job.job_id)
