"""GPU scheduler: dispatch loop with model affinity scoring."""

from __future__ import annotations

import asyncio
import concurrent.futures
from typing import TYPE_CHECKING

import log
from config import SchedulerConfig
from scheduling.job import (
    InferenceJob, JobResult, StageType, WorkStage,
)
from scheduling.queue import JobQueue, WorkGroup
from scheduling.readiness import _estimate_remaining_s

if TYPE_CHECKING:
    from scheduling.worker import GpuWorker


# Session key/group mappings
_SESSION_MAP: dict[StageType, tuple[str, str]] = {
    StageType.GPU_TEXT_ENCODE:     ("sdxl_te",          "sdxl"),
    StageType.GPU_DENOISE:        ("sdxl_unet",        "sdxl"),
    StageType.GPU_VAE_DECODE:     ("sdxl_vae",         "sdxl"),
    StageType.GPU_VAE_ENCODE:     ("sdxl_vae",         "sdxl"),
    StageType.GPU_UPSCALE:        ("upscale",          "upscale"),
    StageType.GPU_BGREMOVE:       ("bgremove",         "bgremove"),
}

_KEY_TO_GROUP: dict[str, str] = {v[0]: v[1] for v in _SESSION_MAP.values()}

# Lightweight groups that can coexist with any other group on the same GPU.
_COMPATIBLE_WITH_ALL: frozenset[str] = frozenset({"bgremove", "upscale"})


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

    def __init__(self, queue: JobQueue, config: SchedulerConfig | None = None):
        self._queue = queue
        self._config = config or SchedulerConfig()
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
        # Give each worker a reference to all workers for cross-GPU failure propagation
        for w in workers:
            w._all_workers = workers

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

        # Fail jobs that have OOM'd on every healthy GPU — they can never
        # be dispatched and would otherwise block the entire work group.
        # Exclude failed GPUs: a job shouldn't survive just because a dead
        # GPU inflates the set (it can never run on a failed GPU anyway).
        all_gpu_uuids = {w.gpu.uuid for w in self._workers if not w.gpu.is_failed}
        if not all_gpu_uuids:
            return  # no healthy GPUs — nothing to dispatch
        for group in list(gpu_groups):
            failed = []
            for job in group.jobs:
                if job.oom_gpu_ids >= all_gpu_uuids:
                    failed.append(job)
            if failed:
                for job in failed:
                    group.jobs.remove(job)
                    log.error(f"  GpuScheduler: Job[{job.job_id}] OOM on all "
                              f"{len(all_gpu_uuids)} GPUs — failing")
                    from datetime import datetime
                    job.completed_at = datetime.utcnow()
                    job.set_result(JobResult(
                        success=False,
                        error="Out of memory on all available GPUs"))
                self._queue.remove(failed)
            if not group.jobs:
                gpu_groups.remove(group)

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

            # Pick the first job not OOM'd on the chosen GPU (skip blocked ones)
            job = None
            for j in best_group.jobs:
                if best_worker.gpu.uuid not in j.oom_gpu_ids:
                    job = j
                    break
            if job is None:
                # Defensive: scoring should prevent this. If reached, it
                # means _score_worker failed to reject a fully-blocked group.
                log.warning(f"  GpuScheduler: All jobs in {best_group} OOM'd on "
                            f"{best_worker.gpu.uuid} — scoring check missed this")
                gpu_groups.remove(best_group)
                dispatched = True
                continue
            best_group.jobs.remove(job)
            self._queue.remove([job])

            log.debug(f"  GpuScheduler: Dispatching 1 job [{best_group.stage}] "
                      f"to GPU [{best_worker.gpu.uuid}] "
                      f"(score={best_score}, active={best_worker.active_count})")

            best_worker.dispatch(best_group.stage, job)
            dispatched = True

            if not best_group.jobs:
                gpu_groups.remove(best_group)

    def _score_worker(self, worker: "GpuWorker", group: WorkGroup) -> int:
        """Score how well a worker matches a work group. Higher is better."""
        if worker.gpu.is_failed:
            return -2**31

        stage = group.stage

        if not worker.can_accept_work(stage):
            return -2**31

        # VRAM budget gate: reject if this GPU can't fit the new stage's
        # model loading cost + working memory alongside active jobs.
        # Use the first non-OOM'd job for budget estimation (the one that
        # will actually be dispatched), not jobs[0] which may be blocked.
        if group.jobs:
            budget_job = next(
                (j for j in group.jobs if worker.gpu.uuid not in j.oom_gpu_ids),
                group.jobs[0],  # fallback: all blocked, OOM check below rejects
            )
            if not worker.check_vram_budget(stage, budget_job):
                return -2**31

        score = 0
        is_idle = worker.is_idle
        loaded_cats = worker.get_loaded_categories()
        required_group = get_session_group(stage.type)

        # ── Time-to-ready (dominant factor) ──────────────────────────
        # Instead of fixed affinity bonuses, score by *how soon* this GPU
        # can actually start the job: wait time + model load time.
        wait_s = _estimate_remaining_s(worker) if not is_idle else 0.0

        missing_vram = 0
        loaded_count = 0
        num_missing = 0
        if stage.required_components:
            for c in stage.required_components:
                if worker.gpu.is_component_loaded(c.fingerprint):
                    loaded_count += 1
                else:
                    missing_vram += c.estimated_vram_bytes
                    num_missing += 1

        # Model load time estimate: configurable MB/s + 0.5s per-component overhead
        # (eviction, empty_cache, tensor allocation)
        load_rate = self._config.load_rate_mb_s * 1024 * 1024
        load_s = (missing_vram / load_rate + 0.5 * num_missing) if num_missing > 0 else 0.0
        estimated_ready_s = wait_s + load_s

        score -= int(estimated_ready_s * 50)  # 50 pts/second penalty

        # Tiebreakers (only matter when estimated_ready_s values are close)
        if stage.required_components:
            if loaded_count == len(stage.required_components):
                score += 100       # prefer "ready now" over "ready in 0.01s"
            elif loaded_count > 0:
                score += 50        # partial cache saves some load time
        if is_idle:
            score += 20            # slight preference for idle GPUs at equal time

        # Cross-group penalty — only when busy. An idle GPU has zero
        # context-switch cost so switching session groups is free.
        # Skip penalty when either side is a lightweight compatible-with-all group.
        if required_group and not is_idle:
            for cat in loaded_cats:
                cat_group = get_session_group_from_key(cat)
                if cat_group and cat_group != required_group:
                    if required_group in _COMPATIBLE_WITH_ALL or cat_group in _COMPATIBLE_WITH_ALL:
                        continue
                    score -= 300

        # Batch size bonus
        score += 50 * len(group.jobs)

        # Starvation prevention — configurable 3-phase ramp:
        #   0..linear_s:  linear (normal scheduling priority)
        #   linear_s..hard_s: quadratic acceleration
        #   >=hard_s:     hard override (+50000, guarantees dispatch)
        age = group.oldest_age_seconds
        linear_s = self._config.starvation_linear_s
        hard_s = self._config.starvation_hard_s
        base_mult = 25 if is_idle else 10
        if age <= linear_s:
            starvation = int(base_mult * age)
        elif age <= hard_s:
            linear_base = base_mult * linear_s
            excess = age - linear_s
            starvation = int(linear_base + base_mult * excess + 0.5 * excess * excess)
        else:
            starvation = 50000
        score += starvation

        # OOM avoidance: hard-reject only if EVERY job in the group already
        # OOM'd on this GPU.  Previously checked only jobs[0], which caused
        # head-of-line blocking when one job OOM'd on all GPUs.
        if group.jobs:
            if all(worker.gpu.uuid in j.oom_gpu_ids for j in group.jobs):
                return -2**31

        return score

    def estimate_available_slots(self) -> dict[str, int]:
        """Estimate available job slots per capability based on real-time VRAM.

        For SDXL, estimates how many more UNet denoise stages (the bottleneck)
        can run across all GPUs.  For simple tasks, checks per-model capacity.
        Sums across all workers.
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
