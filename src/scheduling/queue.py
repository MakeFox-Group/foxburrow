"""Async job queue for the scheduler."""

from __future__ import annotations

import asyncio
import threading
from typing import TYPE_CHECKING

import log

if TYPE_CHECKING:
    from gpu.pool import GpuPool
    from scheduling.job import InferenceJob, JobType


class JobQueue:
    """Thread-safe priority queue for inference jobs.

    Jobs are ordered by priority (lower = higher priority), then by creation time (FIFO).
    Signals the scheduler when new jobs arrive.
    """

    def __init__(self):
        self._jobs: list[InferenceJob] = []
        self._lock = threading.Lock()
        self.scheduler_wake: asyncio.Event | None = None

    @property
    def count(self) -> int:
        with self._lock:
            return len(self._jobs)

    def enqueue(self, job: InferenceJob) -> None:
        with self._lock:
            self._jobs.append(job)
        log.debug(f"  Queue: Enqueued {job} (queue depth: {self.count})")
        self._wake_scheduler()

    def get_ready_jobs(self) -> list[InferenceJob]:
        """Get all pending jobs that have completed CPU preprocessing.

        Returns jobs sorted by priority then creation time.
        Removes completed/cancelled jobs as a side effect.
        """
        with self._lock:
            # Remove completed/cancelled jobs
            self._jobs = [j for j in self._jobs if not j.completion.done()]

            ready = [j for j in self._jobs if j._cpu_ready]
            ready.sort(key=lambda j: (j.priority, j.created_at))
            return ready

    def get_pending_cpu_jobs(self) -> list[InferenceJob]:
        """Get jobs that need CPU preprocessing (not yet _cpu_ready).

        Returns a snapshot list; jobs are NOT removed from the queue.
        """
        with self._lock:
            return [j for j in self._jobs
                    if not j._cpu_ready and not j.completion.done()]

    def remove(self, job: InferenceJob) -> None:
        with self._lock:
            self._jobs = [j for j in self._jobs if j is not job]

    def re_enqueue(self, job: InferenceJob) -> None:
        with self._lock:
            self._jobs.append(job)
        self._wake_scheduler()

    def snapshot(self) -> list[InferenceJob]:
        with self._lock:
            return list(self._jobs)

    def _wake_scheduler(self) -> None:
        if self.scheduler_wake:
            self.scheduler_wake.set()


class AdmissionControl:
    """Gate job submission by per-capability and global in-flight limits.

    A job holds its admission slot from enqueue until completion (all stages
    done, success or failure).  OOM retries and re-enqueues do NOT release
    the slot — the job is still in-flight.

    Limits are dynamic: per-capability limits query GpuPool.available_count()
    live (excludes failed GPUs), so limits shrink automatically if a GPU fails.
    """

    # Map each JobType to its capability bucket.
    # TAG is exempt (bypasses scheduler, runs immediately).
    _CAPABILITY_MAP: dict[str, str] = {
        "SdxlGenerate":        "sdxl",
        "SdxlGenerateLatents": "sdxl",
        "SdxlDecodeLatents":   "sdxl",
        "SdxlEncodeLatents":   "sdxl",
        "SdxlGenerateHires":   "sdxl",
        "SdxlHiresLatents":    "sdxl",
        "Enhance":             "sdxl",
        "Upscale":             "upscale",
        "BGRemove":            "bgremove",
    }

    # Pipeline depth = 1: one job at a time per GPU (no stage pipelining).
    _PIPELINE_DEPTH: dict[str, int] = {
        "sdxl": 1,
        "upscale": 1,
        "bgremove": 1,
    }

    def __init__(self, gpu_pool: GpuPool) -> None:
        self._gpu_pool = gpu_pool
        self._counts: dict[str, int] = {}  # capability -> in-flight count
        self._total: int = 0               # global in-flight GPU jobs
        self._lock = threading.Lock()

    def try_admit(self, job_type: JobType) -> str | None:
        """Try to admit a job.  Returns an error message on rejection, or None on success."""
        cap = self._CAPABILITY_MAP.get(job_type.value)
        if cap is None:
            # Unmapped job types (e.g. TAG) are exempt
            return None

        with self._lock:
            # Per-capability limit: GPUs × pipeline depth (1 per GPU).
            depth = self._PIPELINE_DEPTH.get(cap, 1)
            cap_limit = self._gpu_pool.available_count(cap) * depth
            current = self._counts.get(cap, 0)
            if current >= cap_limit:
                return f"No {cap} capacity: {current}/{cap_limit}"

            # Global limit: idle GPUs + 2 buffer slots for queuing
            num_idle = sum(1 for g in self._gpu_pool.gpus
                           if not g.is_busy and not g.is_failed
                           and not g._trt_building)
            global_limit = num_idle + 2
            if self._total >= global_limit:
                return f"Server at capacity: {self._total}/{global_limit}"

            # Admitted — increment counts
            self._counts[cap] = current + 1
            self._total += 1
            return None

    def release(self, job_type: JobType) -> None:
        """Release the admission slot for a completed job."""
        cap = self._CAPABILITY_MAP.get(job_type.value)
        if cap is None:
            return

        with self._lock:
            current = self._counts.get(cap, 0)
            if current > 0:
                self._counts[cap] = current - 1
            else:
                log.warning(f"AdmissionControl: release({cap}) but count already 0")
            if self._total > 0:
                self._total -= 1
            else:
                log.warning(f"AdmissionControl: release total but count already 0")

    @property
    def max_concurrent(self) -> int:
        """Current global limit: idle GPUs + 2 buffer slots."""
        num_idle = sum(1 for g in self._gpu_pool.gpus
                       if not g.is_busy and not g.is_failed
                       and not g._trt_building)
        return num_idle + 2

    def snapshot(self) -> dict:
        """Return current admission state for the status endpoint."""
        gpus = self._gpu_pool.gpus
        non_failed = [g for g in gpus if not g.is_failed and not g._trt_building]
        num_idle = sum(1 for g in non_failed if not g.is_busy)
        cap_limits = {
            cap: sum(1 for g in non_failed if g.supports_capability(cap))
                 * self._PIPELINE_DEPTH.get(cap, 1)
            for cap in sorted(set(self._CAPABILITY_MAP.values()))
        }
        max_conc = num_idle + 2

        with self._lock:
            return {
                "counts": dict(self._counts),
                "total": self._total,
                "max_concurrent": max_conc,
                "limits": cap_limits,
            }
