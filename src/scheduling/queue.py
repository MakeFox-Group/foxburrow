"""Async job queue with work grouping for the scheduler."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import log

if TYPE_CHECKING:
    from gpu.pool import GpuPool
    from scheduling.job import InferenceJob, JobType, WorkStage, JobResult


@dataclass
class WorkGroup:
    """A group of jobs that all need the same work stage (same model components)."""
    stage: WorkStage
    jobs: list[InferenceJob] = field(default_factory=list)

    @property
    def oldest_age_seconds(self) -> float:
        if not self.jobs:
            return 0
        return (datetime.utcnow() - self.jobs[0].created_at).total_seconds()

    def __str__(self):
        return f"WorkGroup[{self.stage}] ({len(self.jobs)} jobs, oldest={self.oldest_age_seconds:.1f}s)"


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

    def get_work_groups(self) -> list[WorkGroup]:
        """Get all pending jobs grouped by their current stage's required components."""
        from scheduling.job import JobResult

        with self._lock:
            # Remove completed/cancelled jobs
            self._jobs = [j for j in self._jobs if not j.completion.done()]

            groups: dict[str, WorkGroup] = {}

            for job in self._jobs:
                stage = job.current_stage
                if stage is None:
                    continue

                # Build key from stage type + sorted component fingerprints
                if stage.is_cpu_only:
                    key = f"CPU:{stage.type.value}"
                else:
                    comp_keys = sorted(c.fingerprint for c in stage.required_components)
                    key = f"GPU:{stage.type.value}:{':'.join(comp_keys)}"

                if key not in groups:
                    groups[key] = WorkGroup(stage=stage)
                groups[key].jobs.append(job)

            # Sort jobs within each group by priority then creation time
            for group in groups.values():
                group.jobs.sort(key=lambda j: (j.priority, j.created_at))

            return list(groups.values())

    def remove(self, jobs: list[InferenceJob]) -> None:
        to_remove = set(id(j) for j in jobs)
        with self._lock:
            self._jobs = [j for j in self._jobs if id(j) not in to_remove]

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
    done, success or failure).  OOM retries and stage re-enqueues do NOT
    release the slot — the job is still in-flight.

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
            # Per-capability limit: number of non-failed GPUs with this capability
            cap_limit = self._gpu_pool.available_count(cap)
            current = self._counts.get(cap, 0)
            if current >= cap_limit:
                return f"No {cap} capacity: {current}/{cap_limit}"

            # Global limit: active_gpus + floor(active_gpus / 2)
            num_active = sum(1 for g in self._gpu_pool.gpus if not g.is_failed)
            global_limit = num_active + (num_active // 2)
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
        """Current global limit: active_gpus + floor(active_gpus / 2)."""
        num_active = sum(1 for g in self._gpu_pool.gpus if not g.is_failed)
        return num_active + (num_active // 2)

    def snapshot(self) -> dict:
        """Return current admission state for the status endpoint."""
        # Snapshot GPU pool state once to get a consistent view across all
        # derived values (limits, max_concurrent).  GpuPool.gpus is unsynchronized,
        # so we iterate once and derive everything from that single read.
        gpus = self._gpu_pool.gpus
        non_failed = [g for g in gpus if not g.is_failed]
        num_active = len(non_failed)
        cap_limits = {
            cap: sum(1 for g in non_failed if g.supports_capability(cap))
            for cap in sorted(set(self._CAPABILITY_MAP.values()))
        }
        max_conc = num_active + (num_active // 2)

        with self._lock:
            return {
                "counts": dict(self._counts),
                "total": self._total,
                "max_concurrent": max_conc,
                "limits": cap_limits,
            }
