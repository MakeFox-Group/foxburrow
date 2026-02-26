"""Async job queue with work grouping for the scheduler."""

from __future__ import annotations

import asyncio
import threading
from dataclasses import dataclass, field
from datetime import datetime
from typing import TYPE_CHECKING

import log

if TYPE_CHECKING:
    from scheduling.job import InferenceJob, WorkStage, JobResult


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
        log.info(f"  Queue: Enqueued {job} (queue depth: {self.count})")
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
