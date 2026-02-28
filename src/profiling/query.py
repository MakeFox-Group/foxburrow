"""Query engine for profiling trace data."""

from __future__ import annotations

import json
import os
import statistics
from datetime import datetime
from typing import Any

# Reuse the traces directory constant from tracer
_TRACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiling", "traces",
)


def list_trace_files() -> list[dict[str, Any]]:
    """List available trace files with metadata (arch, size, line count)."""
    if not os.path.isdir(_TRACES_DIR):
        return []

    result = []
    for name in sorted(os.listdir(_TRACES_DIR)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(_TRACES_DIR, name)
        arch = name.removesuffix(".jsonl")
        size = os.path.getsize(path)
        line_count = 0
        with open(path, "r", encoding="utf-8") as f:
            for _ in f:
                line_count += 1
        result.append({
            "arch": arch,
            "file": name,
            "size_bytes": size,
            "event_count": line_count,
        })
    return result


def _iter_events(arch: str | None = None) -> list[dict[str, Any]]:
    """Read all events from trace files, optionally filtered by arch."""
    if not os.path.isdir(_TRACES_DIR):
        return []

    events: list[dict[str, Any]] = []
    files = []

    if arch:
        path = os.path.join(_TRACES_DIR, f"{arch}.jsonl")
        if os.path.isfile(path):
            files.append(path)
    else:
        for name in sorted(os.listdir(_TRACES_DIR)):
            if name.endswith(".jsonl"):
                files.append(os.path.join(_TRACES_DIR, name))

    for path in files:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

    return events


def search_events(
    arch: str | None = None,
    job_id: str | None = None,
    event_type: str | None = None,
    model: str | None = None,
    gpu_uuid: str | None = None,
    after: str | None = None,
    before: str | None = None,
    limit: int = 500,
    offset: int = 0,
) -> list[dict[str, Any]]:
    """Search trace events with filters. Returns matching events newest-first."""
    events = _iter_events(arch)

    # Apply filters
    filtered = []
    for ev in events:
        if job_id and ev.get("job_id") != job_id:
            continue
        if event_type and ev.get("type") != event_type:
            continue
        if model and model.lower() not in (ev.get("model") or "").lower():
            continue
        if gpu_uuid and ev.get("gpu_uuid") != gpu_uuid:
            continue
        if after:
            ts = ev.get("ts", "")
            if ts < after:
                continue
        if before:
            ts = ev.get("ts", "")
            if ts > before:
                continue
        filtered.append(ev)

    # Sort newest-first by timestamp
    filtered.sort(key=lambda e: e.get("ts", ""), reverse=True)

    # Paginate
    return filtered[offset:offset + limit]


def aggregate_stats(
    group_by: str = "model",
    event_type: str | None = None,
    model: str | None = None,
    after: str | None = None,
    before: str | None = None,
) -> list[dict[str, Any]]:
    """Aggregate duration_s stats grouped by field.

    Returns count, sum, avg, min, max, p50, p95 for each group.
    """
    events = _iter_events()

    # Filter
    filtered = []
    for ev in events:
        if event_type and ev.get("type") != event_type:
            continue
        if model and model.lower() not in (ev.get("model") or "").lower():
            continue
        if after and ev.get("ts", "") < after:
            continue
        if before and ev.get("ts", "") > before:
            continue
        dur = ev.get("duration_s")
        if dur is None:
            continue
        filtered.append(ev)

    # Group
    groups: dict[str, list[float]] = {}
    for ev in filtered:
        key = str(ev.get(group_by, "unknown"))
        groups.setdefault(key, []).append(ev["duration_s"])

    # Compute stats
    result = []
    for key, durations in sorted(groups.items()):
        durations.sort()
        n = len(durations)
        p50_idx = int(n * 0.50)
        p95_idx = min(int(n * 0.95), n - 1)
        result.append({
            "key": key,
            "count": n,
            "sum_s": round(sum(durations), 6),
            "avg_s": round(statistics.mean(durations), 6),
            "min_s": round(durations[0], 6),
            "max_s": round(durations[-1], 6),
            "p50_s": round(durations[p50_idx], 6),
            "p95_s": round(durations[p95_idx], 6),
        })

    return result


def job_timeline(job_id: str) -> list[dict[str, Any]]:
    """Get all events for a specific job across all GPUs, sorted by timestamp.

    Provides a complete picture of what happened during a job.
    """
    events = _iter_events()
    timeline = [ev for ev in events if ev.get("job_id") == job_id]
    timeline.sort(key=lambda e: e.get("ts", ""))
    return timeline
