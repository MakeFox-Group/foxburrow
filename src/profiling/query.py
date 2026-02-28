"""Query engine for profiling trace data.

Reads JSONL trace files produced by the tracer module. Files are append-only
and chronologically ordered. Readers may see a partial trailing line if a
write is in progress — such lines are silently skipped (at-most-once delivery
for in-flight events).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, Iterator

# Reuse the traces directory constant from tracer
_TRACES_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "profiling", "traces",
)

# Valid fields for group_by aggregation
_VALID_GROUP_BY = {"model", "type", "gpu_uuid", "gpu_arch", "stage",
                   "component", "encoder", "op", "direction"}

_CHUNK_SIZE = 65536


# ── File helpers ─────────────────────────────────────────────────────

def _get_trace_paths(arch: str | None = None) -> list[str]:
    """Get paths to trace files, optionally filtered by arch."""
    if not os.path.isdir(_TRACES_DIR):
        return []

    if arch:
        path = os.path.join(_TRACES_DIR, f"{arch}.jsonl")
        return [path] if os.path.isfile(path) else []

    return sorted(
        os.path.join(_TRACES_DIR, name)
        for name in os.listdir(_TRACES_DIR)
        if name.endswith(".jsonl")
    )


def _count_lines(path: str) -> int:
    """Count newlines in a file efficiently (binary scan, no JSON parsing)."""
    count = 0
    with open(path, "rb") as f:
        while True:
            chunk = f.read(_CHUNK_SIZE)
            if not chunk:
                break
            count += chunk.count(b"\n")
    return count


def _read_lines_reversed(path: str) -> Iterator[str]:
    """Yield non-empty lines from a file in reverse order, reading in chunks from EOF."""
    with open(path, "rb") as f:
        f.seek(0, 2)
        pos = f.tell()
        tail = b""

        while pos > 0:
            to_read = min(_CHUNK_SIZE, pos)
            pos -= to_read
            f.seek(pos)
            chunk = f.read(to_read) + tail
            lines = chunk.split(b"\n")
            tail = lines[0]  # may be partial line spanning previous chunk
            for line in reversed(lines[1:]):
                stripped = line.strip()
                if stripped:
                    yield stripped.decode("utf-8")

        if tail.strip():
            yield tail.strip().decode("utf-8")


def _iter_events_forward(paths: list[str]) -> Iterator[dict[str, Any]]:
    """Yield parsed events from files in forward (chronological) order."""
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    continue


def _matches_filters(
    ev: dict[str, Any],
    job_id: str | None,
    event_type: str | None,
    model: str | None,
    gpu_uuid: str | None,
    after: str | None,
    before: str | None,
) -> bool:
    """Check if an event matches all the given filters."""
    if job_id and ev.get("job_id") != job_id:
        return False
    if event_type and ev.get("type") != event_type:
        return False
    if model and model.lower() not in (ev.get("model") or "").lower():
        return False
    if gpu_uuid and ev.get("gpu_uuid") != gpu_uuid:
        return False
    ts = ev.get("ts", "")
    if after and ts < after:
        return False
    if before and ts > before:
        return False
    return True


# ── Public API ───────────────────────────────────────────────────────

def list_trace_files() -> list[dict[str, Any]]:
    """List available trace files with metadata (arch, size, event count)."""
    if not os.path.isdir(_TRACES_DIR):
        return []

    result = []
    for name in sorted(os.listdir(_TRACES_DIR)):
        if not name.endswith(".jsonl"):
            continue
        path = os.path.join(_TRACES_DIR, name)
        result.append({
            "arch": name.removesuffix(".jsonl"),
            "file": name,
            "size_bytes": os.path.getsize(path),
            "event_count": _count_lines(path),
        })
    return result


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
    """Search trace events with filters. Returns matching events newest-first.

    Uses reverse file reading for efficient recent-event queries. Stops early
    once enough matching events are collected.
    """
    paths = _get_trace_paths(arch)
    if not paths:
        return []

    needed = offset + limit
    filtered: list[dict[str, Any]] = []

    for path in paths:
        for line in _read_lines_reversed(path):
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue

            # When reading reversed and we have an 'after' filter,
            # stop reading this file once we pass the threshold
            if after and ev.get("ts", "") < after:
                break

            if _matches_filters(ev, job_id, event_type, model, gpu_uuid, after, before):
                filtered.append(ev)

        # For single-file queries, we can stop if we have enough
        if len(paths) == 1 and len(filtered) >= needed:
            break

    # For multi-file, sort globally by timestamp (each file is internally ordered)
    if len(paths) > 1:
        filtered.sort(key=lambda e: e.get("ts", ""), reverse=True)

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
    Raises ValueError if group_by is not a valid field name.
    """
    if group_by not in _VALID_GROUP_BY:
        raise ValueError(
            f"Invalid group_by={group_by!r}. "
            f"Valid values: {', '.join(sorted(_VALID_GROUP_BY))}")

    paths = _get_trace_paths()

    # Stream through events, grouping durations
    groups: dict[str, list[float]] = {}
    for ev in _iter_events_forward(paths):
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

        key = str(ev.get(group_by, "unknown"))
        groups.setdefault(key, []).append(dur)

    # Compute stats with correct percentile calculation
    result = []
    for key, durations in sorted(groups.items()):
        durations.sort()
        n = len(durations)
        p50_idx = max(0, math.ceil(n * 0.50) - 1)
        p95_idx = max(0, min(math.ceil(n * 0.95) - 1, n - 1))
        result.append({
            "key": key,
            "count": n,
            "sum_s": round(sum(durations), 6),
            "avg_s": round(sum(durations) / n, 6),
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
    paths = _get_trace_paths()
    timeline = [
        ev for ev in _iter_events_forward(paths)
        if ev.get("job_id") == job_id
    ]
    timeline.sort(key=lambda e: e.get("ts", ""))
    return timeline
