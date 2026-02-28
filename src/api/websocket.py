"""WebSocket streaming for denoise progress, step previews, and completion."""

from __future__ import annotations

import asyncio
import json
from typing import TYPE_CHECKING

from fastapi import WebSocket, WebSocketDisconnect

import log

if TYPE_CHECKING:
    from scheduling.job import InferenceJob


class ProgressStreamer:
    """Streams job progress and server events to connected WebSocket clients."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._push_task: asyncio.Task | None = None

    def set_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Store the asyncio event loop for thread-safe event dispatch."""
        self._loop = loop

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        log.debug(f"  WebSocket: Client connected ({len(self._connections)} total)")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        log.debug(f"  WebSocket: Client disconnected ({len(self._connections)} total)")

    async def broadcast_progress(self, job: "InferenceJob") -> None:
        """Broadcast denoise progress for a job."""
        if not self._connections:
            return

        stage = job.current_stage
        stage_type = stage.type.value if stage else None

        # Get total GPU count from pool
        gpu_count = 0
        try:
            from state import app_state
            gpu_count = len(app_state.gpu_pool.gpus)
        except Exception:
            pass

        # Calculate overall pipeline progress (0.0 - 1.0)
        stage_count = len(job.pipeline)
        if stage_count > 0:
            # Each stage contributes equally to overall progress
            base_progress = job.current_stage_index / stage_count
            # Within the current stage, denoise steps add fractional progress
            if job.denoise_total_steps > 0:
                stage_frac = job.denoise_step / job.denoise_total_steps
            else:
                stage_frac = 0.5  # Non-denoise stages: assume halfway
            overall_progress = base_progress + (stage_frac / stage_count)
        else:
            overall_progress = 0.0

        msg = json.dumps({
            "type": "progress",
            "job_id": job.job_id,
            "step": job.denoise_step,
            "total": job.denoise_total_steps,
            "progress": round(overall_progress, 4),
            "stage": stage_type,
            "stage_index": job.current_stage_index,
            "stage_count": stage_count,
            "gpu_count": gpu_count,
            "gpus": job.active_gpus,
        })

        # Snapshot to avoid mutation during iteration
        async with self._lock:
            snapshot = list(self._connections)

        dead: list[WebSocket] = []
        for ws in snapshot:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    async def broadcast_complete(self, job: "InferenceJob", success: bool,
                                  error: str | None = None) -> None:
        """Broadcast job completion."""
        if not self._connections:
            return

        msg = json.dumps({
            "type": "complete",
            "job_id": job.job_id,
            "success": success,
            "error": error,
            "gpu_time_s": round(job.gpu_time_s, 3),
            "gpu_stages": job.gpu_stage_times,
        })

        async with self._lock:
            snapshot = list(self._connections)

        dead: list[WebSocket] = []
        for ws in snapshot:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    async def broadcast_event(self, event_type: str, data: dict) -> None:
        """Broadcast a typed event to all connected WebSocket clients."""
        if not self._connections:
            return

        try:
            msg = json.dumps({"type": event_type, **data})
        except (TypeError, ValueError) as exc:
            log.error(f"  WebSocket: broadcast_event({event_type!r}) "
                      f"serialization failed: {exc}")
            return

        async with self._lock:
            snapshot = list(self._connections)

        dead: list[WebSocket] = []
        for ws in snapshot:
            try:
                await ws.send_text(msg)
            except Exception:
                dead.append(ws)

        if dead:
            async with self._lock:
                for ws in dead:
                    if ws in self._connections:
                        self._connections.remove(ws)

    def start_status_push(self) -> None:
        """Start background task that pushes status snapshots to connected clients."""
        loop = self._loop
        if loop is None:
            return
        self._push_task = loop.create_task(self._status_push_loop())

    def stop_status_push(self) -> None:
        """Cancel the status push background task."""
        if self._push_task is not None:
            self._push_task.cancel()
            self._push_task = None

    async def _status_push_loop(self) -> None:
        """Push status_update periodically when clients are connected."""
        from api.status_snapshot import compute_status_snapshot
        from state import app_state

        interval_s = app_state.config.scheduler.status_push_interval_s

        try:
            while True:
                await asyncio.sleep(interval_s)
                if not self._connections:
                    continue
                try:
                    snapshot = compute_status_snapshot()
                    await self.broadcast_event("status_update", snapshot)
                except asyncio.CancelledError:
                    raise
                except Exception as exc:
                    log.error(f"  WebSocket: status push failed: {exc}")
        except asyncio.CancelledError:
            log.debug("  WebSocket: status push loop stopped")
            raise

    def fire_event(self, event_type: str, data: dict) -> None:
        """Thread-safe: schedule broadcast_event on the stored event loop.

        Safe to call from any thread — silently no-ops if no loop is set
        (e.g. during early startup before the server is listening).
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        asyncio.run_coroutine_threadsafe(
            self.broadcast_event(event_type, data), loop)


# Global instance
streamer = ProgressStreamer()
