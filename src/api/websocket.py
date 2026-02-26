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
    """Streams job progress to connected WebSocket clients."""

    def __init__(self):
        self._connections: list[WebSocket] = []
        self._lock = asyncio.Lock()

    async def connect(self, ws: WebSocket) -> None:
        await ws.accept()
        async with self._lock:
            self._connections.append(ws)
        log.info(f"  WebSocket: Client connected ({len(self._connections)} total)")

    async def disconnect(self, ws: WebSocket) -> None:
        async with self._lock:
            if ws in self._connections:
                self._connections.remove(ws)
        log.info(f"  WebSocket: Client disconnected ({len(self._connections)} total)")

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


# Global instance
streamer = ProgressStreamer()
