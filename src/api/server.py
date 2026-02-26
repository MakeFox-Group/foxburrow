"""FastAPI app setup."""

import asyncio
import hmac
from contextlib import asynccontextmanager
from typing import Callable, Coroutine, Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.websockets import WebSocket, WebSocketDisconnect

import log
from state import app_state


def _check_secret(headers, query_params) -> bool:
    """Validate API secret from Authorization header or apikey query param.

    Works for both HTTP requests and WebSocket connections — both expose
    .headers and .query_params with the same interface.
    """
    secret = app_state.config.server.secret
    if not secret:
        return True  # No secret configured — auth disabled

    # Check Authorization: Bearer <token> (case-insensitive scheme per RFC 7235)
    auth_header = headers.get("authorization", "")
    if auth_header[:7].lower() == "bearer ":
        token = auth_header[7:]
        if hmac.compare_digest(token, secret):
            return True

    # Check ?apikey= query parameter
    apikey = query_params.get("apikey", "")
    if apikey and hmac.compare_digest(apikey, secret):
        return True

    return False


def create_app(on_startup: Callable[[], Coroutine[Any, Any, None]] | None = None) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = asyncio.create_task(_job_cleanup_loop())
        if on_startup:
            await on_startup()
        yield
        # Graceful shutdown: cancel scheduler and workers before exiting
        log.info("  Shutting down...")
        cleanup_task.cancel()
        scheduler = app_state.scheduler
        if scheduler and scheduler._task:
            scheduler._task.cancel()
        if scheduler:
            for w in scheduler.workers:
                if w._task:
                    w._task.cancel()
        log.info("  Shutdown complete.")

    app = FastAPI(title="foxburrow", version="2.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not _check_secret(request.headers, request.query_params):
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)

    from api.routes import router
    app.include_router(router, prefix="/api")

    from api.websocket import streamer

    @app.websocket("/api/ws")
    async def ws_endpoint(ws: WebSocket):
        if not _check_secret(ws.headers, ws.query_params):
            await ws.close(code=4401, reason="Unauthorized")
            return

        await streamer.connect(ws)
        try:
            while True:
                await ws.receive_text()  # keep-alive
        except Exception:
            pass
        finally:
            await streamer.disconnect(ws)

    return app


async def _job_cleanup_loop():
    """Remove completed jobs older than 5 minutes every 60 seconds."""
    from datetime import datetime, timedelta
    from scheduling.job import JobResult
    from state import app_state

    while True:
        try:
            await asyncio.sleep(60)
            cutoff = datetime.utcnow() - timedelta(minutes=5)
            expired = [
                jid for jid, job in app_state.jobs.items()
                if job.completed_at is not None and job.completed_at < cutoff
            ]
            for jid in expired:
                job = app_state.jobs.get(jid)
                if job and not job.completion.done():
                    job.set_result(JobResult(success=False, error="Job expired"))
                app_state.jobs.pop(jid, None)
                app_state.job_results.pop(jid, None)
        except asyncio.CancelledError:
            break
        except Exception as ex:
            log.log_exception(ex, "Job cleanup loop error")
