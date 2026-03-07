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


class _ExceptionLoggingMiddleware:
    """ASGI middleware that logs unhandled exceptions to output.jsonl.

    Wraps the entire ASGI app so exceptions from any layer (routes, middleware,
    form parsing) are captured.  Only sends a 500 response if headers haven't
    been sent yet — safe with streaming responses.
    """

    def __init__(self, app):
        self.app = app

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        response_started = False
        original_send = send

        async def send_wrapper(message):
            nonlocal response_started
            if message["type"] == "http.response.start":
                response_started = True
            await original_send(message)

        try:
            await self.app(scope, receive, send_wrapper)
        except Exception as ex:
            request = Request(scope)
            log.log_exception(ex, f"Unhandled exception: {request.method} {request.url.path}")
            if not response_started:
                response = JSONResponse(
                    status_code=500, content={"error": "Internal server error"})
                await response(scope, receive, original_send)


def create_app(on_startup: Callable[[], Coroutine[Any, Any, None]] | None = None) -> FastAPI:

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        cleanup_task = asyncio.create_task(_job_cleanup_loop())
        if on_startup:
            await on_startup()
        yield
        # Graceful shutdown
        log.debug("  Shutting down...")
        cleanup_task.cancel()
        if app_state.trt_manager:
            app_state.trt_manager.stop()
        if app_state.fs_watcher:
            await app_state.fs_watcher.stop()
        scheduler = app_state.scheduler
        if scheduler and scheduler._task:
            scheduler._task.cancel()
        if scheduler:
            for w in scheduler.workers:
                if w._task:
                    w._task.cancel()
        log.debug("  Shutdown complete.")

    app = FastAPI(title="foxburrow", version="2.0.0", lifespan=lifespan)

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not _check_secret(request.headers, request.query_params):
            return JSONResponse(status_code=401, content={"error": "Unauthorized"})
        return await call_next(request)

    # Pure ASGI exception-logging middleware — outermost layer.
    # Uses raw ASGI instead of BaseHTTPMiddleware to guarantee exception
    # propagation regardless of Starlette version and to handle streaming
    # responses safely (won't send a new response if headers already sent).
    app.add_middleware(_ExceptionLoggingMiddleware)

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
