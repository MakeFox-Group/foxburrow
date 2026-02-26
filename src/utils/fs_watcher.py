"""Filesystem watcher — auto-detect model changes using watchfiles (inotify)."""

from __future__ import annotations

import asyncio
import os
from typing import TYPE_CHECKING

from watchfiles import awatch, Change, DefaultFilter

import log
from api.websocket import streamer

if TYPE_CHECKING:
    from state import AppState

# ── Filters ──────────────────────────────────────────────────────────
# DefaultFilter already ignores .git, __pycache__, .swp, .DS_Store, etc.
# We add .fpcache to prevent fingerprint-cache writes from re-triggering
# the watcher (fingerprint → write .fpcache → inotify → rescan → repeat).

_IGNORE_NAMES = frozenset({".fpcache"})

# Only react to changes involving these file extensions.
_SDXL_EXTENSIONS = frozenset({".safetensors"})
_LORA_EXTENSIONS = frozenset({".safetensors", ".pt", ".pth", ".ckpt", ".bin"})

# Debounce: wait for this many ms of filesystem silence before yielding.
# 3 seconds is long enough for most multi-GB model file copies to settle.
_DEBOUNCE_MS = 3000


class _IgnoreFpCache(DefaultFilter):
    """DefaultFilter + ignore .fpcache files."""

    def __call__(self, change: Change, path: str) -> bool:
        if os.path.basename(path) in _IGNORE_NAMES:
            return False
        return super().__call__(change, path)


# ── Helpers ──────────────────────────────────────────────────────────

def _has_relevant_extension(path: str, extensions: frozenset[str]) -> bool:
    return os.path.splitext(path)[1].lower() in extensions


def _summarize_changes(changes: set[tuple[Change, str]]) -> str:
    added = sum(1 for c, _ in changes if c == Change.added)
    removed = sum(1 for c, _ in changes if c == Change.deleted)
    modified = sum(1 for c, _ in changes if c == Change.modified)
    parts = []
    if added:
        parts.append(f"+{added}")
    if removed:
        parts.append(f"-{removed}")
    if modified:
        parts.append(f"~{modified}")
    return " ".join(parts) or "0 events"


# ── Watcher ──────────────────────────────────────────────────────────

class FileSystemWatcher:
    """Watches model directories for file changes and triggers automatic rescans.

    Uses ``watchfiles.awatch()`` (Rust ``notify`` crate) which maps to the
    best native backend per-OS: inotify on Linux, FSEvents on macOS,
    ReadDirectoryChangesW on Windows.

    Fires the same WebSocket events as the manual ``/api/rescan-models``
    and ``/api/rescan-loras`` endpoints.
    """

    def __init__(self, app_state: AppState):
        self._app_state = app_state
        self._tasks: list[asyncio.Task] = []

    def start(self) -> None:
        """Start async watcher tasks for all configured model directories."""
        loop = asyncio.get_running_loop()
        state = self._app_state

        models_dir = os.path.normpath(
            os.path.abspath(state.config.server.models_dir))

        sdxl_dir = os.path.join(models_dir, "sdxl")
        if os.path.isdir(sdxl_dir):
            self._tasks.append(loop.create_task(self._watch_sdxl(sdxl_dir)))
            log.info(f"  FileWatcher: Monitoring {sdxl_dir}")

        loras_dir = state.loras_dir
        if loras_dir and os.path.isdir(loras_dir):
            self._tasks.append(loop.create_task(self._watch_loras(loras_dir)))
            log.info(f"  FileWatcher: Monitoring {loras_dir}")

        if not self._tasks:
            log.warning("  FileWatcher: No directories to watch")

    async def stop(self) -> None:
        """Cancel all watcher tasks."""
        for task in self._tasks:
            task.cancel()
        if self._tasks:
            await asyncio.gather(*self._tasks, return_exceptions=True)
        self._tasks.clear()

    # ── LoRA watcher ─────────────────────────────────────────────────

    async def _watch_loras(self, loras_dir: str) -> None:
        try:
            async for changes in awatch(
                loras_dir,
                watch_filter=_IgnoreFpCache(),
                debounce=_DEBOUNCE_MS,
                recursive=True,
            ):
                relevant = {
                    (c, p) for c, p in changes
                    if _has_relevant_extension(p, _LORA_EXTENSIONS)
                    or c == Change.deleted
                }
                if not relevant:
                    continue

                log.info(f"  FileWatcher: LoRA changes detected "
                         f"({_summarize_changes(relevant)})")

                try:
                    from utils.lora_index import rescan_loras
                    loop = asyncio.get_running_loop()
                    await loop.run_in_executor(
                        None, rescan_loras,
                        loras_dir, self._app_state.lora_index)
                except Exception as ex:
                    log.log_exception(ex, "FileWatcher: LoRA rescan failed")

        except asyncio.CancelledError:
            log.info("  FileWatcher: LoRA watcher stopped")
        except Exception as ex:
            log.log_exception(ex, "FileWatcher: LoRA watcher crashed")

    # ── SDXL model watcher ───────────────────────────────────────────

    async def _watch_sdxl(self, sdxl_dir: str) -> None:
        try:
            async for changes in awatch(
                sdxl_dir,
                watch_filter=_IgnoreFpCache(),
                debounce=_DEBOUNCE_MS,
                recursive=True,
            ):
                relevant = {
                    (c, p) for c, p in changes
                    if _has_relevant_extension(p, _SDXL_EXTENSIONS)
                    or c == Change.deleted
                }
                if not relevant:
                    continue

                log.info(f"  FileWatcher: SDXL changes detected "
                         f"({_summarize_changes(relevant)})")

                try:
                    await self._rescan_sdxl()
                except Exception as ex:
                    log.log_exception(ex, "FileWatcher: SDXL rescan failed")

        except asyncio.CancelledError:
            log.info("  FileWatcher: SDXL watcher stopped")
        except Exception as ex:
            log.log_exception(ex, "FileWatcher: SDXL watcher crashed")

    async def _rescan_sdxl(self) -> dict:
        """Rescan SDXL models — same logic as POST /api/rescan-models."""
        from main import discover_sdxl_models
        from utils.model_scanner import ModelScanner
        from config import _auto_threads

        state = self._app_state
        models_dir = os.path.normpath(
            os.path.abspath(state.config.server.models_dir))

        loop = asyncio.get_running_loop()
        fresh = await loop.run_in_executor(
            None, discover_sdxl_models, models_dir)

        old_names = set(state.sdxl_models.keys())
        new_names = set(fresh.keys())

        added = new_names - old_names
        removed = old_names - new_names

        if not added and not removed:
            return {"added": 0, "removed": 0,
                    "unchanged": len(old_names), "total": len(old_names)}

        # Remove deleted models
        for name in removed:
            path = state.sdxl_models.pop(name)
            streamer.fire_event("model_removed", {
                "model_type": "sdxl", "name": name, "path": path})

        # Register new models (fingerprinting in background thread pool)
        if added:
            new_models = {name: fresh[name] for name in added}
            fp_threads = _auto_threads(state.config.threads.fingerprint, 8)
            scanner = ModelScanner(state.registry, state,
                                   max_workers=fp_threads)
            scanner.start(new_models)
            state.model_scanner = scanner

        summary = {
            "added": len(added),
            "removed": len(removed),
            "unchanged": len(old_names & new_names),
            "total": len(state.sdxl_models) + len(added),
        }

        log.info(f"  FileWatcher: SDXL rescan: +{summary['added']} "
                 f"-{summary['removed']} ={summary['unchanged']} "
                 f"(total: {summary['total']})")

        return summary
