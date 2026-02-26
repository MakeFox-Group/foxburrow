"""Main Textual App for the Foxburrow TUI console."""

from __future__ import annotations

import os
import threading
from collections import deque
from datetime import datetime

from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.widgets import Footer, Static

import log
from log import LogLevel
from state import app_state
from tui.screens import GpuDetailScreen, HelpScreen, QueueDetailScreen
from tui.widgets import GpuDashboard, LogPanel


class FoxburrowApp(App):
    """Foxburrow management TUI console."""

    TITLE = "foxburrow"
    SUB_TITLE = "v2.0.0"

    CSS = """
    Screen {
        layout: vertical;
    }

    #status-bar {
        dock: top;
        height: 1;
        background: $accent;
        color: $text;
        padding: 0 1;
    }

    #gpu-dashboard {
        height: auto;
        max-height: 50%;
    }

    #log-panel {
        height: 1fr;
    }

    Footer {
        dock: bottom;
    }
    """

    BINDINGS = [
        Binding("r", "rescan_models", "Rescan Models"),
        Binding("l", "rescan_loras", "Rescan LoRAs"),
        Binding("q", "queue_detail", "Queue"),
        Binding("g", "gpu_detail", "GPU Detail"),
        Binding("space", "toggle_scroll", "Pause/Resume", show=False),
        Binding("question_mark", "help", "Help (?)"),
    ]

    def __init__(self, uvicorn_server=None, **kwargs) -> None:
        super().__init__(**kwargs)
        self._uvicorn_server = uvicorn_server

        # Thread-safe log buffer: producers (any thread) append, TUI timer drains
        self._log_buffer: deque[tuple[str, LogLevel]] = deque(maxlen=10000)
        self._log_lock = threading.Lock()

    def compose(self) -> ComposeResult:
        yield Static(id="status-bar")
        yield GpuDashboard(id="gpu-dashboard")
        yield LogPanel(id="log-panel")
        yield Footer()

    def on_mount(self) -> None:
        """Called when the app is mounted — register timers and log callback."""
        # Register log callback (suppresses stdout, routes INFO+ to our buffer)
        log.set_tui_callback(self._log_callback, min_level=LogLevel.INFO)

        # Dashboard refresh timer (500ms)
        self.set_interval(0.5, self._refresh_dashboard)

        # Log drain timer (100ms)
        self.set_interval(0.1, self._drain_log_buffer)

        # Initial refresh after compose is fully mounted
        self.call_after_refresh(self._refresh_dashboard)

    def _log_callback(self, line: str, level: LogLevel) -> None:
        """Called from any thread by log.write_line(). Just buffers the line."""
        with self._log_lock:
            self._log_buffer.append((line, level))

    def _drain_log_buffer(self) -> None:
        """Drain buffered log lines into the LogPanel (runs on main thread)."""
        with self._log_lock:
            items = list(self._log_buffer)
            self._log_buffer.clear()

        if not items:
            return

        try:
            log_panel = self.query_one("#log-panel", LogPanel)
            for line, level in items:
                log_panel.write_log(line, level)
        except Exception as e:
            log.debug(f"TUI log drain error: {e}")

    def _refresh_dashboard(self) -> None:
        """Refresh GPU dashboard and status bar from app_state."""
        # Update status bar
        try:
            queue_depth = len(app_state.queue.snapshot())
            gpu_count = len(app_state.gpu_pool.gpus)
            now = datetime.now().strftime("%H:%M:%S")

            status = Text()
            status.append(f"  {self.TITLE} {self.SUB_TITLE}", style="bold")
            status.append("  │  ")
            status.append(f"Queue: {queue_depth}")
            status.append("  │  ")
            status.append(f"GPUs: {gpu_count}")
            status.append("  │  ")
            status.append(now, style="dim")

            status_bar = self.query_one("#status-bar", Static)
            status_bar.update(status)
        except Exception as e:
            log.debug(f"TUI status bar refresh error: {e}")

        # Update GPU panels
        try:
            dashboard = self.query_one("#gpu-dashboard", GpuDashboard)
            dashboard.refresh_from_state(app_state)
        except Exception as e:
            log.debug(f"TUI dashboard refresh error: {e}")

    # ── Actions ────────────────────────────────────────────────────

    def action_rescan_models(self) -> None:
        """Rescan SDXL models directory (runs in background thread)."""
        self.run_worker(self._do_rescan_models, thread=True)

    def action_rescan_loras(self) -> None:
        """Rescan LoRA files directory (runs in background thread)."""
        self.run_worker(self._do_rescan_loras, thread=True)

    def action_queue_detail(self) -> None:
        """Show queue detail modal."""
        self.push_screen(QueueDetailScreen())

    def action_gpu_detail(self) -> None:
        """Show GPU detail modal."""
        self.push_screen(GpuDetailScreen())

    def action_toggle_scroll(self) -> None:
        """Toggle log panel auto-scroll."""
        log_panel = self.query_one("#log-panel", LogPanel)
        log_panel.toggle_auto_scroll()

    def action_help(self) -> None:
        """Show help modal."""
        self.push_screen(HelpScreen())

    # ── Background workers ─────────────────────────────────────────

    @staticmethod
    def _do_rescan_models() -> None:
        """Blocking model rescan — called from a worker thread."""
        from main import discover_sdxl_models

        models_dir = os.path.normpath(os.path.abspath(app_state.config.server.models_dir))
        fresh = discover_sdxl_models(models_dir)

        old_names = set(app_state.sdxl_models.keys())
        new_names = set(fresh.keys())

        added = new_names - old_names
        removed = old_names - new_names

        for name in removed:
            app_state.sdxl_models.pop(name, None)

        if added:
            from config import _auto_threads
            from utils.model_scanner import ModelScanner

            new_models = {name: fresh[name] for name in added}
            fp_threads = _auto_threads(app_state.config.threads.fingerprint, 8)
            scanner = ModelScanner(app_state.registry, app_state, max_workers=fp_threads)
            scanner.start(new_models)
            app_state.model_scanner = scanner

        log.info(f"  Model rescan: +{len(added)} -{len(removed)} "
                 f"={len(old_names & new_names)} "
                 f"(total: {len(fresh)})")

    @staticmethod
    def _do_rescan_loras() -> None:
        """Blocking LoRA rescan — called from a worker thread."""
        loras_dir = app_state.loras_dir
        if not loras_dir:
            log.warning("  No LoRA directory configured — skipping rescan")
            return

        from utils.lora_index import rescan_loras
        rescan_loras(loras_dir, app_state.lora_index)

    # ── Shutdown ───────────────────────────────────────────────────

    def action_request_quit(self) -> None:
        """Ctrl+C quits immediately — no confirmation dialog."""
        self.exit()

    def on_unmount(self) -> None:
        """Clean up on exit — restore stdout logging."""
        log.clear_tui_callback()
