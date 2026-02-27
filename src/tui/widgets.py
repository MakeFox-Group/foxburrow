"""Custom Textual widgets for the Foxburrow TUI console."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.widgets import RichLog, Static
from textual.containers import Vertical

from log import LogLevel


# ── Helpers ────────────────────────────────────────────────────────


def _format_bytes(b: int) -> str:
    """Format bytes as a human-readable string (e.g. '3.8GB')."""
    if b >= 1024 ** 3:
        return f"{b / (1024 ** 3):.1f}GB"
    if b >= 1024 ** 2:
        return f"{b / (1024 ** 2):.0f}MB"
    return f"{b / 1024:.0f}KB"


def _vram_bar(used: int, total: int, width: int = 20) -> Text:
    """Build a colored VRAM progress bar."""
    if total == 0:
        return Text("[" + "?" * width + "]")
    ratio = min(used / total, 1.0)
    filled = int(ratio * width)
    empty = width - filled

    if ratio < 0.50:
        color = "green"
    elif ratio < 0.80:
        color = "yellow"
    else:
        color = "red"

    bar = Text("[")
    bar.append("█" * filled, style=color)
    bar.append("░" * empty, style="bright_black")
    bar.append("]")
    return bar


def _progress_bar(current: int, total: int, width: int = 30) -> Text:
    """Build a job progress bar."""
    if total == 0:
        return Text("[" + "░" * width + "]")
    ratio = min(current / total, 1.0)
    filled = int(ratio * width)
    empty = width - filled

    bar = Text("[")
    bar.append("█" * filled, style="cyan")
    bar.append("░" * empty, style="bright_black")
    bar.append("]")
    return bar


# ── GpuPanel ───────────────────────────────────────────────────────


class GpuPanel(Static):
    """Renders status for a single GPU."""

    DEFAULT_CSS = """
    GpuPanel {
        padding: 0 1;
        margin: 0;
    }
    """

    def __init__(self, gpu_uuid: str, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gpu_uuid = gpu_uuid

    def render_gpu(self, gpu, workers) -> None:
        """Update this panel from a GpuInstance and its active worker jobs."""
        text = Text()

        # Header: name, status, capabilities
        status = "BUSY" if gpu.is_busy else "IDLE"
        status_style = "bold red" if gpu.is_busy else "bold green"
        caps = ", ".join(sorted(gpu.capabilities))

        text.append(f"{gpu.name}", style="bold")
        text.append(" (")
        text.append(status, style=status_style)
        text.append(")")
        # Right-align capabilities
        cap_str = f"  {caps}"
        text.append(cap_str, style="dim")
        text.append("\n")

        # VRAM bar
        vram = gpu.get_vram_stats()
        used = vram["allocated"]
        total = vram["total"]
        free = vram["free"]
        pct = (used / total * 100) if total > 0 else 0

        text.append("VRAM: ")
        text.append_text(_vram_bar(used, total))
        text.append(f" {pct:.0f}%  {_format_bytes(used)}/{_format_bytes(total)}")
        text.append(f"  free: {_format_bytes(free)}", style="dim")
        text.append("\n")

        # Loaded models
        cached_info = gpu.get_cached_models_info()
        if cached_info:
            # Group by source
            by_source: dict[str, list[dict]] = {}
            for m in cached_info:
                src = m["source"] or m["category"]
                by_source.setdefault(src, []).append(m)

            parts = []
            for src, models in by_source.items():
                cats = ", ".join(m["category"] for m in models)
                src_vram = sum(m["vram"] for m in models)
                parts.append(f"{src} ({cats}) {_format_bytes(src_vram)}")

            text.append("Loaded: ", style="dim")
            text.append("; ".join(parts))
            text.append("\n")
        else:
            text.append("Loaded: ", style="dim")
            text.append("(none)", style="dim italic")
            text.append("\n")

        # Active jobs
        active_jobs = []
        for w in workers:
            if w.gpu.uuid == gpu.uuid:
                active_jobs = w.active_jobs
                break

        if active_jobs:
            for job in active_jobs:
                stage = job.current_stage
                stage_name = stage.type.value if stage else "?"
                elapsed = 0.0
                if job.started_at:
                    elapsed = (datetime.utcnow() - job.started_at).total_seconds()

                text.append(f"Job[{job.job_id[:6]}]", style="bold cyan")
                text.append(f" {job.type.value} {stage_name}")
                if stage:
                    text.append(f" ({job.current_stage_index + 1}/{len(job.pipeline)})")
                if job.stage_status == "loading":
                    text.append(" loading...", style="bold yellow")
                text.append(f" {elapsed:.1f}s", style="dim")
                text.append("\n")

                # Denoise progress bar (only while denoise stage is running)
                if job.denoise_total_steps > 0 and job.stage_status == "running":
                    text.append("  ")
                    text.append_text(_progress_bar(job.denoise_step, job.denoise_total_steps))
                    pct = job.denoise_step / job.denoise_total_steps * 100
                    text.append(f" {pct:.1f}%  step {job.denoise_step}/{job.denoise_total_steps}")
                    text.append("\n")
        else:
            text.append("(idle)", style="dim italic")
            text.append("\n")

        self.update(text)


# ── GpuDashboard ───────────────────────────────────────────────────


class GpuDashboard(Vertical):
    """Container holding GpuPanel instances — one per GPU."""

    DEFAULT_CSS = """
    GpuDashboard {
        height: auto;
        max-height: 50%;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._panels: dict[str, GpuPanel] = {}

    def refresh_from_state(self, app_state) -> None:
        """Sync panels with current GPU pool state and update each panel."""
        gpus = app_state.gpu_pool.gpus
        workers = app_state.scheduler.workers if app_state.scheduler else []

        current_uuids = {g.uuid for g in gpus}
        panel_uuids = set(self._panels.keys())

        # Remove panels for GPUs that no longer exist
        for uuid in panel_uuids - current_uuids:
            panel = self._panels.pop(uuid)
            panel.remove()

        # Add panels for new GPUs
        for gpu in gpus:
            if gpu.uuid not in self._panels:
                panel = GpuPanel(gpu.uuid)
                self._panels[gpu.uuid] = panel
                self.mount(panel)

        # Update all panels
        for gpu in gpus:
            panel = self._panels.get(gpu.uuid)
            if panel:
                panel.render_gpu(gpu, workers)


# ── LogPanel ───────────────────────────────────────────────────────


_LEVEL_STYLES: dict[LogLevel, str] = {
    LogLevel.DEBUG: "dim",
    LogLevel.INFO: "",
    LogLevel.WARNING: "yellow",
    LogLevel.ERROR: "bold red",
}


class LogPanel(Vertical):
    """Scrollable log panel wrapping a RichLog widget.

    Auto-detects when the user scrolls away from the bottom and pauses
    auto-scroll so new log lines don't snap back.  Resumes when the user
    scrolls back to the bottom or presses Space.
    """

    DEFAULT_CSS = """
    LogPanel {
        height: 1fr;
    }

    LogPanel #tui-rich-log {
        height: 1fr;
        border-top: solid $accent;
    }

    LogPanel #tui-paused-label {
        dock: bottom;
        text-align: right;
        padding: 0 1;
        height: 1;
    }
    """

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        self._auto_scroll = True
        self._rich_log: RichLog | None = None
        self._paused_label: Static | None = None

    def compose(self):
        yield RichLog(id="tui-rich-log", max_lines=5000, wrap=True, auto_scroll=True)
        yield Static("", id="tui-paused-label")

    def on_mount(self) -> None:
        """Resolve widget references after compose is complete."""
        self._rich_log = self.query_one("#tui-rich-log", RichLog)
        self._paused_label = self.query_one("#tui-paused-label", Static)

    def _is_at_bottom(self) -> bool:
        """Check if the RichLog is scrolled to (or near) the bottom."""
        rl = self._rich_log
        if rl is None:
            return True
        # Not enough content to scroll — effectively at bottom
        if rl.max_scroll_y <= 2:
            return True
        return rl.scroll_y >= rl.max_scroll_y - 2

    def _set_paused(self, paused: bool) -> None:
        """Update auto-scroll state and paused indicator."""
        self._auto_scroll = not paused
        if self._rich_log is not None:
            self._rich_log.auto_scroll = self._auto_scroll
        if self._paused_label is not None:
            if paused:
                self._paused_label.update(
                    Text("[PAUSED — Space or scroll to bottom to resume]",
                         style="bold yellow"))
            else:
                self._paused_label.update("")

    def write_batch(self, items: list[tuple[str, LogLevel]]) -> None:
        """Write a batch of log lines, auto-pausing if user scrolled up.

        Called by the drain timer.  Checks scroll position BEFORE writing
        so that a user who scrolled up won't get snapped back to the bottom.
        """
        if self._rich_log is None or not items:
            return

        # Detect user scroll-up: auto_scroll is on but we're not at bottom
        if self._auto_scroll and not self._is_at_bottom():
            self._set_paused(True)

        for line, level in items:
            style = _LEVEL_STYLES.get(level, "")
            text = Text(line, style=style)
            self._rich_log.write(text)

    def check_auto_resume(self) -> None:
        """Resume auto-scroll if the user has scrolled back to the bottom."""
        if self._auto_scroll or self._rich_log is None:
            return
        if self._is_at_bottom():
            self._set_paused(False)

    def toggle_auto_scroll(self) -> None:
        """Manual toggle via Space key."""
        self._set_paused(self._auto_scroll)

    @property
    def is_paused(self) -> bool:
        return not self._auto_scroll
