"""Modal screens for the Foxburrow TUI console."""

from __future__ import annotations

from datetime import datetime

from rich.text import Text
from textual.app import ComposeResult
from textual.binding import Binding
from textual.containers import Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import DataTable, Static

from tui.widgets import _format_bytes, _vram_bar


# ── QueueDetailScreen ──────────────────────────────────────────────


class QueueDetailScreen(ModalScreen):
    """Modal showing all queued jobs in a DataTable."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("q", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    QueueDetailScreen {
        align: center middle;
    }

    QueueDetailScreen > Vertical {
        width: 90%;
        height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    QueueDetailScreen Static.title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("Queue Detail", classes="title")
            yield DataTable(id="queue-table")

    def on_mount(self) -> None:
        table = self.query_one("#queue-table", DataTable)
        table.add_columns("ID", "Type", "Stage", "Priority", "Age")
        self._refresh_table()
        # Refresh every 1s while the modal is open
        self.set_interval(1.0, self._refresh_table)

    def _refresh_table(self) -> None:
        from state import app_state

        table = self.query_one("#queue-table", DataTable)
        table.clear()

        jobs = app_state.queue.snapshot()
        now = datetime.utcnow()

        for job in jobs:
            stage = job.current_stage
            stage_str = stage.type.value if stage else "complete"
            age = (now - job.created_at).total_seconds()

            if age < 60:
                age_str = f"{age:.0f}s"
            elif age < 3600:
                age_str = f"{age / 60:.1f}m"
            else:
                age_str = f"{age / 3600:.1f}h"

            table.add_row(
                job.job_id[:8],
                job.type.value,
                stage_str,
                str(job.priority),
                age_str,
            )

        if not jobs:
            table.add_row("—", "Queue is empty", "", "", "")


# ── GpuDetailScreen ────────────────────────────────────────────────


class GpuDetailScreen(ModalScreen):
    """Modal showing detailed per-GPU info: VRAM breakdown, temperature, utilization."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("g", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    GpuDetailScreen {
        align: center middle;
    }

    GpuDetailScreen > Vertical {
        width: 90%;
        height: 85%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    GpuDetailScreen Static.title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }

    GpuDetailScreen .gpu-detail-card {
        margin-bottom: 1;
        padding: 0 1;
    }
    """

    def compose(self) -> ComposeResult:
        with Vertical():
            yield Static("GPU Detail", classes="title")
            yield VerticalScroll(id="gpu-detail-scroll")

    def on_mount(self) -> None:
        self._build_cards()
        # Refresh every 2s while the modal is open
        self.set_interval(2.0, self._update_cards)

    def _build_cards(self) -> None:
        """Initial card creation — one Static per GPU."""
        from state import app_state

        container = self.query_one("#gpu-detail-scroll", VerticalScroll)
        for gpu in app_state.gpu_pool.gpus:
            card = Static("", id=f"gpu-card-{gpu.device_id}", classes="gpu-detail-card")
            container.mount(card)
        self._update_cards()

    def _update_cards(self) -> None:
        """Update existing card contents in-place (no remove/mount cycle)."""
        from gpu import nvml
        from state import app_state

        for gpu in app_state.gpu_pool.gpus:
            try:
                card = self.query_one(f"#gpu-card-{gpu.device_id}", Static)
            except Exception:
                continue

            text = Text()

            # Header
            text.append(f"━━━ {gpu.name} ", style="bold")
            text.append(f"[{gpu.uuid[:16]}...]", style="dim")
            text.append(f"  CUDA:{gpu.device_id}", style="dim")
            text.append("\n")

            # Temperature and utilization via NVML
            try:
                temp = nvml.get_temperature(gpu.nvml_handle)
                gpu_util, mem_util = nvml.get_utilization(gpu.nvml_handle)
                text.append(f"Temp: {temp}°C", style="bold yellow" if temp >= 80 else "")
                text.append(f"  GPU Util: {gpu_util}%")
                text.append(f"  Mem Util: {mem_util}%")
                text.append("\n")
            except Exception:
                text.append("(NVML stats unavailable)\n", style="dim")

            # VRAM breakdown
            vram = gpu.get_vram_stats()
            text.append("\n")
            text.append("VRAM: ")
            text.append_text(_vram_bar(vram["used"], vram["total"], width=30))
            text.append("\n")
            text.append(f"  Allocated (tensors): {_format_bytes(vram['allocated'])}\n")
            text.append(f"  Reserved (cache):    {_format_bytes(vram['reserved'])}\n")
            text.append(f"  Used (NVML):         {_format_bytes(vram['used'])}\n")
            text.append(f"  Total:               {_format_bytes(vram['total'])}\n")
            text.append(f"  Free (NVML):         {_format_bytes(vram['free'])}\n")

            # Cache contents
            cached = gpu.get_cached_models_info()
            text.append(f"\nModel Cache ({len(cached)} entries):\n")
            if cached:
                for m in cached:
                    vram_str = _format_bytes(m["vram"])
                    text.append(f"  {m['category']:<12}", style="bold")
                    text.append(f" {m['source'] or '?':<20}")
                    text.append(f" {vram_str:>8}")
                    if m["actual_vram"] > 0:
                        text.append("  (measured)", style="dim")
                    else:
                        text.append("  (estimated)", style="dim italic")
                    text.append("\n")
            else:
                text.append("  (empty)\n", style="dim")

            # Capabilities
            caps = ", ".join(sorted(gpu.capabilities))
            text.append(f"\nCapabilities: {caps}\n")

            card.update(text)


# ── HelpScreen ─────────────────────────────────────────────────────


class HelpScreen(ModalScreen):
    """Modal showing all keybindings."""

    BINDINGS = [
        Binding("escape", "dismiss", "Close", show=True),
        Binding("question_mark", "dismiss", "Close", show=False),
    ]

    DEFAULT_CSS = """
    HelpScreen {
        align: center middle;
    }

    HelpScreen > Vertical {
        width: 60;
        height: auto;
        max-height: 80%;
        border: thick $accent;
        background: $surface;
        padding: 1 2;
    }

    HelpScreen Static.title {
        text-align: center;
        text-style: bold;
        margin-bottom: 1;
    }
    """

    def compose(self) -> ComposeResult:
        text = Text()
        bindings = [
            ("r", "Rescan SDXL models"),
            ("l", "Rescan LoRA files"),
            ("q", "Queue detail"),
            ("g", "GPU detail"),
            ("Space", "Pause/resume log scroll"),
            ("d", "Toggle DEBUG log level"),
            ("i", "Toggle INFO log level"),
            ("w", "Toggle WARNING log level"),
            ("e", "Toggle ERROR log level"),
            ("?", "This help screen"),
            ("Ctrl+C", "Quit"),
        ]
        for key, desc in bindings:
            text.append(f"  {key:<12}", style="bold cyan")
            text.append(f"{desc}\n")

        with Vertical():
            yield Static("Keybindings", classes="title")
            yield Static(text)
