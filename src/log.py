"""Thread-safe timestamped logging to stdout + file.

Supports an optional TUI callback: when registered, log lines at or above
a configurable level are routed to the callback instead of stdout.  The
file sink always receives ALL levels (including DEBUG).
"""

import os
import sys
import threading
import traceback
from datetime import datetime
from enum import Enum
from typing import Callable


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


_LEVEL_ORDER: dict[LogLevel, int] = {
    LogLevel.DEBUG: 0,
    LogLevel.INFO: 1,
    LogLevel.WARNING: 2,
    LogLevel.ERROR: 3,
}

_lock = threading.Lock()
_log_file = None
_log_path: str | None = None

# TUI callback support
_tui_callback: Callable[[str, LogLevel], None] | None = None
_tui_min_level: LogLevel = LogLevel.INFO
_suppress_stdout: bool = False


def init_file(path: str) -> None:
    """Open the log file for appending. Creates parent directories if needed."""
    global _log_file, _log_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _log_file = open(path, "a", encoding="utf-8")
    _log_path = path


def get_log_path() -> str | None:
    """Return the active log file path, or None if file logging is not enabled."""
    return _log_path


def set_tui_callback(
    callback: Callable[[str, LogLevel], None],
    min_level: LogLevel = LogLevel.INFO,
) -> None:
    """Register a TUI callback.  When set, stdout is suppressed and lines
    at *min_level* or above are routed to *callback* instead."""
    global _tui_callback, _tui_min_level, _suppress_stdout
    with _lock:
        _tui_callback = callback
        _tui_min_level = min_level
        _suppress_stdout = True


def clear_tui_callback() -> None:
    """Unregister the TUI callback and restore stdout output."""
    global _tui_callback, _suppress_stdout
    with _lock:
        _tui_callback = None
        _suppress_stdout = False


def write_line(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    # Snapshot callback under lock, then call it OUTSIDE the lock to avoid
    # deadlock if the callback (or anything it calls) re-enters log.*.
    cb = None
    with _lock:
        if not _suppress_stdout:
            print(line, flush=True)
        if _log_file is not None:
            _log_file.write(line + "\n")
            _log_file.flush()
        if _tui_callback is not None and _LEVEL_ORDER.get(level, 0) >= _LEVEL_ORDER.get(_tui_min_level, 1):
            cb = _tui_callback
    if cb is not None:
        cb(line, level)


def write(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    cb = None
    with _lock:
        if not _suppress_stdout:
            print(line, end="", flush=True)
        if _log_file is not None:
            _log_file.write(line)
            _log_file.flush()
        if _tui_callback is not None and _LEVEL_ORDER.get(level, 0) >= _LEVEL_ORDER.get(_tui_min_level, 1):
            cb = _tui_callback
    if cb is not None:
        cb(line, level)


def log_exception(ex: BaseException, context: str | None = None) -> None:
    tb = traceback.format_exception(type(ex), ex, ex.__traceback__)
    tb_str = "".join(tb).rstrip()
    if context:
        msg = f"{context}\n{type(ex).__name__}: {ex}\n{tb_str}"
    else:
        msg = f"{type(ex).__name__}: {ex}\n{tb_str}"
    write_line(msg, LogLevel.ERROR)


def debug(message: str) -> None:
    write_line(message, LogLevel.DEBUG)


def info(message: str) -> None:
    write_line(message, LogLevel.INFO)


def warning(message: str) -> None:
    write_line(message, LogLevel.WARNING)


def error(message: str) -> None:
    write_line(message, LogLevel.ERROR)
