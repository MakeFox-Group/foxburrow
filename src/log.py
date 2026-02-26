"""Thread-safe timestamped logging to stdout + file."""

import os
import sys
import threading
import traceback
from datetime import datetime
from enum import Enum


class LogLevel(Enum):
    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


_lock = threading.Lock()
_log_file = None
_log_path: str | None = None


def init_file(path: str) -> None:
    """Open the log file for appending. Creates parent directories if needed."""
    global _log_file, _log_path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    _log_file = open(path, "a", encoding="utf-8")
    _log_path = path


def get_log_path() -> str | None:
    """Return the active log file path, or None if file logging is not enabled."""
    return _log_path


def write_line(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    with _lock:
        print(line, flush=True)
        if _log_file is not None:
            _log_file.write(line + "\n")
            _log_file.flush()


def write(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    with _lock:
        print(line, end="", flush=True)
        if _log_file is not None:
            _log_file.write(line)
            _log_file.flush()


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
