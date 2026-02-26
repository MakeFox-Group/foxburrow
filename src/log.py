"""Thread-safe timestamped logging."""

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


def write_line(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    with _lock:
        print(line, flush=True)


def write(message: str, level: LogLevel = LogLevel.INFO) -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    line = f"[{timestamp}] [{level.value}] {message}"
    with _lock:
        print(line, end="", flush=True)


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
