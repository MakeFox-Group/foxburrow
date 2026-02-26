"""SHA256 file fingerprinting with per-directory JSONL cache.

Each directory containing model files gets a `.fpcache` file — a JSON Lines
file where each line is a JSON object:

    {"file": "name.safetensors", "size": 12345, "mtime": 1708900000.0, "sha256": "ab3f..."}

Keys are filenames (not full paths) since the cache lives in the same directory.
Appends are atomic up to PIPE_BUF (4KB on Linux); each line is ~150 bytes.
Corrupt lines are silently skipped (only that entry is lost).
"""

import hashlib
import json
import os
import threading

import log

_CACHE_FILENAME = ".fpcache"

# In-memory cache: directory path -> {filename: (mtime, size, sha256)}
_dir_caches: dict[str, dict[str, tuple[float, int, str]]] = {}
_lock = threading.Lock()


def _load_dir_cache(dir_path: str) -> dict[str, tuple[float, int, str]]:
    """Load and deduplicate a directory's .fpcache file.

    Returns {filename: (mtime, size, sha256)}. Last entry wins on duplicates.
    Corrupt lines are silently skipped.
    """
    cache_path = os.path.join(dir_path, _CACHE_FILENAME)
    entries: dict[str, tuple[float, int, str]] = {}

    try:
        with open(cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    entries[obj["file"]] = (
                        float(obj["mtime"]),
                        int(obj["size"]),
                        obj["sha256"],
                    )
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning(f"  Fingerprint cache: error reading {cache_path}: {e}")

    return entries


def _get_dir_cache(dir_path: str) -> dict[str, tuple[float, int, str]]:
    """Get the in-memory cache for a directory, loading from disk if needed."""
    with _lock:
        if dir_path not in _dir_caches:
            _dir_caches[dir_path] = _load_dir_cache(dir_path)
        return _dir_caches[dir_path]


def _append_to_cache(dir_path: str, filename: str, mtime: float, size: int, sha256: str) -> None:
    """Append a single entry to the directory's .fpcache file."""
    cache_path = os.path.join(dir_path, _CACHE_FILENAME)
    entry = json.dumps({
        "file": filename,
        "size": size,
        "mtime": mtime,
        "sha256": sha256,
    }, separators=(",", ":"))

    try:
        with open(cache_path, "a") as f:
            f.write(entry + "\n")
    except OSError as e:
        log.warning(f"  Fingerprint cache: error writing {cache_path}: {e}")


def compute(path: str) -> str:
    """Compute SHA256 fingerprint of a file, with per-directory JSONL caching.

    Checks the in-memory + on-disk cache first. On cache miss, hashes the file
    and appends the result to the directory's .fpcache file.
    """
    path = os.path.realpath(path)

    try:
        st = os.stat(path)
    except OSError:
        raise FileNotFoundError(f"Cannot stat file for fingerprinting: {path}")

    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)
    mtime = st.st_mtime
    size = st.st_size

    # Check cache
    dir_cache = _get_dir_cache(dir_path)
    with _lock:
        if filename in dir_cache:
            cached_mtime, cached_size, cached_fp = dir_cache[filename]
            if cached_mtime == mtime and cached_size == size:
                return cached_fp

    # Compute SHA256
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(1 << 20)  # 1MB chunks
            if not chunk:
                break
            sha.update(chunk)

    fingerprint = sha.hexdigest()

    # Update in-memory cache and append to disk
    with _lock:
        dir_cache[filename] = (mtime, size, fingerprint)

    _append_to_cache(dir_path, filename, mtime, size, fingerprint)

    return fingerprint


def try_get_cached(path: str) -> str | None:
    """Return cached fingerprint if mtime+size still match, else None."""
    path = os.path.realpath(path)
    try:
        st = os.stat(path)
    except OSError:
        return None

    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)

    dir_cache = _get_dir_cache(dir_path)
    with _lock:
        if filename in dir_cache:
            cached_mtime, cached_size, cached_fp = dir_cache[filename]
            if cached_mtime == st.st_mtime and cached_size == st.st_size:
                return cached_fp
    return None
