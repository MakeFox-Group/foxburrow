"""SHA256 file fingerprinting with per-directory JSONL cache.

Each directory containing model files gets a `.fpcache` file — a JSON Lines
file where each line is a JSON object:

    {"file": "name.safetensors", "size": 12345, "mtime": 1708900000.0, "sha256": "ab3f..."}

Extra metadata fields (e.g. ``lora_arch``) can be stored alongside the
fingerprint via :func:`set_extra`.  These are appended as additional JSONL
entries and merged on load.  When no metadata is cached for a file,
:func:`get_extra` returns ``None`` — callers should detect on first use
and populate the cache.

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

# In-memory cache: directory path -> {filename: {"mtime": ..., "size": ..., "sha256": ..., **extras}}
_dir_caches: dict[str, dict[str, dict]] = {}
_lock = threading.Lock()


def _load_dir_cache(dir_path: str) -> dict[str, dict]:
    """Load and deduplicate a directory's .fpcache file.

    Returns {filename: {field: value}}. Last entry per filename wins,
    with fields merged from all matching entries.
    Corrupt lines are silently skipped.
    """
    cache_path = os.path.join(dir_path, _CACHE_FILENAME)
    entries: dict[str, dict] = {}

    try:
        with open(cache_path, "r") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                    fname = obj["file"]
                    if fname in entries:
                        # Merge: new fields overwrite old ones
                        entries[fname].update(obj)
                    else:
                        entries[fname] = obj
                except (json.JSONDecodeError, KeyError, ValueError, TypeError):
                    continue
    except FileNotFoundError:
        pass
    except OSError as e:
        log.warning(f"  Fingerprint cache: error reading {cache_path}: {e}")

    return entries


def _get_dir_cache(dir_path: str) -> dict[str, dict]:
    """Get the in-memory cache for a directory, loading from disk if needed.

    Uses double-checked locking so the global lock isn't held during disk I/O.
    """
    with _lock:
        if dir_path in _dir_caches:
            return _dir_caches[dir_path]
    # Load from disk without the global lock
    loaded = _load_dir_cache(dir_path)
    with _lock:
        if dir_path not in _dir_caches:
            _dir_caches[dir_path] = loaded
        return _dir_caches[dir_path]


def _append_to_cache(dir_path: str, entry: dict) -> None:
    """Append a single JSONL entry to the directory's .fpcache file."""
    cache_path = os.path.join(dir_path, _CACHE_FILENAME)
    line = json.dumps(entry, separators=(",", ":"))

    try:
        with open(cache_path, "a") as f:
            f.write(line + "\n")
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
            cached = dir_cache[filename]
            if cached.get("mtime") == mtime and cached.get("size") == size:
                sha = cached.get("sha256")
                if sha:
                    return sha

    # Compute SHA256
    sha = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            chunk = f.read(8 * 1024 * 1024)  # 8MB chunks
            if not chunk:
                break
            sha.update(chunk)

    fp = sha.hexdigest()

    # Update in-memory cache and append to disk (under lock to prevent
    # interleaved writes from concurrent threads in the same directory)
    entry = {"file": filename, "size": size, "mtime": mtime, "sha256": fp}
    with _lock:
        if filename in dir_cache:
            dir_cache[filename].update(entry)
        else:
            dir_cache[filename] = entry
        _append_to_cache(dir_path, entry)

    return fp


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
            cached = dir_cache[filename]
            if cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size:
                return cached.get("sha256")
    return None


def get_extra(path: str, key: str) -> str | None:
    """Return a cached extra metadata field for a file, or None if not cached.

    The value is only returned if the file's mtime+size still match the cache
    (i.e., the file hasn't changed since the metadata was stored).
    Returns None if the file has changed, the key doesn't exist, or there's
    no cache entry at all — callers should detect and populate on first use.
    """
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
            cached = dir_cache[filename]
            if cached.get("mtime") == st.st_mtime and cached.get("size") == st.st_size:
                return cached.get(key)
    return None


def set_extra(path: str, **kwargs) -> None:
    """Store extra metadata fields for a file in the .fpcache.

    Example: ``set_extra("/path/to/lora.safetensors", lora_arch="sdxl")``

    The file's current mtime+size are recorded so stale entries are
    automatically invalidated when the file changes.
    """
    path = os.path.realpath(path)
    try:
        st = os.stat(path)
    except OSError:
        return

    dir_path = os.path.dirname(path)
    filename = os.path.basename(path)

    entry = {"file": filename, "mtime": st.st_mtime, "size": st.st_size, **kwargs}

    dir_cache = _get_dir_cache(dir_path)
    with _lock:
        if filename in dir_cache:
            dir_cache[filename].update(entry)
        else:
            dir_cache[filename] = entry
        _append_to_cache(dir_path, entry)
