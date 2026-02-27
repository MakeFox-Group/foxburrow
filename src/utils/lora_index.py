"""LoRA discovery, indexing, and background fingerprinting."""

from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import log
from api.websocket import streamer
from utils import fingerprint

# Supported LoRA file extensions
_LORA_EXTENSIONS = frozenset({".safetensors", ".pt", ".pth", ".ckpt", ".bin"})


@dataclass
class LoraEntry:
    """A discovered LoRA file."""
    name: str              # filename without extension (lookup key)
    path: str              # absolute path
    fingerprint: str | None  # SHA256, None if not yet computed
    size_bytes: int
    mtime: float


def discover_loras(loras_dir: str) -> dict[str, LoraEntry]:
    """Scan loras_dir recursively and build name -> LoraEntry index.

    LoRA name = filename without extension (A1111/Forge behavior).
    Duplicate filenames across subdirectories: use the most recently modified.
    Fingerprints are populated from cache hits only; misses are left as None.

    Returns the index immediately (names available for /api/status).
    """
    if not os.path.isdir(loras_dir):
        log.warning(f"  LoRA directory not found: {loras_dir}")
        return {}

    index: dict[str, LoraEntry] = {}
    duplicates: dict[str, list[str]] = {}  # name -> [paths] for warnings

    for dirpath, _dirnames, filenames in os.walk(loras_dir):
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in _LORA_EXTENSIONS:
                continue

            full_path = os.path.join(dirpath, fname)
            name = os.path.splitext(fname)[0]

            try:
                st = os.stat(full_path)
            except OSError:
                continue

            entry = LoraEntry(
                name=name,
                path=full_path,
                fingerprint=fingerprint.try_get_cached(full_path),
                size_bytes=st.st_size,
                mtime=st.st_mtime,
            )

            if name in index:
                # Track duplicates for warning
                if name not in duplicates:
                    duplicates[name] = [index[name].path]
                duplicates[name].append(full_path)
                # Keep the most recently modified
                if entry.mtime > index[name].mtime:
                    index[name] = entry
            else:
                index[name] = entry

    # Log duplicate warnings
    for name, paths in duplicates.items():
        log.warning(f"  LoRA duplicate: '{name}' found in {len(paths)} locations, "
                    f"using most recent: {index[name].path}")

    return index


def rescan_loras(
    loras_dir: str,
    current_index: dict[str, LoraEntry],
) -> dict:
    """Rescan loras_dir and update current_index in-place.

    Adds new LoRAs, removes deleted ones, updates changed files.
    Kicks off background hashing for any new entries missing fingerprints.
    Fires WebSocket events automatically:
      - ``model_removed`` for each deleted LoRA
      - ``model_discovered`` for each new/updated LoRA (fingerprint may be null)
      - ``lora_scan_complete`` with the overall summary

    Returns a summary dict with counts of added/removed/updated/unchanged.
    """
    fresh = discover_loras(loras_dir)

    old_names = set(current_index.keys())
    new_names = set(fresh.keys())

    added = new_names - old_names
    removed = old_names - new_names
    updated = 0

    # Remove deleted entries and notify
    for name in removed:
        entry = current_index.pop(name)
        streamer.fire_event("model_removed", {
            "model_type": "lora", "name": name, "path": entry.path})

    # Add new entries and notify
    for name in added:
        entry = fresh[name]
        current_index[name] = entry
        streamer.fire_event("model_discovered", {
            "model_type": "lora", "name": name, "path": entry.path,
            "fingerprint": entry.fingerprint,
            "size_bytes": entry.size_bytes})

    # Check for changes in existing entries (mtime or size changed)
    for name in old_names & new_names:
        old = current_index[name]
        new = fresh[name]
        if old.path != new.path or old.mtime != new.mtime or old.size_bytes != new.size_bytes:
            current_index[name] = new
            updated += 1
            streamer.fire_event("model_discovered", {
                "model_type": "lora", "name": name, "path": new.path,
                "fingerprint": new.fingerprint,
                "size_bytes": new.size_bytes})

    unchanged = len(old_names & new_names) - updated

    # Background-hash any new/updated entries missing fingerprints
    unhashed = [e for e in current_index.values() if e.fingerprint is None]
    if unhashed:
        start_background_hashing(current_index)

    summary = {
        "added": len(added),
        "removed": len(removed),
        "updated": updated,
        "unchanged": unchanged,
        "total": len(current_index),
    }

    log.info(f"  LoRA rescan: +{summary['added']} -{summary['removed']} "
             f"~{summary['updated']} ={summary['unchanged']} "
             f"(total: {summary['total']})")

    streamer.fire_event("lora_scan_complete", summary)

    return summary


def start_background_hashing(
    index: dict[str, LoraEntry],
    max_workers: int | None = None,
    pool: ThreadPoolExecutor | None = None,
    shutdown_pool: bool = False,
) -> threading.Thread:
    """Launch a background daemon thread that hashes all LoRAs with missing fingerprints.

    If *pool* is provided, uses that executor instead of creating one.
    Set *shutdown_pool* to ``True`` if this is the last consumer and should
    shut down the pool when done.

    Returns the daemon thread (for optional join).
    """
    unhashed = [e for e in index.values() if e.fingerprint is None]
    if not unhashed:
        log.debug(f"  LoRA hashing: all {len(index)} entries already cached")
        if shutdown_pool and pool is not None:
            pool.shutdown(wait=False)
        return _make_noop_thread()

    workers = min(max_workers or min(os.cpu_count() or 4, 16), len(unhashed))

    def _manager():
        if pool is not None:
            log.debug(f"  LoRA hashing: {len(unhashed)} files to hash (shared pool)")
        else:
            log.debug(f"  LoRA hashing: {len(unhashed)} files to hash "
                      f"({workers} workers)")
        completed = 0
        last_log = time.monotonic()
        last_completed = 0

        def _hash_one(entry: LoraEntry) -> tuple[LoraEntry, str | None]:
            try:
                fp = fingerprint.compute(entry.path)
                return entry, fp
            except Exception:
                return entry, None

        # Use caller's pool if provided; otherwise create a local one.
        own_pool: ThreadPoolExecutor | None = None
        use_pool = pool
        if use_pool is None:
            use_pool = ThreadPoolExecutor(max_workers=workers)
            own_pool = use_pool

        try:
            for entry, fp in use_pool.map(_hash_one, unhashed):
                if fp is not None:
                    entry.fingerprint = fp
                    # Re-announce with the computed fingerprint
                    streamer.fire_event("model_discovered", {
                        "model_type": "lora", "name": entry.name,
                        "path": entry.path, "fingerprint": fp,
                        "size_bytes": entry.size_bytes})
                completed += 1

                now = time.monotonic()
                # Log progress periodically
                if (now - last_log) >= 10:
                    elapsed = now - last_log
                    done_since = completed - last_completed
                    rate = done_since / elapsed if elapsed > 0 else 0
                    log.debug(f"  LoRA hashing: {completed}/{len(unhashed)} "
                              f"({rate:.0f}/sec)")
                    last_log = now
                    last_completed = completed
        finally:
            if own_pool is not None:
                own_pool.shutdown(wait=True)
            elif shutdown_pool:
                use_pool.shutdown(wait=True)

        log.debug(f"  LoRA hashing: complete ({completed}/{len(unhashed)} hashed)")

    t = threading.Thread(target=_manager, name="lora-hash-manager", daemon=True)
    t.start()
    return t


def _make_noop_thread() -> threading.Thread:
    """Return an already-finished daemon thread (for consistent API)."""
    t = threading.Thread(target=lambda: None, daemon=True)
    t.start()
    t.join()
    return t
