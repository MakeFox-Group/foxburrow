#!/usr/bin/env bash
#
# foxburrow.sh — Create/update venv, install deps, start FoxBurrow server.
#
# Usage:  ./foxburrow.sh          (foreground)
#         ./foxburrow.sh --bg     (background, logs to logs/foxburrow.log)
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_DIR="$PROJECT_ROOT/.venv"
SRC_DIR="$PROJECT_ROOT/src"
REQ_FILE="$PROJECT_ROOT/requirements.txt"
STAMP_FILE="$VENV_DIR/.requirements.stamp"
LOG_DIR="$PROJECT_ROOT/logs"
LOG_FILE="$LOG_DIR/foxburrow.log"

cd "$SRC_DIR"

# ── Check Python version ────────────────────────────────────────────
if ! python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)" 2>/dev/null; then
    echo "ERROR: Python 3.11+ required." >&2
    exit 1
fi

# ── Create venv if missing ──────────────────────────────────────────
if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python venv..."
    python3 -m venv "$VENV_DIR"
fi

# ── Install/update packages if requirements.txt changed ─────────────
req_hash="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
old_hash=""
[ -f "$STAMP_FILE" ] && old_hash="$(cat "$STAMP_FILE")"

if [ "$req_hash" != "$old_hash" ]; then
    echo "Installing/updating packages..."
    if "$VENV_DIR/bin/pip" install --upgrade pip && \
       "$VENV_DIR/bin/pip" install -r "$REQ_FILE"; then
        echo "$req_hash" > "$STAMP_FILE"
        echo "Packages up to date."
    else
        echo "ERROR: pip install failed." >&2
        exit 1
    fi
else
    echo "Packages already up to date."
fi

# ── Kill existing instance if running ───────────────────────────────
PID_FILE="$PROJECT_ROOT/data/foxburrow.pid"
if [ -f "$PID_FILE" ]; then
    old_pid="$(cat "$PID_FILE")"
    if kill -0 "$old_pid" 2>/dev/null; then
        echo "Stopping existing foxburrow (PID $old_pid)..."
        kill "$old_pid"
        # Wait up to 5 seconds for clean shutdown
        for i in $(seq 1 50); do
            kill -0 "$old_pid" 2>/dev/null || break
            sleep 0.1
        done
        # Force kill if still alive
        if kill -0 "$old_pid" 2>/dev/null; then
            echo "Force-killing PID $old_pid..."
            kill -9 "$old_pid" 2>/dev/null || true
        fi
    fi
fi

# ── Launch ──────────────────────────────────────────────────────────
if [ "${1:-}" = "--bg" ]; then
    mkdir -p "$LOG_DIR"
    echo "Starting foxburrow in background (log: $LOG_FILE)..."
    nohup "$VENV_DIR/bin/python" main.py > "$LOG_FILE" 2>&1 &
    disown
    echo "PID: $!"
else
    echo "Starting foxburrow..."
    exec "$VENV_DIR/bin/python" main.py
fi
