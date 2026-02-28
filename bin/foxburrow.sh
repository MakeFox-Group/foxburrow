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

MIN_PYTHON=(3 13)

cd "$PROJECT_ROOT"

# ── Find best Python ──────────────────────────────────────────────
# Prefer python3.13 (best SageAttention 2 / ecosystem support),
# fall back to system python3 if it meets the minimum version.
find_python() {
    local candidate version_ok
    for candidate in python3.13 python3; do
        if ! command -v "$candidate" &>/dev/null; then
            continue
        fi
        version_ok="$("$candidate" -c \
            "import sys; sys.exit(0 if sys.version_info >= (${MIN_PYTHON[0]}, ${MIN_PYTHON[1]}) else 1)" \
            2>/dev/null && echo yes || echo no)"
        if [ "$version_ok" = "yes" ]; then
            echo "$candidate"
            return 0
        fi
    done
    return 1
}

PYTHON="$(find_python)" || {
    echo "ERROR: Python ${MIN_PYTHON[0]}.${MIN_PYTHON[1]}+ required." >&2
    echo "  Searched for: python3.13, python3" >&2
    exit 1
}
PYTHON_VERSION="$("$PYTHON" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")"
echo "Using $PYTHON (Python $PYTHON_VERSION)"

# ── Create or recreate venv ───────────────────────────────────────
# Recreate if the venv's Python version doesn't match the selected one.
recreate_venv=false
if [ -d "$VENV_DIR" ]; then
    if [ -x "$VENV_DIR/bin/python" ]; then
        venv_version="$("$VENV_DIR/bin/python" -c \
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "")"
        if [ "$venv_version" != "$PYTHON_VERSION" ]; then
            echo "Python version changed ($venv_version -> $PYTHON_VERSION), recreating venv..."
            recreate_venv=true
        fi
    else
        recreate_venv=true
    fi
fi

if [ "$recreate_venv" = true ]; then
    rm -rf "$VENV_DIR"
fi

if [ ! -d "$VENV_DIR" ]; then
    echo "Creating Python $PYTHON_VERSION venv..."
    "$PYTHON" -m venv "$VENV_DIR"
fi

# ── PyTorch nightly (CUDA 13.0) ─────────────────────────────────────
# Torch, torchvision, and xformers come from PyTorch's nightly index
# (pre-release builds with Blackwell sm_120 support). These are managed
# separately from requirements.txt to avoid pip resolver conflicts.
TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu130"
TORCH_STAMP="$VENV_DIR/.torch.stamp"
TORCH_WANT="torch torchvision"
TORCH_HASH="$(echo "$TORCH_WANT $TORCH_INDEX" | sha256sum | cut -d' ' -f1)"
old_torch_hash=""
[ -f "$TORCH_STAMP" ] && old_torch_hash="$(cat "$TORCH_STAMP")"

if [ "$TORCH_HASH" != "$old_torch_hash" ]; then
    echo "Installing/updating PyTorch packages from nightly index..."
    if "$VENV_DIR/bin/pip" install --upgrade pip && \
       "$VENV_DIR/bin/pip" install --pre $TORCH_WANT --index-url "$TORCH_INDEX"; then
        echo "$TORCH_HASH" > "$TORCH_STAMP"
    else
        echo "ERROR: PyTorch nightly install failed." >&2
        exit 1
    fi
fi

# ── Install/update packages if requirements.txt changed ─────────────
req_hash="$(sha256sum "$REQ_FILE" | cut -d' ' -f1)"
old_hash=""
[ -f "$STAMP_FILE" ] && old_hash="$(cat "$STAMP_FILE")"

if [ "$req_hash" != "$old_hash" ]; then
    echo "Installing/updating packages..."

    # --no-deps: Install only the packages listed in requirements.txt
    # without resolving transitive dependencies. This prevents pip's
    # resolver from pulling in a different torch version through
    # dependency chains like accelerate → torch>=2.0.0.
    if "$VENV_DIR/bin/pip" install --no-deps -r "$REQ_FILE"; then
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
    nohup "$VENV_DIR/bin/python" src/main.py > "$LOG_FILE" 2>&1 &
    disown
    echo "PID: $!"
else
    echo "Starting foxburrow..."
    exec "$VENV_DIR/bin/python" src/main.py
fi
