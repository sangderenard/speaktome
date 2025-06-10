#!/usr/bin/env bash
# ------------------------------------------------------------------------
# setup_env.sh -- no Unicode, all pip/python in venv unless --no-venv
# ------------------------------------------------------------------------

set -uo pipefail

# Resolve repository root so this script works from any directory
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

ACTIVE_FILE=${SPEAKTOME_ACTIVE_FILE:-/tmp/speaktome_active.json}
export SPEAKTOME_ACTIVE_FILE="$ACTIVE_FILE"

USE_VENV=1
for arg in "$@"; do
  case $arg in
    --no-venv) USE_VENV=0 ;;
  esac
done

# Helper: run a command but never terminate on failure
safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

# Attempt to install a package quietly. If the install outputs data within
# $TIMEOUT seconds the captured output is streamed. Otherwise the process is
# terminated and the log is displayed with a warning.
install_quiet() {
  local TIMEOUT=60
  local log
  log="$(mktemp)"
  ("$@" >"$log" 2>&1) &
  local pid=$!
  local started=0
  local i=0
  while kill -0 "$pid" 2>/dev/null; do
    sleep 1
    i=$((i + 1))
    if [ $started -eq 0 ] && [ -s "$log" ]; then
      started=1
      cat "$log"
      tail -f "$log" --pid="$pid" &
      tail_pid=$!
    fi
    if [ $i -ge $TIMEOUT ] && [ $started -eq 0 ]; then
      kill "$pid" 2>/dev/null
      wait "$pid" 2>/dev/null
      echo "Optional Torch download did not succeed, continuing anyway." >&2
      cat "$log"
      rm -f "$log"
      return 0
    fi
  done
  wait "$pid"
  local status=$?
  if [ $started -eq 1 ]; then
    kill "$tail_pid" 2>/dev/null
    wait "$tail_pid" 2>/dev/null
  else
    cat "$log"
  fi
  rm -f "$log"
  if [ $status -ne 0 ]; then
    echo "Optional Torch download did not succeed, continuing anyway." >&2
  fi
  return 0
}

if [ $USE_VENV -eq 1 ]; then
  safe_run python -m venv .venv
  source .venv/bin/activate
  VENV_PYTHON="./.venv/bin/python"
  VENV_PIP="./.venv/bin/pip"
else
  VENV_PYTHON="python"
  VENV_PIP="pip"
fi

safe_run $VENV_PYTHON -m pip install --upgrade pip

NOEXTRAS=0
ML=0        # flag for full ML extras (transformers, torch_geometric)
FORCE_GPU=0
PREFETCH=0
CODEBASES="speaktome,AGENTS/tools,time_sync"

for arg in "$@"; do
  case $arg in
    --noextras)       NOEXTRAS=1 ;;
    --ml)             ML=1      ;;  # install ML extras too
    --gpu)            FORCE_GPU=1 ;;
    --prefetch)       PREFETCH=1 ;;
    --codebases=*)    CODEBASES="${arg#*=}" ;;
  esac
done


# Always install torch first for GPU safety
if [ "${GITHUB_ACTIONS:-}" = "true" ]; then
    echo "Installing latest stable CPU-only torch (CI environment)"
    install_quiet "$VENV_PIP" install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
elif [ $FORCE_GPU -eq 1 ]; then
    echo "Installing GPU-enabled torch"
    install_quiet "$VENV_PIP" install torch -f https://download.pytorch.org/whl/cu118
else
    echo "Installing latest stable CPU-only torch (default)"
    install_quiet "$VENV_PIP" install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
fi

# If not called from a dev script, launch the dev menu for all codebase/group installs
CALLED_BY_DEV=0
for arg in "$@"; do
  case $arg in
    --from-dev) CALLED_BY_DEV=1 ;;
  esac
done

if [ $CALLED_BY_DEV -eq 0 ]; then
  echo "Launching codebase/group selection tool for editable installs..."
  PIP_CMD="$VENV_PIP" "$VENV_PYTHON" "$SCRIPT_ROOT/AGENTS/tools/dev_group_menu.py" --install --record "$SPEAKTOME_ACTIVE_FILE"
fi

echo "Environment setup complete."
echo "[OK] Environment ready. Activate with 'source .venv/bin/activate'."
TORCH_INFO=$($VENV_PYTHON - <<'PY'
import importlib, sys
spec = importlib.util.find_spec("torch")
if spec is None:
    sys.exit(0)
import torch
cuda = torch.version.cuda if torch.cuda.is_available() else "CPU"
print(f"{torch.__version__} {cuda}")
PY
)
echo "   * Torch = ${TORCH_INFO:-missing}"
echo "Selections recorded to $SPEAKTOME_ACTIVE_FILE"
