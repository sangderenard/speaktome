#!/usr/bin/env bash
# ------------------------------------------------------------------------
# setup_env.sh -- no Unicode, all pip/python in venv unless --no-venv
# ------------------------------------------------------------------------

set -uo pipefail



# Resolve repository root so this script works from any directory
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="$SCRIPT_ROOT/AGENTS/codebase_map.json"

ACTIVE_FILE=${SPEAKTOME_ACTIVE_FILE:-/tmp/speaktome_active.json}
export SPEAKTOME_ACTIVE_FILE="$ACTIVE_FILE"

USE_VENV=1
HEADLESS=0
TORCH_CHOICE=""
CODEBASES=""
GROUPS=()
for arg in "$@"; do
  arg_lc="${arg,,}"
  case $arg_lc in
    -no-venv) USE_VENV=0 ;;
    -headless) HEADLESS=1 ;;
    -torch) TORCH_CHOICE="cpu" ;;
    -gpu|-gpu-torch) TORCH_CHOICE="gpu" ;;
    -notorch|-no-torch) TORCH_CHOICE="" ;;
  esac
done

# Auto-load codebases from map file when running headless
if [ -z "$CODEBASES" ] && [ $HEADLESS -eq 1 ] && [ -f "$MAP_FILE" ]; then
  CODEBASES="$(python - "$MAP_FILE" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print(",".join(d.keys()))
PY
)"
fi

[ -n "$CODEBASES" ] && MENU_ARGS+=("-codebases" "$CODEBASES")
for g in "${GROUPS[@]}"; do
  MENU_ARGS+=("-groups" "$g")
done
echo "[DEBUG] Codebases: ${CODEBASES:-}" >&2
echo "[DEBUG] Groups: ${GROUPS[*]:-}" >&2

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
safe_run $VENV_PYTHON -m pip install wheel
CODEBASES=""
GROUPS=()
MENU_ARGS=()

for arg in "$@"; do
  arg_lc="${arg,,}"
  case $arg_lc in
    -codebases=*|-cb=*) CODEBASES="${arg#*=}" ;;
    -groups=*|-grp=*)   GROUPS+=("${arg#*=}") ;;
  esac
done

if [ -n "$TORCH_CHOICE" ]; then
  if [ "$TORCH_CHOICE" = "gpu" ]; then
    echo "Installing torch with GPU support"
    install_quiet "$VENV_PIP" install torch==2.3.1
  else
    echo "Installing CPU-only torch"
    install_quiet "$VENV_PIP" install torch==2.3.1+cpu -f https://download.pytorch.org/whl/torch_stable.html
  fi
else
  echo "[INFO] Torch not requested; skipping installation."
fi

# If not called from a dev script, launch the dev menu for all codebase/group installs
CALLED_BY_DEV=0
for arg in "$@"; do
  case $arg in
    --from-dev) CALLED_BY_DEV=1 ;;
  esac
done

# Always run dev_group_menu.py at the end
if [ $CALLED_BY_DEV -eq 1 ]; then
  # Headless install of AGENTS/tools
  echo "Installing AGENTS/tools in headless mode..."
  PIP_CMD="$VENV_PIP" "$VENV_PYTHON" "$SCRIPT_ROOT/AGENTS/tools/dev_group_menu.py" --install --codebases tools --record "$SPEAKTOME_ACTIVE_FILE"
  # Then run again with any arguments passed
  echo "Launching codebase/group selection tool for editable installs (from-dev)..."
  PIP_CMD="$VENV_PIP" "$VENV_PYTHON" "$SCRIPT_ROOT/AGENTS/tools/dev_group_menu.py" --install --record "$SPEAKTOME_ACTIVE_FILE" "${MENU_ARGS[@]}"
else
  echo "Launching codebase/group selection tool for editable installs..."
  PIP_CMD="$VENV_PIP" "$VENV_PYTHON" "$SCRIPT_ROOT/AGENTS/tools/dev_group_menu.py" --install --record "$SPEAKTOME_ACTIVE_FILE" "${MENU_ARGS[@]}"
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

# Create `.venv` symlinks in selected codebases so editors opened
# in those directories automatically find the repo environment.
python - "$MAP_FILE" "$SPEAKTOME_ACTIVE_FILE" <<'PY'
import json, os, sys, pathlib

map_file, active_file = sys.argv[1:3]
try:
    mapping = json.load(open(map_file))
    active = json.load(open(active_file))
except Exception:
    sys.exit(0)

root = pathlib.Path(map_file).resolve().parents[1]
env_dir = root / '.venv'
for cb in active.get('codebases', []):
    info = mapping.get(cb, {})
    cb_path = root / info.get('path', cb)
    link = cb_path / '.venv'
    if not link.exists():
        try:
            link.symlink_to(os.path.relpath(env_dir, cb_path))
        except Exception as exc:
            print(f"[WARN] Could not link {link}: {exc}", file=sys.stderr)
PY
