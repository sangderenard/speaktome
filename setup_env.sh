#!/usr/bin/env bash
# ------------------------------------------------------------------------
# setup_env.sh -- no Unicode, all pip/python in venv unless --no-venv
# ------------------------------------------------------------------------

set -uo pipefail

# Initialize argument-related variables early so they exist when referenced
CODEBASES=""
GROUPS=()
MENU_ARGS=()


# Resolve repository root so this script works from any directory
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MAP_FILE="$SCRIPT_ROOT/AGENTS/codebase_map.json"

# Provide ENV_SETUP_BOX for modules that fail to import early
ENV_BOX_FILE="$SCRIPT_ROOT/ENV_SETUP_BOX.md"
ENV_SETUP_BOX="\n$(cat "$ENV_BOX_FILE")\n"
export ENV_SETUP_BOX

ACTIVE_FILE=${SPEAKTOME_ACTIVE_FILE:-/tmp/speaktome_active.json}
export SPEAKTOME_ACTIVE_FILE="$ACTIVE_FILE"

# Always operate from the repository root so the virtual environment is
# created at the top level regardless of the caller's working directory.
cd "$SCRIPT_ROOT"

USE_VENV=1
# Ensure the Poetry build backend is available
if ! python - <<'PY' 2>/dev/null
import importlib.util, sys
sys.exit(0 if importlib.util.find_spec("poetry.core.masonry.api") else 1)
PY
then
  echo "[INFO] poetry-core missing; installing to enable editable builds" >&2
  python -m pip install -q --user 'poetry-core>=1.5.0' || {
    echo "[WARN] Automatic poetry-core install failed; please install it manually" >&2
  }
fi
for arg in "$@"; do
  arg_lc="${arg,,}"
  case $arg_lc in
    -no-venv) USE_VENV=0 ;;
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
  # Use the repo's central virtual environment ('.venv' in repository root)
  # without modifying global Poetry config.
  INSTALL_ARGS="${SPEAKTOME_POETRY_ARGS:---without cpu-torch --without gpu-torch}"
  if [[ "$INSTALL_ARGS" == *"--with"* && "$INSTALL_ARGS" != *"--without"* ]]; then
    echo "[INFO] Torch groups requested; attempting install" >&2
    install_quiet env POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --sync --no-interaction $INSTALL_ARGS
  else
    echo "[INFO] Skipping torch groups" >&2
    safe_run env POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --sync --no-interaction $INSTALL_ARGS
  fi
  VENV_PYTHON="./.venv/bin/python"
  VENV_PIP="./.venv/bin/pip"
else
  VENV_PYTHON="python"
  VENV_PIP="pip"
fi

for arg in "$@"; do
  arg_lc="${arg,,}"
  case $arg_lc in
    -codebases=*|-cb=*) CODEBASES="${arg#*=}" ;;
    -groups=*|-grp=*)   GROUPS+=("${arg#*=}") ;;
  esac
done

[ -n "$CODEBASES" ] && MENU_ARGS+=("-codebases" "$CODEBASES")
for g in "${GROUPS[@]}"; do
  MENU_ARGS+=("-groups" "$g")
done
echo "[DEBUG] Codebases: ${CODEBASES:-}" >&2
echo "[DEBUG] Groups: ${GROUPS[*]:-}" >&2


# If not called from a dev script, launch the dev menu for all codebase/group installs
CALLED_BY_DEV=0
for arg in "$@"; do
  case $arg in
    --from-dev) CALLED_BY_DEV=1 ;;
  esac
done

# Always run dev_group_menu.py at the end
if [ $CALLED_BY_DEV -eq 1 ]; then
  # Install AGENTS/tools without prompts
  echo "Installing AGENTS/tools..."
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

def _find_repo_root(start: pathlib.Path) -> pathlib.Path:
    current = start.resolve()
    required = {
        "speaktome",
        "laplace",
        "tensorprinting",
        "timesync",
        "AGENTS",
        "fontmapper",
        "tensors",
    }
    for parent in [current, *current.parents]:
        if all((parent / name).exists() for name in required):
            return parent
    return current

map_file, active_file = sys.argv[1:3]
try:
    mapping = json.load(open(map_file))
    active = json.load(open(active_file))
except Exception:
    sys.exit(0)

root = _find_repo_root(pathlib.Path(map_file))
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

# Mark the environment so pytest knows setup completed with at least one codebase
PYTEST_MARKER="$SCRIPT_ROOT/.venv/pytest_enabled"
if [ -f "$SPEAKTOME_ACTIVE_FILE" ]; then
  if "$VENV_PYTHON" - <<'PY'
import json, os, sys
path = os.environ.get("SPEAKTOME_ACTIVE_FILE")
try:
    data = json.load(open(path))
    if data.get("codebases"):
        sys.exit(0)
except Exception:
    pass
sys.exit(1)
PY
  then
    touch "$PYTEST_MARKER"
  else
    rm -f "$PYTEST_MARKER"
    echo "Warning: No codebases recorded; pytest will remain disabled." >&2
  fi
else
  rm -f "$PYTEST_MARKER"
  echo "Warning: Active selection file not found; pytest will remain disabled." >&2
fi
