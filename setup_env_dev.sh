#!/usr/bin/env bash
# Developer-oriented environment setup
# Runs standard setup then dumps headers, stubs, and key documents

set -uo pipefail

# Resolve repository root so we can reliably access the venv even after
# changing directories with pushd. This allows the script to be invoked
# from anywhere while still locating `.venv`.
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export SPEAKTOME_ENV_SETUP_BOX="\n+----------------------------------------------------------------------+\n| Imports failed. See ENV_SETUP_OPTIONS.md for environment guidance.  |\n| Missing packages usually mean setup was skipped or incomplete.      |\n+----------------------------------------------------------------------+\n"
ACTIVE_FILE=${SPEAKTOME_ACTIVE_FILE:-/tmp/speaktome_active.json}
export SPEAKTOME_ACTIVE_FILE="$ACTIVE_FILE"
MENU_ARGS=()

safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

# Run the regular setup script (this creates the venv)
USE_VENV=1
TORCH_CHOICE=""
for arg in "$@"; do
  arg_lc="${arg,,}"
  case $arg_lc in
    -no-venv) USE_VENV=0 ;;
    -codebases=*|-cb=*) MENU_ARGS+=("-codebases" "${arg#*=}") ;;
    -groups=*|-grp=*)   MENU_ARGS+=("-groups" "${arg#*=}") ;;
    -torch) TORCH_CHOICE="cpu" ;;
    -gpu|-gpu-torch) TORCH_CHOICE="gpu" ;;
  esac
done
POETRY_ARGS="--without cpu-torch --without gpu-torch"
if [ "$TORCH_CHOICE" = "cpu" ]; then
  POETRY_ARGS="--with cpu-torch"
elif [ "$TORCH_CHOICE" = "gpu" ]; then
  POETRY_ARGS="--with gpu-torch"
fi
export SPEAKTOME_POETRY_ARGS="$POETRY_ARGS"

ARGS=()
for arg in "$@"; do
  case "${arg,,}" in
    -torch|-gpu|-gpu-torch) ;;
    *) ARGS+=("$arg") ;;
  esac
done

safe_run bash "$SCRIPT_ROOT/setup_env.sh" "${ARGS[@]}" --from-dev

# Define the venv Python path (assumes setup_env.sh created it at .venv)
if [ $USE_VENV -eq 1 ]; then
  VENV_PYTHON="$SCRIPT_ROOT/.venv/bin/python"
  VENV_PIP="$SCRIPT_ROOT/.venv/bin/pip"
else
  VENV_PYTHON="python"
  VENV_PIP="pip"
fi

# Fail fast if venv not created
if [ $USE_VENV -eq 1 ]; then
  if [ ! -x "$VENV_PYTHON" ]; then
    echo "Error: Virtual environment Python not found at $VENV_PYTHON" >&2
    # Allow script to continue, subsequent VENV_PYTHON commands will be caught by safe_run
  fi
  if [ ! -x "$VENV_PIP" ]; then
    echo "Error: Virtual environment Pip not found at $VENV_PIP. Ensure setup_env.sh created the .venv correctly." >&2
    # Allow script to continue, subsequent VENV_PIP commands will be handled or fail gracefully
  fi

  # Activate the virtual environment
  VENV_ACTIVATE="$SCRIPT_ROOT/.venv/bin/activate"
  if [ -f "$VENV_ACTIVATE" ]; then
      # shellcheck disable=SC1090
      source "$VENV_ACTIVATE"
  else
      echo "Error: Virtual environment activation script not found at $VENV_ACTIVATE" >&2
      exit 1
  fi

  # Dev dependencies installed via poetry

fi


# Interactive document menu with inactivity timeout
dev_menu() {
  local TIMEOUT=5 choice
  while true; do
    echo "\nDeveloper info menu (timeout ${TIMEOUT}s):"
    echo " 0) Dump headers"
    echo " 1) Stub finder"
    echo " 2) List contributors"
    echo " 3) Preview AGENT_CONSTITUTION.md"
    echo " 4) Preview AGENTS.md"
    echo " 5) Preview LICENSE"
    echo " 6) Preview CODING_STANDARDS.md"
    echo " 7) Preview CONTRIBUTING.md"
    echo " 8) Preview PROJECT_OVERVIEW.md"
    echo " 9) Launch dev group menu"
    echo " q) Quit"
    read -r -t "$TIMEOUT" -p "Select option: " choice || { echo "No input in ${TIMEOUT}s. Exiting."; break; }
    case $choice in
      0) echo "Running dump_headers"; safe_run "$VENV_PYTHON" AGENTS/tools/dump_headers.py speaktome --markdown; TIMEOUT=10 ;;
      1) echo "Running stubfinder"; safe_run "$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome; TIMEOUT=10 ;;
      2) echo "Running list_contributors"; safe_run "$VENV_PYTHON" AGENTS/tools/list_contributors.py; TIMEOUT=10 ;;
      3) echo "Preview AGENT_CONSTITUTION.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md; TIMEOUT=10 ;;
      4) echo "Preview AGENTS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS.md; TIMEOUT=10 ;;
      5) echo "Preview LICENSE"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py LICENSE; TIMEOUT=10 ;;
      6) echo "Preview CODING_STANDARDS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md; TIMEOUT=10 ;;
      7) echo "Preview CONTRIBUTING.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md; TIMEOUT=10 ;;
      8) echo "Preview PROJECT_OVERVIEW.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md; TIMEOUT=10 ;;
      9) echo "Launching dev group menu"; safe_run "$VENV_PYTHON" AGENTS/tools/dev_group_menu.py --record "$SPEAKTOME_ACTIVE_FILE" "${MENU_ARGS[@]}"; TIMEOUT=10 ;;
      q|Q) echo "Exiting."; break ;;
      *) echo "Unknown choice: $choice" ;;
    esac
  done
}
dev_menu
echo "For advanced codebase/group selection, run: python AGENTS/tools/dev_group_menu.py"
echo "Selections recorded to $SPEAKTOME_ACTIVE_FILE"

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

# All options for this script should be used with double-dash GNU-style flags, e.g.:
#   --torch --no-venv --codebases=projectA,projectB --groups=groupX
# Use --torch or --gpu to request torch. If omitted, torch is skipped.
# Do not use single-dash flags with this script.
