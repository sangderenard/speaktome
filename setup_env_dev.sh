#!/usr/bin/env bash
# Developer-oriented environment setup
# Runs standard setup then dumps headers, stubs, and key documents

set -uo pipefail

# Resolve repository root so we can reliably access the venv even after
# changing directories with pushd. This allows the script to be invoked
# from anywhere while still locating `.venv`.
SCRIPT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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
for arg in "$@"; do
  case $arg in
    --no-venv) USE_VENV=0 ;;
  esac
done

safe_run bash "$SCRIPT_ROOT/setup_env.sh" "$@" --from-dev

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
fi

install_speaktome_extras() {
  local SPEAKTOME_DIR="$SCRIPT_ROOT/speaktome"
  if [ ! -d "$SPEAKTOME_DIR" ]; then
    echo "Warning: SpeakToMe directory not found at $SPEAKTOME_DIR. Skipping extras installation." >&2
    return 0
  fi

  # Save current directory and cd into speaktome
  local OLD_DIR
  OLD_DIR="$(pwd)"
  cd "$SPEAKTOME_DIR"

  echo "Attempting to upgrade pip..."
  if ! "$VENV_PYTHON" -m pip install --upgrade pip; then
    echo "Warning: Failed to upgrade pip." >&2
  fi

  # Restore previous directory
  cd "$OLD_DIR"

  echo "Launching codebase/group selection tool..."
  PIP_CMD="$VENV_PIP" "$VENV_PYTHON" "$SCRIPT_ROOT/AGENTS/tools/dev_group_menu.py" --install
}
install_speaktome_extras

# Interactive document menu with inactivity timeout
dev_menu() {
  local TIMEOUT=5 choice
  while true; do
    echo "\nDeveloper info menu (timeout ${TIMEOUT}s):"
    echo " 1) Dump headers"
    echo " 2) Stub finder"
    echo " 3) List contributors"
    echo " 4) Preview AGENT_CONSTITUTION.md"
    echo " 5) Preview AGENTS.md"
    echo " 6) Preview LICENSE"
    echo " 7) Preview CODING_STANDARDS.md"
    echo " 8) Preview CONTRIBUTING.md"
    echo " 9) Preview PROJECT_OVERVIEW.md"
    echo " q) Quit"
    read -r -t "$TIMEOUT" -p "Select option: " choice || { echo "No input in ${TIMEOUT}s. Exiting."; break; }
    case $choice in
      1) echo "Running dump_headers"; safe_run "$VENV_PYTHON" AGENTS/tools/dump_headers.py speaktome --markdown; TIMEOUT=60 ;;
      2) echo "Running stubfinder"; safe_run "$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome; TIMEOUT=60 ;;
      3) echo "Running list_contributors"; safe_run "$VENV_PYTHON" AGENTS/tools/list_contributors.py; TIMEOUT=60 ;;
      4) echo "Preview AGENT_CONSTITUTION.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md; TIMEOUT=60 ;;
      5) echo "Preview AGENTS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS.md; TIMEOUT=60 ;;
      6) echo "Preview LICENSE"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py LICENSE; TIMEOUT=60 ;;
      7) echo "Preview CODING_STANDARDS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md; TIMEOUT=60 ;;
      8) echo "Preview CONTRIBUTING.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md; TIMEOUT=60 ;;
      9) echo "Preview PROJECT_OVERVIEW.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md; TIMEOUT=60 ;;
      q|Q) echo "Exiting."; break ;;
      *) echo "Unknown choice: $choice" ;;
    esac
  done
}
dev_menu
echo "For advanced codebase/group selection, run: python AGENTS/tools/dev_group_menu.py"
