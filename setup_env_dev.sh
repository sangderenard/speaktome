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
safe_run bash "$SCRIPT_ROOT/setup_env.sh" "$@"

# Define the venv Python path (assumes setup_env.sh created it at .venv)
VENV_PYTHON="$SCRIPT_ROOT/.venv/bin/python"
VENV_PIP="$SCRIPT_ROOT/.venv/bin/pip"

# Fail fast if venv not created
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: Virtual environment Python not found at $VENV_PYTHON" >&2
  # Allow script to continue, subsequent VENV_PYTHON commands will be caught by safe_run
fi
if [ ! -x "$VENV_PIP" ]; then
  echo "Error: Virtual environment Pip not found at $VENV_PIP. Ensure setup_env.sh created the .venv correctly." >&2
  # Allow script to continue, subsequent VENV_PIP commands will be handled or fail gracefully
fi


# Explicitly set Python environment variables so all commands run against the
# virtual environment without relying on activation scripts.
PY_VERSION="$($VENV_PYTHON -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')"
export PYTHONHOME="$SCRIPT_ROOT/.venv"
export PYTHONPATH="$SCRIPT_ROOT/.venv/lib/python${PY_VERSION}/site-packages"



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

  echo "Attempting to install SpeakToMe in editable mode..."
  if ! "$VENV_PIP" install -e .; then
    echo "Warning: Failed to install SpeakToMe in editable mode." >&2
  fi

  OPTIONAL_GROUPS=("plot" "ml" "dev")
  for group in "${OPTIONAL_GROUPS[@]}"; do
    read -t 3 -p "Install optional group '$group'? [y/N] (auto-skip in 3s): " reply
    reply=${reply:-N}
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      echo "Attempting to install optional group: $group"
      if ! "$VENV_PIP" install ".[${group}]" ; then
        echo "Warning: Failed to install optional group: $group" >&2
      fi
    else
      echo "Skipping optional group: $group"
    fi
  done

  BACKEND_GROUPS=("numpy" "jax" "ctensor")
  for group in "${BACKEND_GROUPS[@]}"; do
    read -t 3 -p "Install backend group '$group'? Install if you plan to run the corresponding abstract tensor backend. [y/N] (auto-skip in 3s): " reply
    reply=${reply:-N}
    if [[ "$reply" =~ ^[Yy]$ ]]; then
      echo "Attempting to install backend group: $group"
      if ! "$VENV_PIP" install ".[${group}]" ; then
        echo "Warning: Failed to install backend group: $group" >&2
      fi
    else
      echo "Skipping backend group: $group"
    fi
  done

  # Restore previous directory
  cd "$OLD_DIR"
}
install_speaktome_extras

# Interactive document menu with inactivity timeout
dev_menu() {
  local TIMEOUT=60 choice
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
      1) echo "Running dump_headers"; safe_run "$VENV_PYTHON" AGENTS/tools/dump_headers.py speaktome --markdown ;;
      2) echo "Running stubfinder"; safe_run "$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome ;;
      3) echo "Running list_contributors"; safe_run "$VENV_PYTHON" AGENTS/tools/list_contributors.py ;;
      4) echo "Preview AGENT_CONSTITUTION.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md ;;
      5) echo "Preview AGENTS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS.md ;;
      6) echo "Preview LICENSE"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py LICENSE ;;
      7) echo "Preview CODING_STANDARDS.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md ;;
      8) echo "Preview CONTRIBUTING.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md ;;
      9) echo "Preview PROJECT_OVERVIEW.md"; safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md ;;
      q|Q) echo "Exiting."; break ;;
      *) echo "Unknown choice: $choice" ;;
    esac
  done
}
dev_menu
