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
    read -t 3 -p "Install backend group '$group'? [y/N] (auto-skip in 3s): " reply
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

# Optionally run document dump with user confirmation and countdown
echo
read -t 10 -p "Run document dump (headers, stubs, docs)? [Y/n] (auto-yes in 10s): " docdump
docdump=${docdump:-Y}
if [[ "$docdump" =~ ^[Yy]$ ]]; then
  safe_run "$VENV_PYTHON" AGENTS/tools/dump_headers.py speaktome --markdown
  safe_run "$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome
  safe_run "$VENV_PYTHON" AGENTS/tools/list_contributors.py
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/AGENT_CONSTITUTION.md
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS.md
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py LICENSE
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CODING_STANDARDS.md
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/CONTRIBUTING.md
  safe_run "$VENV_PYTHON" AGENTS/tools/preview_doc.py AGENTS/PROJECT_OVERVIEW.md
else
  echo "Document dump skipped."
fi
