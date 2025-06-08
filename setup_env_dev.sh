#!/usr/bin/env bash
# Developer-oriented environment setup
# Runs standard setup then dumps headers, stubs, and key documents

set -uo pipefail

safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

# Run the regular setup script (this creates the venv)
safe_run bash setup_env.sh "$@"

# Define the venv Python path (assumes setup_env.sh created it at .venv)
VENV_PYTHON="./.venv/bin/python"

# Fail fast if venv not created
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: Virtual environment Python not found at $VENV_PYTHON" >&2
  return 0
fi

# Run Python commands using the venv's interpreter
safe_run "$VENV_PYTHON" dump_headers.py speaktome --markdown
safe_run "$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome
safe_run "$VENV_PYTHON" AGENTS/tools/list_contributors.py

# Display important documentation
safe_run cat AGENTS/AGENT_CONSTITUTION.md
safe_run cat AGENTS.md

# Show license
safe_run cat LICENSE

# Show coding standards
safe_run cat AGENTS/CODING_STANDARDS.md
safe_run cat AGENTS/CONTRIBUTING.md

# Show project overview
safe_run cat AGENTS/PROJECT_OVERVIEW.md
