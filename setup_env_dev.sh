#!/usr/bin/env bash
# Developer-oriented environment setup
# Runs standard setup then dumps headers, stubs, and key documents

set -euo pipefail

# Run the regular setup script (this creates the venv)
bash setup_env.sh "$@"

# Define the venv Python path (assumes setup_env.sh created it at .venv)
VENV_PYTHON="./.venv/bin/python"

# Fail fast if venv not created
if [ ! -x "$VENV_PYTHON" ]; then
  echo "Error: Virtual environment Python not found at $VENV_PYTHON"
  exit 1
fi

# Run Python commands using the venv's interpreter
"$VENV_PYTHON" dump_headers.py speaktome --markdown
"$VENV_PYTHON" AGENTS/tools/stubfinder.py speaktome
"$VENV_PYTHON" AGENTS/tools/list_contributors.py

# Display important documentation
cat AGENTS/AGENT_CONSTITUTION.md
cat AGENTS.md

# Show license
cat LICENSE

# Show coding standards
cat AGENTS/CODING_STANDARDS.md
cat AGENTS/CONTRIBUTING.md

# Show project overview
cat AGENTS/PROJECT_OVERVIEW.md
