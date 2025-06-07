#!/usr/bin/env bash
# Simple environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

set -euo pipefail

# Find Python 3 interpreter
PYTHON=$(command -v python3 || command -v python || true)

if [ -z "${PYTHON}" ]; then
    echo "❌ Error: Python 3 is not installed or not found in PATH." >&2
    exit 1
fi

# Create venv if it doesn't exist
if [ ! -d ".venv" ]; then
    "$PYTHON" -m venv .venv
fi

# Activate the venv
source .venv/scripts/activate

# Upgrade pip
pip install --upgrade pip

# Parse arguments
EXTRAS=0
PREFETCH=0
for arg in "$@"; do
    case $arg in
        --extras|--full)
            EXTRAS=1
            ;;
        --prefetch)
            PREFETCH=1
            ;;
    esac
done

# Install core requirements
pip install -r requirements.txt -r requirements-dev.txt

# Optionally install extras
if [[ $EXTRAS -eq 1 ]]; then
    pip install -r optional_requirements.txt
fi

# Optionally prefetch models
if [[ $PREFETCH -eq 1 ]]; then
    bash fetch_models.sh
fi

# Final dev message
echo "✅ Environment setup complete. Activate with: source .venv/bin/activate"
echo
echo "🧑‍💻 Developer Workflow & Testing:"
echo "  1. After activation (source .venv/bin/activate), run 'pytest -v'"
echo "  2. Check the latest log in 'testing/logs/pytest_*.log'"
echo "     The log header explains the testing strategy and faculty system."
echo "  3. Address the first reported FAILED test or critical ERROR."
echo "  4. Consult 'testing/stub_todo.txt' for pending stub tests."
echo "  5. Repeat this cycle until all tests pass and stubs are resolved."
echo
