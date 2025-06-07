#!/usr/bin/env bash
# Simple environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

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

pip install -r requirements.txt -r requirements-dev.txt
if [[ $EXTRAS -eq 1 ]]; then
    pip install -r optional_requirements.txt
fi

if [[ $PREFETCH -eq 1 ]]; then
    bash fetch_models.sh
fi

echo "Environment setup complete. Activate with 'source .venv/bin/activate'."
echo
echo "🧑‍💻 Developer Workflow & Testing:"
echo "  1. After activation (source .venv/bin/activate), run 'pytest -v'"
echo "  2. Check the latest log in 'testing/logs/pytest_*.log'"
echo "     The log header explains the testing strategy and faculty system."
echo "  3. Address the first reported FAILED test or critical ERROR."
echo "  4. Consult 'testing/stub_todo.txt' for pending stub tests."
echo "  5. Repeat this cycle until all tests pass and stubs are resolved."
echo
