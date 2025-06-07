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
