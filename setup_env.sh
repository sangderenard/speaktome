#!/usr/bin/env bash
# Simple environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

if [[ "${1:-}" == "--prefetch" ]]; then
    bash fetch_models.sh
fi

echo "Environment setup complete. Activate with 'source .venv/bin/activate'."
