#!/usr/bin/env bash
# Simple environment setup script for SpeakToMe
# Creates a virtual environment and installs required packages

set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Environment setup complete. Activate with 'source .venv/bin/activate'."
