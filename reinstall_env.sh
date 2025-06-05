#!/usr/bin/env bash
# Reinstall environment from scratch for SpeakToMe
# Removes the .venv directory and runs setup_env.sh

set -euo pipefail

read -r -p "This will delete the .venv directory and reinstall dependencies. Continue? [y/N] " confirm
confirm=${confirm:-N}
if [[ "$confirm" =~ ^[Yy]$ ]]; then
    rm -rf .venv
    bash setup_env.sh "$@"
else
    echo "Aborted."
fi
