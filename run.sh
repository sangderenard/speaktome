#!/usr/bin/env bash
# Entry point wrapper for SpeakToMe using the local virtual environment.

set -euo pipefail

VENV_PY=".venv/bin/python"
if [[ ! -x "$VENV_PY" ]]; then
  echo "Virtual environment not found. See ENV_SETUP_OPTIONS.md." >&2
  exit 1
fi

exec "$VENV_PY" -m speaktome.speaktome "$@"
