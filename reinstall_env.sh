#!/usr/bin/env bash
# Reinstall environment from scratch for SpeakToMe
# Removes the .venv directory and runs setup_env.sh

set -uo pipefail

safe_run() {
  "$@"
  local status=$?
  if [ $status -ne 0 ]; then
    echo "Warning: command '$*' failed with status $status" >&2
  fi
  return 0
}

FORCE=0
if [[ "${1:-}" == "-y" || "${1:-}" == "--yes" ]]; then
    FORCE=1
    shift
fi

if [[ $FORCE -eq 0 ]]; then
    read -r -p "This will delete the .venv directory and reinstall dependencies. Continue? [y/N] " confirm
    confirm=${confirm:-N}
    if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
        echo "Aborted."
        exit 0
    fi
fi

safe_run rm -rf .venv
safe_run bash setup_env.sh "$@"
