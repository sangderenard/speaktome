#!/usr/bin/env bash
# Automated reinstall and demo run for SpeakToMe
# This script performs a non-interactive reinstall and runs a few example commands.
set -euo pipefail

# Reinstall environment
bash reinstall_env.sh -y

# First demo run (non-interactive)
bash run.sh -s "Automation demo" -m 5 --auto_expand 2 -x

# Second demo run with visualization
bash run.sh -s "Visualization demo" -m 5 --final_viz -x

# Optional interactive run if user passes --interactive
if [[ "${1:-}" == "--interactive" ]]; then
    bash run.sh -s "Interactive session" -c
fi
