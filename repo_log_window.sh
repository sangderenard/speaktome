#!/bin/bash
# Print a rolling window of recent git commits.
# Usage: ./repo_log_window.sh [NUM_COMMITS]
NUM=${1:-20}
git --no-pager log -n "$NUM" --oneline --decorate
