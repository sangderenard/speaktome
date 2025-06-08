#!/usr/bin/env bash
# Scan the repository for files above a size threshold and add them to .gitignore.
# Usage: ./gitignore_large_files.sh [SIZE_MB]
set -euo pipefail

CAP_MB=${1:-100}
CAP_BYTES=$((CAP_MB * 1024 * 1024))
GITIGNORE=.gitignore

touch "$GITIGNORE"

# Find files larger than the cap, skipping the .git directory.
while IFS= read -r -d '' file; do
  rel=${file#./}
  if ! grep -Fxq "$rel" "$GITIGNORE"; then
    echo "$rel" >> "$GITIGNORE"
    echo "Added $rel to .gitignore"
  fi
done < <(find . -path ./.git -prune -o -type f -size +"${CAP_BYTES}"c -print0)
