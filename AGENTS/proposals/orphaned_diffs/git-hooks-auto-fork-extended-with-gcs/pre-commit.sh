#!/bin/sh
echo "[HOOK] Checking submodule state before commit..."

# Pull latest submodule commits
git submodule update --init --recursive

# Detect dirty submodules
dirty=$(git submodule foreach --quiet 'git diff --quiet || echo $name')

if [ -n "$dirty" ]; then
  echo "[HOOK] Error: The following submodules have uncommitted changes:"
  echo "$dirty"
  echo "[HOOK] Please commit changes in submodules first."
  exit 1
fi

# Stage updated submodule pointers (if any changed)
git diff --quiet wheelhouse || git add wheelhouse
git diff --quiet wheelhouse/lfsavoider || git add wheelhouse/lfsavoider

echo "[HOOK] Submodule state is clean. Proceeding with commit."

if [ $? -ne 0 ]; then
    echo "Commit failed. Creating forked branch..."
    fork_branch="auto/fork/$(date +%Y%m%d)-$(git rev-parse --short HEAD)"
    git checkout -b "$fork_branch"
    git push -u origin "$fork_branch"
    echo "Changes pushed to forked branch: $fork_branch"
fi
