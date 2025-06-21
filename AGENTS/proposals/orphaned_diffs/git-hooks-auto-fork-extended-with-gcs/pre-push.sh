#!/bin/sh
echo "[HOOK] Verifying clean submodules before push..."

# Ensure submodules are fully updated
git submodule update --init --recursive

# Abort push if submodules are dirty
dirty=$(git submodule foreach --quiet 'git diff --quiet || echo $name')

if [ -n "$dirty" ]; then
  echo "[HOOK] Error: The following submodules are dirty:"
  echo "$dirty"
  echo "[HOOK] Commit submodule changes before pushing."
  exit 1
fi

# Check if submodule pointers changed and need staging
if ! git diff --quiet wheelhouse; then
  echo "[HOOK] Updating submodule pointer: wheelhouse"
  git add wheelhouse
  git commit -m "Auto-commit: Update wheelhouse submodule pointer"
fi

if ! git diff --quiet wheelhouse/lfsavoider; then
  echo "[HOOK] Updating submodule pointer: lfsavoider"
  git add wheelhouse/lfsavoider
  git commit -m "Auto-commit: Update lfsavoider submodule pointer"
fi

echo "[HOOK] Submodules clean and pointers committed. Push allowed."

if [ $? -ne 0 ]; then
    echo "Push failed. Creating forked branch..."
    fork_branch="auto/fork/$(date +%Y%m%d)-$(git rev-parse --short HEAD)"
    git checkout -b "$fork_branch"
    git push -u origin "$fork_branch"
    echo "Changes pushed to forked branch: $fork_branch"
fi
