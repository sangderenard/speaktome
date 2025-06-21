#!/usr/bin/env python3
"""Update all git submodules recursively and verify cleanliness.

This utility mirrors the logic embedded in the archived git hooks. It calls
``git submodule update --init --recursive`` to fetch and initialize each
submodule. Afterwards it checks for uncommitted changes using
``git submodule foreach``. Dirty modules are printed so that automation may
abort or alert a maintainer.
"""
from __future__ import annotations

import subprocess
import sys


def update_submodules() -> int:
    """Run git submodule update and return the subprocess return code."""
    result = subprocess.run(
        ["git", "submodule", "update", "--init", "--recursive"]
    )
    return result.returncode


def list_dirty_submodules() -> list[str]:
    """Return a list of submodules with uncommitted changes."""
    cmd = [
        "bash",
        "-c",
        "git submodule foreach --quiet 'git diff --quiet || echo $name'",
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return [line.strip() for line in proc.stdout.splitlines() if line.strip()]


def main() -> int:
    rc = update_submodules()
    if rc != 0:
        print("submodule update failed", file=sys.stderr)
        return rc
    dirty = list_dirty_submodules()
    if dirty:
        print(
            "dirty submodules detected:\n" + "\n".join(dirty),
            file=sys.stderr,
        )
        return 1
    print("submodules clean and up to date")
    return 0


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
