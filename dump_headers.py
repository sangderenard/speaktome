#!/usr/bin/env python
"""Dump headers of all Python files.

Recursively finds ``.py`` files under the given directory and prints
lines from each file up to the sentinel ``# --- END HEADER ---``.
This allows us to collect standard file headers for logging or
aggregation tasks.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

END_MARKER = "# --- END HEADER ---"


def find_py_files(root: Path) -> list[Path]:
    """Yield Python files under ``root`` ignoring ``.venv`` and ``.git``."""
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".git"} for part in path.parts):
            continue
        yield path


def dump_headers(root: Path) -> None:
    for pyfile in sorted(find_py_files(root)):
        print(f"===== {pyfile} =====")
        try:
            with pyfile.open("r", encoding="utf-8", errors="ignore") as fh:
                for line in fh:
                    print(line.rstrip())
                    if line.rstrip() == END_MARKER:
                        break
        except Exception as exc:  # noqa: BLE001
            print(f"[ERROR] {pyfile}: {exc}", file=sys.stderr)


if __name__ == "__main__":
    root_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
    dump_headers(root_arg.resolve())
