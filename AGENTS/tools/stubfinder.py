#!/usr/bin/env python3
"""Locate high-visibility stub blocks in Python files.

Recursively searches the provided directories for comments containing
``STUB:``. Each discovered block is printed with a thin separator and
its file location.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# --- END HEADER ---

STUB_REGEX = re.compile(r"STUB:")


def extract_stubs(path: Path) -> list[tuple[int, list[str]]]:
    """Return ``(line_number, lines)`` for each stub block in ``path``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    stubs = []
    i = 0
    while i < len(lines):
        if STUB_REGEX.search(lines[i]):
            start = i
            block = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("#"):
                block.append(lines[i])
                i += 1
            stubs.append((start + 1, block))
        else:
            i += 1
    return stubs


def search_paths(paths: list[str]) -> list[tuple[Path, int, list[str]]]:
    """Gather stub blocks from all ``.py`` files under ``paths``."""
    results = []
    for base in paths:
        for file in Path(base).rglob("*.py"):
            if any(part in {".git", ".venv"} for part in file.parts):
                continue
            for lineno, block in extract_stubs(file):
                results.append((file, lineno, block))
    return results


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["."], help="directories to search recursively"
    )
    args = parser.parse_args(argv)

    stubs = search_paths(args.paths)
    for file, lineno, block in stubs:
        print("-" * 40)
        print(f"{file}:{lineno}")
        for line in block:
            print(line)
    if stubs:
        print("-" * 40)


if __name__ == "__main__":
    main()
