#!/usr/bin/env python3
"""Locate high‑visibility stub blocks in Python files and export them.

Recursively searches the provided directories for comments containing
``STUB:`` in the standard format defined by
``AGENTS/CODING_STANDARDS.md``. Each discovered block is written to the
``todo`` directory so agents can track outstanding work.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

# --- END HEADER ---

STUB_SPEC = {
    "fields": [
        "PURPOSE",
        "EXPECTED BEHAVIOR",
        "INPUTS",
        "OUTPUTS",
        "KEY ASSUMPTIONS/DEPENDENCIES",
        "TODO",
        "NOTES",
    ]
}

# Match the exact header line beginning a stub block.
STUB_START_REGEX = re.compile(r"^\s*#\s+##########\s+STUB:\s+(?P<name>.+?)\s+##########\s*$")
# Match the terminating line consisting solely of '# ' followed by many '#'.
STUB_END_REGEX = re.compile(r"^\s*#\s+#{60,}\s*$")

DEFAULT_OUTPUT = Path("todo")


def extract_stubs(path: Path) -> list[tuple[int, list[str]]]:
    """Return ``(line_number, lines)`` for each stub block in ``path``."""
    lines = path.read_text(encoding="utf-8").splitlines()
    stubs = []
    i = 0
    while i < len(lines):
        match = STUB_START_REGEX.match(lines[i])
        if match:
            start = i
            block = [lines[i]]
            i += 1
            while i < len(lines) and lines[i].lstrip().startswith("#"):
                block.append(lines[i])
                if STUB_END_REGEX.match(lines[i]):
                    i += 1
                    break
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
    return sort_results(results)


def sort_results(results: list[tuple[Path, int, list[str]]]) -> list[tuple[Path, int, list[str]]]:
    """Sort stubs by project relevance.

    Priority order:
    1. ``speaktome`` package
    2. ``tests``
    3. ``testing``
    4. ``AGENTS/tools``
    """

    def key(item: tuple[Path, int, list[str]]) -> tuple[int, str, int]:
        path, lineno, _ = item
        rel = str(path)
        if rel.startswith("./"):
            rel = rel[2:]
        if rel.startswith("speaktome/"):
            bucket = 0
        elif rel.startswith("tests/"):
            bucket = 1
        elif rel.startswith("testing/"):
            bucket = 2
        elif rel.startswith("AGENTS/tools/"):
            bucket = 3
        else:
            bucket = 4
        return (bucket, rel, lineno)

    return sorted(results, key=key)


def print_stub(path: Path, lineno: int, block: list[str]) -> None:
    """Pretty-print a stub block."""
    header = {
        "file": str(path),
        "line": lineno,
    }
    print(json.dumps(header) + "\n" + "\n".join(block) + "\n")


def write_stub_files(stubs: list[tuple[Path, int, list[str]]], output: Path) -> None:
    """Write stub blocks to ``output`` directory, one file per stub."""
    output.mkdir(exist_ok=True)
    for existing in output.glob("*.stub.md"):
        existing.unlink()
    for file, lineno, block in stubs:
        rel = file.resolve().relative_to(Path.cwd().resolve())
        name = str(rel).replace("/", "_").replace("\\", "_")
        stub_path = output / f"{name}_L{lineno}.stub.md"
        with stub_path.open("w", encoding="utf-8") as f:
            f.write(f"# Stub from {rel}:{lineno}\n\n")
            f.write("\n".join(block))
        print_stub(rel, lineno, block)
        print(f"Stub written to {stub_path}")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "paths", nargs="*", default=["."], help="directories to search recursively"
    )
    parser.add_argument(
        "--output-dir", "-o", default=DEFAULT_OUTPUT, type=Path,
        help="directory to store .stub.md files",
    )
    args = parser.parse_args(argv)

    stubs = search_paths(args.paths)
    write_stub_files(stubs, Path(args.output_dir))


if __name__ == "__main__":
    main()
