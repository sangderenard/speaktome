#!/usr/bin/env python3
"""Locate high‑visibility stub blocks in Python files and export them.

Recursively searches the provided directories for comments containing
``STUB:``. Each discovered block is written to ``todo`` as a standalone
``.stub.md`` file so agents can track outstanding work.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

# --- END HEADER ---

STUB_REGEX = re.compile(r"STUB:")
DEFAULT_OUTPUT = Path("todo")


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
