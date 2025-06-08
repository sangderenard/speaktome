#!/usr/bin/env python3
"""Display the first and last lines of a file for quick review."""

from __future__ import annotations

import argparse
from pathlib import Path

# --- END HEADER ---


def preview(path: Path, n: int) -> str:
    text = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    if len(text) <= 2 * n:
        return "\n".join(text)
    top = text[:n]
    bottom = text[-n:]
    return "\n".join(top + ["..."] + bottom)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Show head and tail of a file")
    parser.add_argument("file", type=Path)
    parser.add_argument("-n", type=int, default=10, help="Lines from start and end")
    args = parser.parse_args(argv)
    print(preview(args.file, args.n))


if __name__ == "__main__":
    main()
