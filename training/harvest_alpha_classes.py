#!/usr/bin/env python3
"""Harvest Python classes from the Alpha archive.

This utility scans the `training/archive/Alpha` directory for all `.py` files
and copies them into a new flat directory. Each file is renamed using its
modification timestamp in epoch seconds followed by the alphabetically sorted
class names contained within the source file. Duplicate files are detected by
SHA256 hash and skipped. The goal is to quickly compare class definitions
across the archive while preserving temporal information.
"""

from __future__ import annotations

import argparse
import ast
import hashlib
import shutil
from pathlib import Path
from typing import Iterable, List, Dict


def parse_class_names(path: Path) -> List[str]:
    """Return a sorted list of class names defined in `path`."""
    try:
        src = path.read_text(encoding="utf-8", errors="ignore")
    except OSError as exc:
        print(f"Failed to read {path}: {exc}")
        return []
    try:
        tree = ast.parse(src, filename=str(path))
    except SyntaxError as exc:
        print(f"Failed to parse {path}: {exc}")
        return []

    names = [n.name for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
    return sorted(set(names))


def file_hash(path: Path) -> str:
    """Return the SHA256 hash of `path`."""
    h = hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    return h.hexdigest()


def unique_target(dest: Path, name: str) -> Path:
    """Return a unique path within `dest` based on `name`."""
    candidate = dest / name
    if not candidate.exists():
        return candidate
    stem = candidate.stem
    suffix = candidate.suffix
    idx = 1
    while True:
        candidate = dest / f"{stem}_{idx}{suffix}"
        if not candidate.exists():
            return candidate
        idx += 1


def harvest(src: Path, dest: Path, *, dry_run: bool = False) -> None:
    if not src.is_dir():
        raise ValueError(f"Source directory {src} does not exist")
    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    seen_hashes: Dict[str, Path] = {}
    for path in src.rglob("*.py"):
        if not path.is_file():
            continue
        digest = file_hash(path)
        if digest in seen_hashes:
            original = seen_hashes[digest]
            print(f"Duplicate {path} -> already copied as {original.name}")
            continue
        classes = parse_class_names(path)
        timestamp = int(path.stat().st_mtime)
        class_part = "_".join(classes) if classes else "NoClass"
        new_name = f"{timestamp}_{class_part}.py"
        target = unique_target(dest, new_name)
        print(f"{path} -> {target.name}")
        if not dry_run:
            shutil.copy2(path, target)
        seen_hashes[digest] = target
        count += 1
    print(f"Harvested {count} unique files to {dest}")


def main(argv: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Harvest Alpha Python classes")
    parser.add_argument(
        "dest",
        type=Path,
        help="Directory where harvested files will be stored",
    )
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("training/archive/Alpha"),
        help="Source directory to scan (default: training/archive/Alpha)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show actions without copying files",
    )
    args = parser.parse_args(argv)
    harvest(args.src, args.dest, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
