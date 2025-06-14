#!/usr/bin/env python3
"""
########## STUB: sanitize_alpha_data.py ##########
PURPOSE: Scan the massive `training/archive/Alpha` dump and detect
    duplicate files. Optionally remove duplicates so that only one
    canonical copy (the first encountered) remains. This helps clean
    up the dataset where many versions of classes were exported.
EXPECTED BEHAVIOR: When executed it reports duplicate files by hash
    and, if run with `--delete`, moves them into an adjacent
    `duplicates/` folder for manual review. Actual deletion should be
    a conscious decision to preserve data history.
INPUTS: The root directory to scan (default `training/archive/Alpha`).
        Optional command line flags `--delete` and `--dry-run`.
OUTPUTS: Printed report of duplicates. If `--delete` is provided,
        duplicate files are moved to `duplicates/` within the archive
        root.
KEY ASSUMPTIONS/DEPENDENCIES: Only checks file content (SHA1 hash)
        for exact duplicates; does not detect semantic similarity.
TODO:
  - Add logging and progress indication for large datasets.
  - Consider heuristics to choose the "premium" version instead of
    relying on first appearance.
NOTES: This script is intentionally conservative. It is meant to help
    human maintainers clean up the training dump without silently
    removing data. Review the duplicates folder after running with
    `--delete` to confirm nothing important was lost.
#######################################################################
"""
from __future__ import annotations

try:
    import argparse
    import hashlib
    import os
    import shutil
    from pathlib import Path
except Exception:
    from AGENTS.tools.headers.header_utils import ENV_SETUP_BOX
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---


def compute_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def find_duplicates(root: Path):
    """Yield (duplicate_path, canonical_path) tuples."""
    hashes = {}
    for dirpath, _, filenames in os.walk(root):
        for name in filenames:
            file_path = Path(dirpath) / name
            digest = compute_sha1(file_path)
            if digest in hashes:
                yield file_path, hashes[digest]
            else:
                hashes[digest] = file_path


def main():
    parser = argparse.ArgumentParser(description="Deduplicate Alpha dataset")
    parser.add_argument(
        "root", type=Path, nargs="?", default=Path("training/archive/Alpha"),
        help="Root directory to scan for duplicates")
    parser.add_argument(
        "--delete", action="store_true",
        help="Move duplicate files into a 'duplicates' folder")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Only report duplicates without modifying files")
    args = parser.parse_args()

    duplicates = list(find_duplicates(args.root))
    if not duplicates:
        print("No duplicates found")
        return

    dup_dir = args.root / "duplicates"
    if args.delete:
        dup_dir.mkdir(exist_ok=True)

    for dup, canonical in duplicates:
        rel_dup = dup.relative_to(args.root)
        rel_canon = canonical.relative_to(args.root)
        print(f"Duplicate: {rel_dup} -> {rel_canon}")
        if args.delete and not args.dry_run:
            target = dup_dir / rel_dup
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(dup), target)

    print(f"Found {len(duplicates)} duplicate files.")
    if args.delete:
        print(f"Duplicates moved to: {dup_dir}")


if __name__ == "__main__":
    main()
