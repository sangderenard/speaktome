#!/usr/bin/env python3
"""Harvest Python classes from the Alpha archive."""
from __future__ import annotations

try:
    import argparse
    import ast
    import hashlib
    import shutil
    from pathlib import Path
    from typing import Iterable, List, Dict
except Exception:
    from AGENTS.tools.headers.header_utils import ENV_SETUP_BOX
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---

# Determine the script's directory to make default paths relative to it
SCRIPT_DIRECTORY = Path(__file__).resolve().parent
DEFAULT_SRC_DIRECTORY = SCRIPT_DIRECTORY / "archive" / "Alpha"


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
        timestamp_str = str(int(path.stat().st_mtime))
        class_list_str = "_".join(classes) if classes else "NoClass"

        # Define a target maximum length for the entire filename component (e.g., "timestamp_classes.py")
        # NTFS limit is 255. Using 240 as a conservative value to allow for _idx from unique_target.
        TARGET_MAX_FILENAME_LEN = 240

        # Calculate length of fixed parts: timestamp + "_" + ".py"
        base_len = len(timestamp_str) + 1 + 3  # 1 for '_', 3 for '.py'
        max_len_for_processed_class_list = TARGET_MAX_FILENAME_LEN - base_len

        processed_class_list_str: str
        if max_len_for_processed_class_list <= 0:
            # This edge case implies timestamp_str is excessively long or TARGET_MAX_FILENAME_LEN is tiny.
            # Fallback to a short hash of the original class list.
            processed_class_list_str = hashlib.sha256(class_list_str.encode('utf-8')).hexdigest()[:10]
        elif len(class_list_str) <= max_len_for_processed_class_list:
            processed_class_list_str = class_list_str
        else:
            # class_list_str is too long, truncate and append a short hash.
            hash_part = hashlib.sha256(class_list_str.encode('utf-8')).hexdigest()[:8]
            
            # Calculate length for the truncated original class list string part.
            # It must fit: truncated_part + "_" + hash_part
            len_for_truncated_original = max_len_for_processed_class_list - (1 + len(hash_part)) # 1 for '_'

            if len_for_truncated_original < 1:
                # Not enough space for even one char of original + '_' + hash.
                # Use the hash, truncated if necessary to fit max_len_for_processed_class_list.
                processed_class_list_str = hash_part[:max_len_for_processed_class_list]
            else:
                processed_class_list_str = f"{class_list_str[:len_for_truncated_original]}_{hash_part}"
        
        new_name = f"{timestamp_str}_{processed_class_list_str}.py"
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
        default=DEFAULT_SRC_DIRECTORY,
        help="Source directory to scan. Defaults to 'archive/Alpha' relative to the script's directory.",
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
