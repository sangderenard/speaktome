#!/usr/bin/env python3
"""Report AbstractTensor methods compared to documentation."""
from __future__ import annotations

try:
    import ast
    import re
    from pathlib import Path
    from typing import Iterable
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def gather_methods(path: Path, class_name: str) -> list[str]:
    """Return method names defined directly on ``class_name``."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name == class_name:
            return [n.name for n in node.body if isinstance(n, ast.FunctionDef)]
    return []


def parse_doc(md_path: Path) -> set[str]:
    """Return documented method names from ``abstraction_functions.md``."""
    pattern = re.compile(r"-\s*(?:\w+\.)?(\w+)\(")
    methods: set[str] = set()
    for line in md_path.read_text(encoding="utf-8").splitlines():
        m = pattern.search(line)
        if m:
            methods.add(m.group(1))
    return methods


def print_report(actual: set[str], documented: set[str]) -> None:
    print(f"Methods in source: {len(actual)}")
    print(f"Documented methods: {len(documented)}")

    missing = sorted(actual - documented)
    extra = sorted(documented - actual)

    if missing:
        print("\nMissing from documentation:")
        for name in missing:
            print(f" - {name}")

    if extra:
        print("\nDocument lists unknown methods:")
        for name in extra:
            print(f" - {name}")


def main(argv: Iterable[str] | None = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Compare AbstractTensor methods against documentation"
    )
    parser.add_argument(
        "--src",
        default="tensors/abstraction.py",
        type=Path,
        help="Path to abstraction.py",
    )
    parser.add_argument(
        "--doc",
        default="tensors/abstraction_functions.md",
        type=Path,
        help="Path to abstraction_functions.md",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    methods = set(gather_methods(args.src, "AbstractTensor"))
    methods.update(gather_methods(args.src, "ShapeAccessor"))

    documented = parse_doc(args.doc)

    print_report(methods, documented)


if __name__ == "__main__":
    main()
