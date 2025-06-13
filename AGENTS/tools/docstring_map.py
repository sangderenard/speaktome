#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Generate an indented outline of classes and functions with docstrings."""
from __future__ import annotations

try:
    import argparse
    import ast
    from pathlib import Path
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["SPEAKTOME_ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


INDENT = "    "


def _collect(node: ast.AST, indent: int = 0) -> list[str]:
    lines: list[str] = []
    for child in getattr(node, "body", []):
        if isinstance(child, ast.ClassDef):
            lines.append(f"{INDENT * indent}class {child.name}")
            doc = ast.get_docstring(child)
            if doc:
                for ln in doc.splitlines():
                    lines.append(f"{INDENT * (indent + 1)}{ln}")
            lines.extend(_collect(child, indent + 1))
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            lines.append(f"{INDENT * indent}def {child.name}")
            doc = ast.get_docstring(child)
            if doc:
                for ln in doc.splitlines():
                    lines.append(f"{INDENT * (indent + 1)}{ln}")
            lines.extend(_collect(child, indent + 1))
    return lines


def build_map(path: Path) -> str:
    """Return outline for ``path``."""
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    return "\n".join(_collect(tree))


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("file", type=Path, help="Python file to map")
    args = parser.parse_args(argv)
    print(build_map(args.file))


if __name__ == "__main__":
    main()
