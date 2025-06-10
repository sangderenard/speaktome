#!/usr/bin/env python
"""Dump class headers across the ``speaktome`` package.

This utility recursively scans Python files and records each class with a
``HEADER`` attribute or docstring as well as the presence of a
``@staticmethod`` ``test`` method.  The results are printed as JSON and
optionally as Markdown for human inspection.
"""

from __future__ import annotations

import ast
import json
import sys
import os
from pathlib import Path

END_MARKER = "# --- END HEADER ---"


EXCLUDE_DIRS = {
    "archive",
    "third_party",
    "laplace",
    "training",
    "tensor printing",
    "tensor_printing",
}


def find_py_files(root: Path) -> list[Path]:
    """Yield Python files under ``root`` ignoring ``.venv`` and ``.git``."""
    for dirpath, dirnames, filenames in os.walk(root):
        parts = Path(dirpath).parts
        if any(d in parts for d in EXCLUDE_DIRS | {".venv", ".git"}):
            dirnames[:] = []
            continue
        for name in filenames:
            if name.endswith(".py"):
                yield Path(dirpath) / name


def collect_class_info(pyfile: Path) -> list[dict[str, object]]:
    with pyfile.open("r", encoding="utf-8", errors="ignore") as fh:
        source = fh.read()
    try:
        tree = ast.parse(source, filename=str(pyfile))
    except Exception as exc:  # pragma: no cover - parse failure
        print(f"[ERROR] {pyfile}: {exc}", file=sys.stderr)
        return []
    classes = []
    for node in tree.body:
        if isinstance(node, ast.ClassDef):
            header = ast.get_docstring(node)
            for stmt in node.body:
                if (
                    isinstance(stmt, ast.Assign)
                    and any(isinstance(t, ast.Name) and t.id == "HEADER" for t in stmt.targets)
                    and isinstance(stmt.value, ast.Constant)
                    and isinstance(stmt.value.value, str)
                ):
                    header = stmt.value.value
                if isinstance(stmt, ast.FunctionDef) and stmt.name == "test":
                    has_test = any(
                        isinstance(deco, ast.Name) and deco.id == "staticmethod"
                        for deco in stmt.decorator_list
                    )
                    break
            else:
                has_test = False
            classes.append({"class": node.name, "header": header, "has_test": has_test})
    return classes


def dump_headers(root: Path, markdown: bool = False) -> None:
    results = {}
    for pyfile in sorted(find_py_files(root)):
        module = pyfile.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")
        info = collect_class_info(pyfile)
        if info:
            results[module] = info
    print(json.dumps(results, indent=2))

    if markdown:
        for module, items in results.items():
            print(f"\n### {module}")
            for entry in items:
                test = "yes" if entry["has_test"] else "no"
                header = entry["header"] or "(none)"
                print(f"- **{entry['class']}** – test: {test}\n  \n  {header}")


def main() -> None:
    root_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("speaktome")
    md = "--markdown" in sys.argv
    dump_headers(root_arg.resolve(), markdown=md)


if __name__ == "__main__":
    main()
