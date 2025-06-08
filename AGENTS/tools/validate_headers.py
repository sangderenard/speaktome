#!/usr/bin/env python3
"""Static header validation utility for the SPEAKTOME project.

This script walks the ``speaktome/`` package and checks that every class
defines a ``HEADER`` attribute or docstring and provides a
``@staticmethod`` ``test()`` method.  Any violations are printed with the
``[AGENT_ACTIONABLE_ERROR]`` tag for easy parsing.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PACKAGE_ROOT = Path(__file__).parent / "speaktome"


# ########## STUB: header validation helper ##########
# PURPOSE: Ensure each class exposes documentation and a basic test stub.
# EXPECTED BEHAVIOR: Recursively parse Python modules and report missing
# ``HEADER`` or ``test`` implementations.
# INPUTS: Directory path to scan (defaults to ``PACKAGE_ROOT``).
# OUTPUTS: Prints error lines prefixed with ``[AGENT_ACTIONABLE_ERROR]``.
# KEY ASSUMPTIONS/DEPENDENCIES: Relies on ``ast`` for static analysis
# without importing project modules.
# TODO:
#   - Expand to optionally rewrite files with placeholder stubs.
#   - Integrate with pre-commit and CI workflows.
# NOTES: Minimal implementation provided for initial tooling.
# ###########################################################################

def collect_classes(root: Path) -> list[tuple[str, ast.ClassDef]]:
    classes = []
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".git"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse error
            print(f"[AGENT_ACTIONABLE_ERROR] {path}: parse error {exc}")
            continue
        mod_name = path.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append((mod_name, node))
    return classes

def check_class(cls: ast.ClassDef) -> tuple[bool, bool, str | None]:
    doc = ast.get_docstring(cls)
    header = None
    has_test = False
    if doc:
        header = doc
    for stmt in cls.body:
        if isinstance(stmt, ast.Assign):
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "HEADER" and isinstance(stmt.value, ast.Str):
                    header = stmt.value.s
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "test":
            for deco in stmt.decorator_list:
                if isinstance(deco, ast.Name) and deco.id == "staticmethod":
                    has_test = True
    return bool(header), has_test, header


def validate(root: Path) -> int:
    exit_code = 0
    for mod_name, cls in collect_classes(root):
        has_header, has_test, _header = check_class(cls)
        missing = []
        if not has_header:
            missing.append("HEADER")
        if not has_test:
            missing.append("@staticmethod test()")
        if missing:
            exit_code = 1
            print(
                f"[AGENT_ACTIONABLE_ERROR] {mod_name}:{cls.name} missing "
                f"{', '.join(missing)}"
            )
    return exit_code


def main() -> None:
    path = PACKAGE_ROOT
    if len(sys.argv) > 1:
        path = Path(sys.argv[1])
    sys.exit(validate(path))


if __name__ == "__main__":
    main()
