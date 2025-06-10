#!/usr/bin/env python3
"""Static header validation utility for the SPEAKTOME project."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import ast
    from pathlib import Path
    from typing import Iterable
    from .header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

PACKAGE_ROOT = Path(__file__).parent / "speaktome"


# This helper ensures each class exposes documentation and a basic test stub.
# It recursively parses Python modules and reports missing ``HEADER`` or
# ``test`` implementations without importing project modules.  When ``--rewrite``
# is supplied the script will insert minimal placeholders for any missing
# elements.  This makes it easier to bootstrap new modules and keep
# documentation consistent.


def collect_classes(root: Path) -> list[tuple[str, Path, ast.ClassDef]]:
    classes: list[tuple[str, Path, ast.ClassDef]] = []
    for path in root.rglob("*.py"):
        if any(part in {".venv", ".git"} for part in path.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"))
        except Exception as exc:  # pragma: no cover - parse error
            print(f"[AGENT_ACTIONABLE_ERROR] {path}: parse error {exc}")
            continue
        mod_name = (
            path.relative_to(root.parent).with_suffix("").as_posix().replace("/", ".")
        )
        for node in tree.body:
            if isinstance(node, ast.ClassDef):
                classes.append((mod_name, path, node))
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
                if (
                    isinstance(target, ast.Name)
                    and target.id == "HEADER"
                    and isinstance(stmt.value, (ast.Str, ast.Constant))
                ):
                    header = getattr(
                        stmt.value, "s", getattr(stmt.value, "value", None)
                    )
        if isinstance(stmt, ast.FunctionDef) and stmt.name == "test":
            for deco in stmt.decorator_list:
                if isinstance(deco, ast.Name) and deco.id == "staticmethod":
                    has_test = True
    return bool(header), has_test, header


def validate(root: Path, *, rewrite: bool = False) -> int:
    exit_code = 0
    rewrites: dict[Path, list[tuple[ast.ClassDef, bool, bool]]] = {}

    for mod_name, path, cls in collect_classes(root):
        has_header, has_test, _header = check_class(cls)
        missing_header = not has_header
        missing_test = not has_test
        if missing_header or missing_test:
            exit_code = 1
            parts = []
            if missing_header:
                parts.append("HEADER")
            if missing_test:
                parts.append("@staticmethod test()")
            print(
                f"[AGENT_ACTIONABLE_ERROR] {mod_name}:{cls.name} missing "
                f"{', '.join(parts)}"
            )
            if rewrite:
                rewrites.setdefault(path, []).append(
                    (cls, missing_header, missing_test)
                )

    if rewrite:
        for path, items in rewrites.items():
            lines = path.read_text(encoding="utf-8").splitlines()
            for cls, miss_head, miss_test in sorted(
                items, key=lambda t: t[0].lineno, reverse=True
            ):
                insert_at = cls.lineno
                if (
                    cls.body
                    and isinstance(cls.body[0], ast.Expr)
                    and isinstance(cls.body[0].value, (ast.Str, ast.Constant))
                ):
                    insert_at = cls.body[0].end_lineno
                insert_idx = insert_at - 1
                indent = " " * (cls.col_offset + 4)
                new_lines: list[str] = []
                if miss_head:
                    new_lines.append(f"{indent}try:")
                    new_lines.append(f'{indent}    HEADER = "TODO"')
                    new_lines.append(f"{indent}except Exception as exc:")
                    new_lines.append(f"{indent}    import sys")
                    new_lines.append(f"{indent}    print(ENV_SETUP_BOX)")
                    new_lines.append(f"{indent}    sys.exit(1)")
                if miss_test:
                    new_lines.append(f"{indent}@staticmethod")
                    new_lines.append(f"{indent}def test() -> None:")
                    new_lines.append(f"{indent}    pass")
                lines[insert_idx:insert_idx] = new_lines
            path.write_text("\n".join(lines) + "\n", encoding="utf-8")
            exit_code = 0

    return exit_code


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate class headers")
    parser.add_argument("path", nargs="?", default=PACKAGE_ROOT, type=Path)
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Insert missing HEADER strings and test stubs",
    )
    args = parser.parse_args()

    try:
        exit_code = validate(args.path, rewrite=args.rewrite)
    except Exception as exc:  # pragma: no cover - unexpected failure
        print(
            "[AGENT_ACTIONABLE_ERROR] validate_headers failed: "
            f"{exc}.\n{ENV_SETUP_BOX}"
        )
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
