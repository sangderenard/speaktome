#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    import your_modules
except Exception:
    import os
    import sys
    from pathlib import Path
    import json

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensorprinting",
            "timesync",
            "AGENTS",
            "fontmapper",
            "tensors",
            "testenv",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    ROOT = _find_repo_root(Path(__file__))
    MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"

    def guess_codebase(path: Path, map_file: Path = MAP_FILE) -> str | None:
        """Return codebase name owning ``path``.

        Tries ``codebase_map.json`` first, then falls back to scanning path
        components for known codebase names.
        """
        try:
            data = json.loads(map_file.read_text())
        except Exception:
            data = None

        if data:
            for name, info in data.items():
                cb_path = ROOT / info.get("path", name)
                try:
                    path.relative_to(cb_path)
                    return name
                except ValueError:
                    continue
        else:
            candidates = {
                "speaktome",
                "laplace",
                "tensorprinting",
                "timesync",
                "fontmapper",
                "tensors",
                "testenv",
                "tools",
            }
            for part in path.parts:
                if part in candidates:
                    return part

        return None

    def parse_pyproject_groups(pyproject: Path) -> list[str]:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # Python < 3.11
                import tomli as tomllib
        except Exception:
            return []
        try:
            data = tomllib.loads(pyproject.read_text())
        except Exception:
            return []

        groups = set()
        groups.update(data.get("project", {}).get("optional-dependencies", {}).keys())
        tool = data.get("tool", {}).get("poetry", {})
        groups.update(tool.get("group", {}).keys())
        groups.update(tool.get("extras", {}).keys())
        return sorted(groups)

    if "ENV_SETUP_BOX" not in os.environ:
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = _find_repo_root(Path(__file__))
        pyproject = root / "pyproject.toml"
        groups = parse_pyproject_groups(pyproject)
        codebase = guess_codebase(Path(__file__))
        base_cmd = [
            sys.executable,
            "-m",
            "AGENTS.tools.auto_env_setup",
            str(root),
        ]
        if codebase:
            base_cmd.append(f"-codebases={codebase}")
        subprocess.run(base_cmd, check=False)
        for grp in groups:
            subprocess.run(base_cmd + [f"-groups={grp}"], check=False)

    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"[HEADER] import failure in {__file__}")
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
                    and isinstance(stmt.value, ast.Constant)
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
                    and isinstance(cls.body[0].value, ast.Constant)
                ):
                    insert_at = cls.body[0].end_lineno
                insert_idx = insert_at - 1
                indent = " " * (cls.col_offset + 4)
                new_lines: list[str] = []
                if miss_head:
                    new_lines.append(f"{indent}try:")
                    new_lines.append(f'{indent}    HEADER = "TODO"')
                    new_lines.append(f"{indent}except Exception:")
                    new_lines.append(f"{indent}    import os")
                    new_lines.append(f"{indent}    import sys")
                    new_lines.append(
                        f"{indent}    ENV_SETUP_BOX = os.environ['ENV_SETUP_BOX']"
                    )
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
        try:
            env_box = os.environ["ENV_SETUP_BOX"]
        except KeyError:
            env_box = "environment not initialized"
        print(
            "[AGENT_ACTIONABLE_ERROR] validate_headers failed: "
            f"{exc}.\n{env_box}"
        )
        sys.exit(1)
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
