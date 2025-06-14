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

END_MARKER = "# --- END HEADER ---"


EXCLUDE_DIRS = {
    "archive",
    "third_party",
    "laplace",
    "training",
    "tensorprinting",
    "tensorprinting",
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
