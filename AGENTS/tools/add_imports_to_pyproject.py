#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Update ``pyproject.toml`` based on imports in a Python script."""
from __future__ import annotations

try:
    import argparse
    import ast
    import sys
    from pathlib import Path

    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib
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

STD_LIB = set(sys.builtin_module_names) | getattr(sys, "stdlib_module_names", set())
TORCH_PKGS = {
    "torch",
    "torchvision",
    "torchaudio",
    "torchtext",
    "pytorch_lightning",
    "lightning",
}


class PyProject:
    """Simple interface for updating ``pyproject.toml``."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.data = tomllib.loads(path.read_text()) if path.exists() else {}

    # Helper methods to handle poetry style structures
    def base_deps(self) -> dict[str, object]:
        return self.data.setdefault("tool", {}).setdefault("poetry", {}).setdefault(
            "dependencies", {}
        )

    def group_deps(self, group: str) -> dict[str, object]:
        section = (
            self.data.setdefault("tool", {})
            .setdefault("poetry", {})
            .setdefault("group", {})
            .setdefault(group, {})
        )
        return section.setdefault("dependencies", {})

    def ensure_group(self, group: str, optional: bool) -> None:
        grp = (
            self.data.setdefault("tool", {})
            .setdefault("poetry", {})
            .setdefault("group", {})
            .setdefault(group, {})
        )
        if optional:
            grp["optional"] = True
        elif "optional" in grp:
            del grp["optional"]
        grp.setdefault("dependencies", {})

    def add_dependency(self, group: str, package: str) -> None:
        deps = self.group_deps(group) if group else self.base_deps()
        deps.setdefault(package, "*")

    def exists(self, package: str) -> bool:
        if package in self.base_deps():
            return True
        groups = (
            self.data.get("tool", {})
            .get("poetry", {})
            .get("group", {})
        )
        for meta in groups.values():
            if package in meta.get("dependencies", {}):
                return True
        return False

    def write(self) -> None:
        import tomli_w

        self.path.write_text(tomli_w.dumps(self.data))


def extract_imports(py_file: Path) -> set[str]:
    """Return top-level imported package names from ``py_file``."""
    node = ast.parse(py_file.read_text())
    pkgs: set[str] = set()
    for stmt in node.body:
        if isinstance(stmt, ast.Import):
            for alias in stmt.names:
                root = alias.name.split(".")[0]
                pkgs.add(root)
        elif isinstance(stmt, ast.ImportFrom) and stmt.level == 0:
            root = stmt.module.split(".")[0] if stmt.module else ""
            if root:
                pkgs.add(root)
    return pkgs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("script", type=Path, help="Python script to analyze")
    parser.add_argument(
        "--pyproject", type=Path, default=Path("pyproject.toml"), help="pyproject path"
    )
    args = parser.parse_args()

    imports = extract_imports(args.script)
    project = PyProject(args.pyproject)

    local_pkgs = {"speaktome", "fontmapper", "laplace", "tensors", "time_sync", "tensor_printing", "tools"}

    project.ensure_group("unsorted", optional=False)
    project.ensure_group("cpu-torch", optional=True)
    project.ensure_group("gpu-torch", optional=True)

    for pkg in sorted(imports):
        if pkg in STD_LIB or pkg in local_pkgs:
            continue
        if project.exists(pkg):
            continue
        if pkg in TORCH_PKGS or pkg.startswith("torch"):
            project.add_dependency("cpu-torch", pkg)
        else:
            project.add_dependency("unsorted", pkg)

    project.write()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
