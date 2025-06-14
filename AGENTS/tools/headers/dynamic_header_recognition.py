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

# Regex patterns for atomic units within the standard header
HEADER_START_REGEX = re.compile(r"^# --- BEGIN HEADER ---$", re.MULTILINE)
HEADER_END_REGEX = re.compile(r"^# --- END HEADER ---$", re.MULTILINE)
DOCSTRING_REGEX = re.compile(r'^"""(.*?)"""', re.DOTALL | re.MULTILINE)
IMPORT_REGEX = re.compile(r"^import (.+)$", re.MULTILINE)
TRY_BLOCK_REGEX = re.compile(r"^try:$", re.MULTILINE)
EXCEPT_BLOCK_REGEX = re.compile(r"^except (.+):$", re.MULTILINE)


class HeaderNode:
    """Node in a tree representation of the header."""

    def __init__(self, name: str, content: str = "", children: Optional[list["HeaderNode"]] = None) -> None:
        self.name = name
        self.content = content
        self.children: list[HeaderNode] = children or []

    def add_child(self, child: "HeaderNode") -> None:
        self.children.append(child)

    def __str__(self) -> str:  # pragma: no cover - trivial
        return f"{self.name}: {self.content}"

    def __repr__(self) -> str:  # pragma: no cover - debug
        return f"HeaderNode(name={self.name!r}, content={self.content!r}, children={self.children!r})"


ROOT = Path(__file__).resolve().parents[2]
MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"
TEMPLATE_PATH = Path(__file__).with_name("header_template.py")


def guess_codebase(path: Path, map_file: Path = MAP_FILE) -> str | None:
    """Return codebase name owning ``path``.

    The function first consults ``codebase_map.json``. If that file is missing
    or malformed, it falls back to scanning the path components for known
    codebase names so headers can operate even before dependencies are
    installed.
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
            "tools",
        }
        for part in path.parts:
            if part in candidates:
                return part

    return None

def parse_header(text: str) -> HeaderNode:
    """Parse ``text`` into a :class:`HeaderNode` tree."""
    root = HeaderNode("root")
    if not (HEADER_START_REGEX.search(text) and HEADER_END_REGEX.search(text)):
        return root

    doc = DOCSTRING_REGEX.search(text)
    if doc:
        root.add_child(HeaderNode("docstring", doc.group(1).strip()))

    for match in IMPORT_REGEX.finditer(text):
        root.add_child(HeaderNode("import", match.group(1).strip()))

    if TRY_BLOCK_REGEX.search(text):
        root.add_child(HeaderNode("try"))
    if EXCEPT_BLOCK_REGEX.search(text):
        root.add_child(HeaderNode("except"))

    return root


def load_template_tree(path: Path = TEMPLATE_PATH) -> HeaderNode:
    """Return :class:`HeaderNode` tree parsed from the header template."""
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return HeaderNode("root")
    return parse_header(text)


def compare_trees(parsed_tree: HeaderNode, template_tree: HeaderNode) -> list[str]:
    """Return a list of differences between ``parsed_tree`` and ``template_tree``."""
    differences: list[str] = []
    if parsed_tree.name != template_tree.name:
        differences.append(f"Mismatch: {parsed_tree.name} != {template_tree.name}")
    template_children = {child.name for child in template_tree.children}
    parsed_children = {child.name for child in parsed_tree.children}
    for name in template_children - parsed_children:
        differences.append(f"Missing: {name}")
    return differences


def pretty_print_tree(node: HeaderNode, level: int = 0) -> str:
    """Return a markdown representation of ``node`` and its children."""
    indent = "  " * level
    result = f"{indent}- {node.name}: {node.content}\n"
    for child in node.children:
        result += pretty_print_tree(child, level + 1)
    return result


__all__ = [
    "HeaderNode",
    "guess_codebase",
    "parse_header",
    "load_template_tree",
    "compare_trees",
    "pretty_print_tree",
]
