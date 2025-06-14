#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    import AGENTS.tools

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

from typing import Any, Dict, List, Optional
import re

class Header:
    """A class to represent and manipulate headers dynamically."""

    def __init__(self, description: str, components: Optional[Dict[str, str]] = None, error_handling: Optional[Dict[str, str]] = None):
        self.description = description
        self.components = components or {}
        self.error_handling = error_handling or {}

    def __str__(self) -> str:
        """Pretty-print the header as a markdown string."""
        components_str = "\n".join([f"  - {key}: \"{value}\"" for key, value in self.components.items()])
        error_handling_str = "\n".join([f"  - {key}: \"{value}\"" for key, value in self.error_handling.items()])
        return f"""
HEADER:
  - Description: "{self.description}"
  - Components:
{components_str}
  - Error Handling:
{error_handling_str}
"""

    @classmethod
    def from_markdown(cls, markdown: str) -> "Header":
        """Parse a markdown template to create a Header object."""
        description_match = re.search(r"Description: \"(.*?)\"", markdown)
        components_match = re.findall(r"- (\w+): \"(.*?)\"", markdown.split("Components:")[1].split("Error Handling:")[0])
        error_handling_match = re.findall(r"- (\w+): \"(.*?)\"", markdown.split("Error Handling:")[1])

        description = description_match.group(1) if description_match else ""
        components = {key: value for key, value in components_match}
        error_handling = {key: value for key, value in error_handling_match}

        return cls(description, components, error_handling)

    def to_markdown(self) -> str:
        """Convert the Header object back to a markdown string."""
        return str(self)

    def interpret(self, instructions: List[str]) -> None:
        """Interpret procedural instructions for dynamic header manipulation."""
        for instruction in instructions:
            # Example: Add logic for GOTO-style procedural looping
            pass

# Example usage
if __name__ == "__main__":
    markdown_template = """
HEADER:
  - Description: "Template for SPEAKTOME module headers."
  - Components:
      - IMPORT_FAILURE_PREFIX: "Prefix for import failure messages."
      - auto_env_setup: "Module invoked via subprocess to initialize the environment."
      - ENV_SETUP_BOX: "Environment variable for setup box."
  - Error Handling:
      - KeyError: "Raised if ENV_SETUP_BOX is not set."
      - RuntimeError: "Raised if the environment is not initialized."
"""
    header = Header.from_markdown(markdown_template)
    print(header)
