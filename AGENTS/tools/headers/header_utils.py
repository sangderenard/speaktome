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

ENV_BOX_ENV = "ENV_SETUP_BOX"


def get_env_setup_box() -> str:
    """Return the environment setup message."""
    try:
        return os.environ[ENV_BOX_ENV]
    except KeyError as exc:  # pragma: no cover - env missing
        raise RuntimeError("environment not initialized") from exc


ENV_SETUP_BOX = get_env_setup_box()

HEADER_START = "# --- BEGIN HEADER ---"
HEADER_END = "# --- END HEADER ---"
IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"

HEADER_REQUIREMENTS = [
    "'# --- BEGIN HEADER ---' sentinel after shebang",
    "shebang line '#!/usr/bin/env python3'",
    "module docstring",
    "'from __future__ import annotations' before the try block",
    "imports wrapped in a try block",
    (
        "except block imports os, sys and Path, defines _find_repo_root,"
        " checks ENV_SETUP_BOX and prints the message when missing,"
        " then invokes auto_env_setup via subprocess",
    ),
    "'# --- END HEADER ---' sentinel after the except block",
]


def extract_header_import_block(lines: list[str]) -> list[str]:
    """Return lines from the first import to the header end sentinel."""
    try:
        start = next(
            i for i, ln in enumerate(lines) if ln.strip().startswith(("import", "from "))
        )
    except StopIteration:
        return []
    try:
        end = next(i for i, ln in enumerate(lines) if ln.strip() == HEADER_END)
    except StopIteration:
        end = len(lines)
    return lines[start:end]

__all__ = [
    "get_env_setup_box",
    "HEADER_START",
    "HEADER_END",
    "IMPORT_FAILURE_PREFIX",
    "HEADER_REQUIREMENTS",
    "extract_header_import_block",
]
