#!/usr/bin/env python3
# Standard library imports
import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
# --- END HEADER ---

"""Interactive dev environment setup with dynamic codebase discovery.

This script presents a menu asking which codebases you want to work on and
which optional dependency groups to install for each one. It can simply print
the selections or install them automatically when ``--install`` is passed. Use
``--json`` to output the selections in machine readable form.
"""

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "AGENTS" / "CODEBASE_REGISTRY.md"


def discover_codebases(registry_path: Path) -> list[Path]:
    """Return valid codebase directories listed in ``CODEBASE_REGISTRY.md``."""
    pattern = re.compile(r"- \*\*(.+?)\*\*")
    codebases: list[Path] = []
    if not registry_path.exists():
        return codebases
    for line in registry_path.read_text().splitlines():
        match = pattern.match(line.strip())
        if not match:
            continue
        path = ROOT / match.group(1)
        if path.is_dir():
            codebases.append(path)
    return codebases


def extract_groups(toml_path: Path) -> list[str]:
    """Extract optional dependency groups from a ``pyproject.toml`` file."""
    try:
        data = tomllib.loads(toml_path.read_text())
    except Exception:
        return []
    return list(data.get("project", {}).get("optional-dependencies", {}).keys())


def build_codebase_groups() -> dict[str, list[str]]:
    """Construct mapping of codebase name to optional dependency groups."""
    mapping: dict[str, list[str]] = {}
    for cb_path in discover_codebases(REGISTRY):
        groups: list[str] = []
        for toml_file in cb_path.rglob("pyproject.toml"):
            groups = extract_groups(toml_file)
            if groups:
                break
        mapping[cb_path.name] = groups
    return mapping


CODEBASES = build_codebase_groups()


def ask(prompt: str, timeout: int = 3, default: str = "n") -> str:
    """Prompt the user with ``prompt`` and return the lowercase response.

    If no input is provided within ``timeout`` seconds ``default`` is returned.
    """

    import signal

    def handler(signum, frame):  # pragma: no cover - interactive helper
        raise TimeoutError

    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        resp = input(prompt)
        signal.alarm(0)
        return resp.strip().lower() or default
    except TimeoutError:  # pragma: no cover - interactive helper
        print()
        return default


def interactive_selection() -> tuple[list[str], dict[str, list[str]]]:
    """Return user selected codebases and groups."""

    selected_codebases: list[str] = []
    for cb in CODEBASES:
        if ask(f"Work on codebase '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
            selected_codebases.append(cb)

    selected_groups: dict[str, list[str]] = {}
    for cb in selected_codebases:
        groups = CODEBASES[cb]
        selected_groups[cb] = []
        for group in groups:
            if (
                ask(f"Install group '{group}' for '{cb}'? [y/N] (auto-skip in 3s): ")
                == "y"
            ):
                selected_groups[cb].append(group)

    return selected_codebases, selected_groups


def install_selections(
    selections: dict[str, list[str]], *, pip_cmd: str = "pip"
) -> None:
    """Install editable codebases and extras using ``pip_cmd``."""

    for cb, groups in selections.items():
        cb_path = ROOT / cb
        if cb_path.is_dir():
            subprocess.run([pip_cmd, "install", "-e", str(cb_path)], check=False)
            for grp in groups:
                subprocess.run([pip_cmd, "install", f"{cb_path}[{grp}]"], check=False)


def main(argv: list[str] | None = None) -> None:  # pragma: no cover - CLI
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output selections as JSON instead of human readable text",
    )
    parser.add_argument(
        "--install",
        action="store_true",
        help="Install selected codebases and groups using $PIP_CMD",
    )
    args = parser.parse_args(argv)

    cbs, groups = interactive_selection()

    if args.install:
        pip_cmd = os.environ.get("PIP_CMD", "pip")
        install_selections(groups, pip_cmd=pip_cmd)

    if args.json:
        print(json.dumps({"codebases": cbs, "groups": groups}))
    else:
        print("Selected codebases:", cbs)
        print("Selected groups:", groups)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
