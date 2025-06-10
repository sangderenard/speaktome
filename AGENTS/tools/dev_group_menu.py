#!/usr/bin/env python3
# Standard library imports
import argparse
import json
import os
import re
import subprocess
import sys
import threading
import tempfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib

# Platform-specific input handling
if os.name == 'nt':  # Windows
    import msvcrt
    def getch_timeout(timeout):
        """Get a single character with timeout on Windows."""
        result = []
        def input_thread():
            try:
                result.append(msvcrt.getch().decode())
            except Exception:
                result.append(None)
        
        thread = threading.Thread(target=input_thread)
        thread.daemon = True
        thread.start()
        thread.join(timeout)
        return result[0] if result else None

else:  # Unix
    import select
    import termios
    import tty
    
    def getch_timeout(timeout):
        """Get a single character with timeout on Unix."""
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(sys.stdin.fileno())
            if select.select([sys.stdin], [], [], timeout)[0]:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

def ask(prompt, timeout=3, default="n"):
    """Cross-platform input with timeout."""
    print(prompt, end='', flush=True)
    result = getch_timeout(timeout)
    print()  # Move to next line
    return result.lower() if result else default

# --- END HEADER ---

"""Interactive dev environment setup with dynamic codebase discovery.

This script presents a menu asking which codebases you want to work on and
which optional dependency groups to install for each one. It can simply print
the selections or install them automatically when ``--install`` is passed. Use
``--json`` to output the selections in machine readable form.
"""

ROOT = Path(__file__).resolve().parents[2]
REGISTRY = ROOT / "AGENTS" / "CODEBASE_REGISTRY.md"
ACTIVE_ENV = "SPEAKTOME_ACTIVE_FILE"


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


def extract_group_packages(toml_path: Path) -> dict[str, list[str]]:
    """Return optional dependency mapping from ``pyproject.toml``."""
    try:
        data = tomllib.loads(toml_path.read_text())
    except Exception:
        return {}
    return data.get("project", {}).get("optional-dependencies", {})


def build_codebase_groups() -> dict[str, dict[str, list[str]]]:
    """Return mapping of codebase name to group->packages."""
    mapping: dict[str, dict[str, list[str]]] = {}
    for cb_path in discover_codebases(REGISTRY):
        groups: dict[str, list[str]] = {}
        for toml_file in cb_path.rglob("pyproject.toml"):
            groups = extract_group_packages(toml_file)
            if groups:
                break
        mapping[cb_path.name] = groups
    return mapping


CODEBASES = build_codebase_groups()


def interactive_selection() -> tuple[list[str], dict[str, dict[str, list[str]]]]:
    """Return selected codebases mapped to chosen packages."""

    selected_codebases: list[str] = []
    for cb in CODEBASES:
        if ask(f"Work on codebase '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
            selected_codebases.append(cb)

    selected: dict[str, dict[str, list[str]]] = {}
    for cb in selected_codebases:
        groups = CODEBASES[cb]
        selected[cb] = {}
        for group, pkgs in groups.items():
            if (
                ask(
                    f"Install group '{group}' for '{cb}'? [Y/n] (auto-accept in 3s): ",
                    default="y",
                )
                == "y"
            ):
                selected[cb][group] = []
                for pkg in pkgs:
                    if (
                        ask(
                            f"  Install package '{pkg}'? [y/N] (auto-skip in 3s): "
                        )
                        == "y"
                    ):
                        selected[cb][group].append(pkg)

    return selected_codebases, selected


def install_selections(
    selections: dict[str, dict[str, list[str]]], *, pip_cmd: str = "pip"
) -> None:
    """Install selected packages for each codebase using ``pip_cmd``."""

    for cb, groups in selections.items():
        cb_path = ROOT / cb
        if not cb_path.is_dir():
            continue
        subprocess.run([pip_cmd, "install", "-e", str(cb_path)], check=False)
        for pkgs in groups.values():
            for pkg in pkgs:
                subprocess.run([pip_cmd, "install", pkg], check=False)


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
        help="Install selected codebases and packages using $PIP_CMD",
    )
    parser.add_argument(
        "--record",
        metavar="PATH",
        help="Write selections to PATH (default from SPEAKTOME_ACTIVE_FILE)",
        nargs="?",
        const=os.environ.get(ACTIVE_ENV, str(Path(tempfile.gettempdir()) / "speaktome_active.json")),
    )
    args = parser.parse_args(argv)

    cbs, selections = interactive_selection()

    if args.install:
        pip_cmd = os.environ.get("PIP_CMD", "pip")
        install_selections(selections, pip_cmd=pip_cmd)

    if args.record:
        path = Path(args.record)
        try:
            path.write_text(json.dumps({"codebases": cbs, "packages": selections}))
        except OSError as exc:  # pragma: no cover - file write errors
            print(f"[WARN] Could not write selections to {path}: {exc}")

    if args.json:
        print(json.dumps({"codebases": cbs, "packages": selections}))
    else:
        print("Selected codebases:", cbs)
        print("Selected packages:", selections)


if __name__ == "__main__":  # pragma: no cover - CLI entry
    main()
