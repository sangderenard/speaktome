#!/usr/bin/env python
# Standard library imports
import importlib.util
import subprocess
import sys
from pathlib import Path
import tomllib
# --- END HEADER ---

"""Ensure optional dependencies from ``pyproject.toml`` are installed.

This helper reads the optional dependency groups defined in ``pyproject.toml``
and installs any missing packages for each group using ``pip``. Installation
errors are reported but do not abort execution.
"""

PYPROJECT = Path(__file__).resolve().parents[2] / "pyproject.toml"


def parse_extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data.get("project", {}).get("optional-dependencies", {})


def missing_pkgs(pkgs: list[str]) -> list[str]:
    missing: list[str] = []
    for dep in pkgs:
        name = dep.split("[")[0].split("==")[0].split(">=")[0]
        if importlib.util.find_spec(name) is None:
            missing.append(dep)
    return missing


def ensure_group(group: str, pkgs: list[str]) -> None:
    missing = missing_pkgs(pkgs)
    if not missing:
        return
    print(f"Installing missing {group} dependencies: {', '.join(missing)}")
    subprocess.run([sys.executable, "-m", "pip", "install", f".[{group}]"], check=False)


def main() -> int:
    extras = parse_extras()
    if not extras:
        print("No optional dependencies defined.")
        return 0
    for group, pkgs in extras.items():
        ensure_group(group, pkgs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
