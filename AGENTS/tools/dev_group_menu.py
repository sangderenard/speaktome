#!/usr/bin/env python3
# Standard library imports
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python < 3.11
    import tomli as tomllib
# --- END HEADER ---

"""Interactive dev environment setup with dynamic codebase discovery."""

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

def ask(prompt, timeout=3, default="n"):
    import signal
    def handler(signum, frame):
        raise TimeoutError
    signal.signal(signal.SIGALRM, handler)
    signal.alarm(timeout)
    try:
        resp = input(prompt)
        signal.alarm(0)
        return resp.strip().lower() or default
    except TimeoutError:
        print()
        return default

selected_codebases = []
for cb in CODEBASES:
    if ask(f"Work on codebase '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
        selected_codebases.append(cb)

selected_groups = {}
for cb in selected_codebases:
    groups = CODEBASES[cb]
    selected_groups[cb] = []
    for group in groups:
        if ask(f"Install group '{group}' for '{cb}'? [y/N] (auto-skip in 3s): ") == "y":
            selected_groups[cb].append(group)

print("Selected codebases:", selected_codebases)
print("Selected groups:", selected_groups)
# Optionally: write to a file or export as needed