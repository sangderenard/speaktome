#!/usr/bin/env python3
"""Helpers for automatic environment initialization during tests."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path
import tomllib

if os.name == "nt":
    import msvcrt

    def getch_timeout(timeout: float) -> str | None:
        result: list[str] = []
        done = threading.Event()

        def _target() -> None:
            try:
                char = msvcrt.getch()
                if char:
                    result.append(char.decode("utf-8", "ignore"))
            finally:
                done.set()

        threading.Thread(target=_target, daemon=True).start()
        done.wait(timeout)
        return result[0] if result else None
else:
    import select
    import termios
    import tty

    def getch_timeout(timeout: float) -> str | None:
        try:
            fd = sys.stdin.fileno()
        except Exception:
            return None
        old = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ready, _, _ = select.select([sys.stdin], [], [], timeout)
            if ready:
                return sys.stdin.read(1)
            return None
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old)


def ask_no_venv(timeout: float = 5.0) -> bool:
    """Return True if user agrees to system-level install."""
    prompt = f"Install without virtualenv? [y/N] (auto-N in {int(timeout)}s): "
    print(prompt, end="", flush=True)
    ch = getch_timeout(timeout)
    print()
    return ch is not None and ch.lower() == "y"


def parse_pyproject_dependencies(pyproject_path: Path) -> list[str]:
    """Return a priority-sorted list of optional dependency groups from pyproject.toml."""
    # Here we just load them in alphabetical order as an example.
    # You may implement your own priority logic.
    if not pyproject_path.is_file():
        raise FileNotFoundError(f"pyproject.toml not found at {pyproject_path}")
    with pyproject_path.open("rb") as f:
        data = tomllib.load(f)
    optdeps = set()
    project_opts = data.get("project", {}).get("optional-dependencies", {})
    optdeps.update(project_opts.keys())
    poetry_groups = data.get("tool", {}).get("poetry", {}).get("group", {})
    optdeps.update(poetry_groups.keys())
    return sorted(optdeps)


def run_setup_script(project_root: Path | None = None, *, use_venv: bool = True) -> subprocess.CompletedProcess | None:
    """Run the repository setup script and return the result object."""
    if project_root is None:
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
            }
            for parent in [current, *current.parents]:
                if all((parent / name).exists() for name in required):
                    return parent
            return current

        project_root = _find_repo_root(Path(__file__))

    project_root = project_root.resolve()
    pyproject_file = project_root / "pyproject.toml"
    groups = parse_pyproject_dependencies(pyproject_file)
    script = project_root / ("setup_env.ps1" if os.name == "nt" else "setup_env.sh")
    if not script.exists():
        print(f"Error: setup script not found in {project_root}", file=sys.stderr)
        return None

    script = script.resolve()
    base_cmd = [str(script), "-codebases=speaktome"]
    if not use_venv:
        base_cmd.append("-no-venv")

    def invoke(cmd_args: list[str]) -> subprocess.CompletedProcess | None:
        if os.name == "nt":
            full_cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File"] + cmd_args
        else:
            full_cmd = ["bash"] + cmd_args
        try:
            return subprocess.run(full_cmd, check=False, capture_output=True, text=True)
        except Exception:
            return None

    result = invoke(base_cmd)

    for grp in groups:
        invoke(base_cmd + [f"-groups={grp}"])

    return result


def main(argv: list[str] | None = None) -> int:
    """Run the setup script from the command line."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Auto env setup (requires a path to the project root containing pyproject.toml)."
    )
    parser.add_argument(
        "project_root",
        help="Path to the project root (must contain pyproject.toml)."
    )
    parser.add_argument(
        "-no-venv",
        dest="no_venv",
        action="store_true",
        help="Install packages outside a virtualenv (single-dash style)."
    )
    args = parser.parse_args(argv)

    root_path = Path(args.project_root).resolve()
    if not (root_path / "pyproject.toml").is_file():
        print(f"Error: pyproject.toml not found in {root_path}", file=sys.stderr)
        return 1

    result = run_setup_script(root_path, use_venv=(not args.no_venv))
    if result is None:
        print("Error: setup script not found or could not be run.", file=sys.stderr)
        return 1

    sys.stdout.write(result.stdout)
    sys.stderr.write(result.stderr)
    return result.returncode


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
