#!/usr/bin/env python3
"""Helpers for automatic environment initialization during tests."""
from __future__ import annotations

import os
import subprocess
import sys
import threading
from pathlib import Path

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


def run_setup_script(root: Path | None = None, *, use_venv: bool = True) -> subprocess.CompletedProcess | None:
    """Run setup_env.sh or setup_env.ps1 if present and return the result."""
    if root is None:
        root = Path(__file__).resolve().parents[1]
    script = root / ("setup_env.ps1" if os.name == "nt" else "setup_env.sh")
    if not script.exists():
        return None
    if os.name == "nt":
        cmd = ["powershell", "-ExecutionPolicy", "Bypass", "-File", str(script)]
        if not use_venv:
            cmd.append("-no-venv")
    else:
        cmd = ["bash", str(script)]
        if not use_venv:
            cmd.append("-no-venv")
    try:
        return subprocess.run(cmd, check=False, capture_output=True, text=True)
    except Exception:
        return None
