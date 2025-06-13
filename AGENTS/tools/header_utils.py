#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Shared constants for header compliance utilities."""
from __future__ import annotations

try:
    import os
except Exception:
    import os
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensor_printing",
            "time_sync",
            "AGENTS",
            "fontmapper",
            "tensors",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

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
        subprocess.run(
            [
                sys.executable,
                "-m",
                "AGENTS.tools.auto_env_setup",
                str(root),
            ],
            check=False,
        )
    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:  # pragma: no cover - env missing
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
