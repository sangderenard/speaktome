#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Shared constants for header compliance utilities."""
from __future__ import annotations

try:
    import os
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:  # pragma: no cover - env missing
        raise RuntimeError("environment not initialized") from exc
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
        "except block imports sys, invokes the auto_env_setup module via"
        " subprocess, retrieves 'ENV_SETUP_BOX' from the environment, "
        "prints IMPORT_FAILURE_PREFIX and that variable, then exits"
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
