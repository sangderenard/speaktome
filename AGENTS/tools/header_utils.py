#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Shared constants for header compliance utilities."""
from __future__ import annotations

ENV_SETUP_BOX = (
    "\n"
    "+-----------------------------------------------------------------------+\n"
    "| Imports failed. Run setup_env or setup_env_dev and select every    |\n"
    "| project and module you plan to use. Missing packages mean setup was |\n"
    "| skipped or incomplete.                                             |\n"
    "+-----------------------------------------------------------------------+\n"
)

HEADER_START = "# --- BEGIN HEADER ---"
HEADER_END = "# --- END HEADER ---"
IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX, IMPORT_FAILURE_PREFIX
    import sys
except Exception:
    import sys
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

HEADER_REQUIREMENTS = [
    "'# --- BEGIN HEADER ---' sentinel after shebang",
    "shebang line '#!/usr/bin/env python3'",
    "module docstring",
    "'from __future__ import annotations' before the try block",
    "imports wrapped in a try block",
    "except block imports sys then prints IMPORT_FAILURE_PREFIX and ENV_SETUP_BOX and exits",
    "'# --- END HEADER ---' sentinel after the except block",
]

__all__ = [
    "ENV_SETUP_BOX",
    "HEADER_START",
    "HEADER_END",
    "IMPORT_FAILURE_PREFIX",
    "HEADER_REQUIREMENTS",
]
