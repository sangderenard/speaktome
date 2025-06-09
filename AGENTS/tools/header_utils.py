#!/usr/bin/env python3
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

try:
    pass
except Exception:
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---

HEADER_REQUIREMENTS = [
    "shebang line '#!/usr/bin/env python3'",
    "module docstring",
    "'from __future__ import annotations' before the try block",
    "imports wrapped in a try block",
    "except block printing ENV_SETUP_BOX",
    "'# --- END HEADER ---' sentinel after the except block",
]

__all__ = ["ENV_SETUP_BOX", "HEADER_REQUIREMENTS"]
