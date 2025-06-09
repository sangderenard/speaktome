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
    import sys
except Exception:
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = ["ENV_SETUP_BOX"]
