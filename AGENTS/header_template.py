#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX, IMPORT_FAILURE_PREFIX
    import your_modules
except Exception:
    import sys

    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
