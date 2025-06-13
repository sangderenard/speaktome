#!/usr/bin/env python3
"""Meta namespace for agent utilities."""
from __future__ import annotations

try:
    IMPORT_FAILURE_PREFIX = "[HEADER] import failure in"
except Exception:
    import os
    import sys
    ENV_SETUP_BOX = os.environ.get(
        "ENV_SETUP_BOX", "Environment not initialized. See ENV_SETUP_OPTIONS.md"
    )
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = []
