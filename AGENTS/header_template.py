#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import IMPORT_FAILURE_PREFIX
    import your_modules
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
