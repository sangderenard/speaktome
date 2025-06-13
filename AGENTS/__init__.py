#!/usr/bin/env python3
"""Meta namespace for agent utilities."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import IMPORT_FAILURE_PREFIX
except Exception:
    import os
    import sys
    try:
        ENV_SETUP_BOX = os.environ["SPEAKTOME_ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"{IMPORT_FAILURE_PREFIX} {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = []
