#!/usr/bin/env python3
"""Meta namespace for agent utilities."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    import os
    import sys
    ENV_SETUP_BOX = os.environ.get(
        "SPEAKTOME_ENV_SETUP_BOX",
        "Environment setup incomplete. See ENV_SETUP_OPTIONS.md",
    )
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = []
