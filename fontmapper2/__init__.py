#!/usr/bin/env python3
"""High-level FontMapper v2 package."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:  # pragma: no cover - env failure
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from .ascii_mapper import ascii_preview

__all__ = ["ascii_preview"]
