#!/usr/bin/env python3
"""SpeakToMe beam search package."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from .tensors.faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

__all__ = ["Faculty", "DEFAULT_FACULTY", "FORCE_ENV"]
