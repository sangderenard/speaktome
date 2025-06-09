#!/usr/bin/env python3
"""SpeakToMe beam search package."""
from __future__ import annotations

try:
    from .tensors.faculty import Faculty, DEFAULT_FACULTY, FORCE_ENV
except Exception:
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---

__all__ = ["Faculty", "DEFAULT_FACULTY", "FORCE_ENV"]
