#!/usr/bin/env python3
"""FontMapper utilities package."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from .FM16.modules import charset_ops

from .FM16.modules.charset_ops import (
    list_printable_characters,
    generate_checkerboard_pattern,
    generate_variants,
    bytemaps_as_ascii,
    obtain_charset,
)

__all__ = [
    "charset_ops",
    "list_printable_characters",
    "generate_checkerboard_pattern",
    "generate_variants",
    "bytemaps_as_ascii",
    "obtain_charset",
]
