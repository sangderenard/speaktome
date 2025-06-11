#!/usr/bin/env python3
"""Simplified ASCII rendering utilities for FontMapper v2."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    from fontmapper import FM16
    from fontmapper.FM16.modules import charset_ops
    from PIL import Image
    from pathlib import Path
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---

from typing import Iterable


def ascii_preview(font_files: Iterable[str], font_size: int = 12, complexity_level: int = 0) -> str:
    """Return ASCII preview of the generated charset variants."""
    fonts, charset, bitmaps, width, height = charset_ops.obtain_charset(list(font_files), font_size, complexity_level)
    return charset_ops.bytemaps_as_ascii(bitmaps, width, height)

