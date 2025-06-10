#!/usr/bin/env python3
"""Terminal drawing helpers for ASCII frame diffs."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import sys
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def draw_diff(diff_data: list[tuple[int, int, str]]) -> None:
    """Render diff data using ANSI cursor positioning."""
    for y, x, char in diff_data:
        sys.stdout.write(f"\x1b[{y+1};{x+1}H{char}")
    sys.stdout.flush()
