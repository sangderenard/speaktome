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


def draw_diff(diff_data: list[tuple[int, int, str]], base_row: int = 1, base_col: int = 1) -> None:
    """Render diff data using ANSI cursor positioning.
    The content from diff_data is drawn starting at terminal `base_row`, `base_col`.
    `y`, `x` in `diff_data` are 0-indexed offsets from this base position.
    `base_row`, `base_col` are 1-indexed for ANSI terminal compatibility."""
    for y, x, char in diff_data:
        terminal_row = base_row + y
        terminal_col = base_col + x
        sys.stdout.write(f"\x1b[{terminal_row};{terminal_col}H{char}")
    sys.stdout.flush()
