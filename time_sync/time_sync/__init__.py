"""Time synchronization utilities."""
# --- END HEADER ---

from .core import (
    sync_offset,
    get_offset,
    set_offset,
    adjust_datetime,
    now,
    utcnow,
)
from .ascii_digits import (
    compose_ascii_digits,
    print_analog_clock,
    print_digital_clock,
    render_ascii_to_array,
)
from .console import (
    init_colorama_for_windows,
    full_clear_and_reset_cursor,
    reset_cursor_to_top,
)
from ..frame_buffer import AsciiFrameBuffer
from ..render_thread import render_loop
from ..draw import draw_diff
from . import _internet  # exported for tests

__all__ = [
    "sync_offset",
    "get_offset",
    "set_offset",
    "adjust_datetime",
    "now",
    "utcnow",
    "compose_ascii_digits",
    "render_ascii_to_array",
    "print_analog_clock",
    "print_digital_clock",
    "init_colorama_for_windows",
    "full_clear_and_reset_cursor",
    "reset_cursor_to_top",
    "AsciiFrameBuffer",
    "render_loop",
    "draw_diff",
    "_internet",
]
