#!/usr/bin/env python3
"""Triple-buffered ASCII frame diff manager."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import numpy as np
    import threading
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


class AsciiFrameBuffer:
    """Manage ASCII frames for flicker-free terminal rendering."""

    def __init__(self, shape: tuple[int, int]):
        """Allocate buffers for ``shape`` and initialize contents."""
        self.lock = threading.Lock()
        self.shape = shape
        self.buffer_render = np.full(shape, " ", dtype="<U1")
        self.buffer_next = np.full(shape, " ", dtype="<U1")
        self.buffer_display = np.full(shape, " ", dtype="<U1")

    def _resize(self, shape: tuple[int, int]) -> None:
        """Resize internal buffers to ``shape``."""
        self.shape = shape
        self.buffer_render = np.full(shape, " ", dtype="<U1")
        self.buffer_next = np.full(shape, " ", dtype="<U1")
        self.buffer_display = np.full(shape, " ", dtype="<U1")

    def update_render(self, new_data: np.ndarray) -> None:
        """Update the render buffer with ``new_data``.

        Resizes internal buffers if ``new_data`` shape differs from
        the current configuration.
        """
        if new_data.shape != self.shape:
            with self.lock:
                self._resize(new_data.shape)
        with self.lock:
            np.copyto(self.buffer_render, new_data)

    def get_diff_and_promote(self) -> list[tuple[int, int, str]]:
        """Return changed cells since last promotion.

        Copies the render buffer into the next buffer, computes the
        difference against the display buffer, updates the display
        buffer, and returns a list of ``(y, x, char)`` tuples for the
        cells that changed.
        """
        with self.lock:
            np.copyto(self.buffer_next, self.buffer_render)
        diff_mask = self.buffer_next != self.buffer_display
        coords = np.argwhere(diff_mask)
        updates: list[tuple[int, int, str]] = []
        for y, x in coords:
            updates.append((int(y), int(x), str(self.buffer_next[y, x])))
        np.copyto(self.buffer_display, self.buffer_next)
        return updates
