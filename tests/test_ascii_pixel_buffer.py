#!/usr/bin/env python3
"""PixelFrameBuffer regression tests."""

from __future__ import annotations

try:
    import os
    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    import numpy as np
    from ascii_diff.frame_buffer import PixelFrameBuffer
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def test_diff_threshold_behavior() -> None:
    fb = PixelFrameBuffer((1, 2), diff_threshold=10)
    frame1 = np.zeros((1, 2, 3), dtype=np.uint8)
    fb.update_render(frame1)
    fb.get_diff_and_promote()

    frame2 = np.zeros((1, 2, 3), dtype=np.uint8)
    frame2[0, 0] = [5, 5, 0]
    fb.update_render(frame2)
    diff = fb.get_diff_and_promote()
    assert diff == []

    frame3 = np.zeros((1, 2, 3), dtype=np.uint8)
    frame3[0, 0] = [6, 5, 0]
    fb.update_render(frame3)
    diff = fb.get_diff_and_promote()
    assert len(diff) == 1
    assert diff[0][:2] == (0, 0)


def test_force_full_redraw() -> None:
    fb = PixelFrameBuffer((2, 2))
    frame = np.zeros((2, 2, 3), dtype=np.uint8)
    fb.update_render(frame)
    fb.get_diff_and_promote()

    fb.force_full_redraw_next_frame()
    fb.update_render(frame)
    diff = fb.get_diff_and_promote()
    assert len(diff) == 4
