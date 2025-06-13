from __future__ import annotations

try:
    import os
    from timesync.frame_buffer import PixelFrameBuffer
    import numpy as np

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
def test_framebuffer_diff_basic():
    fb = PixelFrameBuffer((2, 3))
    frame = np.zeros((2, 3, 3), dtype=np.uint8)
    frame[0, 0] = [255, 0, 0]
    fb.update_render(frame)
    diff = fb.get_diff_and_promote()
    assert len(diff) == 6
    assert (0, 0, (255, 0, 0)) in diff

    diff = fb.get_diff_and_promote()
    assert diff == []


def test_framebuffer_resize():
    fb = PixelFrameBuffer((1, 2))
    frame1 = np.zeros((1, 2, 3), dtype=np.uint8)
    fb.update_render(frame1)
    fb.get_diff_and_promote()

    frame2 = np.zeros((2, 3, 3), dtype=np.uint8)
    fb.update_render(frame2)
    diff = fb.get_diff_and_promote()
    assert fb.buffer_shape == (2, 3, 3)
    assert len(diff) == 6
