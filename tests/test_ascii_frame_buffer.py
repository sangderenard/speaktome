from time_sync.frame_buffer import AsciiFrameBuffer
import numpy as np


def test_framebuffer_diff_basic():
    fb = AsciiFrameBuffer((2, 3))
    frame = np.array([list("abc"), list("def")])
    fb.update_render(frame)
    diff = fb.get_diff_and_promote()
    assert len(diff) == 6
    assert (0, 0, "a") in diff

    # No changes -> diff empty
    diff = fb.get_diff_and_promote()
    assert diff == []


def test_framebuffer_resize():
    fb = AsciiFrameBuffer((1, 2))
    frame1 = np.array([list("hi")])
    fb.update_render(frame1)
    fb.get_diff_and_promote()

    frame2 = np.array([list("abc"), list("def")])
    fb.update_render(frame2)
    diff = fb.get_diff_and_promote()
    assert fb.shape == (2, 3)
    assert len(diff) == 6
