import numpy as np
from time_sync.draw import get_changed_subunits, default_subunit_to_char_kernel


def test_get_changed_subunits_basic():
    old = np.zeros((2, 2, 3), dtype=np.uint8)
    new = old.copy()
    new[0, 0] = [255, 0, 0]
    subs = get_changed_subunits(old, new, 1, 1)
    assert len(subs) == 1
    y, x, data = subs[0]
    assert (y, x) == (0, 0)
    assert np.array_equal(data, np.array([[[255, 0, 0]]], dtype=np.uint8))


def test_default_kernel_char():
    arr = np.array([[[255, 255, 255]]], dtype=np.uint8)
    char = default_subunit_to_char_kernel(arr)
    assert isinstance(char, str) and len(char) == 1
