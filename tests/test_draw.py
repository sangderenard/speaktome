#!/usr/bin/env python3
"""Tests for ASCII drawing utilities."""
from __future__ import annotations

try:
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
    import numpy as np
    from time_sync.draw import (
        get_changed_subunits,
        default_subunit_batch_to_chars,
        flexible_subunit_kernel,
    )
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


def test_get_changed_subunits_basic():
    old = np.zeros((2, 2, 3), dtype=np.uint8)
    new = old.copy()
    new[0, 0] = [255, 0, 0]
    subs = get_changed_subunits(old, new, 1, 1)
    assert len(subs) == 1
    y, x, data = subs[0]
    assert (y, x) == (0, 0)
    assert np.array_equal(data, np.array([[[255, 0, 0]]], dtype=np.uint8))


def test_get_changed_subunits_edge_short_circuit():
    old = np.zeros((2, 2), dtype=np.uint8)
    new = old.copy()
    subs = get_changed_subunits(old, new, 3, 3)
    assert len(subs) == 1
    y, x, data = subs[0]
    assert (y, x) == (0, 0)
    # Partial subunit is returned without padding
    assert data.shape == (2, 2)
    assert np.array_equal(data, new)


def test_get_changed_subunits_threshold():
    old = np.zeros((1, 1), dtype=np.uint8)
    new = np.array([[1]], dtype=np.uint8)
    # Difference is 1; with threshold 5 no change should be reported
    subs = get_changed_subunits(old, new, 1, 1, loss_threshold=5)
    assert len(subs) == 0
    # Threshold 0 should flag the change
    subs = get_changed_subunits(old, new, 1, 1, loss_threshold=0)
    assert len(subs) == 1


def test_default_kernel_char():
    arr = np.array([[[[255, 255, 255]]]], dtype=np.uint8)
    chars = default_subunit_batch_to_chars(arr)
    assert isinstance(chars[0], str) and len(chars[0]) == 1


def test_flexible_kernel_modes():
    arr = np.array([[[100, 150, 200]]], dtype=np.uint8)

    ascii_char = flexible_subunit_kernel(arr, " .:░▒▓█", mode="ascii")
    assert isinstance(ascii_char, str)

    raw = flexible_subunit_kernel(arr, " .:░▒▓█", mode="raw")
    assert isinstance(raw, np.ndarray)
    assert raw.shape == arr.shape

    hybrid = flexible_subunit_kernel(arr, " .:░▒▓█", mode="hybrid")
    assert isinstance(hybrid, np.ndarray)
    assert hybrid.shape[:2] == arr.shape[:2]
