"""Tests for the :mod:`time_sync` utilities."""

import datetime as dt
from unittest import mock

import pytest

from time_sync import adjust_datetime, compose_ascii_digits, get_offset, set_offset, sync_offset

# --- END HEADER ---


def test_adjust_datetime(monkeypatch):
    base = dt.datetime(2024, 1, 1, 0, 0, 0)
    monkeypatch.setenv("SPEAKTOME_TIME_OFFSET", "10")
    adjusted = adjust_datetime(base)
    assert adjusted == base + dt.timedelta(seconds=10)


def test_sync_offset_fallback(monkeypatch):
    monkeypatch.setenv("SPEAKTOME_TIME_OFFSET", "5")
    with mock.patch("time_sync._internet.fetch_internet_utc", side_effect=OSError):
        val = sync_offset()
    assert val == 5
    assert get_offset() == 5


def test_compose_ascii_digits():
    art = compose_ascii_digits("12")
    lines = art.splitlines()
    assert len(lines) == 5
    assert all(len(line) > 0 for line in lines)
