"""Tests for the :mod:`timesync` utilities."""

import datetime as dt
from unittest import mock

import pytest

ntplib = pytest.importorskip("ntplib")
from timesync import (
    adjust_datetime,
    compose_ascii_digits,
    get_offset,
    set_offset,
    sync_offset,
    print_digital_clock,
    RenderingBackend,
)
from timesync.timesync.theme_manager import ThemeManager

# --- END HEADER ---


def test_adjust_datetime(monkeypatch):
    base = dt.datetime(2024, 1, 1, 0, 0, 0)
    monkeypatch.setenv("SPEAKTOME_TIME_OFFSET", "10")
    adjusted = adjust_datetime(base)
    assert adjusted == base + dt.timedelta(seconds=10)


def test_sync_offset_fallback(monkeypatch):
    monkeypatch.setenv("SPEAKTOME_TIME_OFFSET", "5")
    with mock.patch("timesync._internet.fetch_internet_utc", side_effect=OSError):
        val = sync_offset()
    assert val == 5
    assert get_offset() == 5


def test_compose_ascii_digits():
    art = compose_ascii_digits("12")
    lines = art.splitlines()
    assert len(lines) == 5
    assert all(len(line) > 0 for line in lines)


def test_compose_ascii_digits_dot():
    art = compose_ascii_digits(".")
    lines = art.splitlines()
    assert len(lines) == 5
    assert lines[-1].strip()  # bottom row contains the dot


def test_print_digital_clock(capsys):
    time = dt.datetime(2024, 1, 1, 12, 0, 0)
    print_digital_clock(time)
    captured = capsys.readouterr().out
    assert "12" in captured


def test_list_available_operations():
    rb = RenderingBackend(ThemeManager())
    ops = rb.list_available_operations()
    assert "rotate" in ops
