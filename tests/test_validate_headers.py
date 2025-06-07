"""Tests for the header validation utility."""

import logging
from importlib import reload
from pathlib import Path

import pytest

import validate_headers as vh
# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_validate_headers(tmp_path, monkeypatch) -> None:
    """Check that missing HEADER or test triggers an error code."""
    logger.info("test_validate_headers start")
    pkg = tmp_path / "speaktome"
    pkg.mkdir()
    mod = pkg / "foo.py"
    mod.write_text("class Foo:\n    pass\n")
    monkeypatch.setattr(vh, "PACKAGE_ROOT", pkg)
    code = vh.validate(pkg)
    assert code == 1
    logger.info("test_validate_headers end")
