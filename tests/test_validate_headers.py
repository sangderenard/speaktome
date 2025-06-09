"""Tests for the header validation utility."""

import logging
from importlib import reload
from pathlib import Path

import pytest

import AGENTS.tools.validate_headers as vh
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


def test_validate_headers_rewrite(tmp_path, monkeypatch) -> None:
    """Ensure rewrite mode inserts missing stubs."""
    pkg = tmp_path / "speaktome"
    pkg.mkdir()
    mod = pkg / "bar.py"
    mod.write_text("class Bar:\n    pass\n")
    monkeypatch.setattr(vh, "PACKAGE_ROOT", pkg)
    code = vh.validate(pkg, rewrite=True)
    assert code == 0
    content = mod.read_text()
    assert "HEADER" in content
    assert "def test()" in content
    # Subsequent validation should succeed without rewrites
    code = vh.validate(pkg)
    assert code == 0
