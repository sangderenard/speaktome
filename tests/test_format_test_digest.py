"""Tests for the `format_test_digest` utility."""

import logging
from pathlib import Path

from format_test_digest import format_digest

# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_digest_counts(tmp_path: Path) -> None:
    """Ensure tagged lines are summarised correctly."""
    logger.info("test_digest_counts start")
    log = tmp_path / "sample.log"
    log.write_text(
        """INFO root: [FACULTY_SKIP] some skipped test\n"
        "INFO root: [AGENT_ACTIONABLE_ERROR] something failed\n"
        "INFO root: [TEST_PASS] success\n""",
        encoding="utf-8",
    )
    md = format_digest(log)
    assert "FACULTY_SKIP: 1" in md
    assert "AGENT_ACTIONABLE_ERROR: 1" in md
    assert "TEST_PASS: 1" in md
    logger.info("test_digest_counts end")
