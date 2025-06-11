"""Tests for the `test_all_headers` utility."""

import logging

from AGENTS.tools.test_all_headers import run_test
from tensors.faculty import Faculty

# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_run_test_basic() -> None:
    """Ensure ``run_test`` executes class tests and captures output."""
    logger.info("test_run_test_basic start")
    result = run_test("speaktome.util.token_vocab", "TokenVocabulary", Faculty.PURE_PYTHON)
    assert result["ok"] is True
    assert isinstance(result["stdout"], str)
    assert isinstance(result["stderr"], str)
    logger.info("test_run_test_basic end")
