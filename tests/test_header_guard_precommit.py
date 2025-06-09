"""Tests for header guard precommit checks."""

from pathlib import Path

import AGENTS.tools.header_guard_precommit as hg
# --- END HEADER ---

def test_check_try_header_pass(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text(
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    print('warn')\n# --- END HEADER ---\n"
    )
    assert hg.check_try_header(path) == []

def test_check_try_header_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("import os\n# --- END HEADER ---\n")
    errors = hg.check_try_header(path)
    assert errors
