"""Tests for the guestbook validation helper."""
from __future__ import annotations

try:
    import os
    import shutil
    import logging
    from pathlib import Path
    from importlib import reload
    import pytest

    ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]

    import AGENTS.validate_guestbook as vg
except Exception:
    import sys
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---
SCRIPT = Path('AGENTS/validate_guestbook.py')

logger = logging.getLogger(__name__)


@pytest.fixture
def temp_reports(tmp_path, monkeypatch):
    dst = tmp_path / 'experience_reports'
    dst.mkdir()
    (dst / 'template_experience_report.md').write_text('')
    base = 1749417600
    for i in range(12):
        fname = dst / f'{base + i}_v1_Test{i}.md'
        fname.write_text('x')
    vg_mod = reload(vg)
    monkeypatch.setattr(vg_mod, 'REPORTS_DIR', str(dst))
    monkeypatch.setattr(vg_mod, 'ARCHIVE_DIR', str(dst / 'archive'))
    monkeypatch.setattr(vg_mod, 'STICKIES_FILE', str(dst / 'stickies.txt'))
    return dst


def test_archives_old_files(temp_reports) -> None:
    """Verify that older reports are moved into ``archive/``."""
    logger.info("test_archives_old_files start")
    vg.validate_and_fix()
    vg.archive_old_reports()
    archived_files = sorted(os.listdir(temp_reports / 'archive'))
    assert len(archived_files) == 2
    logger.info("test_archives_old_files end")


