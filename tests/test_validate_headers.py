#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Tests for the header validation utility."""
from __future__ import annotations

try:
    import logging
    from importlib import reload
    from pathlib import Path

    import pytest

    import AGENTS.tools.validate_headers as vh
    from AGENTS.tools.header_utils import get_env_setup_box
except Exception:
    import os
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensor printing",
            "time_sync",
            "AGENTS",
            "fontmapper",
            "tensors",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    if "ENV_SETUP_BOX" not in os.environ:
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = _find_repo_root(Path(__file__))
        groups = os.environ.get("SPEAKTOME_GROUPS", "")
        cmd = [sys.executable, "-m", "AGENTS.tools.auto_env_setup", str(root)]
        if groups:
            for g in groups.split(","):
                cmd.append(f"-groups={g}")
        subprocess.run(cmd, check=False)
    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"[HEADER] import failure in {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
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
    logger.info("test_validate_headers_rewrite start")
    pkg = tmp_path / "speaktome"
    pkg.mkdir()
    mod = pkg / "foo.py"
    mod.write_text("class Foo:\n    pass\n")
    monkeypatch.setattr(vh, "PACKAGE_ROOT", pkg)
    code = vh.validate(pkg, rewrite=True)
    assert code == 0
    assert (pkg / "foo.py").read_text().startswith("#!/usr/bin/env python3")
    logger.info("test_validate_headers_rewrite end")
