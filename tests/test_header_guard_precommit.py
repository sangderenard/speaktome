#!/usr/bin/env python3
"""Tests for header guard precommit checks."""
from __future__ import annotations

try:
    from pathlib import Path
    import AGENTS.tools.header_guard_precommit as hg
    from AGENTS.tools.header_utils import ENV_SETUP_BOX
except Exception:
    print(ENV_SETUP_BOX)
    raise
# --- END HEADER ---


def test_check_try_header_pass(tmp_path: Path) -> None:
    path = tmp_path / "ok.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "# --- BEGIN HEADER ---\n"
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    import sys\n    from pathlib import Path\n    from AGENTS.tools.auto_env_setup import run_setup_script\n    run_setup_script(Path(__file__).resolve().parents[1])\n    print(f'[HEADER] import failure in {__file__}')\n    print(ENV_SETUP_BOX)\n    sys.exit(1)\n# --- END HEADER ---\n"
    )
    assert hg.check_try_header(path) == []


def test_check_try_header_fail(tmp_path: Path) -> None:
    path = tmp_path / "bad.py"
    path.write_text("import os\n# --- END HEADER ---\n")
    errors = hg.check_try_header(path)
    assert "Missing '# --- BEGIN HEADER ---'" in errors
    assert "Missing shebang" in errors
    assert "Missing module docstring" in errors


def test_check_env_print_missing(tmp_path: Path) -> None:
    path = tmp_path / "noprint.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "# --- BEGIN HEADER ---\n"
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    import sys\n    sys.exit(1)\n# --- END HEADER ---\n"
    )
    errors = hg.check_try_header(path)
    assert "Missing 'print(ENV_SETUP_BOX)' in except block" in errors


def test_check_sys_import_missing(tmp_path: Path) -> None:
    path = tmp_path / "nosys.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "# --- BEGIN HEADER ---\n"
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    print(ENV_SETUP_BOX)\n    sys.exit(1)\n# --- END HEADER ---\n"
    )
    errors = hg.check_try_header(path)
    assert "Missing 'import sys' in except block" in errors


def test_check_sys_exit_missing(tmp_path: Path) -> None:
    path = tmp_path / "noexit.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "# --- BEGIN HEADER ---\n"
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    import sys\n    print(ENV_SETUP_BOX)\n# --- END HEADER ---\n"
    )
    errors = hg.check_try_header(path)
    assert "Missing 'sys.exit(1)' in except block" in errors


def test_check_run_setup_missing(tmp_path: Path) -> None:
    path = tmp_path / "nosetup.py"
    path.write_text(
        "#!/usr/bin/env python3\n"
        "# --- BEGIN HEADER ---\n"
        '"""doc"""\n'
        "from __future__ import annotations\n"
        "try:\n    import os\nexcept Exception:\n    import sys\n    print(ENV_SETUP_BOX)\n    sys.exit(1)\n# --- END HEADER ---\n"
    )
    errors = hg.check_try_header(path)
    assert "Missing 'run_setup_script' in except block" in errors
