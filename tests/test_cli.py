#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""CLI entry-point tests."""
from __future__ import annotations

try:
    import subprocess
    import sys
    import pytest

    pytestmark = pytest.mark.requires_torch

    from speaktome.util.cli_permutations import CLIArgumentMatrix
except Exception:
    import os
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensor_printing",
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


def test_help_message() -> None:
    """Invoke ``--help`` and ensure the usage banner appears."""
    result = subprocess.run([
        sys.executable,
        '-m', 'speaktome.speaktome',
        '-h'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower()


def test_basic_combinations() -> None:
    """Run a minimal CLI cycle with a few argument permutations."""
    pytest.importorskip('torch', reason='CLI requires torch for full run')
    pytest.importorskip('transformers', reason='CLI requires transformers for full run with torch')
    matrix = CLIArgumentMatrix()
    matrix.add_option('--max_steps', [1])
    matrix.add_option('--safe_mode', [None])
    combos = matrix.generate()
    for combo in combos:
        result = subprocess.run([
            sys.executable,
            '-m', 'speaktome.speaktome',
            *combo,
            'hi'
        ], capture_output=True, text=True)
        assert result.returncode == 0
