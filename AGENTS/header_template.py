#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Template for SPEAKTOME module headers."""
from __future__ import annotations

try:
    import your_modules
except Exception:
    import os
    import sys
    from pathlib import Path
    import subprocess
    try:
        root = Path(__file__).resolve()
        for parent in [root, *root.parents]:
            if (parent / "pyproject.toml").is_file():
                root = parent
                break
        subprocess.run(
            [
                sys.executable,
                "-m",
                "AGENTS.tools.auto_env_setup",
                str(root),
            ],
            check=False,
        )
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
