#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Tests for the semantic spectrum prototype."""
from __future__ import annotations

try:
    import numpy as np
    from todo.semantic_spectrum_prototype import compute
except Exception:
    import os
    import sys
    from pathlib import Path
    from AGENTS.tools.path_utils import find_repo_root
    if "ENV_SETUP_BOX" not in os.environ:
        root = find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = find_repo_root(Path(__file__))
        subprocess.run(
            [sys.executable, "-m", "AGENTS.tools.auto_env_setup", str(root)],
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

def test_pca_variance_line():
    """PCA on points along a line should explain nearly all variance."""
    X = np.array([[0, 0], [1, 1], [2, 2]], dtype=float)
    coords, var = compute(X)
    assert np.isclose(var, 1.0)
    assert np.isclose(coords[1], 0.0)
