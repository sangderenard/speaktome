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
    import json

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensorprinting",
            "timesync",
            "AGENTS",
            "fontmapper",
            "tensors",
            "testenv",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    ROOT = _find_repo_root(Path(__file__))
    MAP_FILE = ROOT / "AGENTS" / "codebase_map.json"

    def guess_codebase(path: Path, map_file: Path = MAP_FILE) -> str | None:
        """Return codebase name owning ``path``.

        Tries ``codebase_map.json`` first, then falls back to scanning path
        components for known codebase names.
        """
        try:
            data = json.loads(map_file.read_text())
        except Exception:
            data = None

        if data:
            for name, info in data.items():
                cb_path = ROOT / info.get("path", name)
                try:
                    path.relative_to(cb_path)
                    return name
                except ValueError:
                    continue
        else:
            candidates = {
                "speaktome",
                "laplace",
                "tensorprinting",
                "timesync",
                "fontmapper",
                "tensors",
                "testenv",
                "tools",
            }
            for part in path.parts:
                if part in candidates:
                    return part

        return None

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
