#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Interactive tensor operations menu."""
from __future__ import annotations

try:
    import importlib.util
    import os
    import sys
    from typing import Any, Tuple
    import tensors as ta
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

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)


def available_backends() -> list[tuple[str, type[ta.AbstractTensor]]]:
    backends: list[tuple[str, type[ta.AbstractTensor]]] = [
        (name, backend)
        for name, backend in ta.BACKENDS.items()
    ]
    return backends


def main() -> None:
    for name, backend in available_backends():
        print(name, backend)


if __name__ == "__main__":  # pragma: no cover - manual use
    main()
