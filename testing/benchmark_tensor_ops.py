#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Micro benchmark for tensor backends."""
from __future__ import annotations

try:
    import time
    from tensors.faculty import Faculty
    from tensors import get_tensor_operations
except Exception:
    import os
    import sys
    from pathlib import Path

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


def benchmark_sqrt(faculty: Faculty, reps: int = 1000) -> float:
    """Return seconds taken to repeatedly compute sqrt."""
    ops = get_tensor_operations(faculty)
    tensor = ops.tensor_from_list(list(range(100)), dtype=ops.float_dtype, device="cpu")
    start = time.process_time()
    for _ in range(reps):
        ops.sqrt(tensor)
    return time.process_time() - start


if __name__ == "__main__":  # pragma: no cover - manual benchmark
    for fac in Faculty:
        try:
            total = benchmark_sqrt(fac)
        except Exception:
            continue
        print(fac.name, total)
