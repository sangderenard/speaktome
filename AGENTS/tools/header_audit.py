#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Audit header compliance for tools and tests."""
from __future__ import annotations

try:
    import re
    from pathlib import Path
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

ROOT = Path(__file__).resolve().parents[2]
PATTERN = re.compile(r"# --- BEGIN HEADER ---")


def has_header(path: Path) -> bool:
    try:
        text = path.read_text(encoding="utf-8")
    except Exception:
        return False
    return "# --- BEGIN HEADER ---" in text and "# --- END HEADER ---" in text


def audit() -> int:
    """Print files missing headers and return count."""
    dirs = [ROOT / "AGENTS" / "tools", ROOT / "tests", ROOT / "testing"]
    missing: list[Path] = []
    for d in dirs:
        for path in d.rglob("*.py"):
            if not has_header(path):
                missing.append(path.relative_to(ROOT))
    if missing:
        print("Missing headers:")
        for m in missing:
            print(f" - {m}")
    else:
        print("All headers present")
    return len(missing)


if __name__ == "__main__":  # pragma: no cover - manual run
    raise SystemExit(audit())
