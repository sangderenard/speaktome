#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Ensure optional dependencies via auto setup."""
from __future__ import annotations

try:
    import sys
    import subprocess
    from pathlib import Path
    import importlib.util
    try:
        import tomllib
    except ModuleNotFoundError:  # Python < 3.11
        import tomli as tomllib
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

PYPROJECT = Path(__file__).resolve().parent / "pyproject.toml"


def parse_extras() -> dict[str, list[str]]:
    data = tomllib.loads(PYPROJECT.read_text())
    return data.get("project", {}).get("optional-dependencies", {})


def ensure_groups(groups: dict[str, list[str]]) -> None:
    for group in groups:
        subprocess.run(
            [
                sys.executable,
                "-m",
                "AGENTS.tools.auto_env_setup",
                str(PYPROJECT.parent),
                f"-groups={group}",
            ],
            check=False,
        )


def main() -> int:
    extras = parse_extras()
    if not extras:
        print("No optional dependencies defined.")
        return 0
    ensure_groups(extras)
    return 0


if __name__ == "__main__":  # pragma: no cover - manual invocation
    raise SystemExit(main())
