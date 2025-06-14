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

    def parse_pyproject_groups(pyproject: Path) -> list[str]:
        try:
            try:
                import tomllib
            except ModuleNotFoundError:  # Python < 3.11
                import tomli as tomllib
        except Exception:
            return []
        try:
            data = tomllib.loads(pyproject.read_text())
        except Exception:
            return []

        groups = set()
        groups.update(data.get("project", {}).get("optional-dependencies", {}).keys())
        tool = data.get("tool", {}).get("poetry", {})
        groups.update(tool.get("group", {}).keys())
        groups.update(tool.get("extras", {}).keys())
        return sorted(groups)

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
        pyproject = root / "pyproject.toml"
        groups = parse_pyproject_groups(pyproject)
        codebase = guess_codebase(Path(__file__))
        base_cmd = [
            sys.executable,
            "-m",
            "AGENTS.tools.auto_env_setup",
            str(root),
        ]
        if codebase:
            base_cmd.append(f"-codebases={codebase}")
        subprocess.run(base_cmd, check=False)
        for grp in groups:
            subprocess.run(base_cmd + [f"-groups={grp}"], check=False)

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


# ########## STUB: recursive test runner ##########
# PURPOSE: Execute class level ``test()`` methods under each faculty tier.
# EXPECTED BEHAVIOR: Import each class in a subprocess with the environment
# variable ``SPEAKTOME_FACULTY`` set and call ``Class.test()``.
# INPUTS: JSON data produced by ``dump_headers.py`` or scanned directly.
# OUTPUTS: JSON summary of pass/fail results for optional markdown reporting.
# KEY ASSUMPTIONS/DEPENDENCIES: Requires Python interpreter access and the
# ``speaktome`` package importable in subprocesses.
# TODO:
#   - Capture stdout/stderr from tests for logging. [DONE]
#   - Surface structured results for ``format_test_digest.py``. [DONE]
# NOTES: This is a minimal scaffold and does not implement all failure modes.
# ###########################################################################

SCRIPT_PATH = Path(__file__).with_name("dump_headers.py")


def load_headers() -> dict:
    data = subprocess.check_output([sys.executable, str(SCRIPT_PATH)])
    return json.loads(data)


def run_test(mod: str, cls: str, faculty: Faculty) -> dict[str, object]:
    """Return structured results for ``cls.test`` under ``faculty``."""
    env = os.environ.copy()
    env[FORCE_ENV] = faculty.name
    code = (
        "import importlib, sys; "
        f"mod = importlib.import_module('{mod}'); "
        f"cls = getattr(mod, '{cls}'); "
        "getattr(cls, 'test', lambda: None)();"
    )
    proc = subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
    )
    return {
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main() -> None:
    headers = load_headers()
    results = {}
    for mod, items in headers.items():
        for item in items:
            cls = item["class"]
            key = f"{mod}.{cls}"
            results[key] = {}
            for fac in Faculty:
                results[key][fac.name] = run_test(mod, cls, fac)

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
