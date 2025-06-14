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



# ########## header_check_orchestrator ##########
# PURPOSE: Provide a unified CLI that runs ``auto_fix_headers.py``,
#          ``validate_headers.py`` and ``test_all_headers.py`` in
#          sequence.
# EXPECTED BEHAVIOR: This module executes both tools,
#          aggregates their results, and emits a single exit code
#          summarizing success or failure.
# INPUTS: Optional command-line flags controlling paths and verbosity.
# OUTPUTS: Aggregated textual report printed to stdout.
# KEY ASSUMPTIONS/DEPENDENCIES: Assumes both helper scripts are present
#          in ``AGENTS/tools`` and accessible via the current Python
#          interpreter.
# TODO:
#   - Capture and merge stdout/stderr from the called scripts.
# ###########################################################################

def run_checks(path: Path, *, rewrite: bool = False) -> int:
    """Run header repair, validation and header-based tests."""
    script_dir = Path(__file__).resolve().parent
    repair = script_dir / "auto_fix_headers.py"
    validate = script_dir / "validate_headers.py"
    test_all = script_dir / "test_all_headers.py"

    code = 0
    subprocess.run([sys.executable, str(repair), str(path)])
    args = [sys.executable, str(validate), str(path)]
    if rewrite:
        args.append("--rewrite")
    proc = subprocess.run(args)
    code |= proc.returncode
    proc = subprocess.run([sys.executable, str(test_all)])
    code |= proc.returncode
    return code


class HeaderCheckRunner:
    """CLI wrapper for :func:`run_checks`."""

    @staticmethod
    def test() -> None:
        run_checks(Path('.'), rewrite=False)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Repair, validate and test file headers"
    )
    parser.add_argument("path", nargs="?", default=".", type=Path)
    parser.add_argument(
        "--rewrite",
        action="store_true",
        help="Pass --rewrite to validate_headers.py",
    )
    args = parser.parse_args()
    code = run_checks(args.path, rewrite=args.rewrite)
    sys.exit(code)
