#!/usr/bin/env python3
# --- BEGIN HEADER ---
"""Run header repair, validation and tests in one step."""
from __future__ import annotations

try:
    from pathlib import Path
    import subprocess
    import sys
    import os
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

